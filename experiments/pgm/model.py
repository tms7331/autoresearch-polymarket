"""
Semantic Event Graph with CPDs — model.py

Architecture:
- Semantic Layer: sqlite-vec vector DB maps text to canonical event nodes
- Event Nodes: article-derived semantic clusters stored in vector DB
- Bayesian Network: Category × EvidenceLevel → PriceBucket (CPDs learned from data)
- Inference: pgmpy VariableElimination

Pipeline: Text → Embedding → Vector DB match → Event Node → Evidence Level →
          Bayesian Network → Posterior Probability

The agent iterates on this file to improve prediction quality.
"""

import sqlite3
import re
from dataclasses import dataclass, field

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from prepare import Market, Article, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, well-tested
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.55   # cosine sim to merge into existing event node
EVIDENCE_THRESHOLD = 0.40     # min cosine sim for article-market relevance
MAX_CHUNKS_PER_ARTICLE = 3    # sentences to extract per article
MAX_TOTAL_CHUNKS = 5000       # cap to stay within time budget

# BN node states
EVIDENCE_LEVELS = ["none", "low", "medium", "high"]
PRICE_BUCKETS = ["low", "mid", "high"]
BUCKET_MIDPOINTS = np.array([0.17, 0.50, 0.83])

CATEGORY_PATTERNS = {
    "politics": r"\b(president|congress|senate|election|vote|governor|mayor|party|democrat|republican|trump|biden|legislation|bill|act)\b",
    "economics": r"\b(gdp|inflation|interest rate|fed|recession|unemployment|tariff|trade|cpi|jobs|economic)\b",
    "crypto": r"\b(bitcoin|btc|ethereum|eth|crypto|token|blockchain|defi|nft|solana)\b",
    "markets": r"\b(stock|s&p|dow|nasdaq|index|price|above|below|market cap)\b",
    "geopolitics": r"\b(war|ceasefire|military|nato|sanction|invasion|troops|missile|conflict|peace)\b",
    "tech": r"\b(ai|artificial intelligence|openai|google|apple|meta|microsoft|launch|release|chip)\b",
    "sports": r"\b(win|championship|nba|nfl|mlb|nhl|game|match|tournament|playoff|super bowl|world cup)\b",
    "science": r"\b(climate|vaccine|fda|trial|study|space|nasa|launch|species)\b",
}
CATEGORY_LIST = list(CATEGORY_PATTERNS.keys()) + ["other"]
NUM_CATEGORIES = len(CATEGORY_LIST)


# ---------------------------------------------------------------------------
# Prediction Model Interface (must implement these)
# ---------------------------------------------------------------------------

class PredictionModel:
    """Base prediction model. Subclass and override build() and predict()."""

    def build(self, dataset: Dataset):
        """Build the model from articles and market text. Called once."""
        raise NotImplementedError

    def predict(self, market: Market) -> float:
        """Predict probability of "Yes" for a market. Returns float in [0, 1]."""
        raise NotImplementedError

    def predict_batch(self, markets: list[Market]) -> list[tuple[str, float]]:
        """Predict probabilities for a batch of markets. Returns (market_id, prob) pairs."""
        return [(m.id, self.predict(m)) for m in markets]

    def price_event(self, description: str) -> dict:
        """Tool interface: price a free-text event description."""
        raise NotImplementedError

    def stats(self) -> dict:
        """Return model statistics (num_nodes, num_edges, etc.)."""
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_f32(vec: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes for sqlite-vec."""
    return vec.astype(np.float32).tobytes()


def _l2_to_cosine(l2_dist: float) -> float:
    """Convert L2 distance to cosine similarity (for unit-norm vectors).

    For normalized vectors: ||a-b||² = 2(1 - cos(a,b))
    So cos(a,b) = 1 - ||a-b||²/2
    """
    return 1.0 - (l2_dist ** 2) / 2.0


def _categorize(text: str) -> str:
    """Classify text into a category using keyword patterns."""
    text_lower = text.lower()
    scores = {}
    for cat, pattern in CATEGORY_PATTERNS.items():
        scores[cat] = len(re.findall(pattern, text_lower))
    if not scores or max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def _price_bucket(price: float) -> int:
    if price < 0.33:
        return 0  # low
    elif price < 0.66:
        return 1  # mid
    else:
        return 2  # high


def _evidence_level(n_matches: int) -> int:
    if n_matches == 0:
        return 0  # none
    elif n_matches <= 2:
        return 1  # low
    elif n_matches <= 5:
        return 2  # medium
    else:
        return 3  # high


def _bucket_to_prob(dist: np.ndarray) -> float:
    """Convert distribution over [low, mid, high] buckets to a point estimate."""
    return float(np.dot(dist, BUCKET_MIDPOINTS))


def _extract_sentences(text: str, max_n: int = 3) -> list[str]:
    """Extract first N meaningful sentences from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 30][:max_n]


# ---------------------------------------------------------------------------
# Event Node — a canonical event in the semantic graph
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """A node representing a semantic cluster of observations."""
    id: str
    label: str                                    # representative text
    embedding: np.ndarray                         # centroid vector
    aliases: list[str] = field(default_factory=list)  # all phrases mapped here
    observation_count: int = 0                    # how many article chunks mapped


# ---------------------------------------------------------------------------
# Semantic Event Graph
# ---------------------------------------------------------------------------

class SemanticEventGraph(PredictionModel):
    """
    Semantic Event Graph with CPDs.

    Components:
    1. Semantic Layer (sqlite-vec) — resolves text to canonical event nodes
    2. Event Nodes — article-derived clusters, each a semantic unit
    3. Bayesian Network — Category × EvidenceLevel → PriceBucket
    4. CPDs — learned from training data with Laplace smoothing
    5. Inference — pgmpy VariableElimination
    """

    def __init__(self):
        self.embedder = None
        self.db = None              # sqlite connection with vec extension
        self.event_nodes: dict[str, EventNode] = {}
        self.rowid_to_node: dict[int, str] = {}
        self._next_id = 0
        self.bn = None
        self.inference_engine = None
        self.fallback_dist = np.array([1/3, 1/3, 1/3])
        self.num_articles = 0
        self._market_embeddings: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, dataset: Dataset):
        self.num_articles = len(dataset.articles)

        # 1. Load embedding model
        print("  Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 2. Initialize vector DB
        self._init_vec_db()

        # 3. Ingest articles → create event nodes in vector DB
        print("  Ingesting articles into event graph...")
        self._ingest_articles(dataset.articles)
        print(f"  Created {len(self.event_nodes)} event nodes from articles")

        # 4. Embed markets and compute evidence features
        print("  Computing market evidence features...")
        market_texts = [f"{m.question} {m.description[:200]}" for m in dataset.markets]
        market_embs = (
            self.embedder.encode(market_texts, show_progress_bar=False,
                                 normalize_embeddings=True)
            if market_texts else np.array([])
        )

        categories = []
        evidence_levels = []
        price_buckets = []

        for i, market in enumerate(dataset.markets):
            self._market_embeddings[market.id] = market_embs[i]
            cat = _categorize(market.question + " " + market.description)
            n_matches = self._count_evidence(market_embs[i])
            categories.append(cat)
            evidence_levels.append(_evidence_level(n_matches))
            price_buckets.append(_price_bucket(market.market_price))

        # 5. Build Bayesian Network with learned CPDs
        print("  Building Bayesian network...")
        self._build_bn(categories, evidence_levels, price_buckets)

    def _init_vec_db(self):
        """Create sqlite-vec virtual table for event node embeddings."""
        self.db = sqlite3.connect(":memory:")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute(f"""
            CREATE VIRTUAL TABLE vec_events USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

    def _ingest_articles(self, articles: list[Article]):
        """Embed article chunks, create/merge event nodes via vector DB."""
        all_chunks = []
        for article in articles:
            chunks = []
            if article.title and len(article.title) > 20:
                chunks.append(article.title)
            chunks.extend(_extract_sentences(article.body, MAX_CHUNKS_PER_ARTICLE))
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Cap to stay within time budget
        if len(all_chunks) > MAX_TOTAL_CHUNKS:
            all_chunks = all_chunks[:MAX_TOTAL_CHUNKS]

        # Batch embed
        embeddings = self.embedder.encode(
            all_chunks, show_progress_bar=False,
            normalize_embeddings=True, batch_size=256,
        )

        # Map each chunk to an event node (create or merge)
        for chunk, emb in zip(all_chunks, embeddings):
            self._find_or_create_node(chunk, emb)

    def _find_or_create_node(self, text: str, embedding: np.ndarray) -> str:
        """Map text to existing event node (if similar enough) or create new one."""
        # Search vector DB for nearest existing node
        if self.event_nodes:
            rows = self.db.execute("""
                SELECT rowid, distance FROM vec_events
                WHERE embedding MATCH ?
                ORDER BY distance LIMIT 1
            """, [_serialize_f32(embedding)]).fetchall()

            if rows:
                rowid, dist = rows[0]
                sim = _l2_to_cosine(dist)
                if sim >= SIMILARITY_THRESHOLD:
                    node_id = self.rowid_to_node[rowid]
                    node = self.event_nodes[node_id]
                    node.aliases.append(text[:100])
                    node.observation_count += 1
                    return node_id

        # Create new event node
        self._next_id += 1
        node_id = f"evt_{self._next_id:04d}"
        node = EventNode(
            id=node_id,
            label=text[:100],
            embedding=embedding.copy(),
            aliases=[text[:100]],
            observation_count=1,
        )
        self.event_nodes[node_id] = node
        self.db.execute(
            "INSERT INTO vec_events(rowid, embedding) VALUES (?, ?)",
            [self._next_id, _serialize_f32(embedding)],
        )
        self.rowid_to_node[self._next_id] = node_id
        return node_id

    # ------------------------------------------------------------------
    # Evidence computation via vector DB
    # ------------------------------------------------------------------

    def _count_evidence(self, market_embedding: np.ndarray) -> int:
        """Count observation weight of event nodes relevant to a market."""
        if not self.event_nodes:
            return 0

        rows = self.db.execute("""
            SELECT rowid, distance FROM vec_events
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 20
        """, [_serialize_f32(market_embedding)]).fetchall()

        count = 0
        for rowid, dist in rows:
            sim = _l2_to_cosine(dist)
            if sim < EVIDENCE_THRESHOLD:
                break  # sorted by distance — rest will be worse
            node_id = self.rowid_to_node.get(rowid)
            if node_id:
                count += self.event_nodes[node_id].observation_count
        return count

    def _get_evidence_details(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Return (label, similarity) for relevant event nodes."""
        if not self.event_nodes:
            return []

        rows = self.db.execute("""
            SELECT rowid, distance FROM vec_events
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 10
        """, [_serialize_f32(embedding)]).fetchall()

        results = []
        for rowid, dist in rows:
            sim = _l2_to_cosine(dist)
            if sim < EVIDENCE_THRESHOLD:
                break
            node_id = self.rowid_to_node.get(rowid)
            if node_id:
                results.append((self.event_nodes[node_id].label, sim))
        return results

    # ------------------------------------------------------------------
    # Bayesian Network
    # ------------------------------------------------------------------

    def _build_bn(self, categories: list[str], evidence_levels: list[int],
                  price_buckets: list[int]):
        """Build BN: Category × EvidenceLevel → PriceBucket with learned CPDs."""
        n_ev = len(EVIDENCE_LEVELS)
        n_pb = len(PRICE_BUCKETS)

        # Count (category, evidence_level, price_bucket) joint occurrences
        counts = np.zeros((NUM_CATEGORIES, n_ev, n_pb), dtype=float)
        for cat, ev, pb in zip(categories, evidence_levels, price_buckets):
            cat_idx = CATEGORY_LIST.index(cat) if cat in CATEGORY_LIST else NUM_CATEGORIES - 1
            counts[cat_idx, ev, pb] += 1

        # Laplace smoothing
        counts += 1.0

        # P(PriceBucket | Category, EvidenceLevel)
        # pgmpy column order: Category varies slowest, EvidenceLevel varies fastest
        n_combos = NUM_CATEGORIES * n_ev
        cpd_values = np.zeros((n_pb, n_combos))
        for cat_idx in range(NUM_CATEGORIES):
            for ev_idx in range(n_ev):
                col = cat_idx * n_ev + ev_idx
                col_counts = counts[cat_idx, ev_idx, :]
                cpd_values[:, col] = col_counts / col_counts.sum()

        # Category prior (uniform)
        cat_prior = np.ones((NUM_CATEGORIES, 1)) / NUM_CATEGORIES

        # Evidence level prior (from data)
        ev_counts = np.zeros(n_ev)
        for ev in evidence_levels:
            ev_counts[ev] += 1
        ev_counts += 1.0
        ev_prior = (ev_counts / ev_counts.sum()).reshape(-1, 1)

        # Assemble BN
        self.bn = BayesianNetwork([
            ("Category", "PriceBucket"),
            ("EvidenceLevel", "PriceBucket"),
        ])

        cpd_cat = TabularCPD(
            variable="Category",
            variable_card=NUM_CATEGORIES,
            values=cat_prior,
            state_names={"Category": CATEGORY_LIST},
        )
        cpd_ev = TabularCPD(
            variable="EvidenceLevel",
            variable_card=n_ev,
            values=ev_prior,
            state_names={"EvidenceLevel": EVIDENCE_LEVELS},
        )
        cpd_price = TabularCPD(
            variable="PriceBucket",
            variable_card=n_pb,
            values=cpd_values,
            evidence=["Category", "EvidenceLevel"],
            evidence_card=[NUM_CATEGORIES, n_ev],
            state_names={
                "PriceBucket": PRICE_BUCKETS,
                "Category": CATEGORY_LIST,
                "EvidenceLevel": EVIDENCE_LEVELS,
            },
        )

        self.bn.add_cpds(cpd_cat, cpd_ev, cpd_price)
        assert self.bn.check_model()
        self.inference_engine = VariableElimination(self.bn)

        # Fallback distribution
        total = counts.sum(axis=(0, 1))
        self.fallback_dist = total / total.sum()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, market: Market) -> float:
        if self.inference_engine is None:
            return _bucket_to_prob(self.fallback_dist)

        cat = _categorize(market.question + " " + market.description)

        # Use cached embedding if available, else compute
        emb = self._market_embeddings.get(market.id)
        if emb is None:
            emb = self.embedder.encode(
                [f"{market.question} {market.description[:200]}"],
                show_progress_bar=False, normalize_embeddings=True,
            )[0]

        n_matches = self._count_evidence(emb)
        ev_level = EVIDENCE_LEVELS[_evidence_level(n_matches)]

        result = self.inference_engine.query(
            variables=["PriceBucket"],
            evidence={"Category": cat, "EvidenceLevel": ev_level},
        )
        return _bucket_to_prob(result.values)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def price_event(self, description: str) -> dict:
        emb = self.embedder.encode(
            [description], show_progress_bar=False, normalize_embeddings=True,
        )[0]

        cat = _categorize(description)
        n_matches = self._count_evidence(emb)
        ev_level = EVIDENCE_LEVELS[_evidence_level(n_matches)]
        evidence_details = self._get_evidence_details(emb)

        if self.inference_engine is not None:
            result = self.inference_engine.query(
                variables=["PriceBucket"],
                evidence={"Category": cat, "EvidenceLevel": ev_level},
            )
            dist = result.values
        else:
            dist = self.fallback_dist

        prob = _bucket_to_prob(dist)

        factors = [
            {"factor": f"Category: {cat}", "direction": "neutral", "weight": 1.0},
            {"factor": f"Evidence: {ev_level} ({n_matches} matches)",
             "direction": "neutral", "weight": 0.5},
            {"factor": f"P(low)={dist[0]:.2f}, P(mid)={dist[1]:.2f}, P(high)={dist[2]:.2f}",
             "direction": "neutral", "weight": 0.0},
        ]
        for label, sim in evidence_details[:3]:
            factors.append({
                "factor": f"Related: {label[:60]}",
                "direction": "supporting",
                "weight": round(sim, 3),
            })

        return {
            "probability": round(prob, 4),
            "confidence": round(float(1.0 - dist[1]), 2),
            "factors": factors,
            "related_keywords": [label for label, _ in evidence_details[:5]],
        }

    def stats(self) -> dict:
        return {
            "num_nodes": len(self.bn.nodes()) if self.bn else 0,
            "num_edges": len(self.bn.edges()) if self.bn else 0,
            "num_event_nodes": len(self.event_nodes),
            "num_articles": self.num_articles,
        }


# ---------------------------------------------------------------------------
# Model factory — change this to swap models
# ---------------------------------------------------------------------------

def create_model() -> PredictionModel:
    """Create and return the prediction model to use."""
    return SemanticEventGraph()
