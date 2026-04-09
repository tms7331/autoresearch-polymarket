"""
Semantic Event Graph with CPDs — model.py

Architecture:
- Semantic Layer: sqlite-vec vector DB maps text to canonical event nodes
- Event Nodes: article-derived semantic concepts (e.g., "Iranian strikes on Kuwait")
  Each is a binary variable in the Bayesian Network (observed / not observed)
- Market Nodes: prediction markets, each a binary variable (Yes / No)
- Edges: event → market when semantically relevant (top-k by cosine similarity)
- CPDs: P(market=Yes | parent_events) learned from data

Pipeline: Articles → Embeddings → Vector DB → Event Nodes (semantic clusters)
          Markets → Embeddings → Similarity search → Parent assignment
          → Build BN with event nodes as parents of market nodes
          → Inference via direct CPD lookup + calibration + kNN blend

The agent iterates on this file to improve prediction quality.
"""

import os
import pickle
import sqlite3
import re
from dataclasses import dataclass, field

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from prepare import Market, Article, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, well-tested
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.55   # cosine sim to merge into existing event node
EVIDENCE_SIM_THRESHOLD = 0.40 # min cosine sim for event → market edge
MAX_PARENTS_PER_MARKET = 3    # max event-node parents per market (keeps CPD small: 2^3=8)
MAX_CHUNKS_PER_ARTICLE = 3    # sentences to extract per article
MAX_TOTAL_CHUNKS = 5000       # cap to stay within time budget


# ---------------------------------------------------------------------------
# Prediction Model Interface (must implement these)
# ---------------------------------------------------------------------------

class PredictionModel:
    """Base prediction model. Subclass and override build() and predict()."""

    def build(self, dataset: Dataset):
        raise NotImplementedError

    def predict(self, market: Market) -> float:
        raise NotImplementedError

    def predict_batch(self, markets: list[Market]) -> list[tuple[str, float]]:
        return [(m.id, self.predict(m)) for m in markets]

    def price_event(self, description: str) -> dict:
        raise NotImplementedError

    def stats(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_f32(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _l2_to_cosine(l2_dist: float) -> float:
    return 1.0 - (l2_dist ** 2) / 2.0


def _extract_sentences(text: str, max_n: int = 3) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 30][:max_n]


# ---------------------------------------------------------------------------
# Event Node — a canonical concept in the semantic graph
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """A node representing a semantic concept (e.g., 'Iranian strikes on Kuwait')."""
    id: str
    label: str                                    # representative text
    embedding: np.ndarray                         # centroid vector
    aliases: list[str] = field(default_factory=list)
    observation_count: int = 0                    # article chunks mapped here


# ---------------------------------------------------------------------------
# Semantic Event Graph
# ---------------------------------------------------------------------------

class SemanticEventGraph(PredictionModel):
    """
    Semantic Event Graph with CPDs.

    The BN contains two types of nodes:
    - Event nodes (from articles): binary (observed/not), root variables
    - Market nodes: binary (Yes/No), child variables with event-node parents

    Edges connect event nodes to market nodes based on semantic similarity.
    CPDs encode P(market=Yes | parent_event_states).
    """

    def __init__(self):
        self.embedder = None
        self.db = None
        self.event_nodes: dict[str, EventNode] = {}
        self.rowid_to_node: dict[int, str] = {}
        self._next_id = 0

        # Graph structure (direct storage for fast inference)
        self.market_parents: dict[str, list[tuple[str, float]]] = {}  # market_id -> [(evt_id, sim)]
        self.market_cpds: dict[str, dict[int, float]] = {}            # market_id -> {state_combo -> P(Yes)}
        self.base_rate: float = 0.5

        # pgmpy BN (for inspection/export)
        self.bn = None

        # Calibration + kNN
        self._cal_bins: list[tuple[float, float]] = []
        self._market_emb_list: list[tuple[np.ndarray, float]] = []
        self._market_embeddings: dict[str, np.ndarray] = {}
        self._market_prices: dict[str, float] = {}
        self.num_articles: int = 0

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

        # 3. Ingest articles → event nodes
        print("  Ingesting articles into event graph...")
        self._ingest_articles(dataset.articles)
        print(f"  Created {len(self.event_nodes)} event nodes from articles")

        # 4. Embed markets and find parent event nodes
        print("  Linking markets to event nodes...")
        market_texts = [f"{m.question} {m.description[:500]}" for m in dataset.markets]
        market_embs = (
            self.embedder.encode(market_texts, show_progress_bar=False,
                                 normalize_embeddings=True)
            if market_texts else np.array([])
        )

        # Global base rate
        prices = [m.market_price for m in dataset.markets]
        self.base_rate = np.mean(prices) if prices else 0.5

        for i, market in enumerate(dataset.markets):
            self._market_embeddings[market.id] = market_embs[i]
            self._market_prices[market.id] = market.market_price
            self._market_emb_list.append((market_embs[i], market.market_price))

            # Find top-k most similar event nodes
            parents = self._find_parents(market_embs[i])
            self.market_parents[market.id] = parents

            # Build CPD for this market
            self._build_market_cpd(market.id, market.market_price, parents)

        # 5. Build pgmpy BN (for inspection)
        print("  Building Bayesian network...")
        self._build_pgmpy_bn()

        # 6. Calibration
        print("  Learning calibration...")
        self._learn_calibration(dataset.markets)

    def _init_vec_db(self):
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
        all_chunks = []
        for article in articles:
            chunks = []
            if article.title and len(article.title) > 20:
                chunks.append(article.title)
            chunks.extend(_extract_sentences(article.body, MAX_CHUNKS_PER_ARTICLE))
            all_chunks.extend(chunks)

        if not all_chunks:
            return
        if len(all_chunks) > MAX_TOTAL_CHUNKS:
            all_chunks = all_chunks[:MAX_TOTAL_CHUNKS]

        embeddings = self.embedder.encode(
            all_chunks, show_progress_bar=False,
            normalize_embeddings=True, batch_size=256,
        )

        for chunk, emb in zip(all_chunks, embeddings):
            self._find_or_create_node(chunk, emb)

    def _find_or_create_node(self, text: str, embedding: np.ndarray) -> str:
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

        self._next_id += 1
        node_id = f"evt_{self._next_id:04d}"
        node = EventNode(
            id=node_id, label=text[:100], embedding=embedding.copy(),
            aliases=[text[:100]], observation_count=1,
        )
        self.event_nodes[node_id] = node
        self.db.execute(
            "INSERT INTO vec_events(rowid, embedding) VALUES (?, ?)",
            [self._next_id, _serialize_f32(embedding)],
        )
        self.rowid_to_node[self._next_id] = node_id
        return node_id

    # ------------------------------------------------------------------
    # Parent finding and CPD construction
    # ------------------------------------------------------------------

    def _find_parents(self, market_embedding: np.ndarray) -> list[tuple[str, float]]:
        """Find top-k event nodes most similar to this market."""
        if not self.event_nodes:
            return []

        rows = self.db.execute("""
            SELECT rowid, distance FROM vec_events
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT ?
        """, [_serialize_f32(market_embedding), MAX_PARENTS_PER_MARKET * 3]).fetchall()

        parents = []
        for rowid, dist in rows:
            sim = _l2_to_cosine(dist)
            if sim < EVIDENCE_SIM_THRESHOLD:
                break
            node_id = self.rowid_to_node.get(rowid)
            if node_id:
                parents.append((node_id, sim))
            if len(parents) >= MAX_PARENTS_PER_MARKET:
                break

        return parents

    def _build_market_cpd(self, market_id: str, market_price: float,
                          parents: list[tuple[str, float]]):
        """Build CPD: P(market=Yes | parent_event_states).

        Heuristic: each observed parent pushes the prediction from base_rate
        toward the market_price, weighted by its similarity score.
        """
        n = len(parents)
        if n == 0:
            self.market_cpds[market_id] = {0: self.base_rate}
            return

        total_sim = sum(sim for _, sim in parents)
        cpd = {}

        for combo in range(2 ** n):
            # Sum similarity of observed parents
            observed_sim = 0.0
            for j in range(n):
                if combo & (1 << j):
                    observed_sim += parents[j][1]

            # Weight: fraction of total evidence explained by observed parents
            weight = observed_sim / total_sim if total_sim > 0 else 0.0

            # Interpolate: no evidence → base_rate, full evidence → market_price
            p_yes = self.base_rate + (market_price - self.base_rate) * weight
            p_yes = max(0.01, min(0.99, p_yes))
            cpd[combo] = p_yes

        self.market_cpds[market_id] = cpd

    # ------------------------------------------------------------------
    # pgmpy BN (for inspection)
    # ------------------------------------------------------------------

    def _build_pgmpy_bn(self):
        """Build a pgmpy BayesianNetwork from the graph structure."""
        # Collect edges
        edges = []
        active_event_ids = set()
        for market_id, parents in self.market_parents.items():
            mkt_node = f"mkt_{market_id}"
            for evt_id, sim in parents:
                edges.append((evt_id, mkt_node))
                active_event_ids.add(evt_id)

        if not edges:
            return

        self.bn = BayesianNetwork(edges)

        # CPDs for event nodes (roots): P(observed)
        for evt_id in active_event_ids:
            node = self.event_nodes[evt_id]
            p_obs = min(0.95, 0.5 + node.observation_count * 0.05)
            cpd = TabularCPD(
                variable=evt_id, variable_card=2,
                values=[[1 - p_obs], [p_obs]],
                state_names={evt_id: ["not_observed", "observed"]},
            )
            self.bn.add_cpds(cpd)

        # CPDs for market nodes
        for market_id, parents in self.market_parents.items():
            if not parents:
                continue
            mkt_node = f"mkt_{market_id}"
            n = len(parents)
            parent_ids = [evt_id for evt_id, _ in parents]
            cpd_data = self.market_cpds[market_id]

            # Build the values array: shape (2, 2^n)
            # Row 0 = P(No), Row 1 = P(Yes)
            # Columns ordered by pgmpy convention: rightmost parent varies fastest
            n_combos = 2 ** n
            yes_probs = []
            no_probs = []
            for combo in range(n_combos):
                p_yes = cpd_data.get(combo, self.base_rate)
                yes_probs.append(p_yes)
                no_probs.append(1 - p_yes)

            cpd = TabularCPD(
                variable=mkt_node, variable_card=2,
                values=[no_probs, yes_probs],
                evidence=parent_ids,
                evidence_card=[2] * n,
                state_names={
                    mkt_node: ["No", "Yes"],
                    **{eid: ["not_observed", "observed"] for eid in parent_ids},
                },
            )
            self.bn.add_cpds(cpd)

        try:
            assert self.bn.check_model()
        except (AssertionError, ValueError) as e:
            print(f"  Warning: BN validation failed ({e}), continuing without pgmpy BN")
            self.bn = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _learn_calibration(self, markets: list[Market]):
        pairs = []
        for m in markets:
            raw = self._predict_raw(m)
            pairs.append((raw, m.market_price))
        pairs.sort(key=lambda x: x[0])

        n_bins = 30
        self._cal_bins = []
        bin_size = max(1, len(pairs) // n_bins)
        for i in range(0, len(pairs), bin_size):
            chunk = pairs[i:i + bin_size]
            mean_pred = sum(p for p, _ in chunk) / len(chunk)
            mean_actual = sum(a for _, a in chunk) / len(chunk)
            self._cal_bins.append((mean_pred, mean_actual))

    def _calibrate(self, raw_pred: float) -> float:
        if not self._cal_bins:
            return raw_pred
        if raw_pred <= self._cal_bins[0][0]:
            return self._cal_bins[0][1]
        if raw_pred >= self._cal_bins[-1][0]:
            return self._cal_bins[-1][1]
        for i in range(len(self._cal_bins) - 1):
            p0, a0 = self._cal_bins[i]
            p1, a1 = self._cal_bins[i + 1]
            if p0 <= raw_pred <= p1:
                if p1 == p0:
                    return a0
                t = (raw_pred - p0) / (p1 - p0)
                return a0 + t * (a1 - a0)
        return raw_pred

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_raw(self, market: Market) -> float:
        """Raw prediction via direct CPD lookup (all parents observed)."""
        cpd = self.market_cpds.get(market.id)
        parents = self.market_parents.get(market.id, [])
        if not cpd or not parents:
            return self.base_rate

        # All parents observed → combo with all bits set
        combo = (1 << len(parents)) - 1
        return cpd.get(combo, self.base_rate)

    def predict(self, market: Market) -> float:
        raw = self._predict_raw(market)
        calibrated = self._calibrate(raw)

        # kNN blend
        emb = self._market_embeddings.get(market.id)
        if emb is not None and self._market_emb_list:
            sims = []
            for m_emb, m_price in self._market_emb_list:
                sim = float(np.dot(emb, m_emb))
                sims.append((sim, m_price))
            sims.sort(key=lambda x: x[0], reverse=True)
            neighbors = [(s, p) for s, p in sims[1:6] if s > 0.5]
            if neighbors:
                knn_price = sum(s * p for s, p in neighbors) / sum(s for s, _ in neighbors)
                return 0.7 * calibrated + 0.3 * knn_price

        return calibrated

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def price_event(self, description: str) -> dict:
        emb = self.embedder.encode(
            [description], show_progress_bar=False, normalize_embeddings=True,
        )[0]

        # Find relevant event nodes
        evidence_details = self._get_evidence_details(emb)

        # Find similar markets and use their CPDs
        best_market_id = None
        best_sim = -1
        for mid, m_emb in self._market_embeddings.items():
            sim = float(np.dot(emb, m_emb))
            if sim > best_sim:
                best_sim = sim
                best_market_id = mid

        if best_market_id:
            # Use most similar market's raw prediction as basis
            cpd = self.market_cpds.get(best_market_id, {})
            parents = self.market_parents.get(best_market_id, [])
            if parents:
                combo = (1 << len(parents)) - 1
                raw = cpd.get(combo, self.base_rate)
            else:
                raw = self.base_rate
            prob = self._calibrate(raw)
        else:
            prob = self.base_rate

        factors = [
            {"factor": f"Most similar market (sim={best_sim:.3f})",
             "direction": "neutral", "weight": best_sim},
        ]
        for label, sim in evidence_details[:5]:
            factors.append({
                "factor": f"Event: {label[:60]}",
                "direction": "supporting",
                "weight": round(sim, 3),
            })

        return {
            "probability": round(prob, 4),
            "confidence": round(best_sim, 2) if best_sim > 0 else 0.0,
            "factors": factors,
            "related_keywords": [label for label, _ in evidence_details[:5]],
        }

    def _get_evidence_details(self, embedding: np.ndarray) -> list[tuple[str, float]]:
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
            if sim < EVIDENCE_SIM_THRESHOLD:
                break
            node_id = self.rowid_to_node.get(rowid)
            if node_id:
                results.append((self.event_nodes[node_id].label, sim))
        return results

    def stats(self) -> dict:
        n_edges = sum(len(p) for p in self.market_parents.values())
        active_events = set()
        for parents in self.market_parents.values():
            for evt_id, _ in parents:
                active_events.add(evt_id)
        return {
            "num_nodes": len(active_events) + len(self.market_parents),
            "num_edges": n_edges,
            "num_event_nodes": len(self.event_nodes),
            "num_active_event_nodes": len(active_events),
            "num_market_nodes": len(self.market_parents),
            "num_articles": self.num_articles,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        embedder = self.embedder
        db = self.db
        self.embedder = None
        self.db = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.embedder = embedder
        self.db = db

    @classmethod
    def load(cls, path: str) -> "SemanticEventGraph":
        with open(path, "rb") as f:
            model = pickle.load(f)
        model._rebuild_vec_db()
        return model

    def _rebuild_vec_db(self):
        self._init_vec_db()
        for rowid, node_id in self.rowid_to_node.items():
            node = self.event_nodes[node_id]
            self.db.execute(
                "INSERT INTO vec_events(rowid, embedding) VALUES (?, ?)",
                [rowid, _serialize_f32(node.embedding)],
            )

    def load_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "model.pkl")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model() -> PredictionModel:
    return SemanticEventGraph()
