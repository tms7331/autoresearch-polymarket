"""
arpm-memory Memory System — the file you modify.

Implements the memory bank that backs the agent's memory_lookup tool.
Memories are concrete facts derived from news articles — events, statements,
data points — NOT market questions.  Backed by sqlite-vec for fast vector
similarity search.

**This is the ONLY file the experiment loop modifies.**
"""

import math
import hashlib
import re
import sqlite3
from collections import defaultdict

import numpy as np
import sqlite_vec

from prepare import (
    Article, Market, MemoryTriplet, Dataset,
    categorize_text, extract_keywords,
    Q_LEARNING_RATE, RETRIEVAL_LAMBDA,
)

# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    def __init__(self, max_features: int = 2048):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, documents: list[list[str]]):
        df = defaultdict(int)
        for doc in documents:
            for term in set(doc):
                df[term] += 1
        n_docs = len(documents)
        sorted_terms = sorted(df.keys(), key=lambda t: df[t], reverse=True)
        filtered = [
            t for t in sorted_terms
            if 2 <= df[t] <= int(n_docs * 0.8) + 1
        ]
        if len(filtered) < 50:
            filtered = sorted_terms
        selected = filtered[: self.max_features]
        self.vocab = {term: i for i, term in enumerate(selected)}
        n = len(documents)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((n + 1) / (df.get(term, 0) + 1)) + 1

    def transform(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens or not self.vocab:
            return vec
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = (0.5 + 0.5 * count / max_tf) * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def _serialize_f32(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# Article -> Memory extraction
# ---------------------------------------------------------------------------

BOILERPLATE = {
    "follow us on", "sign up here", "subscribe to", "click here",
    "additional research by", "produced by", "graphics by",
    "reporting by", "editing by", "compiled by", "written by",
    "bbc africa", "on facebook", "on twitter", "on instagram",
}


def _split_into_facts(article: Article) -> list[str]:
    """Split an article into meaningful factual chunks."""
    text = article.summary.strip()
    if not text:
        return [article.title] if article.title else []

    prefix_parts = []
    if article.published:
        prefix_parts.append(article.published)
    if article.source and article.source != "local":
        prefix_parts.append(article.source)
    prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= 300:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            group = []
            group_len = 0
            for sent in sentences:
                group.append(sent)
                group_len += len(sent)
                if group_len >= 150 and len(group) >= 2:
                    chunks.append(" ".join(group))
                    group = []
                    group_len = 0
            if group:
                chunks.append(" ".join(group))

    facts = []
    for chunk in chunks:
        if len(chunk) < 40:
            continue
        if not re.search(r'[A-Z]', chunk):
            continue
        chunk_lower = chunk.lower()
        if any(bp in chunk_lower for bp in BOILERPLATE):
            continue
        if len(chunk) < 60 and not re.search(r'\d', chunk):
            continue
        facts.append(f"{prefix}{chunk}")

    if article.title:
        facts.insert(0, f"{prefix}{article.title}")

    return facts


# ---------------------------------------------------------------------------
# Memory Model Interface
# ---------------------------------------------------------------------------

class MemoryModel:
    def build(self, dataset: Dataset) -> list[MemoryTriplet]:
        raise NotImplementedError

    def tool_lookup(self, query: str) -> dict:
        raise NotImplementedError

    def start_prediction(self, market_id: str):
        pass

    def update_q_values(self, market_id: str, reward: float):
        raise NotImplementedError

    def get_memories(self) -> list[MemoryTriplet]:
        return []

    def stats(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# sqlite-vec backed implementation — article-derived memories
# ---------------------------------------------------------------------------

class SqliteVecMemoryModel(MemoryModel):
    """Memory bank where each memory is a concrete fact from a news article."""

    def __init__(self, embedding_dim: int = 2048, k1: int = 30, k2: int = 8,
                 q_blend: float = 0.1):
        self.embedding_dim = embedding_dim
        self.k1 = k1
        self.k2 = k2
        self.q_blend = q_blend

        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.category_rates: dict[str, float] = {}
        self.global_base_rate: float = 0.5

        self._current_market_id: str | None = None
        self._retrieved_for_market: dict[str, set[str]] = {}

        self.db = sqlite3.connect(":memory:")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_tables()

    def _init_tables(self):
        cur = self.db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                fact          TEXT,
                category      TEXT,
                source_title  TEXT,
                source_date   TEXT,
                source_url    TEXT,
                q_value       REAL DEFAULT 0.0,
                q_updates     INTEGER DEFAULT 0
            )
        """)
        cur.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                id        TEXT PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
        """)
        self.db.commit()

    def build(self, dataset: Dataset) -> list[MemoryTriplet]:
        # 1. Extract facts from all articles
        all_facts: list[tuple[str, Article]] = []
        for article in dataset.articles:
            for fact in _split_into_facts(article):
                all_facts.append((fact, article))
        print(f"  Extracted {len(all_facts)} facts from {len(dataset.articles)} articles")

        # 2. Fit TF-IDF
        corpus_tokens = []
        for fact, _ in all_facts:
            corpus_tokens.append(extract_keywords(fact))
        for m in dataset.train_markets:
            corpus_tokens.append(extract_keywords(f"{m.question} {m.description}"))
        self.vectorizer.fit(corpus_tokens)

        # 3. Category base rates from test market odds
        cat_odds: dict[str, list[float]] = defaultdict(list)
        for market in dataset.active_markets:
            if market.outcome_prices and market.outcome_prices.get("Yes") is not None:
                cat = categorize_text(market.question)
                cat_odds[cat].append(market.outcome_prices["Yes"])
        for cat, odds_list in cat_odds.items():
            self.category_rates[cat] = sum(odds_list) / len(odds_list)
        all_odds = [o for odds_list in cat_odds.values() for o in odds_list]
        self.global_base_rate = sum(all_odds) / len(all_odds) if all_odds else 0.5

        # 4. Insert facts as memories
        triplets: list[MemoryTriplet] = []
        cur = self.db.cursor()
        seen_ids = set()

        for fact, article in all_facts:
            mem_id = hashlib.sha256(fact.encode()).hexdigest()[:16]
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)

            category = categorize_text(fact)
            tokens = extract_keywords(fact)
            embedding = self.vectorizer.transform(tokens)

            cur.execute(
                "INSERT OR IGNORE INTO memories VALUES (?,?,?,?,?,?,?,?)",
                (mem_id, fact, category, article.title,
                 article.published or "", article.url or "", 0.0, 0),
            )
            cur.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (mem_id, _serialize_f32(embedding)),
            )
            triplets.append(MemoryTriplet(
                id=mem_id, intent=fact, intent_keywords=tokens,
                experience=fact, experience_type="observation",
                category=category, source_articles=[article.id],
                source_market=None, q_value=0.0, q_updates=0,
                created_at=article.published or "",
                prediction_at_creation=None,
            ))

        self.db.commit()

        # 5. Bootstrap Q-values
        self._train_q_values(dataset)
        return triplets

    def _train_q_values(self, dataset: Dataset):
        test_odds = {}
        for m in dataset.active_markets:
            if m.outcome_prices and m.outcome_prices.get("Yes") is not None:
                test_odds[m.id] = m.outcome_prices["Yes"]

        import random
        rng = random.Random(42)
        sample = rng.sample(dataset.train_markets, min(200, len(dataset.train_markets)))

        for market in sample:
            if market.id not in test_odds:
                continue
            target = test_odds[market.id]
            retrieved = self._retrieve(market.question)
            if not retrieved:
                continue
            cat = categorize_text(market.question)
            base = self.category_rates.get(cat, self.global_base_rate)
            reward = 1.0 - (base - target) ** 2

            cur = self.db.cursor()
            for mem_id, _dist, q in retrieved:
                new_q = q + Q_LEARNING_RATE * (reward - q)
                cur.execute(
                    "UPDATE memories SET q_value = ?, q_updates = q_updates + 1 WHERE id = ?",
                    (new_q, mem_id),
                )
        self.db.commit()

    def _retrieve(self, query: str, exclude_ids: set | None = None,
                  ) -> list[tuple[str, float, float]]:
        exclude_ids = exclude_ids or set()
        tokens = extract_keywords(query)
        query_vec = self.vectorizer.transform(tokens)
        if np.linalg.norm(query_vec) < 1e-8:
            return []

        cur = self.db.cursor()
        fetch_k = self.k1 + len(exclude_ids) + 5
        rows = cur.execute(
            """
            SELECT v.id, v.distance, m.q_value
            FROM memory_vec v
            JOIN memories m ON m.id = v.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_f32(query_vec), fetch_k),
        ).fetchall()

        candidates = [
            (r[0], r[1], r[2]) for r in rows if r[0] not in exclude_ids
        ][: self.k1]
        if not candidates:
            return []

        dists = np.array([c[1] for c in candidates], dtype=np.float64)
        qs = np.array([c[2] for c in candidates], dtype=np.float64)
        d_mean, d_std = dists.mean(), max(dists.std(), 1e-8)
        q_mean, q_std = qs.mean(), max(qs.std(), 1e-8)

        scored = []
        for mem_id, dist, q in candidates:
            z_sim = -(dist - d_mean) / d_std
            z_q = (q - q_mean) / q_std
            composite = (1 - self.q_blend) * z_sim + self.q_blend * z_q
            scored.append((mem_id, dist, q, composite))

        scored.sort(key=lambda x: -x[3])
        return [(s[0], s[1], s[2]) for s in scored[: self.k2]]

    def start_prediction(self, market_id: str):
        self._current_market_id = market_id
        self._retrieved_for_market[market_id] = set()

    def tool_lookup(self, query: str) -> dict:
        retrieved = self._retrieve(query)
        if self._current_market_id:
            for mem_id, _, _ in retrieved:
                self._retrieved_for_market[self._current_market_id].add(mem_id)

        cur = self.db.cursor()
        memories_out = []
        for mem_id, dist, q in retrieved:
            row = cur.execute(
                "SELECT fact, category, source_title, source_date "
                "FROM memories WHERE id = ?", (mem_id,),
            ).fetchone()
            if not row:
                continue
            memories_out.append({
                "fact": row[0],
                "category": row[1],
                "source": row[2],
                "date": row[3],
                "relevance": round(1.0 / (1.0 + dist), 3),
                "q_value": round(q, 3),
            })

        cat = categorize_text(query)
        return {
            "memories": memories_out,
            "category": cat,
            "category_base_rate": round(
                self.category_rates.get(cat, self.global_base_rate), 3
            ),
            "num_total_memories": cur.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0],
        }

    def update_q_values(self, market_id: str, reward: float):
        mem_ids = self._retrieved_for_market.get(market_id, set())
        if not mem_ids:
            return
        cur = self.db.cursor()
        for mem_id in mem_ids:
            row = cur.execute(
                "SELECT q_value FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()
            if row:
                new_q = row[0] + Q_LEARNING_RATE * (reward - row[0])
                cur.execute(
                    "UPDATE memories SET q_value = ?, q_updates = q_updates + 1 WHERE id = ?",
                    (new_q, mem_id),
                )
        self.db.commit()

    def get_memories(self) -> list[MemoryTriplet]:
        cur = self.db.cursor()
        rows = cur.execute(
            "SELECT id, fact, category, source_title, q_value, q_updates FROM memories"
        ).fetchall()
        return [
            MemoryTriplet(
                id=r[0], intent=r[1], intent_keywords=[],
                experience=r[1], experience_type="observation",
                category=r[2], source_articles=[], source_market=None,
                q_value=r[4], q_updates=r[5], created_at="",
                prediction_at_creation=None,
            )
            for r in rows
        ]

    def stats(self) -> dict:
        cur = self.db.cursor()
        row = cur.execute(
            "SELECT COUNT(*), AVG(q_value), MAX(q_value), MIN(q_value) FROM memories"
        ).fetchone()
        n_cats = cur.execute(
            "SELECT COUNT(DISTINCT category) FROM memories"
        ).fetchone()[0]
        return {
            "num_memories": row[0],
            "avg_q_value": round(row[1] or 0, 4),
            "max_q_value": round(row[2] or 0, 4),
            "min_q_value": round(row[3] or 0, 4),
            "num_categories": n_cats,
            "embedding_dim": self.embedding_dim,
        }


def create_model() -> MemoryModel:
    return SqliteVecMemoryModel(embedding_dim=2048, k1=30, k2=8, q_blend=0.1)
