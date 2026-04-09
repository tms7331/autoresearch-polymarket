"""
arpm-memory Memory System — the file you modify.

Implements a MemRL-inspired memory bank that learns to predict market outcomes
by storing (intent, experience, Q-value) triplets and using two-phase retrieval.

This is the baseline implementation. Improve it.
"""

import math
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from prepare import (
    Article, Market, MemoryTriplet, Dataset,
    categorize_text, extract_keywords,
    Q_LEARNING_RATE, RETRIEVAL_LAMBDA,
)

# ---------------------------------------------------------------------------
# Memory Prediction Model Interface (must implement these)
# ---------------------------------------------------------------------------

class MemoryPredictionModel:
    """Base memory prediction model. Subclass and override."""

    def build(self, dataset: Dataset) -> list[MemoryTriplet]:
        """Build memory bank from training data. Returns list of memories."""
        raise NotImplementedError

    def predict(self, market: Market, dataset: Dataset) -> float:
        """Predict probability that market resolves Yes. Returns float in [0, 1]."""
        raise NotImplementedError

    def predict_batch(self, markets: list[Market], dataset: Dataset) -> list[tuple[str, float]]:
        """Predict probabilities for a batch of markets."""
        return [(m.id, self.predict(m, dataset)) for m in markets]

    def update_q_values(self, market: Market, reward: float):
        """Update Q-values for memories that were used to predict this market."""
        raise NotImplementedError

    def predict_market(self, question: str, dataset: Dataset) -> dict:
        """Tool interface: predict a free-text market question."""
        raise NotImplementedError

    def get_memories(self) -> list[MemoryTriplet]:
        """Return the current memory bank."""
        return []

    def stats(self) -> dict:
        """Return model statistics."""
        return {}


# ---------------------------------------------------------------------------
# Baseline: Category Base Rate + Keyword Similarity + Q-Value Retrieval
# ---------------------------------------------------------------------------

class BaselineMemoryModel(MemoryPredictionModel):
    """Baseline MemRL-style memory system.

    Memory construction: One memory per training market, built from
    linked article headlines + category + outcome.

    Retrieval: Two-phase:
      Phase A — keyword overlap similarity (top-k1 candidates)
      Phase B — re-rank by blending similarity score with Q-value

    Q-value update: EMA rule from MemRL paper.

    Prediction: Weighted average of retrieved memory predictions,
    blended with category base rate.
    """

    def __init__(self):
        self.memories: list[MemoryTriplet] = []
        self.memory_by_id: dict[str, MemoryTriplet] = {}
        self.category_rates: dict[str, float] = {}
        self.global_base_rate: float = 0.5
        # Track which memories were used for each prediction (for Q-updates)
        self._last_retrieved: dict[str, list[str]] = {}  # market_id -> [memory_ids]

    def build(self, dataset: Dataset) -> list[MemoryTriplet]:
        """Build memory bank from resolved training markets."""

        # Compute category base rates
        cat_counts = defaultdict(lambda: [0, 0])  # [total, resolved_yes]
        for market in dataset.train_markets:
            cat = categorize_text(market.question)
            cat_counts[cat][0] += 1
            if market.resolved:
                cat_counts[cat][1] += 1

        for cat, (total, yes) in cat_counts.items():
            self.category_rates[cat] = yes / total if total > 0 else 0.5

        total = len(dataset.train_markets)
        total_yes = sum(1 for m in dataset.train_markets if m.resolved)
        self.global_base_rate = total_yes / total if total > 0 else 0.5

        # Create one memory per training market
        for market in dataset.train_markets:
            # Gather linked article context
            linked_ids = dataset.article_to_market.get(market.id, [])
            article_context = []
            for aid in linked_ids[:10]:  # cap at 10 articles
                art = dataset.article_by_id.get(aid)
                if art:
                    article_context.append(f"- {art.title}")

            # Build experience text
            outcome = "Yes" if market.resolved else "No"
            experience_parts = [
                f"Market: {market.question}",
                f"Category: {categorize_text(market.question)}",
                f"Outcome: {outcome}",
                f"Volume: ${market.volume:,.0f}" if market.volume else "",
            ]
            if article_context:
                experience_parts.append("Related news:")
                experience_parts.extend(article_context[:5])

            experience = "\n".join(p for p in experience_parts if p)

            # Create memory triplet
            mem_id = hashlib.sha256(f"market:{market.id}".encode()).hexdigest()[:16]
            memory = MemoryTriplet(
                id=mem_id,
                intent=market.question,
                intent_keywords=extract_keywords(market.question),
                experience=experience,
                experience_type="success" if market.resolved else "failure",
                category=categorize_text(market.question),
                source_articles=linked_ids[:10],
                source_market=market.id,
                q_value=0.0,
                q_updates=0,
                created_at=market.resolution_date or "",
                prediction_at_creation=1.0 if market.resolved else 0.0,
            )
            self.memories.append(memory)
            self.memory_by_id[memory.id] = memory

        # Run Q-learning on training data: simulate predictions and update Q-values
        self._train_q_values(dataset)

        return self.memories

    def _train_q_values(self, dataset: Dataset):
        """Simulate predictions on training markets to bootstrap Q-values."""
        for market in dataset.train_markets:
            # Retrieve memories (excluding the market's own memory)
            own_mem_id = hashlib.sha256(f"market:{market.id}".encode()).hexdigest()[:16]
            retrieved = self._retrieve(
                market.question,
                k1=10, k2=5,
                exclude_ids={own_mem_id},
            )

            if not retrieved:
                continue

            # The reward is based on whether retrieved memories would help predict correctly
            outcome = 1.0 if market.resolved else 0.0
            # Simple prediction from retrieved memories
            pred = self._aggregate_predictions(retrieved, market)
            # Reward = 1 - Brier score for this prediction
            error = (pred - outcome) ** 2
            reward = 1.0 - error

            # Update Q-values for retrieved memories
            for mem_id, _sim, _q in retrieved:
                mem = self.memory_by_id.get(mem_id)
                if mem:
                    mem.q_value = mem.q_value + Q_LEARNING_RATE * (reward - mem.q_value)
                    mem.q_updates += 1

    def _keyword_similarity(self, keywords_a: list[str], keywords_b: list[str]) -> float:
        """Compute keyword overlap similarity (Jaccard-like)."""
        if not keywords_a or not keywords_b:
            return 0.0
        set_a = set(keywords_a)
        set_b = set(keywords_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _retrieve(
        self,
        query: str,
        k1: int = 10,
        k2: int = 5,
        exclude_ids: set = None,
    ) -> list[tuple[str, float, float]]:
        """Two-phase retrieval from MemRL paper.

        Phase A: Top-k1 by keyword similarity, with sparsity threshold.
        Phase B: Re-rank by composite score = (1-lambda)*sim + lambda*Q.

        Returns: list of (memory_id, similarity, q_value) tuples.
        """
        if not self.memories:
            return []

        exclude_ids = exclude_ids or set()
        query_keywords = extract_keywords(query)

        # Phase A: similarity-based recall
        candidates = []
        for mem in self.memories:
            if mem.id in exclude_ids:
                continue
            sim = self._keyword_similarity(query_keywords, mem.intent_keywords)
            if sim > 0.0:  # sparsity threshold: must have at least some overlap
                candidates.append((mem.id, sim, mem.q_value))

        if not candidates:
            return []

        # Sort by similarity, take top-k1
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:k1]

        # Phase B: re-rank by composite score
        # z-score normalization
        sims = [c[1] for c in candidates]
        qs = [c[2] for c in candidates]

        sim_mean = sum(sims) / len(sims) if sims else 0
        sim_std = (sum((s - sim_mean) ** 2 for s in sims) / len(sims)) ** 0.5 if len(sims) > 1 else 1.0
        q_mean = sum(qs) / len(qs) if qs else 0
        q_std = (sum((q - q_mean) ** 2 for q in qs) / len(qs)) ** 0.5 if len(qs) > 1 else 1.0

        sim_std = max(sim_std, 1e-8)
        q_std = max(q_std, 1e-8)

        scored = []
        for mem_id, sim, q in candidates:
            z_sim = (sim - sim_mean) / sim_std
            z_q = (q - q_mean) / q_std
            composite = (1 - RETRIEVAL_LAMBDA) * z_sim + RETRIEVAL_LAMBDA * z_q
            scored.append((mem_id, sim, q, composite))

        scored.sort(key=lambda x: -x[3])
        return [(s[0], s[1], s[2]) for s in scored[:k2]]

    def _aggregate_predictions(
        self, retrieved: list[tuple[str, float, float]], market: Market
    ) -> float:
        """Aggregate retrieved memory signals into a single probability estimate."""
        if not retrieved:
            cat = categorize_text(market.question)
            return self.category_rates.get(cat, self.global_base_rate)

        # Weighted average: weight by similarity * (1 + q_value)
        total_weight = 0.0
        weighted_sum = 0.0
        for mem_id, sim, q in retrieved:
            mem = self.memory_by_id.get(mem_id)
            if not mem:
                continue
            # The prediction signal from this memory: did the similar market resolve yes?
            signal = mem.prediction_at_creation if mem.prediction_at_creation is not None else 0.5
            weight = sim * (1.0 + max(q, 0.0))
            weighted_sum += signal * weight
            total_weight += weight

        if total_weight == 0:
            cat = categorize_text(market.question)
            return self.category_rates.get(cat, self.global_base_rate)

        memory_pred = weighted_sum / total_weight

        # Blend with category base rate (shrinkage toward prior)
        cat = categorize_text(market.question)
        base_rate = self.category_rates.get(cat, self.global_base_rate)
        confidence = min(1.0, len(retrieved) / 5.0)  # more memories = more confidence

        final = confidence * memory_pred + (1 - confidence) * base_rate
        return max(0.01, min(0.99, final))

    def predict(self, market: Market, dataset: Dataset) -> float:
        """Predict market outcome using memory retrieval."""
        retrieved = self._retrieve(market.question, k1=10, k2=5)

        # Track which memories were used (for Q-value updates)
        self._last_retrieved[market.id] = [r[0] for r in retrieved]

        return self._aggregate_predictions(retrieved, market)

    def update_q_values(self, market: Market, reward: float):
        """Update Q-values for memories used to predict this market."""
        mem_ids = self._last_retrieved.get(market.id, [])
        for mem_id in mem_ids:
            mem = self.memory_by_id.get(mem_id)
            if mem:
                mem.q_value = mem.q_value + Q_LEARNING_RATE * (reward - mem.q_value)
                mem.q_updates += 1

    def predict_market(self, question: str, dataset: Dataset) -> dict:
        """Tool interface: predict a free-text market question."""
        # Create a synthetic market for prediction
        synthetic = Market(
            id="query",
            question=question,
            description="",
            category=categorize_text(question),
            outcome_prices={},
            volume=0, liquidity=0, active=True,
        )

        retrieved = self._retrieve(question, k1=10, k2=5)
        prob = self._aggregate_predictions(retrieved, synthetic)

        # Build response
        retrieved_info = []
        for mem_id, sim, q in retrieved:
            mem = self.memory_by_id.get(mem_id)
            if mem:
                retrieved_info.append({
                    "intent": mem.intent[:100],
                    "q_value": round(q, 3),
                    "similarity": round(sim, 3),
                })

        factors = []
        cat = categorize_text(question)
        factors.append({
            "factor": f"Category base rate ({cat})",
            "direction": "neutral",
            "weight": round(self.category_rates.get(cat, self.global_base_rate), 3),
        })
        for mem_id, sim, q in retrieved[:3]:
            mem = self.memory_by_id.get(mem_id)
            if mem:
                direction = "up" if mem.prediction_at_creation and mem.prediction_at_creation > 0.5 else "down"
                factors.append({
                    "factor": f"Similar market: {mem.intent[:60]}",
                    "direction": direction,
                    "weight": round(sim * (1 + max(q, 0)), 3),
                })

        return {
            "probability": round(prob, 4),
            "confidence": round(min(1.0, len(retrieved) / 5.0), 2),
            "retrieved_memories": retrieved_info,
            "factors": factors,
        }

    def get_memories(self) -> list[MemoryTriplet]:
        return self.memories

    def stats(self) -> dict:
        q_values = [m.q_value for m in self.memories]
        return {
            "num_memories": len(self.memories),
            "avg_q_value": round(sum(q_values) / len(q_values), 4) if q_values else 0,
            "max_q_value": round(max(q_values), 4) if q_values else 0,
            "min_q_value": round(min(q_values), 4) if q_values else 0,
            "num_categories": len(self.category_rates),
        }


# ---------------------------------------------------------------------------
# Model factory — change this to swap models
# ---------------------------------------------------------------------------

def create_model() -> MemoryPredictionModel:
    """Create and return the memory prediction model to use.
    Modify this function to try different models.
    """
    return BaselineMemoryModel()
