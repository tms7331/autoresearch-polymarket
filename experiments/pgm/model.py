"""
ARPM Prediction Model — the file you modify.

Builds a probabilistic graphical model from Polymarket data and news articles,
then uses it to price prediction markets.

This is the baseline implementation. Improve it.
"""

import math
import re
from collections import defaultdict

from prepare import Market, Article, Dataset

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
        """Tool interface: price a free-text event description.

        This is the function an LLM calls to get a probability estimate.
        """
        raise NotImplementedError

    def stats(self) -> dict:
        """Return model statistics (num_nodes, num_edges, etc.)."""
        return {}

# ---------------------------------------------------------------------------
# Baseline Model: Article Keyword Graph
# ---------------------------------------------------------------------------

# Simple keyword categories for market questions
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


def _categorize(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for cat, pattern in CATEGORY_PATTERNS.items():
        matches = re.findall(pattern, text_lower)
        scores[cat] = len(matches)
    if not scores or max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text."""
    text_lower = text.lower()
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because", "but", "and",
            "or", "if", "while", "this", "that", "these", "those", "it", "its",
            "any", "what", "which", "who", "whom", "their", "they", "them", "he",
            "she", "his", "her", "we", "our", "you", "your", "my", "me"}
    words = re.findall(r'\b[a-z]+\b', text_lower)
    return [w for w in words if w not in stop and len(w) > 2]


class BaselineModel(PredictionModel):
    """Simple baseline: keyword co-occurrence graph from articles.

    The PGM is an undirected graph where:
    - Nodes are keywords extracted from articles and market text
    - Edges connect keywords that co-occur in the same article
    - Edge weights are co-occurrence counts
    - Prediction uses keyword overlap between market text and articles
      to estimate how much evidence supports "Yes" vs "No"
    """

    def __init__(self):
        self.article_keyword_counts = defaultdict(int)  # keyword -> count across articles
        self.keyword_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.global_base_rate = 0.5
        self.num_nodes = 0
        self.num_edges = 0
        self.num_articles = 0

    def build(self, dataset: Dataset):
        """Build the keyword graph from articles."""
        # Index article keywords — these are the evidence nodes in the PGM
        self.num_articles = len(dataset.articles)
        for article in dataset.articles:
            kws = _extract_keywords(article.text)
            for kw in set(kws):
                self.article_keyword_counts[kw] += 1
            # Co-occurrence graph (cap per article to avoid quadratic blowup)
            unique_kw = list(set(kws))[:50]
            for i, kw1 in enumerate(unique_kw):
                for kw2 in unique_kw[i+1:]:
                    self.keyword_cooccurrence[kw1][kw2] += 1
                    self.keyword_cooccurrence[kw2][kw1] += 1

        self.num_nodes = len(self.article_keyword_counts)
        self.num_edges = sum(len(v) for v in self.keyword_cooccurrence.values()) // 2

    def predict(self, market: Market) -> float:
        """Predict using keyword overlap between market and articles.

        The baseline simply returns 0.5 (no opinion) adjusted by how much
        article evidence is available for this market's topic. This is
        intentionally naive — the experiment loop should improve it.
        """
        keywords = _extract_keywords(market.question + " " + market.description)
        if not keywords or self.num_articles == 0:
            return 0.5

        # How many of this market's keywords appear in articles?
        unique_kw = set(keywords)
        hits = sum(1 for kw in unique_kw if kw in self.article_keyword_counts)
        coverage = hits / len(unique_kw) if unique_kw else 0

        # Weighted keyword salience: how often do these keywords appear?
        total_mentions = sum(self.article_keyword_counts.get(kw, 0) for kw in unique_kw)
        salience = min(1.0, total_mentions / (self.num_articles * 3))

        # Baseline: return 0.5 (the model has no directional signal yet,
        # just awareness of topic coverage). The experiment loop should
        # add directional signal from article content.
        return 0.5

    def price_event(self, description: str) -> dict:
        """Price a free-text event description."""
        cat = _categorize(description)
        keywords = _extract_keywords(description)

        factors = [
            {"factor": f"Category: {cat}", "direction": "neutral", "weight": 0.0},
        ]

        unique_kw = set(keywords)
        hits = sum(1 for kw in unique_kw if kw in self.article_keyword_counts)
        coverage = hits / len(unique_kw) if unique_kw else 0

        for kw in sorted(unique_kw):
            count = self.article_keyword_counts.get(kw, 0)
            if count > 0:
                factors.append({
                    "factor": f"Keyword '{kw}' ({count} article mentions)",
                    "direction": "neutral",
                    "weight": round(count / max(self.num_articles, 1), 3),
                })

        # Find related keywords via co-occurrence
        related = []
        for kw in list(unique_kw)[:5]:
            neighbors = self.keyword_cooccurrence.get(kw, {})
            for neighbor, count in sorted(neighbors.items(), key=lambda x: -x[1])[:3]:
                if neighbor not in unique_kw:
                    related.append(f"{neighbor} (co-occurs with '{kw}' {count}x)")

        return {
            "probability": 0.5,
            "confidence": round(coverage, 2),
            "factors": factors[:10],
            "related_keywords": related[:5],
        }

    def stats(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_articles": self.num_articles,
        }


# ---------------------------------------------------------------------------
# Model factory — change this to swap models
# ---------------------------------------------------------------------------

def create_model() -> PredictionModel:
    """Create and return the prediction model to use.
    Modify this function to try different models.
    """
    return BaselineModel()
