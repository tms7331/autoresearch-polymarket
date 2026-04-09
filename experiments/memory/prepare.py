"""
Article loading, Polymarket data, memory infrastructure, and evaluation for arpm-memory.

This file is FIXED — do not modify. The agent modifies model.py only.

Articles are read from data/articles/ at the repo root.
Polymarket markets are read from data/markets/ at the repo root.

Usage:
    uv run prepare.py          # load articles + markets, print dataset summary
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 180          # model build + inference time budget in seconds (3 minutes)
VAL_FRACTION = 0.3         # fraction of resolved markets held out for validation
MIN_CONFIDENCE = 0.1       # minimum confidence threshold for a prediction to count
Q_LEARNING_RATE = 0.3      # alpha for Q-value EMA updates (from MemRL paper)
RETRIEVAL_LAMBDA = 0.5     # blend weight: 0=pure similarity, 1=pure Q-value

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTICLES_DIR = os.path.join(REPO_ROOT, "data", "articles")
MARKETS_DIR = os.path.join(REPO_ROOT, "data", "markets")
MEMORY_DIR = os.path.join(os.path.expanduser("~"), ".cache", "arpm-memory", "memory")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """A single news article."""
    id: str
    title: str
    summary: str
    url: str
    source: str
    published: str          # ISO 8601
    fetched: str            # ISO 8601

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Market:
    """A Polymarket prediction market."""
    id: str                          # market ID or slug
    question: str                    # the market question
    description: str                 # longer description
    category: str                    # topic tag
    outcome_prices: dict             # {"Yes": 0.73, "No": 0.27} — current or final prices
    volume: float                    # total volume traded
    liquidity: float                 # current liquidity
    active: bool                     # still tradeable?
    resolved: Optional[bool] = None  # None=unresolved, True=Yes won, False=No won
    resolution_date: Optional[str] = None
    created_at: Optional[str] = None
    end_date: Optional[str] = None
    slug: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryTriplet:
    """A single memory in the MemRL-style memory bank.

    z (intent)     = embedding/representation of the query context
    e (experience) = the analysis, reasoning, and news context
    Q (utility)    = learned score: does this memory help predict correctly?
    """
    id: str
    intent: str                      # the question/context this memory was created for
    intent_keywords: list[str]       # extracted keywords for lightweight matching
    experience: str                  # the stored analysis/reasoning
    experience_type: str             # "success", "failure", "observation"
    category: str                    # topic category
    source_articles: list[str]       # article IDs that informed this memory
    source_market: Optional[str]     # market ID if linked to a specific market
    q_value: float = 0.0            # utility score, learned via RL
    q_updates: int = 0              # how many times Q has been updated
    created_at: str = ""            # when this memory was created
    prediction_at_creation: Optional[float] = None  # what we predicted when creating this

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Article loading (from local articles/ folder)
# ---------------------------------------------------------------------------

def _article_id(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def load_articles() -> list[Article]:
    """Load articles from the local articles/ folder.

    Supports three formats:
    - .json files: must have at least "title" and "summary" fields.
      Optional: "url", "source", "published", "fetched".
    - .md files: first line (or # heading) is the title, rest is the summary.
    - .txt files: first line is the title, rest is the summary.
    """
    articles = []
    if not os.path.isdir(ARTICLES_DIR):
        print(f"Warning: articles directory not found: {ARTICLES_DIR}")
        return articles

    for fname in sorted(os.listdir(ARTICLES_DIR)):
        path = os.path.join(ARTICLES_DIR, fname)
        if not os.path.isfile(path):
            continue

        try:
            if fname.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
                article = Article(
                    id=data.get("id", _article_id(path)),
                    title=data.get("title", fname),
                    summary=data.get("summary", data.get("content", "")),
                    url=data.get("url", ""),
                    source=data.get("source", "local"),
                    published=data.get("published", ""),
                    fetched=data.get("fetched", ""),
                )
                articles.append(article)

            elif fname.endswith(".md") or fname.endswith(".txt"):
                with open(path) as f:
                    content = f.read().strip()
                if not content:
                    continue
                lines = content.split("\n", 1)
                title = lines[0].lstrip("# ").strip()
                summary = lines[1].strip() if len(lines) > 1 else ""
                article = Article(
                    id=_article_id(path),
                    title=title,
                    summary=summary,
                    url="",
                    source="local",
                    published="",
                    fetched="",
                )
                articles.append(article)
        except Exception as e:
            print(f"  Warning: failed to load article {fname}: {e}")

    print(f"Articles: loaded {len(articles)} from {ARTICLES_DIR}")
    return articles


# ---------------------------------------------------------------------------
# Polymarket data (loaded from local data_polymarket/ text files)
# ---------------------------------------------------------------------------

def _parse_market_file(path: str) -> Optional[Market]:
    """Parse a market from a structured text file.

    Expected format:
        Market: <question>
        ID: <id>
        URL: <url>

        Volume: $<amount>
        Odds: Yes: <pct>% | No: <pct>%
        End Date: <iso timestamp>

        --- Resolution Criteria ---
        <description text>
    """
    try:
        with open(path) as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: failed to read {path}: {e}")
        return None

    try:
        # Parse Market: line
        m = re.search(r'^Market:\s*(.+)$', content, re.MULTILINE)
        question = m.group(1).strip() if m else ""

        # Parse ID: line
        m = re.search(r'^ID:\s*(.+)$', content, re.MULTILINE)
        market_id = m.group(1).strip() if m else ""

        # Parse URL: line
        m = re.search(r'^URL:\s*(.+)$', content, re.MULTILINE)
        url = m.group(1).strip() if m else ""
        slug = url.rstrip("/").split("/")[-1] if url else ""

        # Parse Volume: line (e.g. "$429" or "$1,234,567")
        volume = 0.0
        m = re.search(r'^Volume:\s*\$?([\d,\.]+)', content, re.MULTILINE)
        if m:
            try:
                volume = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        # Parse Odds: line (e.g. "Yes: 83.0% | No: 17.0%")
        outcome_prices = {}
        m = re.search(r'^Odds:\s*(.+)$', content, re.MULTILINE)
        if m:
            odds_str = m.group(1)
            for part in odds_str.split("|"):
                part = part.strip()
                odds_match = re.match(r'(\w+):\s*([\d.]+)%', part)
                if odds_match:
                    label = odds_match.group(1)
                    pct = float(odds_match.group(2)) / 100.0
                    outcome_prices[label] = pct

        # Parse End Date: line
        m = re.search(r'^End Date:\s*(.+)$', content, re.MULTILINE)
        end_date = m.group(1).strip() if m else ""

        # Parse resolution criteria (everything after "--- Resolution Criteria ---")
        description = ""
        m = re.search(r'---\s*Resolution Criteria\s*---\s*\n(.*)', content, re.DOTALL)
        if m:
            description = m.group(1).strip()

        # Determine category from question text
        category = categorize_text(question + " " + description)

        if not question:
            return None

        return Market(
            id=market_id,
            question=question,
            description=description,
            category=category,
            outcome_prices=outcome_prices,
            volume=volume,
            liquidity=0.0,
            active=True,       # all local markets are treated as active/unresolved
            resolved=None,
            resolution_date=None,
            created_at="",
            end_date=end_date,
            slug=slug,
        )
    except Exception as e:
        print(f"  Warning: failed to parse market from {path}: {e}")
        return None


def load_markets() -> list[Market]:
    """Load all markets from the data_polymarket/ folder.

    Reads .txt and .md files with the structured market format.
    """
    markets = []
    if not os.path.isdir(MARKETS_DIR):
        print(f"Warning: markets directory not found: {MARKETS_DIR}")
        return markets

    for fname in sorted(os.listdir(MARKETS_DIR)):
        path = os.path.join(MARKETS_DIR, fname)
        if not os.path.isfile(path):
            continue
        if not (fname.endswith(".txt") or fname.endswith(".md")):
            continue
        m = _parse_market_file(path)
        if m:
            markets.append(m)

    print(f"Markets: loaded {len(markets)} from {MARKETS_DIR}")
    return markets


# ---------------------------------------------------------------------------
# News-to-market linking
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = {
    "economics": r"\b(gdp|inflation|interest rate|fed|central bank|recession|unemployment|trade|tariff|fiscal|monetary)\b",
    "politics": r"\b(election|vote|parliament|congress|president|minister|policy|legislation|sanction|diplomat)\b",
    "technology": r"\b(ai|artificial intelligence|machine learning|startup|ipo|acquisition|cybersecurity|quantum|chip|semiconductor)\b",
    "science": r"\b(study|research|discovery|climate|vaccine|trial|species|genome|telescope|particle)\b",
    "conflict": r"\b(war|military|attack|ceasefire|troops|missile|nuclear|invasion|conflict|defense)\b",
    "markets": r"\b(stock|market|shares|index|dow|nasdaq|s&p|bond|yield|commodity|oil|gold|crypto|bitcoin)\b",
    "health": r"\b(pandemic|outbreak|disease|who|drug|fda|hospital|virus|treatment|mortality)\b",
    "sports": r"\b(nba|nfl|mlb|nhl|soccer|football|basketball|baseball|hockey|championship|playoffs|tournament)\b",
}


def categorize_text(text: str) -> str:
    """Categorize text by keyword matching."""
    text_lower = text.lower()
    scores = {}
    for cat, pattern in CATEGORY_PATTERNS.items():
        matches = re.findall(pattern, text_lower)
        scores[cat] = len(matches)
    if max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from text for lightweight matching."""
    # Remove common stop words and extract significant terms
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "that",
        "this", "these", "those", "what", "which", "who", "whom", "it", "its",
        "he", "she", "they", "them", "his", "her", "their", "we", "us", "our",
    }
    # Tokenize and filter
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    # Also extract capitalized proper nouns from original text
    proper = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    keywords.extend([p.lower() for p in proper])
    # Deduplicate preserving order
    seen = set()
    result = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result[:50]  # cap at 50 keywords


def link_articles_to_markets(
    articles: list[Article], markets: list[Market]
) -> dict[str, list[str]]:
    """Link articles to relevant markets by keyword overlap.

    Returns: dict mapping market_id -> list of article_ids.
    """
    # Build keyword index for markets
    market_keywords = {}
    for market in markets:
        text = f"{market.question} {market.description}"
        market_keywords[market.id] = set(extract_keywords(text))

    # Score each article against each market
    links = {m.id: [] for m in markets}
    for article in articles:
        art_keywords = set(extract_keywords(f"{article.title} {article.summary}"))
        for market in markets:
            mk = market_keywords[market.id]
            if not mk:
                continue
            overlap = len(art_keywords & mk)
            # Require at least 2 keyword overlap
            if overlap >= 2:
                links[market.id].append(article.id)

    return links


# ---------------------------------------------------------------------------
# Memory bank persistence
# ---------------------------------------------------------------------------

def save_memory_bank(memories: list[MemoryTriplet]):
    """Save memory bank to disk."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = os.path.join(MEMORY_DIR, "bank.json")
    with open(path, "w") as f:
        json.dump([m.to_dict() for m in memories], f, indent=2)


def load_memory_bank() -> list[MemoryTriplet]:
    """Load memory bank from disk."""
    path = os.path.join(MEMORY_DIR, "bank.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [MemoryTriplet.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """The complete dataset for memory building and evaluation."""
    articles: list[Article]
    train_markets: list[Market]       # resolved markets for building memory + Q-learning
    val_markets: list[Market]         # resolved markets held out for evaluation
    active_markets: list[Market]      # still-open markets (for reference/context)
    article_to_market: dict           # market_id -> [article_ids]
    article_by_id: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.article_by_id = {a.id: a for a in self.articles}


def load_dataset() -> Dataset:
    """Load articles and markets, split into train/val.

    Articles are read from the local articles/ folder.
    Markets are loaded from the polymarket cache.

    Returns a Dataset with all markets and articles, linked by keyword overlap.
    Markets that have been resolved (resolved != None) go into train/val splits.
    Unresolved markets are listed as active (prediction targets).
    """
    articles = load_articles()
    markets = load_markets()

    if not markets:
        print("No markets found. Add .txt files to data_polymarket/.")

    # Separate resolved vs active
    resolved = [m for m in markets if m.resolved is not None]
    active = [m for m in markets if m.resolved is None]

    # Split resolved into train/val (if any exist)
    if resolved:
        n_val = max(1, int(len(resolved) * VAL_FRACTION))
        train_markets = resolved[:-n_val] if len(resolved) > n_val else []
        val_markets = resolved[-n_val:]
    else:
        train_markets = []
        val_markets = []

    # Link articles to all markets
    article_to_market = link_articles_to_markets(articles, markets)

    print(f"Dataset: {len(articles)} articles, {len(train_markets)} train markets, "
          f"{len(val_markets)} val markets, {len(active)} active markets")

    return Dataset(
        articles=articles,
        train_markets=train_markets,
        val_markets=val_markets,
        active_markets=active,
        article_to_market=article_to_market,
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_brier(
    predictions: list[tuple[str, float]],
    val_markets: list[Market],
) -> dict:
    """Evaluate predictions using Brier score and related metrics.

    Args:
        predictions: list of (market_id, predicted_probability) tuples.
            Probabilities should be in [0, 1] for the market resolving Yes/True.
        val_markets: list of validation markets with known resolutions.

    Returns:
        dict with brier_score, log_loss, calibration_err, coverage, and details.
    """
    val_lookup = {m.id: m for m in val_markets if m.resolved is not None}
    pred_lookup = {mid: prob for mid, prob in predictions}

    matched = []
    for mid, market in val_lookup.items():
        if mid in pred_lookup:
            prob = max(0.0, min(1.0, pred_lookup[mid]))
            outcome = 1.0 if market.resolved else 0.0
            matched.append((prob, outcome))

    if len(matched) == 0:
        return {
            "brier_score": 1.0,
            "log_loss": 10.0,
            "calibration_err": 1.0,
            "coverage": 0.0,
            "num_markets_eval": 0,
        }

    # Brier score
    brier = sum((p - o) ** 2 for p, o in matched) / len(matched)

    # Log loss
    eps = 1e-15
    log_loss = -sum(
        o * math.log(max(p, eps)) + (1 - o) * math.log(max(1 - p, eps))
        for p, o in matched
    ) / len(matched)

    # Calibration error (binned)
    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for p, o in matched:
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, o))
    cal_err = 0.0
    cal_count = 0
    for bin_list in bins:
        if bin_list:
            mean_pred = sum(p for p, _ in bin_list) / len(bin_list)
            mean_outcome = sum(o for _, o in bin_list) / len(bin_list)
            cal_err += abs(mean_pred - mean_outcome) * len(bin_list)
            cal_count += len(bin_list)
    cal_err = cal_err / cal_count if cal_count > 0 else 1.0

    # Coverage
    coverage = len(matched) / len(val_lookup) if val_lookup else 0.0

    return {
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "calibration_err": round(cal_err, 6),
        "coverage": round(coverage, 4),
        "num_markets_eval": len(matched),
    }


def evaluate_q_correlation(
    memories: list[MemoryTriplet],
    predictions: list[tuple[str, float]],
    val_markets: list[Market],
) -> float:
    """Evaluate whether Q-values correlate with actual prediction accuracy.

    Returns Pearson correlation between memory Q-values and prediction success.
    A high correlation means the Q-values are good predictors of memory usefulness.
    """
    if not memories or not predictions:
        return 0.0

    val_lookup = {m.id: m for m in val_markets if m.resolved is not None}
    pred_lookup = {mid: prob for mid, prob in predictions}

    # For each memory linked to a market, compare Q-value vs prediction error
    q_values = []
    errors = []
    for mem in memories:
        if mem.source_market and mem.source_market in val_lookup and mem.source_market in pred_lookup:
            market = val_lookup[mem.source_market]
            prob = pred_lookup[mem.source_market]
            outcome = 1.0 if market.resolved else 0.0
            error = abs(prob - outcome)
            accuracy = 1.0 - error
            q_values.append(mem.q_value)
            errors.append(accuracy)

    if len(q_values) < 3:
        return 0.0

    q_arr = np.array(q_values)
    e_arr = np.array(errors)

    # Pearson correlation
    if np.std(q_arr) < 1e-10 or np.std(e_arr) < 1e-10:
        return 0.0

    correlation = np.corrcoef(q_arr, e_arr)[0, 1]
    return round(float(correlation), 4) if not np.isnan(correlation) else 0.0


# ---------------------------------------------------------------------------
# Main — run as script to verify data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Articles:  {ARTICLES_DIR}")
    print(f"Markets:   {MARKETS_DIR}")
    print()

    dataset = load_dataset()
    print()

    print("Dataset summary:")
    print(f"  Articles:       {len(dataset.articles)}")
    print(f"  Train markets:  {len(dataset.train_markets)}")
    print(f"  Val markets:    {len(dataset.val_markets)}")
    print(f"  Active markets: {len(dataset.active_markets)}")
    linked = sum(1 for v in dataset.article_to_market.values() if v)
    print(f"  Markets with linked articles: {linked}")

    if dataset.active_markets:
        print()
        print("Active markets:")
        for m in dataset.active_markets:
            odds = m.outcome_prices.get("Yes", "?")
            if isinstance(odds, float):
                odds = f"{odds:.0%}"
            print(f"  [{m.id}] {m.question}  (Yes: {odds})")
