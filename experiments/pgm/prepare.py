"""
Data loading and evaluation for ARPM experiments.
Reads Polymarket markets from data/markets/ and articles from data/articles/.

Usage:
    python prepare.py     # load data and show stats

Both data folders are populated by separate tools:
  - data/markets/   — .txt files with market info (from polymarket scraper)
  - data/articles/  — .txt files with news articles (from news scraper)
"""

import os
import sys
import re
import math
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 180          # model build + inference time budget in seconds (3 minutes)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POLYMARKET_DIR = os.path.join(_REPO_ROOT, "data", "markets")
ARTICLES_DIR = os.path.join(_REPO_ROOT, "data", "articles")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """A single Polymarket prediction market."""
    id: str
    question: str
    description: str                 # resolution criteria
    url: str
    volume: float                    # in dollars
    outcomes: list[str]              # e.g. ["Yes", "No"]
    outcome_prices: list[float]      # e.g. [0.83, 0.17] — probabilities
    end_date: str                    # ISO 8601
    filename: str = ""               # source file

    @property
    def market_price(self) -> float:
        """The current Yes probability."""
        if self.outcomes and self.outcome_prices:
            for i, outcome in enumerate(self.outcomes):
                if outcome.lower() == "yes" and i < len(self.outcome_prices):
                    return self.outcome_prices[i]
            return self.outcome_prices[0] if self.outcome_prices else 0.5
        return 0.5


@dataclass
class Article:
    """A news article loaded from data_articles/."""
    filename: str
    text: str

    @property
    def title(self) -> str:
        """First line of the article, or filename if empty."""
        lines = self.text.strip().split("\n")
        return lines[0].strip() if lines else self.filename

    @property
    def body(self) -> str:
        """Everything after the first line."""
        lines = self.text.strip().split("\n")
        return "\n".join(lines[1:]).strip() if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Market file parser
# ---------------------------------------------------------------------------

def _parse_market_file(filepath: str) -> Optional[Market]:
    """Parse a Polymarket .txt file into a Market object.

    Expected format:
        Market: <question>
        ID: <numeric id>
        URL: <url>

        Volume: <dollar amount>
        Odds: Yes: <pct>% | No: <pct>%
        End Date: <iso date>

        --- Resolution Criteria ---

        <description text>
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not text.strip():
        return None

    # Parse header fields
    question_match = re.search(r'^Market:\s*(.+)$', text, re.MULTILINE)
    id_match = re.search(r'^ID:\s*(\S+)', text, re.MULTILINE)
    url_match = re.search(r'^URL:\s*(\S+)', text, re.MULTILINE)
    volume_match = re.search(r'^Volume:\s*\$?([\d,\.]+[KMB]?)', text, re.MULTILINE)
    odds_match = re.search(r'^Odds:\s*Yes:\s*([\d.]+)%\s*\|\s*No:\s*([\d.]+)%', text, re.MULTILINE)
    end_date_match = re.search(r'^End Date:\s*(.+)$', text, re.MULTILINE)

    if not question_match:
        return None

    question = question_match.group(1).strip()
    market_id = id_match.group(1).strip() if id_match else ""
    url = url_match.group(1).strip() if url_match else ""
    end_date = end_date_match.group(1).strip() if end_date_match else ""

    # Parse volume
    volume = 0.0
    if volume_match:
        vol_str = volume_match.group(1).replace(",", "")
        multiplier = 1.0
        if vol_str.endswith("K"):
            multiplier = 1_000
            vol_str = vol_str[:-1]
        elif vol_str.endswith("M"):
            multiplier = 1_000_000
            vol_str = vol_str[:-1]
        elif vol_str.endswith("B"):
            multiplier = 1_000_000_000
            vol_str = vol_str[:-1]
        try:
            volume = float(vol_str) * multiplier
        except ValueError:
            pass

    # Parse odds
    outcomes = ["Yes", "No"]
    outcome_prices = [0.5, 0.5]
    if odds_match:
        try:
            yes_pct = float(odds_match.group(1)) / 100.0
            no_pct = float(odds_match.group(2)) / 100.0
            outcome_prices = [yes_pct, no_pct]
        except ValueError:
            pass

    # Parse resolution criteria (everything after the --- line)
    criteria_match = re.search(r'---\s*Resolution Criteria\s*---\s*\n(.*)', text, re.DOTALL)
    description = criteria_match.group(1).strip() if criteria_match else ""

    return Market(
        id=market_id,
        question=question,
        description=description,
        url=url,
        volume=volume,
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        end_date=end_date,
        filename=os.path.basename(filepath),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_markets() -> list[Market]:
    """Load all market .txt files from data_polymarket/."""
    if not os.path.isdir(POLYMARKET_DIR):
        return []
    markets = []
    for fname in sorted(os.listdir(POLYMARKET_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(POLYMARKET_DIR, fname)
        market = _parse_market_file(path)
        if market:
            markets.append(market)
    return markets


def load_articles() -> list[Article]:
    """Load all article .txt files from data_articles/."""
    if not os.path.isdir(ARTICLES_DIR):
        return []
    articles = []
    for fname in sorted(os.listdir(ARTICLES_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(ARTICLES_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip():
            articles.append(Article(filename=fname, text=text))
    return articles


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """The complete dataset for model building and evaluation."""
    markets: list[Market]            # all markets (model predicts, eval compares to market price)
    articles: list[Article]          # news articles for PGM evidence

    # Convenience
    market_by_id: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.market_by_id = {m.id: m for m in self.markets}


def load_dataset() -> Dataset:
    """Load markets and articles from disk.

    Returns a Dataset object.
    """
    markets = load_markets()
    articles = load_articles()

    print(f"Dataset: {len(markets)} markets, {len(articles)} articles")

    return Dataset(
        markets=markets,
        articles=articles,
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(predictions: list[tuple[str, float]], markets: list[Market]) -> dict:
    """Evaluate predictions against market prices.

    The ground truth for each market is its current market price (the
    probability implied by Polymarket trading). The model's job is to predict
    this price from the market's question, description, and available articles.

    Args:
        predictions: list of (market_id, predicted_probability) tuples.
            Probabilities should be in [0, 1] for the "Yes" outcome.
        markets: list of markets with known market prices.

    Returns:
        dict with brier_score, log_loss, calibration_err, coverage, and details.
    """
    market_lookup = {m.id: m for m in markets}
    pred_lookup = {mid: prob for mid, prob in predictions}

    # Match predictions to markets
    matched = []
    for mid, market in market_lookup.items():
        if mid in pred_lookup:
            pred = max(0.001, min(0.999, pred_lookup[mid]))  # clamp to avoid log(0)
            truth = market.market_price
            matched.append((pred, truth, market))

    if len(matched) == 0:
        return {
            "brier_score": 1.0,
            "log_loss": 10.0,
            "calibration_err": 1.0,
            "coverage": 0.0,
            "num_markets_eval": 0,
            "mean_abs_error": 1.0,
        }

    # Brier score: MSE between predicted probability and market price
    brier = sum((p - t) ** 2 for p, t, _ in matched) / len(matched)

    # Log loss (treating market price as soft label)
    eps = 1e-15
    log_loss = -sum(
        t * math.log(max(p, eps)) + (1 - t) * math.log(max(1 - p, eps))
        for p, t, _ in matched
    ) / len(matched)

    # Mean absolute error
    mae = sum(abs(p - t) for p, t, _ in matched) / len(matched)

    # Calibration error (binned): group predictions by predicted prob,
    # check if mean market price in each bin matches mean prediction
    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for p, t, _ in matched:
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, t))
    cal_err = 0.0
    cal_count = 0
    for bin_list in bins:
        if bin_list:
            mean_pred = sum(p for p, _ in bin_list) / len(bin_list)
            mean_truth = sum(t for _, t in bin_list) / len(bin_list)
            cal_err += abs(mean_pred - mean_truth) * len(bin_list)
            cal_count += len(bin_list)
    cal_err = cal_err / cal_count if cal_count > 0 else 1.0

    # Coverage: fraction of markets that got a prediction
    coverage = len(matched) / len(market_lookup) if market_lookup else 0.0

    return {
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "calibration_err": round(cal_err, 6),
        "mean_abs_error": round(mae, 6),
        "coverage": round(coverage, 4),
        "num_markets_eval": len(matched),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect ARPM data")
    args = parser.parse_args()

    dataset = load_dataset()
    print()
    print("Markets:")
    for m in dataset.markets:
        print(f"  [p={m.market_price:.2f}] (id={m.id}) {m.question}")
    print()

    if dataset.articles:
        print(f"Articles ({len(dataset.articles)}):")
        for a in dataset.articles[:10]:
            print(f"  [{a.filename}] {a.title[:80]}")
    else:
        print(f"No articles found in {ARTICLES_DIR}/")
        print("  Place .txt files there (one article per file) to use news as PGM evidence.")
