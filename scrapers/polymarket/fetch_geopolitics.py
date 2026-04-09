#!/usr/bin/env python3
"""
Fetch geopolitics-related markets from Polymarket and save each as a text file.

Uses the `polymarket` CLI to search for markets matching geopolitics keywords,
deduplicates results, filters for active/open markets, and writes individual
text files to polymarketmarkets/data/.
"""

import json
import os
import random
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR_TEST = os.path.join(PROJECT_ROOT, "data", "markets_test")
DATA_DIR_TRAIN = os.path.join(PROJECT_ROOT, "data", "markets_train")
DATA_DIR_VALIDATION = os.path.join(PROJECT_ROOT, "data", "markets_validation")

# Search terms covering a broad range of geopolitics topics
SEARCH_TERMS = [
    "war",
    "Ukraine",
    "Russia",
    "China",
    "Taiwan",
    "Iran",
    "Israel",
    "Gaza",
    "NATO",
    "ceasefire",
    "sanctions",
    "tariff",
    "trade war",
    "nuclear",
    "North Korea",
    "military",
    "invasion",
    "territory",
    "annexation",
    "Syria",
    "Yemen",
    "Houthi",
    "missile",
    "peacekeeping",
    "Zelenskyy",
    "Putin",
    "Xi Jinping",
    "geopolitics",
    "coup",
    "regime",
    "embargo",
    "diplomacy",
    "peace deal",
    "India Pakistan",
    "South China Sea",
    "Arctic",
]

SEARCH_LIMIT = 50  # per query

# Keywords that must appear in the market question or description for it to
# count as geopolitics-relevant. This filters out noise from broad search terms
# like "territory" or "nuclear" that also match sports/crypto/entertainment.
RELEVANCE_KEYWORDS = [
    # Countries / regions
    "ukraine", "russia", "china", "taiwan", "iran", "israel", "gaza", "palestine",
    "syria", "yemen", "houthi", "north korea", "dprk", "iraq", "afghanistan",
    "pakistan", "india", "turkey", "erdogan", "saudi", "lebanon", "hezbollah",
    "hamas", "europe", "eu ", "european union", "nato", "arctic", "south china sea",
    "crimea", "donbas", "donetsk", "luhansk", "kursk", "kherson", "zaporizhzhia",
    "cuba", "venezuela", "myanmar", "sudan", "ethiopia", "libya", "somalia",
    "philippines", "japan", "south korea",
    # Leaders
    "zelenskyy", "zelensky", "putin", "xi jinping", "khamenei", "kim jong",
    "modi", "netanyahu", "erdoğan", "macron", "scholz", "starmer",
    "al-sharaa", "assad",
    # Concepts
    "war", "ceasefire", "peace deal", "peace treaty", "invasion", "annex",
    "military", "troops", "missile", "nuclear weapon", "nuclear war",
    "sanction", "embargo", "tariff", "trade deal", "trade war", "trade agreement",
    "peacekeeping", "coup", "regime", "territory", "occupation", "blockade",
    "geopolit", "diplomacy", "diplomatic", "foreign policy", "defense spending",
    "arms deal", "weapons", "airstrikes", "airstrike", "bombing",
    "conflict", "security guarantee", "martial law",
]

# If the question matches any of these patterns, skip it regardless of keyword matches.
# This filters sports, entertainment, and social-media markets that mention country names.
EXCLUSION_PATTERNS = [
    "masters", "augusta", "golf", "tennis", "ufc", "boxing", "nfl", "nba", "nhl",
    "mlb", "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "world cup", "grand slam", "olympic", "medal",
    "album", "song", "spotify", "tiktok", "youtube", "follower", "subscriber",
    "gta vi", "gta 6", "video game", "fortnite", "valorant",
    "bitcoin", "ethereum", "solana", "dogecoin", "memecoin", "crypto price",
    "how many posts", "how many tweets", "post.*posts from",
]


def is_geopolitics_relevant(market: dict) -> bool:
    """Check if a market's question + description mention geopolitics keywords."""
    question = market.get("question", "").lower()
    description = market.get("description", "").lower()
    text = question + " " + description

    # Exclude sports/entertainment/social-media noise
    for pat in EXCLUSION_PATTERNS:
        if re.search(pat, question):
            return False

    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def run_polymarket_search(query: str) -> list[dict]:
    """Run a polymarket CLI search and return parsed JSON results."""
    cmd = ["polymarket", "markets", "search", query, "-o", "json", "--limit", str(SEARCH_LIMIT)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  Warning: search for '{query}' returned exit code {result.returncode}", file=sys.stderr)
            return []
        return json.loads(result.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        print(f"  Warning: search for '{query}' failed: {e}", file=sys.stderr)
        return []


def parse_odds(outcomes_str: str, prices_str: str) -> str:
    """Parse outcomes and prices into a readable odds string."""
    try:
        outcomes = json.loads(outcomes_str)
        prices = json.loads(prices_str)
    except (json.JSONDecodeError, TypeError):
        return "N/A"

    parts = []
    for outcome, price in zip(outcomes, prices):
        try:
            pct = float(price) * 100
            parts.append(f"{outcome}: {pct:.1f}%")
        except (ValueError, TypeError):
            parts.append(f"{outcome}: N/A")
    return " | ".join(parts)


def format_volume(volume_str: str) -> str:
    """Format volume as a readable dollar amount."""
    try:
        vol = float(volume_str)
        if vol >= 1_000_000:
            return f"${vol / 1_000_000:.2f}M"
        elif vol >= 1_000:
            return f"${vol / 1_000:.1f}K"
        else:
            return f"${vol:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def slugify(text: str) -> str:
    """Convert a question string to a safe filename."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:120].strip("-")


def market_to_text_test(market: dict) -> str:
    """Format a market dict into the test file content (includes odds)."""
    question = market.get("question", "Unknown")
    volume = format_volume(market.get("volume", "0"))
    odds = parse_odds(market.get("outcomes", "[]"), market.get("outcomePrices", "[]"))
    description = market.get("description", "No description available.").strip()
    end_date = market.get("endDate") or market.get("umaEndDate") or "N/A"
    market_id = market.get("id", "unknown")
    slug = market.get("slug", "")

    lines = [
        f"Market: {question}",
        f"ID: {market_id}",
        f"URL: https://polymarket.com/event/{slug}",
        f"",
        f"Volume: {volume}",
        f"Odds: {odds}",
        f"End Date: {end_date}",
        f"",
        f"--- Resolution Criteria ---",
        f"",
        description,
    ]
    return "\n".join(lines)


def market_to_text_train(market: dict) -> str:
    """Format a market dict into the train file content (no odds)."""
    question = market.get("question", "Unknown")
    description = market.get("description", "No description available.").strip()
    end_date = market.get("endDate") or market.get("umaEndDate") or "N/A"
    market_id = market.get("id", "unknown")
    slug = market.get("slug", "")

    lines = [
        f"Market: {question}",
        f"ID: {market_id}",
        f"URL: https://polymarket.com/event/{slug}",
        f"",
        f"End Date: {end_date}",
        f"",
        f"--- Resolution Criteria ---",
        f"",
        description,
    ]
    return "\n".join(lines)


def main():
    os.makedirs(DATA_DIR_TEST, exist_ok=True)
    os.makedirs(DATA_DIR_TRAIN, exist_ok=True)
    os.makedirs(DATA_DIR_VALIDATION, exist_ok=True)

    # Collect all markets, deduplicating by ID
    seen_ids: set[str] = set()
    all_markets: list[dict] = []

    print(f"Searching Polymarket for geopolitics markets ({len(SEARCH_TERMS)} search terms)...")
    for term in SEARCH_TERMS:
        print(f"  Searching: {term}")
        markets = run_polymarket_search(term)
        for m in markets:
            mid = m.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_markets.append(m)

    print(f"\nFound {len(all_markets)} unique markets total.")

    # Filter: active, not closed, and geopolitics-relevant
    active_markets = [
        m for m in all_markets
        if m.get("active") and not m.get("closed") and is_geopolitics_relevant(m)
    ]
    print(f"Active, open, and geopolitics-relevant: {len(active_markets)} markets.")

    # Sort by volume descending
    active_markets.sort(key=lambda m: float(m.get("volume", "0") or "0"), reverse=True)

    # Clear old files
    for data_dir in (DATA_DIR_TEST, DATA_DIR_TRAIN, DATA_DIR_VALIDATION):
        for f in os.listdir(data_dir):
            if f.endswith(".txt"):
                os.remove(os.path.join(data_dir, f))

    # Split: 10% validation (held out entirely), rest go to train+test
    random.shuffle(active_markets)
    n_validation = max(1, len(active_markets) // 10)
    validation_markets = active_markets[:n_validation]
    remaining_markets = active_markets[n_validation:]

    print(f"Split: {len(validation_markets)} validation, {len(remaining_markets)} train/test.")

    # Write validation markets (full info including odds)
    for market in validation_markets:
        question = market.get("question", "unknown")
        slug = slugify(question)
        if not slug:
            slug = f"market-{market.get('id', 'unknown')}"
        filename = f"{slug}.txt"

        with open(os.path.join(DATA_DIR_VALIDATION, filename), "w") as f:
            f.write(market_to_text_test(market))

    # Write remaining markets to train and test
    for market in remaining_markets:
        question = market.get("question", "unknown")
        slug = slugify(question)
        if not slug:
            slug = f"market-{market.get('id', 'unknown')}"
        filename = f"{slug}.txt"

        with open(os.path.join(DATA_DIR_TEST, filename), "w") as f:
            f.write(market_to_text_test(market))

        with open(os.path.join(DATA_DIR_TRAIN, filename), "w") as f:
            f.write(market_to_text_train(market))

    print(f"\nWrote {len(validation_markets)} validation files to {DATA_DIR_VALIDATION}/")
    print(f"Wrote {len(remaining_markets)} market files to {DATA_DIR_TEST}/ and {DATA_DIR_TRAIN}/")


if __name__ == "__main__":
    main()
