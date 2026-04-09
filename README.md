# Autoresearch

Autonomous experimentation loops that ingest news and prediction market data, then iteratively build models to predict real-world event probabilities. Polymarket prices and resolutions serve as ground truth signal. The LLM runs the experiment loop: modify model code, evaluate against markets, keep improvements, discard regressions.

Two parallel experiments explore different approaches to the same problem: can an LLM-driven loop learn to price geopolitical events?

## Project Structure

```
autoresearch/
├── pyproject.toml
│
├── scrapers/
│   ├── news/               # News scrapers (GDELT, GNews, Guardian, Reuters, RSS, etc.)
│   └── polymarket/         # Polymarket geopolitics market fetcher
│
├── experiments/
│   ├── pgm/                # Probabilistic graphical model approach
│   └── memory/             # MemRL-inspired memory system approach
│
├── data/
│   ├── articles/           # Scraped news articles (plain text)
│   └── markets/            # Polymarket market definitions + resolution criteria
│
└── reports/                # Analysis outputs and tracked results
```

## Scrapers

**News** (`scrapers/news/`): Pulls articles from multiple sources -- GDELT, GNews, The Guardian, MediaStack, NewsAPI, Reuters, RSS feeds, and Wikipedia current events. Run all with `run_all.py`.

**Polymarket** (`scrapers/polymarket/`): Fetches geopolitics prediction markets from Polymarket, including odds, volume, resolution criteria, and end dates.

## Experiments

Both experiments follow the same autoresearch loop: modify `model.py`, run, evaluate Brier score against Polymarket, keep or discard.

### PGM (`experiments/pgm/`)

Builds a probabilistic graphical model that encodes relationships between news topics, keywords, and market outcomes. The PGM is a tool the LLM can call -- given a natural-language event description, it returns a probability estimate with supporting factors.

### Memory (`experiments/memory/`)

Builds a MemRL-inspired memory bank of (intent, experience, utility) triplets. The system learns which memories are useful for prediction through Q-value reinforcement on market resolutions. Two-phase retrieval: semantic similarity first, then Q-value re-ranking.

## Evaluation

Both experiments optimize **Brier score** (lower is better) against Polymarket. Active markets provide consensus probabilities; resolved markets provide binary ground truth. Secondary metrics: calibration error, log-loss, coverage.
