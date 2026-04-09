# Autoresearch

A variation of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) applied to geopolitical forecasting. Autonomous experimentation loops ingest news and prediction market data, then iteratively build models to predict real-world event probabilities. The goal is a **geopolitical deep research agent** -- one that continuously reads the news, builds structured knowledge, and learns to forecast geopolitical events by calibrating against Polymarket prices and resolutions.

An LLM agent runs the experiment loop: modify model code, evaluate against markets, keep improvements, discard regressions.

## The Memory System (MemRL)

The core approach is inspired by the [MemRL paper](https://arxiv.org/pdf/2601.03192) (Memory-Enhanced Reinforcement Learning), adapted for geopolitical forecasting.

The system maintains a **memory bank** of factual triplets extracted from news articles:

- **Intent (z):** The context or query a memory was created for -- keywords, topic, semantic content.
- **Experience (e):** A concrete fact from the news -- an event, statement, data point, or development.
- **Utility (Q):** A learned Q-value representing how useful this memory has been for making accurate predictions.

### How the Q-Learning Works

When the agent encounters a prediction market question (e.g., "Will Russia and Ukraine reach a ceasefire by July 2025?"), it retrieves relevant memories and produces a probability estimate. The system then learns which memories were actually useful:

1. **Retrieval:** For a given query, the memory bank performs two-phase retrieval:
   - **Phase A (similarity search):** TF-IDF vectors are stored in a sqlite-vec database. The top k1=30 nearest neighbors are retrieved by cosine distance.
   - **Phase B (Q-value re-ranking):** Candidates are re-scored using a composite: `(1 - q_blend) * z_similarity + q_blend * z_q_value`, where z-scores normalize both dimensions. The top k2=8 memories are returned.

2. **Prediction:** A Claude agent receives the retrieved memories and produces a probability estimate for the market question.

3. **Q-value update:** After each prediction, every retrieved memory's Q-value is updated via exponential moving average:
   ```
   Q_new = Q_old + alpha * (reward - Q_old)
   ```
   where `alpha = 0.3` and `reward` is derived from prediction accuracy (Brier score against market odds or resolution).

4. **Bootstrap:** Before live predictions, Q-values are bootstrapped by sampling 200 training markets, retrieving memories for each, and rewarding based on category base-rate accuracy. This gives the system a warm start -- memories about active geopolitical topics get higher initial Q-values than noise.

Over time, the memory bank learns to surface the facts that actually matter for prediction -- high-Q memories about ongoing conflicts, trade negotiations, and political developments rise to the top, while boilerplate and irrelevant facts sink.

The q_blend parameter (currently 0.1) controls how much Q-values influence retrieval vs. pure similarity. This is intentionally low early on -- as Q-values accumulate more updates, the blend can be increased to rely more heavily on learned utility.

## The PGM Approach

The second experiment builds a **Probabilistic Graphical Model** (Bayesian Network) that maps news events to market outcomes. Where the memory system gives an LLM agent better information to reason with, the PGM approach tries to encode the reasoning itself as structure:

- **Semantic event nodes** are created by clustering article passages using sentence-transformer embeddings (all-MiniLM-L6-v2) and sqlite-vec. Each node represents a canonical concept (e.g., "NATO expansion", "Iran nuclear negotiations").
- **Market nodes** are connected to relevant event nodes by cosine similarity. Each market has a Conditional Probability Distribution (CPD) learned from training data: `P(market_outcome | parent_event_states)`.
- **Inference** uses Variable Elimination over the Bayesian network. Given observed events from current news, the model computes posterior probabilities for each market. Post-hoc calibration adjusts for systematic bias.

Both experiments serve the same broader goal: building agents that use prediction markets as a ground-truth calibration signal. Polymarket provides consensus probabilities from real money on the line -- a far better training signal than expert surveys or historical base rates. The PGM encodes this calibration structurally, while the memory system learns it through reinforcement.

## Project Structure

```
autoresearch/
├── scrapers/
│   ├── news/                  # 8 news scrapers (GDELT, GNews, Guardian, Reuters, etc.)
│   └── polymarket/            # Polymarket geopolitics market fetcher
│
├── experiments/
│   ├── memory/                # MemRL memory system (model.py, agent.py, run.py, prepare.py)
│   └── pgm/                   # Probabilistic graphical model (model.py, run.py, prepare.py)
│
├── data/
│   ├── articles/              # Scraped news articles (plain text)
│   ├── markets_train/         # Polymarket markets for model building (no odds)
│   ├── markets_test/          # Polymarket markets for evaluation (with odds)
│   └── markets_validation/    # 10% held-out markets for final evaluation
│
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Install dependencies
uv sync

# Some scrapers require API keys in .env:
#   GUARDIAN_API_KEY, NEWSAPI_KEY, MEDIASTACK_KEY, GNEWS_KEY
#   ANTHROPIC_API_KEY (for running the Claude agent in experiments)

# Reuters scraper uses Browserbase for JS rendering:
#   BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID
```

## 1. Generating the Dataset

The dataset has two parts: news articles and Polymarket markets. Both are scraped at runtime and stored in `data/` (gitignored, only `.gitkeep` files are committed).

### Scrape news articles

```bash
# Run all 8 scrapers (GDELT, GNews, Guardian, MediaStack, NewsAPI, Reuters, RSS, Wikipedia)
cd scrapers/news && python run_all.py

# Or run individual scrapers
python gdelt.py
python guardian.py
python reuters.py
# etc.
```

Output: plain text files in `data/articles/`, one per article, with source/date/URL headers.

### Fetch Polymarket markets

```bash
cd scrapers/polymarket && python fetch_geopolitics.py
```

This searches Polymarket for geopolitics-related markets (war, sanctions, tariffs, elections, etc.), then splits them into three sets:

- **`data/markets_train/`** -- Markets with questions and metadata but **no odds**. Used by the model to build its knowledge structure.
- **`data/markets_test/`** -- Markets **with odds** (Yes/No percentages). Used for evaluation during the experiment loop.
- **`data/markets_validation/`** -- 10% random held-out split with full market info. Used only for final evaluation, never seen during training.

### Prepare experiment data

Each experiment has a `prepare.py` that loads articles and markets into its internal data structures. This runs automatically as part of `run.py`, but you can run it standalone to verify the data:

```bash
cd experiments/memory && uv run prepare.py
cd experiments/pgm && uv run prepare.py
```

## 2. Running Experiments

Both experiments follow the same autoresearch loop: an LLM agent modifies `model.py`, runs the experiment, evaluates against Polymarket, and keeps or discards the change.

### Memory experiment

```bash
cd experiments/memory

# Fast mode: 10-market subset (~3 min)
uv run run.py

# Full mode: all validation markets (~20 min)
uv run run.py --full
```

The runner builds the memory bank from articles, bootstraps Q-values, then uses the Claude agent to predict validation market probabilities. Results are evaluated by Brier score (lower is better), calibration error, log-loss, and coverage.

### PGM experiment

```bash
cd experiments/pgm

# Run (~3 min)
uv run run.py
```

Builds the semantic event graph from articles, learns CPDs from training markets, then predicts validation market probabilities using Bayesian inference.

### The experiment loop

The agent-driven loop (described in each experiment's `program.md`) works like this:

1. Read `model.py` and understand the current approach
2. Make a targeted modification to `model.py`
3. Commit the change
4. Run `uv run run.py` and capture metrics
5. If Brier score improved: keep the commit, log to `results.tsv`
6. If Brier score worsened: revert (`git reset --hard HEAD~1`), log as discarded
7. Repeat

Only `model.py` is modified -- `prepare.py`, `run.py`, and `agent.py` are fixed infrastructure.

## 3. Dashboards and Inspection

Each experiment has two visualization tools that generate standalone HTML files (no dependencies, open in any browser).

### Experiment dashboard

Shows the history of all experiment runs: Brier score trend, keep/discard/crash counts, and a log of every experiment with metrics.

```bash
# Memory experiment dashboard
cd experiments/memory && uv run gen_dashboard.py
open dashboard.html

# PGM experiment dashboard
cd experiments/pgm && uv run gen_dashboard.py
open dashboard.html
```

Reads from `results.tsv` (and `experiments.md` for the PGM dashboard) to generate an SVG chart of Brier score over time and a table of all runs.

### Model inspector

Deep-dive into the model's internal state: what it learned, how it retrieves, and where predictions go wrong.

```bash
# Memory inspector -- Q-value distributions, category coverage, retrieval samples, prediction scatter
cd experiments/memory && uv run inspect_model.py
open inspector.html

# PGM inspector -- event graph structure, CPD tables, calibration curve, bipartite graph visualization
cd experiments/pgm && uv run inspect_model.py
open inspector.html
```

**Memory inspector** shows:
- Memory bank overview (total memories, categories, Q-value stats)
- Q-value distribution histogram
- Top/bottom memories by Q-value
- Retrieval samples for validation markets (what the agent sees)
- Prediction vs. market odds scatter plot

**PGM inspector** shows:
- Graph structure (bipartite SVG: event nodes to market nodes)
- Event node detail cards with connected markets and CPD influence
- Market detail cards with parent events and full CPD tables
- Calibration curve

## Evaluation

Both experiments optimize **Brier score** (lower is better) against Polymarket consensus probabilities. Active markets provide price-based ground truth; resolved markets provide binary outcomes. Secondary metrics: calibration error, log-loss, coverage (fraction of markets the model produces predictions for).
