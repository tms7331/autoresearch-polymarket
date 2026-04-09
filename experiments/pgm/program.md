# arpm — Semantic Event Graph with CPDs

This is an experiment to have the LLM autonomously build and refine a **Semantic Event Graph** that prices real-world events, using news as input.

The model maps text observations to canonical event nodes via a vector database, links them in a Bayesian network with explicit conditional probability distributions (CPDs), and runs inference to produce probability estimates.

## Overview

The system has three phases that form a loop:

1. **Ingest** — load markets and articles from `data/`. Embed article chunks and cluster them into canonical event nodes in a sqlite-vec vector database.
2. **Model** — build a Bayesian Network where evidence features (derived from semantic matching between articles and markets) feed into learned CPDs to produce market price estimates.
3. **Evaluate** — score the model's predictions against Polymarket market prices (Brier score).

Polymarket prices are treated as ground truth. The model's job is to predict what the market thinks — and eventually, to find where the market might be wrong.

The LLM iterates on the model-building code (`model.py`) to improve prediction quality, using the same keep/discard discipline as autoresearch.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`). The branch `arpm/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b arpm/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `experiments/pgm/program.md` — this file. The rules of engagement.
   - `experiments/pgm/prepare.py` — data loading and evaluation. Do not modify.
   - `experiments/pgm/model.py` — the file you modify. Semantic event graph construction, inference, and the `price_event` tool interface.
4. **Verify data exists**: Check that `data/` contains market and article files (`data/markets_train/`, `data/articles/`, etc.). If empty, tell the human to run the scrapers first (`scrapers/polymarket/` and `scrapers/news/`).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

### System: Semantic Event Graph with CPDs

The model has five components:

#### 1. Semantic Layer (sqlite-vec Vector DB)

Resolves different phrasings into the same canonical event node.

Process:
- Input text → generate embedding (sentence-transformers, `all-MiniLM-L6-v2`)
- Nearest neighbor search in sqlite-vec
- If cosine similarity > threshold → map to existing node
- Else → create new node

Each node corresponds to a semantic cluster of observations.

#### 2. Event Nodes

Each event node is derived from article content. Fields:
- `id` — unique identifier (e.g. `evt_0042`)
- `label` — representative text
- `embedding` — centroid vector
- `aliases` — all phrases mapped to this node
- `observation_count` — how many article chunks mapped here

#### 3. Evidence Computation

For each market, the model queries the vector DB to find semantically related event nodes. This produces an **evidence level** (none/low/medium/high) based on the count and relevance of matching article-derived observations.

#### 4. Bayesian Network (pgmpy)

Current structure:
```
Category (9 states) ──────┐
                           ├──→ PriceBucket (3 states: low/mid/high)
EvidenceLevel (4 states) ─┘
```

- **Category**: derived from keyword pattern matching on market text
- **EvidenceLevel**: derived from semantic search against article event nodes
- **PriceBucket**: the prediction target, converted to a point probability estimate

CPDs are learned from training data with Laplace smoothing.

#### 5. Inference

Given a market:
1. Classify category (keyword matching)
2. Embed market text, search vector DB for related evidence
3. Discretize evidence strength into level
4. Set evidence on BN nodes (Category, EvidenceLevel)
5. Run VariableElimination → posterior over PriceBucket
6. Convert bucket distribution to point probability

### Data Layout

All data lives in the `data/` directory at the repo root.

- `data/articles/*.txt` — scraped news articles (plain text, one per file).
- `data/markets_train/*.txt` — Polymarket markets **without odds**. Contains: question, ID, URL, end date, and resolution criteria.
- `data/markets_test/*.txt` — Same markets **with odds**. Use for evaluating predictions.
- `data/markets_validation/*.txt` — 10% held-out set **with odds**. NEVER use during training or model building.

### Data Rules

**Freely available for training/building the model:**
- `data/articles/` — all articles, unrestricted. Use for event node creation, semantic matching, etc.
- `data/markets_train/` — all market text (question, description, end date). No odds present.

**Available with care (contains labels):**
- `data/markets_test/` — includes actual odds. May use for evaluation and aggregate statistics. Do not memorize individual market prices.

**NEVER use during training or model building:**
- `data/markets_validation/` — held-out evaluation set. Never read or incorporate.

**Overfitting guidance:** Ask: "would this help predict a brand-new market I've never seen?" If yes, it's legitimate. If it only helps on markets whose prices you've already seen, it's leakage.

### Data Ingestion (prepare.py — fixed)

- Reads market `.txt` files and article `.txt` files from `data/`.
- Parses into `Market` and `Article` objects.
- Returns a `Dataset` containing all markets and articles.
- Provides an `evaluate()` function that scores predictions against market prices.

### Model (model.py — you modify this)

Current baseline: `SemanticEventGraph` class implementing:
- sqlite-vec vector DB for article event nodes
- sentence-transformers for embeddings
- pgmpy Bayesian Network (Category × EvidenceLevel → PriceBucket)
- Variable elimination inference

Must implement the `PredictionModel` interface (build, predict, predict_batch, price_event, stats).

### Evaluation (prepare.py — fixed)

- **Primary metric: Brier score** (lower is better): MSE between predicted probability and Polymarket price.
- **Secondary metrics**: log-loss, calibration error, mean absolute error, coverage.

### Tool Interface

```python
result = model.price_event("Will the Fed raise interest rates at the next meeting?")
# Returns:
# {
#     "probability": 0.73,
#     "confidence": 0.6,
#     "factors": [
#         {"factor": "Category: economics", "direction": "neutral", "weight": 1.0},
#         {"factor": "Evidence: medium (4 matches)", ...},
#         {"factor": "Related: Fed signals hawkish stance...", ...},
#     ],
#     "related_keywords": ["Fed signals...", "CPI data shows..."],
# }
```

## Experimentation

Each experiment runs locally. The script runs for a **fixed time budget of 3 minutes** (wall clock, excluding startup). Launch with: `cd experiments/pgm && uv run run.py`.

**What you CAN do:**
- Modify `experiments/pgm/model.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify `experiments/pgm/prepare.py`. It is read-only.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation harness.

**The goal is simple: get the lowest Brier score** on the test set — while building a model that generalizes to the held-out validation set.

## Output format

Once the script finishes it prints a summary:

```
---
brier_score:      0.2150
log_loss:         0.5832
calibration_err:  0.0412
mean_abs_error:   0.3200
coverage:         0.85
num_markets_eval: 150
num_nodes:        3
num_edges:        2
num_event_nodes:  342
num_articles:     1500
total_seconds:    45.3
```

Extract the key metric:
```
grep "^brier_score:" run.log
```

## Logging results

### experiments.md

After every experiment run, append the results to `experiments/pgm/experiments.md`. This is the canonical log of all experiments and their outcomes. Format each entry as follows:

```markdown
## Experiment N — <short title>

**Commit:** `<7-char hash>`
**Status:** keep | discard | crash
**Description:** <1-2 sentence summary of what was tried and why>

| Metric | Value |
|---|---|
| brier_score | 0.XXXXXX |
| log_loss | 0.XXXXXX |
| calibration_err | 0.XXXXXX |
| mean_abs_error | 0.XXXXXX |
| coverage | X.XXXX |
| num_markets_eval | N |
| num_event_nodes | N |
| total_seconds | N.N |

**Notes:** <any observations — what improved, what regressed, hypotheses for next run>
```

Number experiments sequentially starting from 1. If the run crashed, record the error message in the Notes field. Always append — never delete or rewrite previous entries.

### results.tsv

Also log a one-line summary to `results.tsv` (tab-separated) for machine-readable tracking.

The TSV has a header row and 5 columns:

```
commit	brier_score	coverage	status	description
```

1. git commit hash (short, 7 chars)
2. brier_score achieved (e.g. 0.2150) — use 0.000000 for crashes
3. coverage (fraction of events model can price) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

### Dashboard

`gen_dashboard.py` generates a standalone HTML dashboard (`dashboard.html`) from `experiments.md` and `results.tsv`. Run it to get a visual overview of experiment progress:

```bash
cd experiments/pgm && uv run gen_dashboard.py
open dashboard.html
```

The dashboard shows: best Brier score, summary stats (total/kept/discarded/crashed), a Brier score trend chart, and a full experiment log table. It reads from both data sources — preferring the richer `experiments.md` when available, falling back to `results.tsv`.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `arpm/apr9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `model.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `cd experiments/pgm && uv run run.py > run.log 2>&1`
5. Read out the results: `grep "^brier_score:\|^coverage:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv)
8. If brier_score improved (lower), keep the commit
9. If brier_score is equal or worse, git reset back

**Timeout**: Each experiment should take ~3 minutes. If a run exceeds 6 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous.

## Ideas to explore

These are starting points for improving the baseline. Use your judgment.

### Semantic Layer Improvements
- **Embedding model**: Try different models (larger, domain-specific)
- **Similarity threshold tuning**: Different thresholds for node merging vs. evidence matching
- **Article chunking**: Better sentence extraction, paragraph-level embeddings, or full-article embeddings
- **Node deduplication**: More aggressive merging, centroid updates, hierarchical clustering

### Graph Structure
- **More BN nodes**: Add volume, time-to-expiry, article recency, sentiment as BN variables
- **Per-market event nodes**: Individual event nodes as BN parents of specific markets (not just aggregated evidence level)
- **Cross-market dependencies**: Markets sharing evidence nodes should influence each other
- **Category refinement**: Use embeddings for classification instead of keyword regex

### CPD Learning
- **Finer price buckets**: 5 or 10 buckets instead of 3 for better resolution
- **Evidence level granularity**: More bins, or continuous evidence features
- **CPD update from outcomes**: Adjust CPDs using observed market resolutions
- **Structure learning**: Let pgmpy learn the graph structure from data
- **Parameter estimation**: Use pgmpy's built-in estimators (MLE, Bayesian)

### News Integration
- **Sentiment analysis**: Does the article sentiment about a topic shift predictions?
- **Temporal signals**: Recency of articles, frequency of mentions over time
- **Entity extraction**: Named entities as graph nodes
- **Topic modeling**: Latent topics from articles as BN variables

### Calibration & Inference
- **Post-hoc calibration**: Platt scaling, isotonic regression on predictions
- **Belief propagation**: For larger graphs where variable elimination is slow
- **Ensemble**: Combine semantic graph predictions with other baselines
- **Confidence weighting**: Weight predictions by evidence strength
