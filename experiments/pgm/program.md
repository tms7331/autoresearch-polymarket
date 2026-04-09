# arpm — Auto Research Prediction Model

This is an experiment to have the LLM autonomously build and refine a probabilistic graphical model (PGM) that prices real-world events, using news as input.

The PGM is designed to be a **tool that an LLM can call**: given a natural-language description of an event, it returns a probability estimate grounded in observed news signals.

## Overview

The system has three phases that form a loop:

1. **Ingest** — pull prediction markets from Polymarket (and optionally news).
2. **Model** — build/update a PGM that encodes relationships between topics, keywords, and market outcomes.
3. **Evaluate** — score the model's predictions against Polymarket market prices (Brier score).

The key insight: Polymarket prices are treated as ground truth. They represent the crowd's best estimate of event probabilities. The model's job is to predict what the market thinks — and eventually, to find where the market might be wrong.

The LLM iterates on the model-building code (`model.py`) to improve prediction quality, using the same keep/discard discipline as autoresearch.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`). The branch `arpm/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b arpm/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `arpm/program.md` — this file. The rules of engagement.
   - `arpm/prepare.py` — news ingestion, event extraction, data loading, evaluation. Do not modify.
   - `arpm/model.py` — the file you modify. PGM construction, inference, and the `price_event` tool interface.
4. **Verify data exists**: Check that `~/.cache/arpm/` contains news data. If not, tell the human to run `cd arpm && uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

### Data Ingestion (prepare.py — fixed)

- Fetches prediction markets from Polymarket via the `polymarket` CLI.
- Caches market data locally in `~/.cache/arpm/`.
- Markets include: question, description, outcomes, current prices, volume, liquidity, resolution status.
- Reads news articles from `arpm/articles/` (plain `.txt` files, one per article, populated by a separate scraper).
- Splits data into:
  - **Train**: Resolved (closed) markets with known binary outcomes (Yes/No).
  - **Val**: Active markets where the current market price is the ground truth probability.
- Provides a `load_dataset()` function that returns train markets, val markets, event context, and articles.

### Probabilistic Graphical Model (model.py — you modify this)

- Constructs a PGM from market data.
- The model should capture:
  - **Category dynamics**: how do different market categories behave?
  - **Keyword relationships**: which keywords/topics predict which outcomes?
  - **Cross-market dependencies**: if market X is priced high, how does it affect market Y?
  - **Volume/liquidity signals**: do high-volume markets behave differently?
- Must implement the `PredictionModel` interface (defined in model.py).

### Evaluation (prepare.py — fixed)

- **Primary metric: Brier score** (lower is better): MSE between predicted probability and Polymarket's market price.
- The model predicts P(Yes) for each active market; the market price is the ground truth.
- **Secondary metrics**: log-loss, calibration error, mean absolute error, coverage.
- **Backtest metric**: Brier score on resolved markets (binary outcome as ground truth).

### Tool Interface

The model exposes a `price_event(description: str) -> dict` function that an LLM can call:

```python
result = model.price_event("Will the Fed raise interest rates at the next meeting?")
# Returns:
# {
#     "probability": 0.73,
#     "confidence": 0.6,        # how much evidence backs this estimate
#     "factors": [               # key factors influencing the estimate
#         {"factor": "Category base rate (economics)", "direction": "neutral", "weight": 0.45},
#         {"factor": "Keyword 'interest' (rate=0.62, n=15)", "direction": "up", "weight": 2.8},
#     ],
#     "related_markets": [       # similar resolved markets
#         "[Yes] Fed raises rates above 5.5%?",
#         "[No] Fed cuts rates before June 2025?",
#     ],
# }
```

## Experimentation

Each experiment runs locally. The script runs for a **fixed time budget of 3 minutes** (wall clock, excluding startup). Launch with: `cd arpm && uv run run.py`.

**What you CAN do:**
- Modify `arpm/model.py` — this is the only file you edit. Graph structure, inference algorithms, feature engineering, priors, everything about the model.

**What you CANNOT do:**
- Modify `arpm/prepare.py`. It is read-only (news ingestion, event extraction, evaluation).
- Install new packages beyond what's in `arpm/pyproject.toml`.
- Modify the evaluation harness.

**The goal is simple: get the lowest Brier score** on the validation set. Secondary goals: improve calibration, increase coverage, and produce useful factor explanations.

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
brier_resolved:   0.2400
num_resolved:     80
num_nodes:        1500
num_edges:        4200
total_seconds:    12.3
```

Extract the key metric:
```
grep "^brier_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	brier_score	coverage	status	description
```

1. git commit hash (short, 7 chars)
2. brier_score achieved (e.g. 0.2150) — use 0.000000 for crashes
3. coverage (fraction of events model can price) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `arpm/apr9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `model.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `cd arpm && uv run run.py > run.log 2>&1`
5. Read out the results: `grep "^brier_score:\|^coverage:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv)
8. If brier_score improved (lower), keep the commit
9. If brier_score is equal or worse, git reset back

**Timeout**: Each experiment should take ~3 minutes. If a run exceeds 6 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try different graph structures, different inference algorithms, different feature representations, different priors.

## Ideas to explore

These are starting points, not a roadmap. Use your judgment.

- **Graph structure**: Bayesian network vs. Markov random field vs. factor graph
- **Keyword embeddings**: Use co-occurrence patterns to find latent topic structure
- **Volume/liquidity features**: High-volume markets may have more efficient prices
- **Cross-market dependencies**: Markets that share keywords should influence each other
- **Temporal features**: How does time-to-expiry affect pricing?
- **Category-specific models**: Different PGM structures for politics vs. crypto vs. sports
- **Hierarchical priors**: Share strength across related market types
- **Ensemble methods**: Combine multiple simple models
- **Feature engineering**: What features of market questions predict outcomes?
- **Calibration**: Post-hoc calibration (Platt scaling, isotonic regression)
- **Information propagation**: How should evidence flow through the graph?
- **Market microstructure**: Does bid/ask spread or order book depth carry signal?
- **News integration**: When news sources are added, use them as additional evidence nodes
