# arpm-memory — Auto Research Prediction Memory

This is an experiment to have the LLM autonomously build and refine a **MemRL-inspired memory system** that learns to predict real-world events by ingesting news and using prediction market (Polymarket) resolutions as ground truth signal.

The core idea: instead of building a static model, we build a **self-evolving memory bank** where each memory is a (intent, experience, utility) triplet. The system learns which memories are actually useful for prediction through reinforcement learning on market outcomes.

## Key Concepts (from MemRL)

- **Memory Triplets**: Each memory is `(z, e, Q)` where:
  - `z` (intent) = embedding of the question/market being predicted
  - `e` (experience) = the analysis, reasoning, and news context that produced a prediction
  - `Q` (utility) = learned score reflecting whether this memory leads to correct predictions
- **Two-Phase Retrieval**: Phase A retrieves by semantic similarity; Phase B re-ranks by blending similarity with Q-value
- **Q-Value Learning**: `Q_new = Q_old + alpha * (reward - Q_old)` where reward comes from Polymarket resolution
- **Store successes AND failures**: Failed predictions with good failure reflections are high-utility near-misses

## Overview

The system has three phases that form a loop:

1. **Ingest** — pull recent news + active/resolved Polymarket markets.
2. **Memory** — build/update a memory bank from news analysis, linking memories to market predictions.
3. **Evaluate** — score predictions against resolved markets (Brier score), update Q-values.

The LLM iterates on the memory system code (`model.py`) to improve prediction quality, using the same keep/discard discipline as autoresearch.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`). The branch `arpm-memory/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b arpm-memory/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `arpm-memory/program.md` — this file. The rules of engagement.
   - `arpm-memory/prepare.py` — news ingestion, Polymarket data, memory infrastructure, evaluation. Do not modify.
   - `arpm-memory/model.py` — the file you modify. Memory system construction, retrieval, Q-value updates, prediction.
4. **Verify data exists**: Check that `~/.cache/arpm-memory/` contains data. If not, tell the human to run `cd arpm-memory && uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

### News Ingestion (prepare.py — fixed)

- Fetches news articles from configured RSS feeds and caches them locally.
- Fetches active and resolved Polymarket markets via the `polymarket` CLI.
- Links news articles to relevant markets by keyword/topic matching.
- Splits data by time: markets resolved before cutoff for training, active/recent for evaluation.
- Provides a `load_dataset()` function that returns train data, val data, and market metadata.

### Polymarket Signal (prepare.py — fixed)

- Active markets provide current consensus probabilities (wisdom of crowds).
- Resolved markets provide binary ground truth (the reward signal for Q-learning).
- Market metadata includes: question, category, volume, liquidity, outcome prices.
- The gap between our memory-system prediction and market resolution is the Brier score we optimize.

### Memory System (model.py — you modify this)

- Constructs a MemRL-style memory bank from news + market data.
- The memory system should implement:
  - **Storage**: How to construct memory triplets from news articles and market context.
  - **Retrieval**: Two-phase retrieval (semantic similarity + Q-value re-ranking).
  - **Q-Value Updates**: How to update utility scores when market outcomes are observed.
  - **Prediction**: How to combine retrieved memories into a probability estimate.
  - **Experience Summarization**: How to distill news context into useful memory content.
- Must implement the `MemoryPredictionModel` interface (defined in prepare.py).

### Evaluation (prepare.py — fixed)

- The evaluation metric is **Brier score** (lower is better): mean squared error between predicted probabilities and binary outcomes.
- Resolved Polymarket markets provide ground truth.
- The model predicts each market's outcome probability using only information available before resolution.
- Secondary metrics: calibration error, log-loss, coverage, and Q-value correlation (do Q-values predict memory usefulness?).

### Tool Interface

The model exposes a `predict_market(question: str) -> dict` function:

```python
result = model.predict_market("Will the Fed raise interest rates at the June meeting?")
# Returns:
# {
#     "probability": 0.73,
#     "confidence": 0.6,
#     "retrieved_memories": [           # which memories were used
#         {"intent": "...", "q_value": 0.82, "similarity": 0.91},
#     ],
#     "factors": [                      # key factors from retrieved memories
#         {"factor": "CPI above expectations", "direction": "up", "weight": 0.3},
#         {"factor": "Fed chair hawkish comments", "direction": "up", "weight": 0.25},
#     ],
#     "polymarket_comparison": 0.71,    # current market price for reference
# }
```

## Experimentation

Each experiment runs locally. The script runs for a **fixed time budget of 3 minutes** (wall clock, excluding startup). Launch with: `cd arpm-memory && uv run run.py`.

**What you CAN do:**
- Modify `arpm-memory/model.py` — this is the only file you edit. Memory construction, retrieval algorithms, Q-value update rules, embedding strategies, prediction aggregation, everything about the memory system.

**What you CANNOT do:**
- Modify `arpm-memory/prepare.py`. It is read-only (news ingestion, market data, evaluation).
- Install new packages beyond what's in `arpm-memory/pyproject.toml`.
- Modify the evaluation harness.

**The goal is simple: get the lowest Brier score** on the validation set of resolved markets. Secondary goals: improve calibration, increase coverage, achieve high Q-value-to-accuracy correlation.

## Output format

Once the script finishes it prints a summary:

```
---
brier_score:      0.2150
log_loss:         0.5832
calibration_err:  0.0412
coverage:         0.85
q_correlation:    0.72
num_memories:     1500
num_markets_eval: 50
total_seconds:    185.3
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
3. coverage (fraction of markets the model can price) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `arpm-memory/apr9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `model.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `cd arpm-memory && uv run run.py > run.log 2>&1`
5. Read out the results: `grep "^brier_score:\|^coverage:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv)
8. If brier_score improved (lower), keep the commit
9. If brier_score is equal or worse, git reset back

**Timeout**: Each experiment should take ~3 minutes. If a run exceeds 6 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try different retrieval strategies, different Q-value update rules, different embedding approaches, different memory representations.

## Ideas to explore

These are starting points, not a roadmap. Use your judgment.

- **Retrieval strategies**: Pure similarity vs. Q-weighted vs. diversity-promoting retrieval
- **Memory construction**: What makes a good memory triplet? Raw headlines vs. summarized analysis vs. entity-focused
- **Q-value dynamics**: Learning rate alpha, temporal decay, per-category Q-values
- **Embedding approaches**: TF-IDF, keyword overlap, entity co-occurrence, category-aware embeddings
- **Temporal awareness**: Recent news should weigh more; markets have time horizons
- **Category specialization**: Separate memory banks per topic (politics, economics, tech)
- **Experience quality**: Success summaries vs. failure reflections vs. both
- **Prediction aggregation**: Weighted average of retrieved memory predictions vs. Bayesian combination
- **Market features**: Use Polymarket volume, liquidity, price history as additional signals
- **Ensemble memories**: Multiple retrieval strategies combined
- **Memory consolidation**: Merge similar memories, prune low-Q stale memories
- **Cross-domain transfer**: Can memories from one topic help predict another?
