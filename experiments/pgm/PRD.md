# ARPM — Auto Research Prediction Model

## Product Requirements Document

### Problem

Prediction markets like Polymarket produce well-calibrated probability estimates for real-world events, but they require active human participation and are limited to the specific questions someone has listed. There's no general-purpose tool that can take an arbitrary event description and return a grounded probability estimate.

Meanwhile, news articles contain a dense stream of signals about the world — who is doing what, what trends are forming, which conflicts are escalating — but this information is unstructured and difficult to reason about quantitatively.

### Solution

ARPM builds a **probabilistic graphical model (PGM)** that combines two information sources:

1. **Prediction market data** (Polymarket) — structured probabilities for specific events, used as ground truth for calibration and evaluation.
2. **News articles** — unstructured text providing broad context about entities, topics, and relationships, used as evidence for the PGM.

The PGM encodes relationships between entities, topics, and outcomes. Given a natural-language event description, it returns a probability estimate grounded in both market data and news signals.

### Key Insight

This system is built using the **autoresearch methodology**: an LLM autonomously iterates on the model code in a tight experiment loop (modify → run → evaluate → keep/discard). The human sets up the scaffolding and data pipeline; the LLM does the research.

### Architecture

```
┌──────────────┐     ┌──────────────┐
│   articles/  │     │  Polymarket  │
│  (text files │     │  (via CLI)   │
│  from scraper│     │              │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────┐
│          prepare.py              │
│  - reads articles from disk      │
│  - fetches markets via poly CLI  │
│  - builds Dataset object         │
│  - provides evaluation harness   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│           model.py               │
│  - builds PGM from articles +   │
│    market data                   │
│  - predicts market probabilities │
│  - exposes price_event() tool    │
│  *** THIS IS WHAT THE LLM       │
│      ITERATES ON ***             │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│            run.py                │
│  - loads data                    │
│  - builds model                  │
│  - evaluates against Polymarket  │
│  - prints metrics                │
└──────────────────────────────────┘
```

### Data Sources

#### Prediction Markets (Polymarket)

- Fetched via the `polymarket` CLI tool.
- **Resolved markets** (closed) serve as training data — we know the binary outcome.
- **Active markets** (open) serve as validation — the current market price is treated as ground truth.
- Markets are filtered by minimum volume/liquidity to exclude noise.

#### News Articles

- Provided as plain text files in `arpm/articles/`.
- Populated by a separate scraper (out of scope for this project).
- Each file is one article. The model reads whatever is there at experiment time.
- Articles provide contextual signals: entity mentions, topic trends, sentiment, temporal patterns.
- The PGM uses articles as evidence nodes that influence market probability estimates.

### Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Brier score** (primary) | MSE between predicted prob and market price | Lower is better |
| Log loss | Cross-entropy against market price | Lower is better |
| Calibration error | Binned prediction-vs-reality gap | Lower is better |
| Mean absolute error | Average |pred - market price| | Lower is better |
| Coverage | Fraction of markets the model can price | Higher is better |
| Brier (resolved) | MSE on closed markets against binary outcome | Secondary |

### The Experiment Loop

Following the autoresearch pattern:

1. LLM modifies `model.py` with an experimental idea
2. Commits the change
3. Runs `uv run run.py` (3 minute time budget)
4. Reads metrics from output
5. If Brier score improved → keep the commit
6. If Brier score worsened → revert
7. Log results to `results.tsv`
8. Repeat indefinitely

### Tool Interface

The end product is a function an LLM can call:

```python
result = model.price_event("Will the EU impose new tariffs on Chinese EVs?")
# → {
#     "probability": 0.68,
#     "confidence": 0.55,
#     "factors": [...],
#     "related_markets": [...]
# }
```

This enables any LLM to get grounded probability estimates for arbitrary events, backed by a model trained against real prediction market data.

### Constraints

- `prepare.py` is read-only during experiments (data loading + evaluation).
- `model.py` is the only file modified during the experiment loop.
- 3 minute time budget per experiment run.
- Only dependencies listed in `pyproject.toml` are available.
- No GPU required — this is a structured data / graphical model problem.

### Out of Scope

- News scraping (handled by a separate tool that writes to `articles/`).
- Trading or placing bets on Polymarket.
- Real-time market monitoring or alerting.
- UI or web interface.

### Success Criteria

1. The model achieves a Brier score meaningfully below the naive baseline (predicting 0.5 for everything = Brier 0.25 on average).
2. The model's `price_event()` function returns sensible, explainable estimates for arbitrary event descriptions.
3. The autonomous experiment loop runs for 10+ iterations and shows measurable improvement.
4. Articles from the `articles/` folder demonstrably improve predictions vs. market-data-only.
