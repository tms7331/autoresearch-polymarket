# arpm-memory — Auto Research Prediction Memory

An LLM agent predicts real-world events using a **memory_lookup** tool backed by a sqlite-vec memory bank. The memory bank is built from news articles — each memory is a concrete fact (an event, quote, data point). The experiment loop tunes `model.py` to improve what the memory tool returns, which improves the agent's predictions.

Each run **wipes and rebuilds the memory bank from scratch** with the current parameters. There is no persistent state between runs — the entire memory system is reconstructed from articles + configuration each time.

## How It Works

```
Articles (data/articles/)
    ↓ _split_into_facts()          ← you tune this
    ↓ TF-IDF embedding             ← you tune this
    ↓ sqlite-vec INSERT
    ↓ Q-value bootstrap            ← you tune this
    = Memory bank (in-memory SQLite)
    ↓
Agent asks: "Will Iran test a nuclear weapon?"
    ↓ tool_lookup()                ← you tune this
    ↓ KNN retrieval + re-ranking
    ↓ returns relevant facts to agent
    ↓
Agent reasons over facts → {"probability": 0.08}
    ↓
Evaluate prediction vs Polymarket odds → Brier score
```

## File Roles

| File | Role | Modifiable? |
|---|---|---|
| `experiments/memory/model.py` | Memory system: fact extraction, storage, retrieval, Q-values | **YES — only file you edit** |
| `experiments/memory/agent.py` | Claude agent harness with memory_lookup tool | No |
| `experiments/memory/run.py` | Experiment runner: build → predict → evaluate | No |
| `experiments/memory/prepare.py` | Data loading, evaluation metrics | No |
| `experiments/memory/inspect_model.py` | Generates inspector.html for debugging | No |
| `experiments/memory/gen_dashboard.py` | Generates dashboard.html from results.tsv | No |

## What You Tune in model.py

Every run wipes the in-memory SQLite database and rebuilds it. You're tuning:

### 1. Fact Extraction (`_split_into_facts()`)
How articles become memories. This is the most important function.

**Current approach**: Split article text on paragraph/sentence boundaries, filter boilerplate, prefix with date/source.

**Known issues to fix**:
- Article metadata (author bylines, URLs, "Reporting by...") leaks through as memories
- Some chunks are too vague to be useful ("He said it was important")
- No entity extraction — facts don't highlight who/what/where

**Things to try**:
- Stricter boilerplate filtering (author lines, URL lines, navigation text)
- Minimum information density — require named entities, numbers, or quoted speech
- Smarter chunking — keep related sentences together, don't split mid-thought
- Include article title as context prefix so facts are self-contained

### 2. Embedding (`TfidfVectorizer`)
How memories and queries are converted to vectors for similarity search.

**Current approach**: TF-IDF with 2048-dim vocabulary, augmented TF, L2 normalized.

**Things to try**:
- Vocabulary size (`max_features`) — bigger catches more terms but is sparser
- IDF weighting formula
- Term filtering thresholds (min/max document frequency)
- Character n-grams or bigrams for better entity matching
- Sentence-transformers for dense embeddings (already in dependencies)

### 3. Retrieval (`_retrieve()`, `tool_lookup()`)
How memories are found and ranked when the agent calls the tool.

**Current approach**: Phase A = sqlite-vec KNN (top k1=30), Phase B = re-rank by 90% similarity + 10% Q-value (top k2=8).

**Known issues to fix**:
- Irrelevant memories with inflated Q-values can pollute results
- Zero-vector queries (no vocab overlap) return nothing — agent gets no help

**Things to try**:
- k1, k2 values — more candidates vs. more filtering
- q_blend weight — how much Q-value matters vs. pure similarity
- Distance threshold — reject memories below a minimum similarity
- Diversity — avoid returning 5 memories about the same sub-event
- Enrich tool_lookup output — add category base rates, memory count, confidence signal

### 4. Q-Value Bootstrap (`_train_q_values()`)
How memories get initial utility scores before the agent runs.

**Current approach**: Sample 200 train markets, retrieve memories for each, reward all retrieved memories equally based on category base rate accuracy.

**Known issues to fix**:
- Rewards all retrieved memories equally — junk memories that co-occur with useful ones get inflated Q-values
- Only uses category base rate as prediction signal, not individual memory quality

**Things to try**:
- Weight reward by similarity (closer memories get more credit)
- Only reward top-1 or top-3 most similar, not all retrieved
- Multiple bootstrap passes
- Per-memory reward based on whether the memory's specific content is predictive
- Temporal decay — older memories get less initial Q-value

### 5. Factory Parameters (`create_model()`)
The constructor arguments: `embedding_dim`, `k1`, `k2`, `q_blend`.

## What Memories Should Be

**Memories must be concrete facts from news articles, NOT market questions.**

**BAD** (do not produce these):
- "Will the Fed raise interest rates in June?" — a question, not a memory
- "Author: Luke Harding Date: 2026-04-08" — metadata, not content
- "He said it was important" — vague fragment with no standalone meaning
- "Follow us on Twitter @BBCAfrica" — boilerplate

**GOOD** (this is what we want):
- "Defense Secretary Pete Hegseth said preventing Iran from obtaining a nuclear weapon is non-negotiable, adding Tehran must agree to full disarmament."
- "The Federal Reserve held interest rates steady at its May meeting. Inflation remains above the 2% target at 2.8%."
- "Russia evacuated 198 more staff from Iran's Bushehr Nuclear Power Plant as an airstrike killed an Iranian security guard."
- "The Academy Award-winning US actor won his third Oscar on Sunday, but skipped the ceremony to visit Ukraine."

Each memory should be: **concrete** (specific events/data), **factual** (from article content), **informative** (carries predictive signal), **standalone** (makes sense without seeing the original article).

## Data Access Rules

| Source | Path | Usage | Contents |
|---|---|---|---|
| Articles | `data/articles/` | Freely use | News articles (plain text) — **source of all memories** |
| Train markets | `data/markets_train/` | Freely use | Market questions, metadata — no odds |
| Test markets | `data/markets_test/` | Selectively use | Same markets with odds — for Q-value calibration |
| Validation markets | `data/markets_validation/` | **NEVER use in training** | Held-out markets — evaluation only |

## Setup

1. **Read the in-scope files**:
   - `experiments/memory/program.md` — this file
   - `experiments/memory/model.py` — the file you modify
   - `experiments/memory/agent.py` — the fixed agent harness (understand what it expects from `tool_lookup()`)
   - `experiments/memory/run.py` — the runner (understand fast vs full mode)
2. **Verify data exists**: `data/articles/`, `data/markets_train/`, `data/markets_test/`, `data/markets_validation/` should all have files.
3. **Verify Claude Code subscription**: The agent uses claude-agent-sdk which runs on the Claude Code subscription.
4. **Confirm and go**.

## The Experiment Loop

LOOP FOREVER:

1. **Read model.py** to understand current state.
2. **Identify one thing to improve** — pick from the known issues or ideas above.
3. **Save a backup**: `cp experiments/memory/model.py experiments/memory/model.py.bak`
4. **Make the change** in model.py. Keep changes focused — one idea per iteration.
5. **Run**: `cd experiments/memory && uv run run.py > run.log 2>&1`
6. **Check results**: `grep "^brier_score:\|^coverage:" run.log`
   - If empty, it crashed — run `tail -n 50 run.log` and fix.
7. **Log results** to `experiments/memory/experiments.md` (see Logging Results below). Always append — never delete or rewrite previous entries.
8. **Keep or discard**:
   - If brier_score improved → keep the change, delete the backup: `rm experiments/memory/model.py.bak`
   - If equal or worse → restore: `cp experiments/memory/model.py.bak experiments/memory/model.py && rm experiments/memory/model.py.bak`
9. **Inspect quality** (periodically): Run `uv run inspect_model.py` and examine:
   - Are the top Q-value memories actually useful facts, or garbage?
   - Do the retrieval samples return relevant content for the market questions?
   - Is the prediction scatter plot improving (dots closer to diagonal)?
10. **Go to step 1.**

**Fast mode** (`uv run run.py`) evaluates on 10 validation markets (~3 min).
**Full mode** (`uv run run.py --full`) evaluates on all 75 (~20 min). Use for final benchmarks.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as crash.

**NEVER STOP**: Do NOT pause to ask the human. If you run out of ideas, run the inspector, examine what's broken, and fix it.

## Logging Results

### experiments.md

After every experiment run, append the results to `experiments/memory/experiments.md`. This is the canonical log of all experiments and their outcomes. Format each entry as follows:

```markdown
## Experiment N — <short title>

**Status:** keep | discard | crash
**Description:** <1-2 sentence summary of what was tried and why>

| Metric | Value |
|---|---|
| brier_score | 0.XXXXXX |
| calibration_err | 0.XXXXXX |
| coverage | X.XXXX |
| num_memories | N |
| total_seconds | N.N |

**Notes:** <any observations — what improved, what regressed, hypotheses for next run>
```

Number experiments sequentially starting from 1. If the run crashed, record the error message in the Notes field. Always append — never delete or rewrite previous entries.

## Current Baseline

```
brier_score:      0.051939
calibration_err:  0.137500
coverage:         1.0000
num_memories:     4275
embedding_dim:    2048
q_blend:          0.05
```

Key finding: reducing q_blend from 0.1 to 0.05 gave the biggest improvement so far (0.086 → 0.052). Q-values were over-weighted, letting noisy bootstrap scores pollute retrieval.

Known problems remaining:
- Article metadata leaks into memories (author bylines, URLs, navigation text) — but filtering attempts have all regressed, possibly because filtering changes memory count which disrupts Q-value bootstrap
- Q-value bootstrap rewards all retrieved memories equally (inflates junk)
- TF-IDF misses queries with no vocab overlap (returns empty)
- No entity extraction — retrieval quality is mediocre for specific named entities
- Calibration (0.138) still has room for improvement — k2=5 showed 0.084 calibration but worse brier
