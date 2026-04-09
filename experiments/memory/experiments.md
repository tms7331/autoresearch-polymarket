# Experiment Log

## Experiment 1 — Baseline

**Status:** baseline
**Description:** Unmodified model.py baseline run.

| Metric | Value |
|---|---|
| brier_score | 0.085755 |
| coverage | 1.0000 |
| num_memories | 3350 |

**Notes:** Starting point for all comparisons.

## Experiment 2 — Stricter boilerplate + metadata stripping

**Status:** discard
**Description:** More aggressive boilerplate and metadata filtering. Removed too much useful content.

| Metric | Value |
|---|---|
| brier_score | 0.092239 |
| coverage | 1.0000 |

**Notes:** Filtering was too aggressive, removed useful factual content along with boilerplate.

## Experiment 3 — Metadata line filtering + real title extraction

**Status:** discard
**Description:** Tried to filter metadata lines and extract cleaner titles. Too aggressive.

| Metric | Value |
|---|---|
| brier_score | 0.091747 |
| coverage | 1.0000 |

**Notes:** Slightly less regressed than experiment 2 but still worse than baseline.

## Experiment 4 — Similarity-weighted Q-value bootstrap with 2 passes

**Status:** discard
**Description:** Weighted Q-value rewards by similarity score and ran 2 bootstrap passes.

| Metric | Value |
|---|---|
| brier_score | 0.095799 |
| coverage | 1.0000 |

**Notes:** Regressed. More complex Q-value bootstrap didn't help.

## Experiment 5 — Bigrams in TF-IDF + 3072 dim

**Status:** discard
**Description:** Added bigrams to TF-IDF and increased embedding dimension to 3072.

| Metric | Value |
|---|---|
| brier_score | 0.133753 |
| coverage | 1.0000 |

**Notes:** Major regression. Bigrams made vectors too sparse.

## Experiment 6 — Pure similarity (q_blend=0) + k2=10

**Status:** discard
**Description:** Disabled Q-value influence entirely and returned more results.

| Metric | Value |
|---|---|
| brier_score | 0.113611 |
| coverage | 1.0000 |

**Notes:** Removing Q-values entirely hurts. Some Q-value signal is useful.

## Experiment 7 — Enriched tool_lookup with calibration hints

**Status:** discard
**Description:** Added calibration hints and extra context to tool_lookup output.

| Metric | Value |
|---|---|
| brier_score | 0.126211 |
| coverage | 1.0000 |

**Notes:** Too verbose output confused the agent.

## Experiment 8 — Gentle URL/email/byline filtering

**Status:** discard
**Description:** Minimal metadata filtering — only URL lines, email lines, and "Reporting by" bylines.

| Metric | Value |
|---|---|
| brier_score | 0.138911 |
| calibration_err | 0.229500 |
| coverage | 1.0000 |
| num_memories | 4275 |
| total_seconds | 191.8 |

**Notes:** Even gentle filtering regressed. The filtered lines may have carried useful signal, or the memory count change disrupted Q-value bootstrap.

## Experiment 9 — Lower q_blend (0.1 -> 0.05)

**Status:** keep
**Description:** Halved Q-value influence in re-ranking so retrieval relies more on pure similarity.

| Metric | Value |
|---|---|
| brier_score | 0.051939 |
| calibration_err | 0.137500 |
| coverage | 1.0000 |
| num_memories | 4275 |
| total_seconds | 180.6 |

**Notes:** Significant improvement! Q-values were over-weighted at 0.1, letting noisy Q-scores pollute retrieval. 0.05 keeps a small Q-value signal while prioritizing similarity. New best.

## Experiment 10 — k2=5 (fewer results returned)

**Status:** discard
**Description:** Reduced results returned from 8 to 5, aiming for higher precision.

| Metric | Value |
|---|---|
| brier_score | 0.072667 |
| calibration_err | 0.084300 |
| coverage | 1.0000 |
| num_memories | 4275 |
| total_seconds | 179.3 |

**Notes:** Brier regressed vs experiment 9 (0.073 vs 0.052) but calibration improved significantly (0.084 vs 0.138). Agent may need more context to make good predictions even if fewer results are more precise.

## Experiment 11 — k1=50 (larger candidate pool)

**Status:** crash (interrupted before completion)
**Description:** Increased initial KNN candidate pool from 30 to 50 before re-ranking.

**Notes:** Run was interrupted, no results.
