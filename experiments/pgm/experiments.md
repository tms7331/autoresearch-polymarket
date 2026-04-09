# ARPM Experiments

## Experiment 1 — Baseline: Semantic Event Graph

**Commit:** `475f9ff`
**Status:** keep
**Description:** Initial implementation of Semantic Event Graph with CPDs. Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings, sqlite-vec for vector DB, and pgmpy BN with Category x EvidenceLevel -> PriceBucket structure. CPDs learned from training data with Laplace smoothing.

| Metric | Value |
|---|---|
| brier_score | 0.087676 |
| log_loss | 0.532888 |
| calibration_err | 0.077317 |
| mean_abs_error | 0.253347 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.5 |

**Notes:** 276 event nodes created from 330 articles (some merged via cosine similarity). Predictions cluster around a few values (0.17, 0.21, 0.27, 0.34, 0.50, 0.83) due to only 3 price buckets x 4 evidence levels x 9 categories = 108 distinct CPD lookups. Model runs well under the 3-minute budget at 8.5s. Main improvement opportunities: finer price buckets, richer evidence features, per-market event node parents in the BN.

## Experiment 2 — 5 price buckets

**Commit:** `475f9ff`
**Status:** keep
**Description:** Increased price bucket granularity from 3 (low/mid/high) to 5 (very_low/low/mid/high/very_high) with midpoints 0.10, 0.30, 0.50, 0.70, 0.90.

| Metric | Value |
|---|---|
| brier_score | 0.085057 |
| log_loss | 0.525520 |
| calibration_err | 0.056283 |
| mean_abs_error | 0.241070 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.0 |

**Notes:** Modest improvement. Finer buckets allow the model to distinguish very_low (<0.20) from low (0.20-0.40) which matters for the many markets priced below 0.20. Calibration error also improved.

## Experiment 3 — 10 price buckets

**Commit:** `475f9ff`
**Status:** discard
**Description:** Pushed to 10 price buckets for maximum resolution.

| Metric | Value |
|---|---|
| brier_score | 0.086589 |
| log_loss | 0.529083 |
| calibration_err | 0.060051 |
| mean_abs_error | 0.244255 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.2 |

**Notes:** Worse than 5 buckets. CPD table becomes too sparse with 10 x 4 x 9 = 360 cells for only 681 markets. Laplace smoothing dominates, washing out learned signal. Sweet spot appears to be 5 buckets.

## Experiment 4 — Similarity-weighted evidence

**Commit:** `475f9ff`
**Status:** keep
**Description:** Changed evidence scoring from raw observation count to cosine-similarity-weighted sum. Each matching event node contributes its similarity score times its observation count.

| Metric | Value |
|---|---|
| brier_score | 0.084884 |
| log_loss | 0.525132 |
| calibration_err | 0.057607 |
| mean_abs_error | 0.241060 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.5 |

**Notes:** Small but consistent improvement. Weighting by similarity gives more credit to highly relevant evidence vs. tangentially related matches. Evidence level thresholds adjusted for the new score scale.

## Experiment 5 — Add Relevance node to BN

**Commit:** `475f9ff`
**Status:** keep
**Description:** Added a Relevance node (4 states: irrelevant/weak/moderate/strong) based on max cosine similarity between market and best-matching event node. BN now has 4 nodes and 3 edges: Category, EvidenceLevel, Relevance all parent PriceBucket.

| Metric | Value |
|---|---|
| brier_score | 0.083615 |
| log_loss | 0.521047 |
| calibration_err | 0.068254 |
| mean_abs_error | 0.239746 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.7 |

**Notes:** Relevance captures a different signal than evidence level: how closely the *best* matching article relates to the market, vs. the total volume of evidence. Both matter. CPD table now 5 x 9 x 4 x 4 = 720 cells — approaching sparsity limit for 681 markets.

## Experiment 6 — Lower evidence threshold (0.30)

**Commit:** `475f9ff`
**Status:** discard
**Description:** Lowered the evidence threshold from 0.40 to 0.30 cosine similarity to capture weaker but potentially useful article-market matches.

| Metric | Value |
|---|---|
| brier_score | 0.084310 |
| log_loss | 0.522980 |
| calibration_err | 0.064491 |
| mean_abs_error | 0.239166 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.9 |

**Notes:** Slightly worse. The weaker matches at 0.30-0.40 cosine similarity introduce noise that outweighs any additional signal. The 0.40 threshold is a better balance.

## Experiment 7 — 6 evidence levels

**Commit:** `475f9ff`
**Status:** discard
**Description:** Expanded evidence levels from 4 to 6 (none/trace/low/medium/high/very_high) for finer granularity.

| Metric | Value |
|---|---|
| brier_score | 0.084117 |
| log_loss | 0.522424 |
| calibration_err | 0.075929 |
| mean_abs_error | 0.242254 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.2 |

**Notes:** Worse. CPD table grows to 5 x 9 x 6 x 4 = 1080 cells for 681 markets — heavily Laplace-smoothed. The 4-level evidence discretization is the right granularity for this data size.

## Experiment 8 — Post-hoc calibration

**Commit:** `475f9ff`
**Status:** keep
**Description:** Added binned calibration step after BN inference. Learns a mapping from raw BN predictions to calibrated values using 20-bin isotonic-style interpolation on training data.

| Metric | Value |
|---|---|
| brier_score | 0.078692 |
| log_loss | 0.506187 |
| calibration_err | 0.012602 |
| mean_abs_error | 0.210765 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.8 |

**Notes:** Biggest single improvement so far. Calibration error dropped from 0.068 to 0.013 — the BN outputs were systematically biased and calibration corrects this. The raw BN predictions cluster in a narrow range; calibration spreads them to better match the actual price distribution.

## Experiment 9 — 5 article chunks per article

**Commit:** `475f9ff`
**Status:** discard
**Description:** Increased article chunk extraction from 3 to 5 sentences per article to capture more signal.

| Metric | Value |
|---|---|
| brier_score | 0.079502 |
| log_loss | 0.507996 |
| calibration_err | 0.036203 |
| mean_abs_error | 0.219987 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 398 |
| total_seconds | 8.2 |

**Notes:** More event nodes (398 vs 276) but slightly worse Brier score. The additional sentences from deeper in articles are less informative (ledes carry most signal). Extra nodes may also dilute evidence relevance scores.

## Experiment 10 — Tighter clustering threshold (0.65)

**Commit:** `475f9ff`
**Status:** discard
**Description:** Raised event node merging threshold from 0.55 to 0.65 cosine similarity to create more fine-grained event clusters.

| Metric | Value |
|---|---|
| brier_score | 0.086332 |
| log_loss | 0.529180 |
| calibration_err | 0.055767 |
| mean_abs_error | 0.208091 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 409 |
| total_seconds | 7.5 |

**Notes:** Significantly worse. 409 event nodes (vs 276) are too fine-grained — fragments the evidence and reduces the observation count per node, making the evidence level signal weaker. The 0.55 threshold produces better semantic clusters.

## Experiment 11 — BN + kNN ensemble blend

**Commit:** `475f9ff`
**Status:** keep
**Description:** Blend calibrated BN prediction (70%) with a kNN estimate (30%) based on similarity-weighted average of the 5 most similar markets' prices. Uses the same sentence-transformer embeddings already computed for market questions.

| Metric | Value |
|---|---|
| brier_score | 0.060937 |
| log_loss | 0.464706 |
| calibration_err | 0.060607 |
| mean_abs_error | 0.186908 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.8 |

**Notes:** Massive improvement — Brier score dropped from 0.079 to 0.061. The kNN component leverages the fact that semantically similar markets tend to have similar prices. This compensates for the BN's coarse discretization by pulling predictions toward the actual price neighborhood. Calibration error rose slightly (0.013 to 0.061) because the blend isn't separately calibrated — a second calibration pass could help.
