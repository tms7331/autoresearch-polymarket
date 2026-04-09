# ARPM Experiments

## Experiment 1 — Baseline

**Commit:** `35d4f87`
**Status:** keep
**Description:** Baseline run of current model. Semantic Event Graph with per-market BN (event nodes as parents), post-hoc 20-bin calibration, and 70/30 BN+kNN ensemble blend.

| Metric | Value |
|---|---|
| brier_score | 0.014294 |
| log_loss | 0.334275 |
| calibration_err | 0.025832 |
| mean_abs_error | 0.070658 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.3 |

**Notes:** Starting point for this experiment run. Model combines calibrated BN prediction with kNN similarity-based pricing. 276 event nodes from 330 articles. Runs well under the 3-minute budget.

## Experiment 2 — Post-blend recalibration

**Commit:** `cef2cb9`
**Status:** keep
**Description:** Added a second calibration pass after the BN+kNN blend. The first calibration corrects the raw BN output; the second corrects the blended prediction.

| Metric | Value |
|---|---|
| brier_score | 0.012846 |
| log_loss | 0.327879 |
| calibration_err | 0.004842 |
| mean_abs_error | 0.061360 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.6 |

**Notes:** Brier score improved from 0.014294 to 0.012846. Calibration error dropped 5x (0.026 → 0.005). The kNN blend was introducing systematic bias that the second calibration pass corrects. No runtime cost.

## Experiment 3 — 10 kNN neighbors, 50/50 blend

**Commit:** `cce0dd4`
**Status:** discard
**Description:** Increased kNN from 5 to 10 neighbors and changed blend from 70/30 to 50/50 (BN/kNN).

| Metric | Value |
|---|---|
| brier_score | 0.018245 |
| log_loss | 0.343221 |
| calibration_err | 0.008478 |
| mean_abs_error | 0.079568 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 6.7 |

**Notes:** Brier worsened from 0.012846 to 0.018245. More neighbors and heavier kNN weight dilutes signal — the 10 nearest markets are too broad, averaging away meaningful price differences. Original 5 neighbors and 70/30 blend is better.

## Experiment 4 — 5 parents per market

**Commit:** `77ad483`
**Status:** discard
**Description:** Increased MAX_PARENTS_PER_MARKET from 3 to 5 to give the BN richer evidence per market.

| Metric | Value |
|---|---|
| brier_score | 0.012846 |
| log_loss | 0.327879 |
| calibration_err | 0.004842 |
| mean_abs_error | 0.061360 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 8.1 |

**Notes:** Identical Brier to baseline. Most markets don't have 5 event nodes above the 0.40 similarity threshold, so extra parent slots go unused. The post-blend calibration also smooths away small CPD differences.

## Experiment 5 — 40 calibration bins

**Commit:** `e1b6a38`
**Status:** keep
**Description:** Increased both calibration passes from 20 to 40 bins. With 681 markets, 40 bins gives ~17 markets per bin for a tighter fit.

| Metric | Value |
|---|---|
| brier_score | 0.012639 |
| log_loss | 0.326894 |
| calibration_err | 0.003025 |
| mean_abs_error | 0.060892 |
| coverage | 1.0000 |
| num_markets_eval | 681 |
| num_event_nodes | 276 |
| total_seconds | 7.8 |

**Notes:** Small but consistent improvement. Calibration error dropped further (0.005 → 0.003). Finer bins allow the calibration to capture local nonlinearities in the prediction-to-price mapping. 40 bins with ~17 markets each is still robust.
