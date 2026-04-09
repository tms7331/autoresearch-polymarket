"""
ARPM experiment runner. Builds the model, runs predictions, evaluates.
Usage: uv run run.py
"""

import time
import sys

from prepare import TIME_BUDGET, load_dataset, evaluate
from model import create_model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()

# Load data
print("Loading dataset...")
dataset = load_dataset()
print()

if not dataset.markets:
    print("No markets found in data_polymarket/. Add .txt market files first.")
    sys.exit(1)

# Build model
print("Building model...")
t_build_start = time.time()
model = create_model()
model.build(dataset)
t_build = time.time() - t_build_start
print(f"Model built in {t_build:.1f}s")
model_stats = model.stats()
for k, v in model_stats.items():
    print(f"  {k}: {v}")
print()

# Check time budget
elapsed = time.time() - t_start
if elapsed > TIME_BUDGET:
    print(f"FAIL: model build exceeded time budget ({elapsed:.1f}s > {TIME_BUDGET}s)")
    sys.exit(1)

# Predict on all markets
print(f"Predicting on {len(dataset.markets)} markets...")
t_pred_start = time.time()
predictions = model.predict_batch(dataset.markets)
t_pred = time.time() - t_pred_start
print(f"Predictions complete in {t_pred:.1f}s")
print()

# Evaluate against market prices
print("Evaluating against market prices...")
results = evaluate(predictions, dataset.markets)

t_end = time.time()
total_seconds = t_end - t_start

# Per-market details
print("\nPer-market predictions:")
market_lookup = {m.id: m for m in dataset.markets}
for mid, pred in predictions:
    m = market_lookup[mid]
    err = abs(pred - m.market_price)
    print(f"  pred={pred:.2f}  actual={m.market_price:.2f}  err={err:.2f}  {m.question[:70]}")

# Final summary
print("\n---")
print(f"brier_score:      {results['brier_score']:.6f}")
print(f"log_loss:         {results['log_loss']:.6f}")
print(f"calibration_err:  {results['calibration_err']:.6f}")
print(f"mean_abs_error:   {results['mean_abs_error']:.6f}")
print(f"coverage:         {results['coverage']:.4f}")
print(f"num_markets_eval: {results['num_markets_eval']}")
print(f"num_nodes:        {model_stats.get('num_nodes', 0)}")
print(f"num_edges:        {model_stats.get('num_edges', 0)}")
print(f"num_articles:     {model_stats.get('num_articles', 0)}")
print(f"total_seconds:    {total_seconds:.1f}")
