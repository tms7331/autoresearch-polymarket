"""
arpm-memory experiment runner. Builds memory system, runs predictions, evaluates.
Usage: uv run run.py
"""

import time
import sys

from prepare import (
    TIME_BUDGET, load_dataset, evaluate_brier, evaluate_q_correlation,
    save_memory_bank,
)
from model import create_model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()

# Load data
print("Loading dataset...")
dataset = load_dataset()
print()

# Build memory system
print("Building memory system...")
t_build_start = time.time()
model = create_model()
memories = model.build(dataset)
t_build = time.time() - t_build_start
print(f"Memory system built in {t_build:.1f}s")
model_stats = model.stats()
for k, v in model_stats.items():
    print(f"  {k}: {v}")
print()

# Check time budget
elapsed = time.time() - t_start
if elapsed > TIME_BUDGET:
    print(f"FAIL: memory build exceeded time budget ({elapsed:.1f}s > {TIME_BUDGET}s)")
    sys.exit(1)

# Predict on validation markets
print(f"Predicting on {len(dataset.val_markets)} validation markets...")
t_pred_start = time.time()
predictions = model.predict_batch(dataset.val_markets, dataset)
t_pred = time.time() - t_pred_start
print(f"Predictions complete in {t_pred:.1f}s")
print()

# Update Q-values based on validation outcomes (online learning simulation)
print("Updating Q-values from validation outcomes...")
for market in dataset.val_markets:
    if market.resolved is not None:
        pred_dict = {mid: p for mid, p in predictions}
        if market.id in pred_dict:
            outcome = 1.0 if market.resolved else 0.0
            error = (pred_dict[market.id] - outcome) ** 2
            reward = 1.0 - error
            model.update_q_values(market, reward)
print()

# Evaluate
print("Evaluating...")
results = evaluate_brier(predictions, dataset.val_markets)

# Q-value correlation
q_corr = evaluate_q_correlation(model.get_memories(), predictions, dataset.val_markets)

# Save memory bank for persistence across runs
save_memory_bank(model.get_memories())

t_end = time.time()
total_seconds = t_end - t_start

# Final summary
print("---")
print(f"brier_score:      {results['brier_score']:.6f}")
print(f"log_loss:         {results['log_loss']:.6f}")
print(f"calibration_err:  {results['calibration_err']:.6f}")
print(f"coverage:         {results['coverage']:.4f}")
print(f"q_correlation:    {q_corr:.4f}")
print(f"num_memories:     {model_stats.get('num_memories', 0)}")
print(f"num_markets_eval: {results['num_markets_eval']}")
print(f"total_seconds:    {total_seconds:.1f}")
