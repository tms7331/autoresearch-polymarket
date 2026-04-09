"""
arpm-memory experiment runner.

Builds the memory system (model.py), then runs the Claude agent (agent.py) on
validation markets.

This file is FIXED infrastructure — do not modify.

Usage:
    uv run run.py              # fast mode: 10-market subset (~3 min)
    uv run run.py --full       # full mode: all validation markets (~20 min)
"""

import argparse
import hashlib
import time
import sys

from prepare import (
    TIME_BUDGET, load_dataset, evaluate_brier, evaluate_q_correlation,
    save_memory_bank,
)
from model import create_model
from agent import predict_market

FAST_SUBSET_SIZE = 10

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Evaluate on all validation markets")
args = parser.parse_args()

t_start = time.time()

print("Loading dataset...")
dataset = load_dataset()
print()

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

elapsed = time.time() - t_start
if elapsed > TIME_BUDGET:
    print(f"FAIL: memory build exceeded time budget ({elapsed:.1f}s > {TIME_BUDGET}s)")
    sys.exit(1)

eval_markets = dataset.val_markets
if not args.full and len(eval_markets) > FAST_SUBSET_SIZE:
    sorted_markets = sorted(eval_markets, key=lambda m: hashlib.md5(m.id.encode()).hexdigest())
    eval_markets = sorted_markets[:FAST_SUBSET_SIZE]
    print(f"Fast mode: evaluating on {FAST_SUBSET_SIZE}/{len(dataset.val_markets)} validation markets")
    print(f"  (use --full to evaluate on all {len(dataset.val_markets)} markets)")
    print()

n_markets = len(eval_markets)
print(f"Running agent on {n_markets} validation markets...")
predictions = []
agent_errors = 0

for i, market in enumerate(eval_markets):
    t_market = time.time()
    model.start_prediction(market.id)
    try:
        result = predict_market(market.question, model)
        prob = result["probability"]
        predictions.append((market.id, prob))
        dt = time.time() - t_market
        print(f"  [{i+1}/{n_markets}] {market.question[:60]}...  p={prob:.3f}  ({dt:.1f}s)")
    except Exception as e:
        agent_errors += 1
        predictions.append((market.id, 0.5))
        print(f"  [{i+1}/{n_markets}] ERROR: {e}")

print()
if agent_errors:
    print(f"Agent errors: {agent_errors}/{n_markets}")

print("Updating Q-values from validation outcomes...")
pred_dict = {mid: p for mid, p in predictions}
for market in eval_markets:
    if market.outcome_prices and market.outcome_prices.get("Yes") is not None:
        if market.id in pred_dict:
            market_odds = market.outcome_prices["Yes"]
            error = (pred_dict[market.id] - market_odds) ** 2
            reward = 1.0 - error
            model.update_q_values(market.id, reward)
print()

print("Evaluating...")
results = evaluate_brier(predictions, eval_markets)
q_corr = evaluate_q_correlation(model.get_memories(), predictions, eval_markets)
save_memory_bank(model.get_memories())

t_end = time.time()
total_seconds = t_end - t_start

mode = "full" if args.full else "fast"
print("---")
print(f"mode:             {mode}")
print(f"brier_score:      {results['brier_score']:.6f}")
print(f"log_loss:         {results['log_loss']:.6f}")
print(f"calibration_err:  {results['calibration_err']:.6f}")
print(f"coverage:         {results['coverage']:.4f}")
print(f"q_correlation:    {q_corr:.4f}")
print(f"num_memories:     {model_stats.get('num_memories', 0)}")
print(f"num_markets_eval: {results['num_markets_eval']}")
print(f"agent_errors:     {agent_errors}")
print(f"total_seconds:    {total_seconds:.1f}")
