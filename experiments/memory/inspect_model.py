"""
inspect_model.py — Generate an HTML inspector for the memory system.

Usage:
    uv run inspect_model.py              # generate inspector.html
    open inspector.html                  # view in browser

Loads the memory model and dataset, then produces a visual deep-dive into
the memory bank: Q-value distributions, category coverage, embedding
neighborhoods, retrieval quality, and prediction comparisons.
"""

import os
import sys
import hashlib
from collections import defaultdict
from html import escape

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from prepare import load_dataset, categorize_text, extract_keywords

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspector.html")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def prob_color(p):
    """Map probability to a blue-to-red color."""
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        t = p / 0.5
        r, g, b = int(59 + t * 196), int(130 + t * 125), int(246 + t * 9)
    else:
        t = (p - 0.5) / 0.5
        r, g, b = int(255 - t * 21), int(255 - t * 167), int(255 - t * 243)
    return f"rgb({r},{g},{b})"


def q_color(q):
    """Map Q-value to green intensity."""
    q = max(0.0, min(1.0, q))
    g = int(100 + q * 155)
    return f"rgb(16, {g}, 69)"


# ---------------------------------------------------------------------------
# Section: Memory Bank Overview
# ---------------------------------------------------------------------------

def render_overview(model, dataset):
    stats = model.stats()
    memories = model.get_memories()

    q_values = [m.q_value for m in memories]
    q_arr = np.array(q_values) if q_values else np.array([0.0])

    cat_counts = defaultdict(int)
    cat_q = defaultdict(list)
    for m in memories:
        cat_counts[m.category] += 1
        cat_q[m.category].append(m.q_value)

    # Category bars
    cats_sorted = sorted(cat_counts.keys(), key=lambda c: cat_counts[c], reverse=True)
    max_count = max(cat_counts.values()) if cat_counts else 1
    bar_max = 300

    cat_html = ""
    for cat in cats_sorted:
        n = cat_counts[cat]
        avg_q = np.mean(cat_q[cat])
        bw = max(4, (n / max_count) * bar_max)
        cat_html += f"""
        <div class="cat-row">
          <span class="cat-label">{cat}</span>
          <div class="cat-bar" style="width:{bw:.0f}px;background:{q_color(avg_q)}"></div>
          <span class="cat-count">{n} memories (avg Q: {avg_q:.3f})</span>
        </div>"""

    return f"""
    <section>
      <h2>Memory Bank Overview</h2>
      <div class="stats">
        <div class="stat"><div class="stat-value">{stats['num_memories']}</div><div class="stat-label">Total Memories</div></div>
        <div class="stat"><div class="stat-value">{stats['num_categories']}</div><div class="stat-label">Categories</div></div>
        <div class="stat"><div class="stat-value">{stats['avg_q_value']:.4f}</div><div class="stat-label">Avg Q-Value</div></div>
        <div class="stat"><div class="stat-value">{stats['max_q_value']:.4f}</div><div class="stat-label">Max Q-Value</div></div>
        <div class="stat"><div class="stat-value">{stats.get('embedding_dim', '?')}</div><div class="stat-label">Embedding Dim</div></div>
      </div>

      <h3>Memories by Category</h3>
      <div class="cat-chart">
        {cat_html}
      </div>
    </section>
    """


# ---------------------------------------------------------------------------
# Section: Q-Value Distribution
# ---------------------------------------------------------------------------

def render_q_distribution(model):
    memories = model.get_memories()
    q_values = [m.q_value for m in memories]
    if not q_values:
        return "<section><h2>Q-Value Distribution</h2><p>No memories.</p></section>"

    # Histogram: 20 bins from 0 to 1
    n_bins = 20
    bins = [0] * n_bins
    for q in q_values:
        idx = min(int(q * n_bins), n_bins - 1)
        idx = max(0, idx)
        bins[idx] += 1

    max_bin = max(bins) or 1
    w, h = 600, 180
    pad = 40
    bar_w = (w - 2 * pad) / n_bins

    bars = ""
    labels = ""
    for i, count in enumerate(bins):
        bh = max(1, (count / max_bin) * (h - 2 * pad))
        x = pad + i * bar_w
        y = h - pad - bh
        q_mid = (i + 0.5) / n_bins
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{bh:.1f}" fill="{q_color(q_mid)}" rx="2"/>\n'
        if i % 4 == 0:
            labels += f'<text x="{x:.1f}" y="{h - 8}" font-size="10" fill="#6b7280">{i / n_bins:.1f}</text>\n'

    # Axis
    axis = f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#d1d5db" stroke-width="1"/>\n'

    q_arr = np.array(q_values)
    stats_text = f'<text x="{w - pad}" y="20" text-anchor="end" font-size="11" fill="#6b7280">mean={q_arr.mean():.3f}  std={q_arr.std():.3f}  median={np.median(q_arr):.3f}</text>'

    return f"""
    <section>
      <h2>Q-Value Distribution</h2>
      <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
        {axis}
        {bars}
        {labels}
        {stats_text}
      </svg>
    </section>
    """


# ---------------------------------------------------------------------------
# Section: Top & Bottom Memories
# ---------------------------------------------------------------------------

def render_top_bottom_memories(model):
    memories = model.get_memories()
    if not memories:
        return ""

    sorted_mems = sorted(memories, key=lambda m: m.q_value, reverse=True)
    top = sorted_mems[:10]
    bottom = sorted_mems[-10:]

    def mem_table(mems, title):
        rows = ""
        for m in mems:
            q_bg = q_color(m.q_value)
            rows += f"""
            <tr>
              <td><span class="q-badge" style="background:{q_bg}">{m.q_value:.3f}</span></td>
              <td>{m.q_updates}</td>
              <td><span class="cat-tag">{m.category}</span></td>
              <td>{escape(m.intent[:80])}</td>
            </tr>"""
        return f"""
        <h3>{title}</h3>
        <table class="mem-table">
          <thead><tr><th>Q-Value</th><th>Updates</th><th>Category</th><th>Market Question</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    return f"""
    <section>
      <h2>Memory Quality</h2>
      {mem_table(top, "Top 10 Memories (highest Q-value)")}
      {mem_table(bottom, "Bottom 10 Memories (lowest Q-value)")}
    </section>
    """


# ---------------------------------------------------------------------------
# Section: Retrieval Quality — sample queries
# ---------------------------------------------------------------------------

def render_retrieval_samples(model, dataset):
    """Show what the memory_lookup tool returns for a few sample queries."""
    samples = []

    # Pick diverse validation markets
    val = dataset.val_markets
    if len(val) >= 5:
        step = len(val) // 5
        samples = [val[i] for i in range(0, len(val), step)][:5]
    else:
        samples = val[:5]

    html = '<section><h2>Retrieval Samples</h2>\n'
    html += '<p class="note">What the memory_lookup tool returns for sample validation markets.</p>\n'

    for market in samples:
        result = model.tool_lookup(market.question)
        odds = market.outcome_prices.get("Yes", "?")
        odds_str = f"{odds:.0%}" if isinstance(odds, float) else str(odds)

        html += f'<div class="retrieval-card">\n'
        html += f'  <div class="retrieval-query">{escape(market.question)}</div>\n'
        html += f'  <div class="retrieval-meta">Market odds: <b>{odds_str}</b> | Category: <b>{result.get("category", "?")}</b> | Base rate: <b>{result.get("category_base_rate", "?")}</b></div>\n'

        mems = result.get("memories", [])
        if mems:
            html += '  <table class="retrieval-table">\n'
            html += '    <thead><tr><th>Q</th><th>Relevance</th><th>Source</th><th>Fact</th></tr></thead>\n'
            html += '    <tbody>\n'
            for m in mems:
                q = m.get("q_value", 0)
                rel = m.get("relevance", 0)
                html += f'    <tr><td><span class="q-badge" style="background:{q_color(q)}">{q:.3f}</span></td>'
                html += f'<td>{rel:.3f}</td>'
                html += f'<td>{escape(str(m.get("source", ""))[:40])}</td>'
                html += f'<td>{escape(str(m.get("fact", ""))[:120])}</td></tr>\n'
            html += '    </tbody></table>\n'
        else:
            html += '  <p class="empty">No memories retrieved</p>\n'
        html += '</div>\n'

    html += '</section>\n'
    return html


# ---------------------------------------------------------------------------
# Section: Prediction Comparison (if run data is available)
# ---------------------------------------------------------------------------

def render_prediction_scatter(predictions, val_markets):
    """SVG scatter plot: our prediction vs market odds."""
    if not predictions:
        return ""

    val_lookup = {
        m.id: m.outcome_prices.get("Yes")
        for m in val_markets
        if m.outcome_prices and m.outcome_prices.get("Yes") is not None
    }
    pred_lookup = dict(predictions)

    points = []
    for mid, pred in pred_lookup.items():
        if mid in val_lookup and val_lookup[mid] is not None:
            points.append((val_lookup[mid], pred))

    if len(points) < 2:
        return ""

    w, h = 450, 450
    pad = 50

    svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="font-family: -apple-system, sans-serif;">\n'

    # Grid
    for i in range(6):
        val = i / 5
        x = pad + val * (w - 2 * pad)
        y = h - pad - val * (h - 2 * pad)
        svg += f'<line x1="{pad}" y1="{y}" x2="{w - pad}" y2="{y}" stroke="#e5e7eb"/>\n'
        svg += f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{h - pad}" stroke="#e5e7eb"/>\n'
        svg += f'<text x="{pad - 8}" y="{y + 4}" text-anchor="end" font-size="10" fill="#6b7280">{val:.1f}</text>\n'
        svg += f'<text x="{x}" y="{h - pad + 16}" text-anchor="middle" font-size="10" fill="#6b7280">{val:.1f}</text>\n'

    # Diagonal (perfect prediction)
    svg += f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" stroke="#d1d5db" stroke-width="1" stroke-dasharray="5,3"/>\n'

    # Points
    for market_odds, pred in points:
        px = pad + market_odds * (w - 2 * pad)
        py = h - pad - pred * (h - 2 * pad)
        error = abs(pred - market_odds)
        color = prob_color(1 - error)  # greener = closer
        svg += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.7"/>\n'

    # Labels
    svg += f'<text x="{w // 2}" y="{h - 8}" text-anchor="middle" font-size="12" fill="#374151">Market Odds (Polymarket)</text>\n'
    svg += f'<text x="14" y="{h // 2}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90, 14, {h // 2})">Agent Prediction</text>\n'

    # MSE annotation
    mse = np.mean([(p - m) ** 2 for m, p in points])
    svg += f'<text x="{w - pad}" y="{pad - 10}" text-anchor="end" font-size="11" fill="#6b7280">Brier = {mse:.4f} (n={len(points)})</text>\n'

    svg += '</svg>\n'

    return f"""
    <section>
      <h2>Prediction vs Market Odds</h2>
      <p class="note">Each dot is one market. Dashed line = perfect prediction. Greener = closer to market odds.</p>
      {svg}
    </section>
    """


# ---------------------------------------------------------------------------
# Section: Per-market breakdown table
# ---------------------------------------------------------------------------

def render_market_table(predictions, val_markets):
    if not predictions:
        return ""

    val_lookup = {m.id: m for m in val_markets}
    pred_lookup = dict(predictions)

    rows_data = []
    for mid, pred in pred_lookup.items():
        m = val_lookup.get(mid)
        if m and m.outcome_prices and m.outcome_prices.get("Yes") is not None:
            odds = m.outcome_prices["Yes"]
            error = (pred - odds) ** 2
            rows_data.append((error, odds, pred, m.question, m.category))

    rows_data.sort(key=lambda r: r[0], reverse=True)  # worst first

    rows_html = ""
    for error, odds, pred, question, category in rows_data:
        err_color = prob_color(min(1.0, error * 5))  # scale for visibility
        rows_html += f"""
        <tr>
          <td style="color:{err_color};font-weight:bold">{error:.4f}</td>
          <td><span class="prob-badge" style="background:{prob_color(odds)}">{odds:.0%}</span></td>
          <td><span class="prob-badge" style="background:{prob_color(pred)}">{pred:.0%}</span></td>
          <td><span class="cat-tag">{category}</span></td>
          <td>{escape(question[:70])}</td>
        </tr>"""

    return f"""
    <section>
      <h2>Per-Market Breakdown</h2>
      <p class="note">Sorted by squared error (worst predictions first).</p>
      <table class="market-table">
        <thead><tr><th>Error</th><th>Market</th><th>Agent</th><th>Category</th><th>Question</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
    """


# ---------------------------------------------------------------------------
# Assemble HTML
# ---------------------------------------------------------------------------

def generate_html(sections: list[str]) -> str:
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ARPM Memory — Model Inspector</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f9fafb; color: #111827; padding: 2rem;
      max-width: 1100px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 2rem; }}
    h2 {{ font-size: 1.2rem; margin: 2.5rem 0 1rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1rem; margin: 1.5rem 0 0.75rem; color: #374151; }}
    .note {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1rem; }}
    .stats {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
    .stat {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem 1.2rem; }}
    .stat-value {{ font-size: 1.3rem; font-weight: bold; font-family: "SF Mono", monospace; }}
    .stat-label {{ font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}

    /* Category bars */
    .cat-chart {{ margin: 1rem 0; }}
    .cat-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }}
    .cat-label {{ width: 100px; font-size: 0.8rem; text-align: right; color: #374151; }}
    .cat-bar {{ height: 20px; border-radius: 3px; min-width: 4px; }}
    .cat-count {{ font-size: 0.75rem; color: #6b7280; }}
    .cat-tag {{ background: #f3f4f6; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; color: #374151; }}

    /* Q badges */
    .q-badge {{ color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.75rem; font-family: "SF Mono", monospace; }}

    /* Prob badges */
    .prob-badge {{ color: white; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: bold; font-family: "SF Mono", monospace; }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; font-size: 0.8rem; margin: 0.5rem 0; }}
    th {{ background: #f3f4f6; text-align: left; padding: 0.5rem 0.7rem; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #374151; }}
    td {{ padding: 0.4rem 0.7rem; border-top: 1px solid #f3f4f6; }}
    tr:hover {{ background: #fafbfc; }}
    .empty {{ color: #9ca3af; font-style: italic; font-size: 0.85rem; }}

    /* Retrieval cards */
    .retrieval-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
    .retrieval-query {{ font-weight: 600; margin-bottom: 0.5rem; }}
    .retrieval-meta {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 0.5rem; }}
    .retrieval-table {{ font-size: 0.78rem; }}

    /* SVG */
    svg {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}

    section {{ margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>ARPM Memory — Model Inspector</h1>
  <p class="subtitle">sqlite-vec memory bank with MemRL Q-learning &mdash; generated {generated_at}</p>
  {body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_dataset()

    print("Building memory model...")
    model = create_model()
    model.build(dataset)

    # Run predictions on validation markets for the scatter plot / table
    print("Running predictions on validation markets...")
    predictions = []
    for market in dataset.val_markets:
        model.start_prediction(market.id)
        result = model.tool_lookup(market.question)
        # Simple heuristic prediction from retrieved memories (no agent call)
        mems = result.get("memories", [])
        if mems:
            # Use category base rate as a simple heuristic prediction
            # (the real prediction comes from the agent, not here)
            pred = result.get("category_base_rate", 0.5)
        else:
            pred = result.get("category_base_rate", 0.5)
        pred = max(0.01, min(0.99, pred))
        predictions.append((market.id, pred))

    print("Generating HTML...")
    sections = [
        render_overview(model, dataset),
        render_q_distribution(model),
        render_top_bottom_memories(model),
        render_retrieval_samples(model, dataset),
        render_prediction_scatter(predictions, dataset.val_markets),
        render_market_table(predictions, dataset.val_markets),
    ]

    html = generate_html(sections)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Inspector written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
