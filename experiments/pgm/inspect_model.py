"""
inspect_model.py — Generate an HTML inspector for the cached Semantic Event Graph.

Usage:
    uv run inspect_model.py              # generate inspector.html
    open inspector.html                  # view in browser
"""

import os
import sys
import itertools
from collections import defaultdict
from html import escape

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    SemanticEventGraph, CACHE_PATH,
    EVIDENCE_SIM_THRESHOLD, MAX_PARENTS_PER_MARKET,
)
from prepare import load_dataset

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspector.html")


def load_model():
    if not os.path.exists(CACHE_PATH):
        print(f"No cached model at {CACHE_PATH}. Run: uv run run.py")
        sys.exit(1)
    return SemanticEventGraph.load(CACHE_PATH)


def clean_label(text):
    """Strip metadata prefixes to find actual content."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("Source:", "Author:", "Date:", "URL:", "===", "---")) and len(line) > 20:
            return line[:120]
    return text[:120]


def price_color(p):
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        t = p / 0.5
        r, g, b = int(59 + t * 196), int(130 + t * 125), int(246 + t * 9)
    else:
        t = (p - 0.5) / 0.5
        r, g, b = int(255 - t * 21), int(255 - t * 167), int(255 - t * 243)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Section: Event node detail cards with connected markets
# ---------------------------------------------------------------------------

def render_event_nodes(model, markets_by_id):
    """Show each active event node with its connected markets and CPD influence."""
    # Find which markets each event node connects to
    evt_to_markets = defaultdict(list)
    for market_id, parents in model.market_parents.items():
        for evt_id, sim in parents:
            evt_to_markets[evt_id].append((market_id, sim))

    # Sort by number of market connections
    active_events = [(eid, model.event_nodes[eid]) for eid in evt_to_markets]
    active_events.sort(key=lambda x: len(evt_to_markets[x[0]]), reverse=True)

    html = f'<p class="subtitle">{len(active_events)} event nodes are connected to markets. Each node is a semantic concept derived from article clustering.</p>\n'

    for evt_id, node in active_events:
        connected = evt_to_markets[evt_id]
        connected.sort(key=lambda x: x[1], reverse=True)
        label = clean_label(node.label)

        # Unique meaningful aliases
        seen = {label[:50]}
        unique_aliases = []
        for alias in node.aliases:
            clean = clean_label(alias)
            short = clean[:50]
            if short not in seen and len(clean) > 20:
                seen.add(short)
                unique_aliases.append(escape(clean))
            if len(unique_aliases) >= 3:
                break

        html += '<div class="evt-detail">\n'
        html += f'  <div class="evt-header">'
        html += f'    <span class="evt-id">{evt_id}</span>'
        html += f'    <span class="evt-obs">{node.observation_count} article mentions</span>'
        html += f'    <span class="evt-conn">{len(connected)} markets</span>'
        html += f'  </div>\n'
        html += f'  <div class="evt-label">{escape(label)}</div>\n'

        if unique_aliases:
            html += '  <div class="evt-aliases">Also: ' + " | ".join(unique_aliases[:3]) + '</div>\n'

        # Connected markets with similarity and CPD influence
        html += '  <div class="evt-markets">\n'
        for mid, sim in connected[:8]:
            m = markets_by_id.get(mid)
            if not m:
                continue
            price = m.market_price
            # Show what the CPD says: P(Yes) with vs without this parent
            cpd = model.market_cpds.get(mid, {})
            parents = model.market_parents.get(mid, [])
            parent_idx = next((i for i, (eid, _) in enumerate(parents) if eid == evt_id), None)

            influence = ""
            if parent_idx is not None and len(parents) > 0:
                n = len(parents)
                all_on = (1 << n) - 1
                this_off = all_on ^ (1 << parent_idx)
                p_with = cpd.get(all_on, model.base_rate)
                p_without = cpd.get(this_off, model.base_rate)
                delta = p_with - p_without
                direction = "+" if delta >= 0 else ""
                influence = f'<span class="influence {"inf-pos" if delta >= 0 else "inf-neg"}">{direction}{delta:.3f}</span>'

            html += f'    <div class="evt-market-row">'
            html += f'      <span class="sim-badge">{sim:.3f}</span>'
            html += f'      <span class="mkt-price" style="background:{price_color(price)};color:{"#fff" if price > 0.6 or price < 0.3 else "#111"}">{price:.0%}</span>'
            html += f'      {influence}'
            html += f'      <span class="mkt-question">{escape(m.question[:80])}</span>'
            html += f'    </div>\n'

        if len(connected) > 8:
            html += f'    <div class="evt-more">+{len(connected) - 8} more markets</div>\n'
        html += '  </div>\n'
        html += '</div>\n'

    return html


# ---------------------------------------------------------------------------
# Section: Market detail cards with parent event nodes and CPD tables
# ---------------------------------------------------------------------------

def render_market_details(model, markets):
    """Show sample markets with their parent event nodes and full CPD tables."""
    # Pick diverse markets: some with many parents, some with few, various prices
    market_list = []
    for m in markets:
        parents = model.market_parents.get(m.id, [])
        if parents:
            market_list.append(m)

    # Sort by number of parents descending, take a diverse sample
    market_list.sort(key=lambda m: len(model.market_parents.get(m.id, [])), reverse=True)

    # Pick 15: 5 with most parents, 5 random from middle, 5 with fewest
    sample = market_list[:5]
    if len(market_list) > 15:
        mid_start = len(market_list) // 3
        sample += market_list[mid_start:mid_start + 5]
        sample += market_list[-5:]
    else:
        sample = market_list[:15]

    html = f'<p class="subtitle">Showing {len(sample)} markets with their parent event nodes and conditional probability tables.</p>\n'

    for m in sample:
        parents = model.market_parents.get(m.id, [])
        cpd = model.market_cpds.get(m.id, {})
        price = m.market_price

        html += '<div class="mkt-detail">\n'
        html += f'  <div class="mkt-header">'
        html += f'    <span class="mkt-price-lg" style="background:{price_color(price)};color:{"#fff" if price > 0.6 or price < 0.3 else "#111"}">{price:.0%}</span>'
        html += f'    <span class="mkt-title">{escape(m.question[:120])}</span>'
        html += f'  </div>\n'

        # Parent event nodes
        html += '  <div class="mkt-parents">\n'
        for i, (evt_id, sim) in enumerate(parents):
            node = model.event_nodes.get(evt_id)
            label = clean_label(node.label) if node else evt_id
            html += f'    <div class="parent-node">'
            html += f'      <span class="parent-arrow">&#x2192;</span>'
            html += f'      <span class="sim-badge">{sim:.3f}</span>'
            html += f'      <code>{evt_id}</code> {escape(label)}'
            html += f'    </div>\n'

        # CPD table
        n = len(parents)
        if n > 0 and cpd:
            html += '    <div class="cpd-section">\n'
            html += '      <table class="cpd-mini">\n'
            html += '        <thead><tr>'
            for i, (evt_id, _) in enumerate(parents):
                html += f'<th>{evt_id}</th>'
            html += '<th>P(Yes)</th></tr></thead>\n'
            html += '        <tbody>\n'

            for combo in range(2 ** n):
                html += '          <tr>'
                for j in range(n):
                    state = "obs" if combo & (1 << j) else "-"
                    cls = "state-on" if state == "obs" else "state-off"
                    html += f'<td class="{cls}">{state}</td>'
                p_yes = cpd.get(combo, model.base_rate)
                html += f'<td class="cpd-val" style="background:{price_color(p_yes)};color:{"#fff" if p_yes > 0.6 or p_yes < 0.3 else "#111"}">{p_yes:.3f}</td>'
                html += '</tr>\n'

            html += '        </tbody></table>\n'
            html += '    </div>\n'

        html += '  </div>\n'
        html += '</div>\n'

    return html


# ---------------------------------------------------------------------------
# Section: Graph overview SVG
# ---------------------------------------------------------------------------

def render_graph_overview(model, markets_by_id):
    """Bipartite graph SVG: event nodes on left, market nodes on right."""
    # Show a subset: top 20 most-connected event nodes and their markets
    evt_conn_count = defaultdict(int)
    for parents in model.market_parents.values():
        for evt_id, _ in parents:
            evt_conn_count[evt_id] += 1

    top_events = sorted(evt_conn_count.items(), key=lambda x: x[1], reverse=True)[:20]
    top_evt_ids = {eid for eid, _ in top_events}

    # Find markets connected to these events
    connected_markets = set()
    for mid, parents in model.market_parents.items():
        if any(eid in top_evt_ids for eid, _ in parents):
            connected_markets.add(mid)

    # Limit markets shown
    shown_markets = sorted(connected_markets)[:40]

    left_count = len(top_events)
    right_count = len(shown_markets)
    row_h = 28
    pad = 20
    left_w = 350
    right_w = 400
    gap = 150
    w = left_w + gap + right_w + 2 * pad
    h = max(left_count, right_count) * row_h + 2 * pad + 40

    svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="font-family: -apple-system, sans-serif;">\n'

    # Header
    svg += f'<text x="{pad + left_w // 2}" y="{pad}" text-anchor="middle" font-size="12" font-weight="bold" fill="#2563eb">Event Nodes (concepts)</text>\n'
    svg += f'<text x="{pad + left_w + gap + right_w // 2}" y="{pad}" text-anchor="middle" font-size="12" font-weight="bold" fill="#7c3aed">Market Nodes</text>\n'

    # Event nodes (left)
    evt_positions = {}
    for i, (evt_id, count) in enumerate(top_events):
        y = pad + 20 + i * row_h
        evt_positions[evt_id] = y + row_h // 2
        node = model.event_nodes.get(evt_id)
        label = clean_label(node.label) if node else evt_id
        label = label[:45]
        # Background
        alpha = min(0.3, 0.05 + count * 0.02)
        svg += f'<rect x="{pad}" y="{y}" width="{left_w}" height="{row_h - 4}" rx="4" fill="rgba(37,99,235,{alpha:.2f})" stroke="#93c5fd" stroke-width="0.5"/>\n'
        svg += f'<text x="{pad + 5}" y="{y + row_h // 2 + 1}" font-size="10" fill="#1e40af" font-weight="500">{escape(label)}</text>\n'
        svg += f'<text x="{pad + left_w - 5}" y="{y + row_h // 2 + 1}" text-anchor="end" font-size="9" fill="#6b7280">{count}m</text>\n'

    # Market nodes (right)
    mkt_positions = {}
    rx = pad + left_w + gap
    for i, mid in enumerate(shown_markets):
        y = pad + 20 + i * row_h
        mkt_positions[mid] = y + row_h // 2
        m = markets_by_id.get(mid)
        if not m:
            continue
        label = m.question[:50]
        price = m.market_price
        svg += f'<rect x="{rx}" y="{y}" width="{right_w}" height="{row_h - 4}" rx="4" fill="{price_color(price)}22" stroke="{price_color(price)}" stroke-width="0.5"/>\n'
        svg += f'<text x="{rx + 5}" y="{y + row_h // 2 + 1}" font-size="10" fill="#374151">{escape(label)}</text>\n'
        svg += f'<text x="{rx + right_w - 5}" y="{y + row_h // 2 + 1}" text-anchor="end" font-size="9" fill="#6b7280" font-weight="bold">{price:.0%}</text>\n'

    # Edges
    for mid in shown_markets:
        parents = model.market_parents.get(mid, [])
        for evt_id, sim in parents:
            if evt_id in evt_positions and mid in mkt_positions:
                x1 = pad + left_w
                y1 = evt_positions[evt_id]
                x2 = rx
                y2 = mkt_positions[mid]
                alpha = min(0.5, sim * 0.8)
                svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgba(107,114,128,{alpha:.2f})" stroke-width="1"/>\n'

    svg += '</svg>'
    return svg


# ---------------------------------------------------------------------------
# Section: Calibration
# ---------------------------------------------------------------------------

def render_calibration_svg(model):
    cal_bins = getattr(model, "_cal_bins", [])
    if not cal_bins:
        return "<p>No calibration data.</p>"
    w, h = 380, 380
    pad = 50
    svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="font-family: -apple-system, sans-serif;">\n'
    for i in range(6):
        val = i / 5
        x = pad + val * (w - 2 * pad)
        y = h - pad - val * (h - 2 * pad)
        svg += f'<line x1="{pad}" y1="{y}" x2="{w - pad}" y2="{y}" stroke="#e5e7eb"/>\n'
        svg += f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{h - pad}" stroke="#e5e7eb"/>\n'
        svg += f'<text x="{pad - 8}" y="{y + 4}" text-anchor="end" font-size="10" fill="#6b7280">{val:.1f}</text>\n'
        svg += f'<text x="{x}" y="{h - pad + 16}" text-anchor="middle" font-size="10" fill="#6b7280">{val:.1f}</text>\n'
    svg += f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" stroke="#d1d5db" stroke-width="1.5" stroke-dasharray="6,3"/>\n'
    points = []
    for raw, cal in cal_bins:
        x = pad + max(0, min(1, raw)) * (w - 2 * pad)
        y = h - pad - max(0, min(1, cal)) * (h - 2 * pad)
        points.append((x, y))
    if len(points) > 1:
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg += f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.5"/>\n'
    for x, y in points:
        svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>\n'
    svg += f'<text x="{w // 2}" y="{h - 8}" text-anchor="middle" font-size="11" fill="#374151">Raw CPD Prediction</text>\n'
    svg += f'<text x="14" y="{h // 2}" text-anchor="middle" font-size="11" fill="#374151" transform="rotate(-90, 14, {h // 2})">Calibrated Output</text>\n'
    svg += '</svg>'
    return svg


# ---------------------------------------------------------------------------
# Full HTML
# ---------------------------------------------------------------------------

def generate_html(model, markets):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    stats = model.stats()
    markets_by_id = {m.id: m for m in markets}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Inspector — Semantic Event Graph</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f9fafb; color: #111827;
    max-width: 1200px; margin: 0 auto; padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.25rem; margin: 3rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e5e7eb; }}
  h3 {{ font-size: 1.05rem; margin: 2rem 0 0.75rem; color: #374151; }}
  p.subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1rem; }}
  code {{ background: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.78rem; font-family: "SF Mono", monospace; }}

  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .stat {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem 1.2rem; min-width: 110px; }}
  .stat-value {{ font-size: 1.3rem; font-weight: 700; color: #2563eb; }}
  .stat-label {{ font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}

  .graph-container {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; overflow-x: auto; }}

  /* Event node detail cards */
  .evt-detail {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .evt-header {{ display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.4rem; flex-wrap: wrap; }}
  .evt-id {{ font-family: monospace; font-size: 0.8rem; color: #2563eb; font-weight: 600; }}
  .evt-obs {{ background: #eff6ff; color: #2563eb; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }}
  .evt-conn {{ background: #f3e8ff; color: #7c3aed; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }}
  .evt-label {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem; }}
  .evt-aliases {{ font-size: 0.78rem; color: #6b7280; margin-bottom: 0.5rem; }}
  .evt-markets {{ border-top: 1px solid #f3f4f6; padding-top: 0.5rem; }}
  .evt-market-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; font-size: 0.82rem; }}
  .evt-more {{ color: #9ca3af; font-size: 0.78rem; font-style: italic; margin-top: 0.3rem; }}

  .sim-badge {{ display: inline-block; background: #f3f4f6; padding: 0.1rem 0.35rem; border-radius: 3px; font-family: monospace; font-size: 0.72rem; min-width: 42px; text-align: center; }}
  .mkt-price {{ display: inline-block; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.72rem; font-weight: 700; min-width: 35px; text-align: center; }}
  .mkt-question {{ color: #374151; }}
  .influence {{ font-family: monospace; font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  .inf-pos {{ background: #dcfce7; color: #166534; }}
  .inf-neg {{ background: #fee2e2; color: #991b1b; }}

  /* Market detail cards */
  .mkt-detail {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  .mkt-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; }}
  .mkt-price-lg {{ padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 1rem; font-weight: 700; min-width: 50px; text-align: center; }}
  .mkt-title {{ font-weight: 600; font-size: 0.95rem; }}
  .mkt-parents {{ margin-left: 0.5rem; }}
  .parent-node {{ margin: 0.3rem 0; font-size: 0.85rem; display: flex; align-items: center; gap: 0.4rem; }}
  .parent-arrow {{ color: #9ca3af; font-size: 1.1rem; }}

  /* CPD mini-tables */
  .cpd-section {{ margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #f3f4f6; }}
  .cpd-mini {{ border-collapse: collapse; font-size: 0.78rem; }}
  .cpd-mini th {{ background: #f3f4f6; padding: 0.3rem 0.5rem; border: 1px solid #e5e7eb; font-size: 0.72rem; font-weight: 600; text-align: center; }}
  .cpd-mini td {{ padding: 0.3rem 0.5rem; border: 1px solid #e5e7eb; text-align: center; }}
  .state-on {{ background: #dbeafe; color: #1e40af; font-weight: 600; }}
  .state-off {{ background: #f9fafb; color: #9ca3af; }}
  .cpd-val {{ font-weight: 700; }}

  .cal-container {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; display: inline-block; }}
</style>
</head>
<body>

<h1>Model Inspector</h1>
<p class="subtitle">Semantic Event Graph &mdash; {now} &mdash; Event concept nodes &#x2192; Market prediction nodes</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{stats['num_event_nodes']}</div><div class="stat-label">Event Nodes (total)</div></div>
  <div class="stat"><div class="stat-value">{stats['num_active_event_nodes']}</div><div class="stat-label">Active in BN</div></div>
  <div class="stat"><div class="stat-value">{stats['num_market_nodes']}</div><div class="stat-label">Market Nodes</div></div>
  <div class="stat"><div class="stat-value">{stats['num_edges']}</div><div class="stat-label">Edges</div></div>
  <div class="stat"><div class="stat-value">{stats['num_articles']}</div><div class="stat-label">Articles</div></div>
  <div class="stat"><div class="stat-value">{model.base_rate:.2f}</div><div class="stat-label">Base Rate</div></div>
</div>

<h2>Graph Structure</h2>
<p class="subtitle">Bipartite graph: event concept nodes (left, blue) connect to market nodes (right, colored by price). Showing top 20 most-connected event nodes. Line opacity = similarity strength.</p>
<div class="graph-container">
{render_graph_overview(model, markets_by_id)}
</div>

<h2>Event Nodes &mdash; Concepts from Articles</h2>
<p class="subtitle">Each event node is a semantic cluster: multiple article passages that describe the same concept. Connected markets inherit evidence from this node. The <span class="influence inf-pos">+0.05</span> / <span class="influence inf-neg">-0.03</span> badges show how much observing this event shifts the market probability vs. not observing it.</p>
{render_event_nodes(model, markets_by_id)}

<h2>Market Nodes &mdash; CPD Tables</h2>
<p class="subtitle">Each market is a BN node with parent event nodes. The CPD table shows P(Yes) for every combination of parent states (obs = event observed, - = not observed). Each row is one possible world.</p>
{render_market_details(model, markets)}

<h2>Calibration Curve</h2>
<p class="subtitle">Maps raw CPD predictions to calibrated probabilities. Dashed = perfect calibration.</p>
<div class="cal-container">
{render_calibration_svg(model)}
</div>

</body>
</html>"""


def main():
    print("Loading model...")
    model = load_model()
    print("Loading dataset...")
    dataset = load_dataset()
    print("Generating inspector HTML...")
    html = generate_html(model, dataset.markets)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Inspector written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
