"""
gen_dashboard.py — Generate an HTML dashboard from experiment results.

Run via: cd experiments/memory && uv run gen_dashboard.py
Output:  experiments/memory/dashboard.html (open in browser)

Reads results.tsv and generates a standalone HTML overview of all runs.
"""

import os
import csv
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(SCRIPT_DIR, "results.tsv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "dashboard.html")


def read_tsv() -> list[dict]:
    if not os.path.exists(TSV_PATH):
        return []
    rows = []
    with open(TSV_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["brier_score"] = float(row.get("brier_score", 0))
                row["coverage"] = float(row.get("coverage", 0))
            except (ValueError, TypeError):
                row["brier_score"] = 0.0
                row["coverage"] = 0.0
            rows.append(row)
    return rows


def generate_svg_chart(rows: list[dict], width: int = 700, height: int = 220) -> str:
    scores = []
    for e in rows:
        try:
            s = float(e.get("brier_score", 0))
            if s > 0:
                scores.append(s)
        except (ValueError, TypeError):
            continue

    if len(scores) < 2:
        return ""

    pad = 50
    cw = width - 2 * pad
    ch = height - 2 * pad

    y_min = min(scores) * 0.9
    y_max = max(scores) * 1.1
    if y_max == y_min:
        y_max = y_min + 0.01

    points = []
    for i, s in enumerate(scores):
        x = pad + (i / (len(scores) - 1)) * cw
        y = pad + (1 - (s - y_min) / (y_max - y_min)) * ch
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>'
        for x, y in points
    )

    y_labels = ""
    for i in range(5):
        val = y_min + (y_max - y_min) * (1 - i / 4)
        y = pad + (i / 4) * ch
        y_labels += f'<text x="{pad - 5}" y="{y + 4}" text-anchor="end" font-size="11" fill="#6b7280">{val:.4f}</text>'
        y_labels += f'<line x1="{pad}" y1="{y}" x2="{width - pad}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'

    x_labels = ""
    for i in range(len(scores)):
        x = pad + (i / (len(scores) - 1)) * cw if len(scores) > 1 else pad
        x_labels += f'<text x="{x}" y="{height - 8}" text-anchor="middle" font-size="11" fill="#6b7280">#{i + 1}</text>'

    return f"""
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      {y_labels}
      {x_labels}
      <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>
      {dots}
    </svg>
    """


def generate_html(rows: list[dict]) -> str:
    total = len(rows)
    keeps = sum(1 for e in rows if e.get("status", "").lower() == "keep")
    discards = sum(1 for e in rows if e.get("status", "").lower() == "discard")
    crashes = sum(1 for e in rows if e.get("status", "").lower() == "crash")

    best_brier = None
    for e in rows:
        try:
            s = float(e.get("brier_score", 0))
            if s > 0 and (best_brier is None or s < best_brier):
                best_brier = s
        except (ValueError, TypeError):
            continue

    best_str = f"{best_brier:.6f}" if best_brier else "N/A"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    table_rows = ""
    for i, e in enumerate(rows):
        status = e.get("status", "?")
        sc = {"keep": "status-keep", "discard": "status-discard", "crash": "status-crash"}.get(status.lower(), "")
        table_rows += f"""
        <tr>
          <td>{i + 1}</td>
          <td><code>{e.get('commit', '—')}</code></td>
          <td><span class="{sc}">{status}</span></td>
          <td>{e.get('brier_score', '—')}</td>
          <td>{e.get('coverage', '—')}</td>
          <td class="desc">{e.get('description', '—')}</td>
        </tr>"""

    chart = generate_svg_chart(rows)
    chart_section = f'<section><h2>Brier Score Trend</h2>{chart}</section>' if chart else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ARPM Memory — Experiment Dashboard</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
      background: #f9fafb; color: #111827; padding: 2rem;
      max-width: 1100px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 2rem; }}
    h2 {{ font-size: 1.1rem; margin: 2rem 0 1rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
    .stats {{ display: flex; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap; }}
    .stat {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.5rem; }}
    .stat-value {{ font-size: 1.4rem; font-weight: bold; }}
    .stat-label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; font-size: 0.85rem; }}
    th {{ background: #f3f4f6; text-align: left; padding: 0.6rem 0.8rem; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #374151; }}
    td {{ padding: 0.5rem 0.8rem; border-top: 1px solid #f3f4f6; }}
    tr:hover {{ background: #f9fafb; }}
    td code {{ background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.8rem; }}
    .desc {{ max-width: 350px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .status-keep {{ color: #059669; font-weight: bold; }}
    .status-discard {{ color: #d97706; }}
    .status-crash {{ color: #dc2626; font-weight: bold; }}
    section {{ margin-bottom: 2rem; }}
    svg {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>ARPM Memory — Experiment Dashboard</h1>
  <p class="subtitle">MemRL memory bank + Claude agent &mdash; generated {generated_at}</p>

  <div class="stats">
    <div class="stat"><div class="stat-value">{best_str}</div><div class="stat-label">Best Brier Score</div></div>
    <div class="stat"><div class="stat-value">{total}</div><div class="stat-label">Total Experiments</div></div>
    <div class="stat"><div class="stat-value">{keeps}</div><div class="stat-label">Kept</div></div>
    <div class="stat"><div class="stat-value">{discards}</div><div class="stat-label">Discarded</div></div>
    <div class="stat"><div class="stat-value">{crashes}</div><div class="stat-label">Crashed</div></div>
  </div>

  {chart_section}

  <section>
    <h2>Experiment Log</h2>
    <table>
      <thead><tr><th>#</th><th>Commit</th><th>Status</th><th>Brier</th><th>Coverage</th><th>Description</th></tr></thead>
      <tbody>
        {table_rows if table_rows else '<tr><td colspan="6" style="text-align:center;color:#6b7280;padding:2rem;">No experiments yet. Run the loop to populate results.tsv.</td></tr>'}
      </tbody>
    </table>
  </section>
</body>
</html>"""


def main():
    rows = read_tsv()
    html = generate_html(rows)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  Data source: results.tsv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
