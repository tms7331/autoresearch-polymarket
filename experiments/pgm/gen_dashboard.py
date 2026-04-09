"""
gen_dashboard.py — Generate an HTML dashboard from experiment results.

Run via: cd experiments/pgm && uv run gen_dashboard.py
Output:  experiments/pgm/dashboard.html (open in browser)

This script reads results.tsv and experiments.md, then generates a single
standalone HTML file with an overview of all experiment runs.

== INSTRUCTIONS FOR CLAUDE ==

When asked to improve this dashboard, follow these guidelines:

DATA SOURCES:
- results.tsv: tab-separated file with columns: commit, brier_score, coverage, status, description
- experiments.md: markdown file with detailed per-experiment entries including all metrics and notes

HTML GENERATION:
- Output a single self-contained HTML file (inline CSS, no external deps)
- Keep the design clean and minimal — no frameworks, no JavaScript libraries
- Use a monospace or system font stack for a technical/data feel

CURRENT SECTIONS:
1. Header — title, date generated, best Brier score achieved
2. Summary stats — total experiments, keeps, discards, crashes
3. Results table — all experiments in order with key metrics and status
4. Brier score trend — simple inline SVG chart showing score over experiments

FUTURE IDEAS (implement when asked):
- Per-market prediction breakdown (would need to parse run.log files)
- Side-by-side comparison of two experiments
- Category-level accuracy breakdown
- Evidence strength distribution visualization
- Colored cells for metric thresholds (green=good, red=bad)
- Filter/sort controls (would need minimal JS)
"""

import os
import csv
import re
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(SCRIPT_DIR, "results.tsv")
EXPERIMENTS_MD_PATH = os.path.join(SCRIPT_DIR, "experiments.md")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "dashboard.html")


def read_tsv() -> list[dict]:
    """Read results.tsv into a list of dicts."""
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


def parse_experiments_md() -> list[dict]:
    """Parse experiments.md for richer per-experiment data."""
    if not os.path.exists(EXPERIMENTS_MD_PATH):
        return []

    with open(EXPERIMENTS_MD_PATH, "r") as f:
        content = f.read()

    experiments = []
    # Split on ## Experiment headers
    sections = re.split(r'^## Experiment ', content, flags=re.MULTILINE)

    for section in sections[1:]:  # skip preamble before first experiment
        entry = {}
        # Parse experiment number and title
        header_match = re.match(r'(\d+)\s*[—–-]\s*(.+)', section)
        if header_match:
            entry["number"] = int(header_match.group(1))
            entry["title"] = header_match.group(2).strip()

        # Parse metrics from table
        for metric in ["brier_score", "log_loss", "calibration_err",
                       "mean_abs_error", "coverage", "num_markets_eval",
                       "num_event_nodes", "total_seconds"]:
            match = re.search(rf'\|\s*{metric}\s*\|\s*([\d.]+)\s*\|', section)
            if match:
                entry[metric] = match.group(1)

        # Parse status
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', section)
        if status_match:
            entry["status"] = status_match.group(1)

        # Parse commit
        commit_match = re.search(r'\*\*Commit:\*\*\s*`(\w+)`', section)
        if commit_match:
            entry["commit"] = commit_match.group(1)

        # Parse description
        desc_match = re.search(r'\*\*Description:\*\*\s*(.+)', section)
        if desc_match:
            entry["description"] = desc_match.group(1).strip()

        # Parse notes
        notes_match = re.search(r'\*\*Notes:\*\*\s*(.+?)(?=\n##|\Z)', section, re.DOTALL)
        if notes_match:
            entry["notes"] = notes_match.group(1).strip()

        if entry:
            experiments.append(entry)

    return experiments


def generate_svg_chart(experiments: list[dict], width: int = 600, height: int = 200) -> str:
    """Generate an inline SVG line chart of Brier scores over experiments."""
    scores = []
    for e in experiments:
        try:
            score = float(e.get("brier_score", 0))
            if score > 0:
                scores.append(score)
        except (ValueError, TypeError):
            continue

    if len(scores) < 2:
        return ""

    padding = 40
    chart_w = width - 2 * padding
    chart_h = height - 2 * padding

    y_min = min(scores) * 0.9
    y_max = max(scores) * 1.1
    if y_max == y_min:
        y_max = y_min + 0.01

    points = []
    for i, score in enumerate(scores):
        x = padding + (i / (len(scores) - 1)) * chart_w
        y = padding + (1 - (score - y_min) / (y_max - y_min)) * chart_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>'
        for x, y in points
    )

    # Y-axis labels
    y_labels = ""
    for i in range(5):
        val = y_min + (y_max - y_min) * (1 - i / 4)
        y = padding + (i / 4) * chart_h
        y_labels += f'<text x="{padding - 5}" y="{y + 4}" text-anchor="end" font-size="11" fill="#6b7280">{val:.4f}</text>'
        y_labels += f'<line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'

    # X-axis labels
    x_labels = ""
    for i in range(len(scores)):
        x = padding + (i / (len(scores) - 1)) * chart_w if len(scores) > 1 else padding
        x_labels += f'<text x="{x}" y="{height - 8}" text-anchor="middle" font-size="11" fill="#6b7280">#{i + 1}</text>'

    return f"""
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      {y_labels}
      {x_labels}
      <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>
      {dots}
    </svg>
    """


def generate_html(tsv_rows: list[dict], md_experiments: list[dict]) -> str:
    """Generate the full HTML dashboard."""
    # Prefer experiments.md data if available, fall back to TSV
    experiments = md_experiments if md_experiments else tsv_rows

    # Summary stats
    total = len(experiments)
    keeps = sum(1 for e in experiments if e.get("status", "").lower() == "keep")
    discards = sum(1 for e in experiments if e.get("status", "").lower() == "discard")
    crashes = sum(1 for e in experiments if e.get("status", "").lower() == "crash")

    best_brier = None
    for e in experiments:
        try:
            score = float(e.get("brier_score", 0))
            if score > 0 and (best_brier is None or score < best_brier):
                best_brier = score
        except (ValueError, TypeError):
            continue

    best_brier_str = f"{best_brier:.6f}" if best_brier else "N/A"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build table rows
    table_rows = ""
    for i, e in enumerate(experiments):
        num = e.get("number", i + 1)
        status = e.get("status", "?")
        status_class = {
            "keep": "status-keep",
            "discard": "status-discard",
            "crash": "status-crash",
        }.get(status.lower(), "")

        brier = e.get("brier_score", "—")
        coverage = e.get("coverage", "—")
        commit = e.get("commit", "—")
        desc = e.get("description", e.get("title", "—"))
        log_loss = e.get("log_loss", "—")
        cal_err = e.get("calibration_err", "—")
        seconds = e.get("total_seconds", "—")

        table_rows += f"""
        <tr>
          <td>{num}</td>
          <td><code>{commit}</code></td>
          <td><span class="{status_class}">{status}</span></td>
          <td>{brier}</td>
          <td>{log_loss}</td>
          <td>{cal_err}</td>
          <td>{coverage}</td>
          <td>{seconds}</td>
          <td class="desc">{desc}</td>
        </tr>"""

    chart_svg = generate_svg_chart(experiments)
    chart_section = f"""
    <section>
      <h2>Brier Score Trend</h2>
      {chart_svg}
    </section>
    """ if chart_svg else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ARPM Experiment Dashboard</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: "SF Mono", "Fira Code", "Consolas", monospace;
      background: #f9fafb;
      color: #111827;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .subtitle {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 2rem; }}
    h2 {{ font-size: 1.1rem; margin: 2rem 0 1rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
    .stats {{
      display: flex;
      gap: 2rem;
      margin-bottom: 2rem;
    }}
    .stat {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1rem 1.5rem;
    }}
    .stat-value {{ font-size: 1.4rem; font-weight: bold; }}
    .stat-label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      font-size: 0.85rem;
    }}
    th {{
      background: #f3f4f6;
      text-align: left;
      padding: 0.6rem 0.8rem;
      font-weight: 600;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #374151;
    }}
    td {{
      padding: 0.5rem 0.8rem;
      border-top: 1px solid #f3f4f6;
    }}
    tr:hover {{ background: #f9fafb; }}
    td code {{ background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.8rem; }}
    .desc {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .status-keep {{ color: #059669; font-weight: bold; }}
    .status-discard {{ color: #d97706; }}
    .status-crash {{ color: #dc2626; font-weight: bold; }}
    section {{ margin-bottom: 2rem; }}
    svg {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>ARPM Experiment Dashboard</h1>
  <p class="subtitle">Semantic Event Graph with CPDs &mdash; generated {generated_at}</p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{best_brier_str}</div>
      <div class="stat-label">Best Brier Score</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total}</div>
      <div class="stat-label">Total Experiments</div>
    </div>
    <div class="stat">
      <div class="stat-value">{keeps}</div>
      <div class="stat-label">Kept</div>
    </div>
    <div class="stat">
      <div class="stat-value">{discards}</div>
      <div class="stat-label">Discarded</div>
    </div>
    <div class="stat">
      <div class="stat-value">{crashes}</div>
      <div class="stat-label">Crashed</div>
    </div>
  </div>

  {chart_section}

  <section>
    <h2>Experiment Log</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Commit</th>
          <th>Status</th>
          <th>Brier</th>
          <th>Log Loss</th>
          <th>Cal. Err</th>
          <th>Coverage</th>
          <th>Time (s)</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {table_rows if table_rows else '<tr><td colspan="9" style="text-align:center;color:#6b7280;padding:2rem;">No experiments yet. Run the experiment loop to populate.</td></tr>'}
      </tbody>
    </table>
  </section>
</body>
</html>"""


def main():
    tsv_rows = read_tsv()
    md_experiments = parse_experiments_md()

    html = generate_html(tsv_rows, md_experiments)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  Data sources: results.tsv ({len(tsv_rows)} rows), experiments.md ({len(md_experiments)} entries)")


if __name__ == "__main__":
    main()
