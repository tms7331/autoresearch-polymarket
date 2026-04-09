"""Run all scrapers and produce a summary report."""

import time
import os
import sys

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(__file__))

SCRAPERS = [
    ("GDELT", "gdelt"),
    ("Google News RSS", "gnews"),
    ("Guardian API", "guardian"),
    ("RSS Feeds (multi)", "rss_feeds"),
    ("Wikipedia Current Events", "wikipedia_current"),
    ("Reuters (Browserbase)", "reuters"),
    ("NewsAPI", "newsapi"),
    ("Mediastack", "mediastack"),
]


def main():
    results = {}

    for name, module_name in SCRAPERS:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            mod = __import__(module_name)
            articles = mod.scrape()
            elapsed = time.time() - start
            results[name] = {
                "count": len(articles),
                "time": elapsed,
                "status": "OK",
                "files": articles,
            }
        except Exception as e:
            elapsed = time.time() - start
            results[name] = {
                "count": 0,
                "time": elapsed,
                "status": f"ERROR: {e}",
                "files": [],
            }
            print(f"  ERROR: {e}")

    # Print summary
    print(f"\n\n{'='*60}")
    print("SCRAPER RESULTS SUMMARY")
    print(f"{'='*60}")
    total = 0
    for name, info in sorted(results.items(), key=lambda x: -x[1]["count"]):
        total += info["count"]
        status = info["status"]
        print(f"  {name:30s} | {info['count']:4d} articles | {info['time']:6.1f}s | {status}")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    file_count = len([f for f in os.listdir(data_dir) if f.endswith(".txt")]) if os.path.isdir(data_dir) else 0
    print(f"\n  Total articles saved: {total}")
    print(f"  Files in data/: {file_count}")


if __name__ == "__main__":
    main()
