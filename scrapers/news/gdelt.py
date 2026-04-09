"""
GDELT Project scraper — free, unlimited, massive global events database.
Uses the GDELT DOC 2.0 API to search for recent articles.
"""

import time
import requests
from utils import save_article, fetch_full_text

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Fewer queries to avoid rate limiting; use sourcelang:english to filter
QUERIES = [
    "Iran war sourcelang:english",
    "Iran conflict military sourcelang:english",
    "Iran nuclear sourcelang:english",
    "Middle East geopolitics Iran sourcelang:english",
]


def scrape(max_per_query: int = 15) -> list[str]:
    saved = []
    seen_urls = set()

    for i, query in enumerate(QUERIES):
        if i > 0:
            time.sleep(5)  # Rate limit: GDELT throttles rapid requests

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": max_per_query,
            "format": "json",
            "sort": "DateDesc",
            "timespan": "14d",
        }

        try:
            resp = requests.get(GDELT_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [gdelt] query '{query}' failed: {e}")
            continue

        articles = data.get("articles", [])
        print(f"  [gdelt] query '{query}' returned {len(articles)} articles")

        for art in articles:
            url = art.get("url", "")
            title = art.get("title", "")
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            date = art.get("seendate", "")
            source = art.get("domain", "")

            # GDELT gives URLs but not full text — fetch it
            text = fetch_full_text(url)
            if len(text) < 200:
                print(f"  [gdelt] skipping (too short): {title[:60]}")
                continue

            path = save_article(
                "gdelt", title, text,
                url=url, author=source, date=date
            )
            saved.append(path)
            print(f"  [gdelt] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[gdelt] Starting GDELT scraper...")
    results = scrape()
    print(f"[gdelt] Done — saved {len(results)} articles")
