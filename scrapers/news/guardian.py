"""
The Guardian API scraper — free API key available, full article text.
Uses the Content API: https://open-platform.theguardian.com/
NOTE: Uses the test API key (works for development, rate-limited).
"""

import requests
from utils import save_article

# The Guardian provides a test key for development
API_KEY = "test"
API_URL = "https://content.guardianapis.com/search"

QUERIES = [
    "Iran war",
    "Iran conflict",
    "Iran military strikes",
    "Iran nuclear",
    "Iran Israel",
    "Middle East geopolitics",
    "Iran US tensions",
    "Persian Gulf",
]


def scrape(max_per_query: int = 10) -> list[str]:
    saved = []
    seen_urls = set()

    for query in QUERIES:
        params = {
            "q": query,
            "api-key": API_KEY,
            "show-fields": "bodyText,byline,firstPublicationDate",
            "page-size": max_per_query,
            "order-by": "newest",
            "section": "world",
        }

        try:
            resp = requests.get(API_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [guardian] query '{query}' failed: {e}")
            continue

        results = data.get("response", {}).get("results", [])
        print(f"  [guardian] query '{query}' returned {len(results)} results")

        for item in results:
            url = item.get("webUrl", "")
            title = item.get("webTitle", "")
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            fields = item.get("fields", {})
            text = fields.get("bodyText", "")
            author = fields.get("byline", "")
            date = fields.get("firstPublicationDate", "")

            if len(text) < 200:
                print(f"  [guardian] skipping (too short): {title[:60]}")
                continue

            path = save_article(
                "guardian", title, text,
                url=url, author=author, date=date
            )
            saved.append(path)
            print(f"  [guardian] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[guardian] Starting Guardian scraper...")
    results = scrape()
    print(f"[guardian] Done — saved {len(results)} articles")
