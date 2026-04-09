"""
NewsAPI.org scraper — aggregates 80k+ sources.
Free tier: 100 requests/day, headlines only (no full text on free plan).
We fetch headlines then scrape full text from the source URLs.

NOTE: Requires a free API key from https://newsapi.org/register
Falls back to "everything" endpoint search without key (will fail on free tier
for production domains, but works from localhost/dev).
"""

import requests
from utils import save_article, fetch_full_text

# Free tier developer key — get one at newsapi.org/register
# Leave empty to skip this scraper
API_KEY = ""

API_URL = "https://newsapi.org/v2/everything"

QUERIES = [
    "Iran war",
    "Iran military conflict",
    "Iran strikes",
    "Iran nuclear",
    "Iran Israel",
    "Iran geopolitics Middle East",
]


def scrape(max_per_query: int = 10) -> list[str]:
    if not API_KEY:
        print("  [newsapi] SKIPPED — no API key set. Get one free at newsapi.org/register")
        return []

    saved = []
    seen_urls = set()

    for query in QUERIES:
        params = {
            "q": query,
            "apiKey": API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_per_query,
        }

        try:
            resp = requests.get(API_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [newsapi] query '{query}' failed: {e}")
            continue

        articles = data.get("articles", [])
        print(f"  [newsapi] query '{query}' returned {len(articles)} articles")

        for art in articles:
            url = art.get("url", "")
            title = art.get("title", "")
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)

            author = art.get("author", "")
            date = art.get("publishedAt", "")
            source_name = art.get("source", {}).get("name", "")

            # NewsAPI free tier truncates content — fetch full text
            text = fetch_full_text(url)
            if len(text) < 200:
                # Use the description as fallback
                text = art.get("description", "") or art.get("content", "")

            if len(text) < 100:
                continue

            path = save_article(
                "newsapi", title, text,
                url=url, author=f"{author} ({source_name})", date=date
            )
            saved.append(path)
            print(f"  [newsapi] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[newsapi] Starting NewsAPI scraper...")
    results = scrape()
    print(f"[newsapi] Done — saved {len(results)} articles")
