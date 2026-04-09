"""
Mediastack API scraper — free tier, 500 requests/month.
https://mediastack.com/

NOTE: Requires a free API key from mediastack.com
"""

import requests
from utils import save_article, fetch_full_text

API_KEY = ""  # Get free key at mediastack.com
API_URL = "http://api.mediastack.com/v1/news"

KEYWORDS = "iran,iran war,iran conflict,middle east war,iran nuclear"


def scrape(limit: int = 50) -> list[str]:
    if not API_KEY:
        print("  [mediastack] SKIPPED — no API key. Get one free at mediastack.com")
        return []

    saved = []

    params = {
        "access_key": API_KEY,
        "keywords": KEYWORDS,
        "languages": "en",
        "sort": "published_desc",
        "limit": limit,
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [mediastack] API call failed: {e}")
        return []

    articles = data.get("data", [])
    print(f"  [mediastack] returned {len(articles)} articles")

    for art in articles:
        url = art.get("url", "")
        title = art.get("title", "")
        if not url or not title:
            continue

        author = art.get("author", "")
        date = art.get("published_at", "")
        description = art.get("description", "")

        text = fetch_full_text(url)
        if len(text) < 200:
            text = description or ""

        if len(text) < 100:
            continue

        path = save_article(
            "mediastack", title, text,
            url=url, author=author, date=date
        )
        saved.append(path)
        print(f"  [mediastack] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[mediastack] Starting Mediastack scraper...")
    results = scrape()
    print(f"[mediastack] Done — saved {len(results)} articles")
