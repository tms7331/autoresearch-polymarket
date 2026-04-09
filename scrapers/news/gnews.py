"""
Google News RSS scraper — no API key needed.
Google News provides RSS feeds for search queries.
Follows Google News redirect URLs to get actual article URLs, then extracts text.
"""

import requests
import feedparser
from utils import save_article, fetch_full_text

GNEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

QUERIES = [
    "Iran war",
    "Iran conflict military",
    "Iran strikes",
    "Iran nuclear",
    "Iran Israel war",
    "Iran Middle East geopolitics",
    "Iran US tensions",
]


def decode_google_news_url(google_url: str) -> str:
    """Decode the base64-encoded real URL from a Google News RSS link."""
    import base64
    # Google News RSS URLs encode the real URL in base64 in the path
    # Format: https://news.google.com/rss/articles/CBMi<base64>?oc=5
    try:
        # Extract the article ID from the URL
        path = google_url.split("/articles/")[1].split("?")[0]
        # The ID is base64url encoded; try decoding with padding
        # Strip the leading "CBMi" prefix marker (4 chars)
        encoded = path
        # Try to decode — Google uses a protobuf wrapper
        padded = encoded + "=" * (4 - len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        # The real URL is embedded in the decoded bytes after some protobuf framing
        # Look for http:// or https:// in the decoded bytes
        decoded_str = decoded.decode("utf-8", errors="ignore")
        for prefix in ["https://", "http://"]:
            idx = decoded_str.find(prefix)
            if idx >= 0:
                url = decoded_str[idx:]
                # Clean up any trailing garbage
                for end_char in ["\x00", "\x12", "\x1a", "\x22"]:
                    if end_char in url:
                        url = url[:url.index(end_char)]
                return url.strip()
    except Exception:
        pass
    # Fallback: try following redirects
    try:
        resp = requests.get(google_url, allow_redirects=True, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
        if resp.url != google_url:
            return resp.url
    except Exception:
        pass
    return ""


def scrape() -> list[str]:
    saved = []
    seen_titles = set()

    for query in QUERIES:
        url = GNEWS_RSS.format(query=query.replace(" ", "+"))
        try:
            feed = feedparser.parse(url)
            entries = feed.entries
        except Exception as e:
            print(f"  [gnews] query '{query}' failed: {e}")
            continue

        print(f"  [gnews] query '{query}' returned {len(entries)} entries")

        for entry in entries[:10]:  # Cap per query
            title = entry.get("title", "")
            link = entry.get("link", "")
            date = entry.get("published", "")
            source = entry.get("source", {}).get("title", "")

            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Decode the Google News URL to get the real article URL
            real_url = decode_google_news_url(link) if link else ""
            text = fetch_full_text(real_url) if real_url else ""

            if len(text) < 200:
                # Some sites block scraping — skip rather than save junk
                print(f"  [gnews] skipping (can't extract text): {title[:50]}")
                continue

            path = save_article(
                "gnews", title, text,
                url=real_url, author=source, date=date
            )
            saved.append(path)
            print(f"  [gnews] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[gnews] Starting Google News RSS scraper...")
    results = scrape()
    print(f"[gnews] Done — saved {len(results)} articles")
