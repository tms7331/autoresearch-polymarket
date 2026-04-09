"""
Multi-source RSS feed scraper.
Pulls from Al Jazeera, BBC, AP, NPR, and other international news feeds.
Filters for geopolitics / Iran-related content.
"""

import re
import feedparser
from utils import save_article, fetch_full_text

# RSS feeds with good international / Middle East coverage
FEEDS = {
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_middleeast": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "ap_topnews": "https://rsshub.app/apnews/topics/world-news",
    "npr_world": "https://feeds.npr.org/1004/rss.xml",
    "reuters_world": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "france24": "https://www.france24.com/en/middle-east/rss",
    "dw_world": "https://rss.dw.com/rdf/rss-en-world",
    "middleeasteye": "https://www.middleeasteye.net/rss",
    "arabnews": "https://www.arabnews.com/rss.xml",
}

# Keywords to filter for geopolitics / Iran
KEYWORDS = re.compile(
    r"iran|tehran|persian gulf|strait of hormuz|irgc|khamenei|"
    r"middle east.*war|israel.*iran|iran.*strike|iran.*nuclear|"
    r"iran.*sanction|hezbollah|houthi|proxy war|geopolit|"
    r"iran.*military|iran.*conflict|iran.*us |iran.*america",
    re.IGNORECASE,
)


def is_relevant(title: str, summary: str = "") -> bool:
    text = f"{title} {summary}"
    return bool(KEYWORDS.search(text))


def scrape() -> list[str]:
    saved = []
    seen_titles = set()

    for feed_name, feed_url in FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries
        except Exception as e:
            print(f"  [rss/{feed_name}] failed to parse: {e}")
            continue

        print(f"  [rss/{feed_name}] found {len(entries)} entries")
        relevant = 0

        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")
            date = entry.get("published", "")
            author = entry.get("author", feed_name)

            if not is_relevant(title, summary):
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)
            relevant += 1

            # Try to get full text
            text = fetch_full_text(link) if link else ""
            if len(text) < 200:
                # Fall back to summary
                text = summary if summary else title

            path = save_article(
                f"rss_{feed_name}", title, text,
                url=link, author=author, date=date
            )
            saved.append(path)
            print(f"  [rss/{feed_name}] saved: {title[:60]}")

        if relevant == 0:
            print(f"  [rss/{feed_name}] no Iran/geopolitics matches")

    return saved


if __name__ == "__main__":
    print("[rss] Starting RSS feed scraper...")
    results = scrape()
    print(f"[rss] Done — saved {len(results)} articles")
