"""
Wikipedia Current Events scraper.
Wikipedia maintains a daily current events portal with structured summaries
of world events. Good for getting a factual overview of ongoing conflicts.
"""

import re
import requests
from bs4 import BeautifulSoup
from utils import save_article

CURRENT_EVENTS_URL = "https://en.wikipedia.org/wiki/Portal:Current_events"

KEYWORDS = re.compile(
    r"iran|tehran|irgc|khamenei|persian gulf|hormuz|hezbollah|houthi|"
    r"middle east.*conflict|israel.*iran|iran.*strike|iran.*nuclear",
    re.IGNORECASE,
)


def scrape() -> list[str]:
    saved = []

    try:
        resp = requests.get(CURRENT_EVENTS_URL, timeout=20, headers={
            "User-Agent": "NewsScraperBot/1.0 (research project)"
        })
        resp.raise_for_status()
    except Exception as e:
        print(f"  [wikipedia] Failed to fetch current events: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # Find all date sections with their event lists
    for desc in soup.find_all("div", class_="description"):
        items = desc.find_all("li")
        for item in items:
            text = item.get_text(strip=True)
            if not KEYWORDS.search(text):
                continue

            # Use first 80 chars as title
            title = text[:80].strip()
            if len(text) < 50:
                continue

            # Extract any linked article URLs for more context
            links = item.find_all("a", href=True)
            url = ""
            for link in links:
                href = link.get("href", "")
                if href.startswith("/wiki/") and ":" not in href:
                    url = f"https://en.wikipedia.org{href}"
                    break

            path = save_article(
                "wikipedia", title, text,
                url=url or CURRENT_EVENTS_URL
            )
            saved.append(path)
            print(f"  [wikipedia] saved: {title[:60]}")

    return saved


if __name__ == "__main__":
    print("[wikipedia] Starting Wikipedia Current Events scraper...")
    results = scrape()
    print(f"[wikipedia] Done — saved {len(results)} articles")
