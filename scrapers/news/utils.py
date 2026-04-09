"""Shared utilities for all scrapers."""

import os
import re
import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def sanitize_filename(title: str, max_len: int = 80) -> str:
    """Turn an article title into a safe filename slug."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[\s_-]+", "_", title)
    return title[:max_len].rstrip("_")


def save_article(scraper_name: str, title: str, text: str,
                 url: str = "", author: str = "", date: str = "") -> str:
    """Save an article to data/ and return the filepath."""
    slug = sanitize_filename(title)
    if not slug:
        slug = "untitled"
    filename = f"{scraper_name}_{slug}.txt"
    filepath = os.path.join(DATA_DIR, filename)

    header = f"{title}\n"
    header += f"Source: {scraper_name}\n"
    if author:
        header += f"Author: {author}\n"
    if date:
        header += f"Date: {date}\n"
    if url:
        header += f"URL: {url}\n"
    header += "=" * 72 + "\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + text.strip() + "\n")

    return filepath


def fetch_full_text(url: str, timeout: int = 15) -> str:
    """Best-effort extraction of article body text from a URL."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove script/style/nav elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Try common article containers
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
    return text
