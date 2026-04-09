"""
Reuters scraper using Browserbase for headless browser access.

Browserbase CDP quirk: only the first page load on the default page
(pages[0]) gets full DOM access via evaluate(). Subsequent new_page()
tabs get empty DOMs. We work around this by:
1. Using one session + default page for section browsing
2. Using a second session + default page for article extraction,
   navigating between articles via page.goto() and using innerText
   (which works even when getElementsByTagName doesn't)
"""

import os
import time
import requests as http_requests
from dotenv import load_dotenv
from utils import save_article

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")

SECTION_URLS = [
    "https://www.reuters.com/world/iran/",
]


def create_session() -> str:
    for attempt in range(3):
        resp = http_requests.post(
            "https://api.browserbase.com/v1/sessions",
            headers={
                "x-bb-api-key": BROWSERBASE_API_KEY,
                "Content-Type": "application/json",
            },
            json={"projectId": BROWSERBASE_PROJECT_ID},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  [reuters] Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 402:
            raise RuntimeError("Browserbase free plan minutes exhausted. Upgrade at browserbase.com/plans")
        resp.raise_for_status()
        return resp.json()["id"]
    resp.raise_for_status()  # Will raise on the last attempt's error


def connect(pw):
    """Create a Browserbase session and return (browser, page)."""
    session_id = create_session()
    connect_url = (
        f"wss://connect.browserbase.com"
        f"?apiKey={BROWSERBASE_API_KEY}&sessionId={session_id}"
    )
    browser = pw.chromium.connect_over_cdp(connect_url)
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    page = context.pages[0] if context.pages else context.new_page()
    return browser, page


def scrape() -> list[str]:
    if not BROWSERBASE_API_KEY or not BROWSERBASE_PROJECT_ID:
        print("  [reuters] ERROR: Missing BROWSERBASE credentials in .env")
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [reuters] ERROR: playwright not installed")
        return []

    saved = []
    seen_urls = set()
    article_urls = []

    with sync_playwright() as pw:
        # --- Session 1: Discover article links ---
        print("  [reuters] Session 1: discovering articles...")
        browser1, page1 = connect(pw)

        for section_url in SECTION_URLS:
            try:
                print(f"  [reuters] Loading: {section_url}")
                page1.goto(section_url, wait_until="domcontentloaded", timeout=30000)
                page1.wait_for_timeout(10000)

                for _ in range(6):
                    page1.evaluate("window.scrollBy(0, 1500)")
                    page1.wait_for_timeout(1200)

                # Extract links via innerText parsing as fallback
                links = page1.evaluate("""() => {
                    const base = window.location.origin;
                    const anchors = document.getElementsByTagName('a');
                    const results = [];
                    const seen = new Set();
                    for (let i = 0; i < anchors.length; i++) {
                        const a = anchors[i];
                        const raw = a.getAttribute('href') || '';
                        if (!raw) continue;
                        const href = raw.startsWith('/') ? base + raw : raw;
                        const text = (a.textContent || '').trim().substring(0, 150);
                        if (text.length > 20
                            && href.includes('reuters.com')
                            && /202\\d/.test(href)
                            && !/(video|pictures|graphics)/.test(href)
                            && !seen.has(href)) {
                            seen.add(href);
                            results.push({ href, text });
                        }
                    }
                    return results;
                }""")

                print(f"  [reuters]   -> {len(links)} article links")
                for link in links:
                    href = link["href"]
                    if href not in seen_urls:
                        seen_urls.add(href)
                        article_urls.append((href, link["text"]))

            except Exception as e:
                print(f"  [reuters] Error: {e}")

        browser1.close()
        print(f"  [reuters] Found {len(article_urls)} unique articles")

        if not article_urls:
            return []

        # --- Extract articles: one fresh session per article ---
        # Browserbase CDP only fully renders the first page load per
        # session. Subsequent navigations get empty DOMs. So we pay
        # the cost of one session per article.
        cap = min(len(article_urls), 20)

        for i, (url, link_text) in enumerate(article_urls[:cap]):
            browser_a = None
            try:
                print(f"  [reuters] [{i+1}/{cap}] {link_text[:50]}...")
                browser_a, page_a = connect(pw)

                page_a.goto(url, wait_until="domcontentloaded", timeout=30000)
                page_a.wait_for_timeout(8000)

                try:
                    page_a.wait_for_selector("h1", timeout=10000)
                except Exception:
                    pass

                title = ""
                try:
                    title = page_a.inner_text("h1", timeout=5000)
                except Exception:
                    title = link_text

                date = ""
                try:
                    date = page_a.get_attribute("time", "datetime", timeout=3000) or ""
                except Exception:
                    pass

                text = ""
                for selector in [
                    "article",
                    '[class*="article-body"]',
                    '[class*="ArticleBody"]',
                    "main",
                ]:
                    try:
                        raw = page_a.inner_text(selector, timeout=5000)
                        lines = [
                            l.strip() for l in raw.split("\n")
                            if len(l.strip()) > 60
                        ]
                        if len(lines) > 3:
                            text = "\n\n".join(lines)
                            break
                    except Exception:
                        continue

                if len(text) < 150:
                    print(f"  [reuters]   skip (too short)")
                    continue

                path = save_article(
                    "reuters", title, text,
                    url=url, author="Reuters", date=date
                )
                saved.append(path)
                print(f"  [reuters]   saved!")

            except RuntimeError as e:
                # Quota exhausted — stop trying
                print(f"  [reuters]   {e}")
                break
            except Exception as e:
                print(f"  [reuters]   error: {e}")
            finally:
                if browser_a:
                    browser_a.close()

    return saved


if __name__ == "__main__":
    print("[reuters] Starting Reuters/Browserbase scraper...")
    results = scrape()
    print(f"[reuters] Done — saved {len(results)} articles")
