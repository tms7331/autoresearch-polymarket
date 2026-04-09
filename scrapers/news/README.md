# News Scrapers — Geopolitics / Iran War Focus

Collection of news scrapers pulling articles on international geopolitics, with a strong focus on the Iran conflict. Each scraper targets a different source — some use public APIs, others use RSS feeds, and some use browser-based scraping (via Browserbase) for paywalled or JS-heavy sites.

## Output Format

All articles are saved to `data/` as plain text files:

```
data/{scraper_name}_{sanitized_article_title}.txt
```

Each file contains a header block (source, author, date, url) followed by the full article text.

## Scrapers

| Scraper | Source | Method | Notes |
|---------|--------|--------|-------|
| `newsapi.py` | NewsAPI.org | REST API | Aggregates 80k+ sources, free tier 100 req/day |
| `gdelt.py` | GDELT Project | REST API | Massive global events database, free & unlimited |
| `rss_feeds.py` | Multiple (Al Jazeera, BBC, AP, etc.) | RSS/Atom | Reliable, no auth needed, summaries + links |
| `reuters.py` | Reuters | Browserbase | JS-heavy site, needs headless browser |
| `guardian.py` | The Guardian | REST API | Free API, full article text |
| `mediastack.py` | Mediastack | REST API | Free tier, 500 req/month |
| `newsdata.py` | NewsData.io | REST API | Free tier, 200 req/day |
| `ap_rss.py` | Associated Press | RSS | Wire service, high quality |
| `aljazeera_rss.py` | Al Jazeera | RSS | Strong Middle East coverage |
| `bbc_rss.py` | BBC World | RSS | Broad international coverage |

## Usage

```bash
cd scrapers
pip install -r requirements.txt
python run_all.py          # runs every scraper
python gdelt.py            # run individually
```

## Dependencies

See `requirements.txt`. Key packages: `requests`, `feedparser`, `beautifulsoup4`, `playwright`, `python-dotenv`.
