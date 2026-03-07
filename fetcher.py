"""
Fetches geopolitical news from GDELT RSS feeds and NewsAPI.
No API key required for GDELT.
"""

import feedparser
import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# RSS feed registry — free, no key needed
# Each entry: url, source label, region bucket
RSS_FEEDS = [
    # ── BBC (regional split gives good bucketing) ──────────────────────────
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml",               "source": "BBC",          "region": "Conflict & War"},
    {"url": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",   "source": "BBC",          "region": "Middle East"},
    {"url": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",          "source": "BBC",          "region": "Asia Pacific"},
    {"url": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",        "source": "BBC",          "region": "Europe"},
    {"url": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml", "source": "BBC",          "region": "Americas"},
    {"url": "https://feeds.bbci.co.uk/news/world/africa/rss.xml",        "source": "BBC",          "region": "Africa"},
    # ── Reuters ────────────────────────────────────────────────────────────
    {"url": "https://feeds.reuters.com/reuters/worldNews",               "source": "Reuters",      "region": "Conflict & War"},
    # ── Al Jazeera ─────────────────────────────────────────────────────────
    {"url": "https://www.aljazeera.com/xml/rss/all.xml",                 "source": "Al Jazeera",   "region": "Middle East"},
    # ── The Guardian ───────────────────────────────────────────────────────
    {"url": "https://www.theguardian.com/world/rss",                     "source": "The Guardian", "region": "Conflict & War"},
    # ── France 24 ──────────────────────────────────────────────────────────
    {"url": "https://www.france24.com/en/rss",                           "source": "France24",     "region": "Europe"},
]

# Country → (lat, lon) for map plotting
COUNTRY_COORDS = {
    "ukraine": (48.3794, 31.1656), "russia": (61.5240, 105.3188),
    "china": (35.8617, 104.1954), "taiwan": (23.6978, 120.9605),
    "israel": (31.0461, 34.8516), "gaza": (31.3547, 34.3088),
    "iran": (32.4279, 53.6880), "usa": (37.0902, -95.7129),
    "united states": (37.0902, -95.7129), "north korea": (40.3399, 127.5101),
    "south korea": (35.9078, 127.7669), "india": (20.5937, 78.9629),
    "pakistan": (30.3753, 69.3451), "syria": (34.8021, 38.9968),
    "yemen": (15.5527, 48.5164), "sudan": (12.8628, 30.2176),
    "ethiopia": (9.1450, 40.4897), "myanmar": (21.9162, 95.9560),
    "afghanistan": (33.9391, 67.7100), "iraq": (33.2232, 43.6793),
    "turkey": (38.9637, 35.2433), "venezuela": (6.4238, -66.5897),
    "haiti": (18.9712, -72.2852), "france": (46.2276, 2.2137),
    "germany": (51.1657, 10.4515), "uk": (55.3781, -3.4360),
    "brazil": (-14.2350, -51.9253), "mexico": (23.6345, -102.5528),
    "japan": (36.2048, 138.2529), "saudi arabia": (23.8859, 45.0792),
}


def fetch_rss_news(max_per_feed: int = 6) -> list[dict]:
    """Fetch articles from all RSS feeds."""
    articles = []
    for feed_cfg in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_cfg["url"])
            for entry in feed.entries[:max_per_feed]:
                articles.append({
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", entry.get("title", "")).strip(),
                    "url": entry.get("link", ""),
                    "source": feed_cfg["source"],
                    "region": feed_cfg["region"],
                    "published": entry.get("published", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as e:
            print(f"Error fetching {feed_cfg['source']}: {e}")
    return articles


def fetch_newsapi(query: str = "geopolitical conflict war sanctions", max_results: int = 20) -> list[dict]:
    """Fetch from NewsAPI if key is set."""
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key or api_key == "your-newsapi-key-here":
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title": a.get("title", ""),
                "summary": a.get("description", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "region": "Global",
                "published": a.get("publishedAt", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def infer_coordinates(text: str) -> tuple[float, float] | None:
    """Try to extract a country from text and return coordinates."""
    text_lower = text.lower()
    for country, coords in COUNTRY_COORDS.items():
        if country in text_lower:
            return coords
    return None


def get_all_news() -> list[dict]:
    """Fetch from all sources and deduplicate."""
    articles = fetch_rss_news() + fetch_newsapi()
    # Attach coordinates
    for a in articles:
        coords = infer_coordinates(a["title"] + " " + a.get("summary", ""))
        a["lat"] = coords[0] if coords else None
        a["lon"] = coords[1] if coords else None
    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        key = a["title"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique
