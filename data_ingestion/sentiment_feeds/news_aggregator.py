"""News sentiment aggregator — multi-source (plan §1.2 sentiment_feeds).

Sources (verified reachable, no API keys):
  1. Yahoo Finance news search   — query1.finance.yahoo.com/v1/finance/search
  2. Google News RSS             — news.google.com/rss/search
  3. Yahoo RSS headline feed     — feeds.finance.yahoo.com/rss/2.0/headline
  4. StockTwits                  — native Bullish/Bearish counts (public API)

GDELT (gdelt_fetcher.py) and Reddit (reddit_fetcher.py) remain as best-effort
sources; on networks where they are blocked they simply yield no events.
Every event is persisted to `sentiment_events` with a lexicon score in [-1, +1]
(StockTwits uses its native bullish/bearish ratio) and consumed by
feature_engineering/sentiment_features.py.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from core.db import Storage, get_storage
from core.events import EVENT_SENTIMENT, bus
from core.logging import get_logger
from data_ingestion.sentiment_feeds._lexicon import score_text

log = get_logger(__name__)

YAHOO_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
GOOGLE_RSS = "https://news.google.com/rss/search"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"
STOCKTWITS = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) agonistes/0.1"}

SOURCE_WEIGHTS = {"NEWS": 0.9, "GOOGLE": 0.8, "YAHOO_RSS": 0.8, "STOCKTWITS": 0.7}


def _persist(storage: Storage, symbol: str, source: str, score: float,
             headline: str, url: str = "", full_text: str = "",
             created_at: str = "") -> dict:
    storage.write_sentiment_event(
        symbol=symbol.upper(), source=source, score=score,
        headline=headline, url=url, source_weight=SOURCE_WEIGHTS.get(source, 0.8),
        full_text=full_text, created_at=created_at)
    return {"symbol": symbol.upper(), "source": source, "score": score,
            "headline": headline, "url": url, "full_text": full_text,
            "created_at": created_at}


def fetch_yahoo_news(symbol: str, storage: Storage, news_count: int = 10) -> list[dict]:
    resp = requests.get(YAHOO_SEARCH, params={"q": symbol, "newsCount": news_count},
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    events = []
    for item in resp.json().get("news", []):
        title = item.get("title", "")
        score = score_text(title)
        events.append(_persist(storage, symbol, "NEWS", score, title,
                               item.get("link", ""),
                               full_text=item.get("description", ""),
                               created_at=_fmt_ts(item.get("providerPublishTime"))))
    return events


def _fmt_ts(epoch) -> str:
    """Epoch seconds/millis → ISO 8601 UTC string, else ''."""
    if epoch in (None, ""):
        return ""
    try:
        e = float(epoch)
    except (TypeError, ValueError):
        return ""
    if e > 1e11:  # milliseconds
        e = e / 1000
    return pd.Timestamp(e, unit="s", tz="UTC").isoformat(sep=" ")


def fetch_google_news(symbol: str, storage: Storage, days: int = 1) -> list[dict]:
    resp = requests.get(
        GOOGLE_RSS,
        params={"q": f'"{symbol}" when:{days}d', "hl": "en-US", "gl": "US",
                "ceid": "US:en"},
        headers=HEADERS, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"n": "http://www.w3.org/2005/Atom"}
    events = []
    for item in root.findall("n:entry", ns):
        title = item.findtext("n:title", default="", namespaces=ns)
        link = item.find("n:link", ns)
        summary = item.findtext("n:summary", default="", namespaces=ns)
        published = item.findtext("n:published", default="", namespaces=ns)
        events.append(_persist(storage, symbol, "GOOGLE", score_text(title),
                               title, link.get("href") if link is not None else "",
                               full_text=summary, created_at=published))
    return events


def fetch_yahoo_rss(symbol: str, storage: Storage) -> list[dict]:
    resp = requests.get(YAHOO_RSS, params={"s": symbol, "region": "US",
                                           "lang": "en-US"},
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    events = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        desc = item.findtext("description", default="")
        pub = item.findtext("pubDate", default="")
        events.append(_persist(storage, symbol, "YAHOO_RSS", score_text(title),
                               title, link, full_text=desc, created_at=pub))
    return events


def fetch_stocktwits(symbol: str, storage: Storage, limit: int = 20) -> list[dict]:
    """StockTwits native sentiment — Bullish/Bearish ratios per message."""
    resp = requests.get(STOCKTWITS.format(symbol=symbol.lower()),
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    events = []
    for msg in resp.json().get("messages", [])[:limit]:
        body = (msg.get("body") or "")[:300]
        sent = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
        if sent == "Bullish":
            score = 0.6
        elif sent == "Bearish":
            score = -0.6
        else:
            score = score_text(body)
        events.append(_persist(storage, symbol, "STOCKTWITS", score, body,
                               f"https://stocktwits.com/symbol/{symbol.upper()}",
                               full_text=body,
                               created_at=_fmt_ts(msg.get("created_at"))))
    return events


FETCHERS = {
    "yahoo": fetch_yahoo_news,
    "google": fetch_google_news,
    "yahoo_rss": fetch_yahoo_rss,
    "stocktwits": fetch_stocktwits,
}


def fetch_news_events(symbol: str, days: int = 1, storage: Storage | None = None,
                      sources: list[str] | None = None) -> list[dict]:
    """Fetch news sentiment for a symbol from all reachable sources."""
    storage = storage or get_storage()
    sources = sources or list(FETCHERS)
    events: list[dict] = []
    for name in sources:
        try:
            events.extend(FETCHERS[name](symbol, storage, days=days) if name == "google"
                          else FETCHERS[name](symbol, storage))
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001
            log.debug("news source %s failed for %s: %s", name, symbol, e)
    import asyncio
    for e in events:
        asyncio.run(bus.publish(EVENT_SENTIMENT, e))
    log.info("News sentiment: %d events for %s", len(events), symbol)
    return events


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        events = fetch_news_events(t)
        print(f"{t}: {len(events)} events")
        for e in events[:5]:
            print(f"  [{e['source']:10s}] {e['score']:+.2f}  {e['headline'][:70]}")
