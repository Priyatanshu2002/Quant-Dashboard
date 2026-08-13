"""GDELT news-event fetcher (free API, no key).

Pulls the most recent articles mentioning a symbol and lexicons them into
[-1, +1] sentiment events persisted to storage.
"""
from __future__ import annotations

import requests

from core.db import Storage, get_storage
from core.events import EVENT_SENTIMENT, bus
from core.logging import get_logger
from data_ingestion.sentiment_feeds._lexicon import score_text

log = get_logger(__name__)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
SOURCE_WEIGHT = 0.8


def fetch_gdelt_events(symbol: str, maxrecords: int = 25,
                       timespan: str = "24h",
                       storage: Storage | None = None) -> list[dict]:
    """Fetch recent GDELT articles for a symbol and persist scored events."""
    storage = storage or get_storage()
    resp = requests.get(
        GDELT_DOC,
        params={"query": f'"{symbol}"', "mode": "artlist", "maxrecords": maxrecords,
                "timespan": timespan, "format": "json"},
        timeout=30)
    if resp.status_code != 200:
        log.warning("GDELT returned %s for %s", resp.status_code, symbol)
        return []
    articles = resp.json().get("articles", [])
    events = []
    for art in articles:
        title = art.get("title", "")
        score = score_text(title)
        url = art.get("url", "")
        storage.write_sentiment_event(
            symbol=symbol.upper(), source="GDELT", score=score,
            headline=title, url=url, source_weight=SOURCE_WEIGHT)
        events.append({"symbol": symbol.upper(), "source": "GDELT",
                       "score": score, "headline": title, "url": url})
    log.info("GDELT: %d events for %s", len(events), symbol)
    import asyncio
    for e in events:
        asyncio.run(bus.publish(EVENT_SENTIMENT, e))
    return events


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        fetch_gdelt_events(t)
