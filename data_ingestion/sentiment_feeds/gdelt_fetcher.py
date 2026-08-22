"""GDELT news-event fetcher (free API, no key).

Pulls the most recent articles mentioning a symbol and lexicons them into
[-1, +1] sentiment events persisted to storage. Includes politeness pacing
(rate-limit guard + inter-symbol delay) so rapid sweeps don't trigger GDELT's
429 quota errors.
"""
from __future__ import annotations

import time

import requests

from core.db import Storage, get_storage
from core.events import EVENT_SENTIMENT, bus
from core.logging import get_logger
from data_ingestion.sentiment_feeds._lexicon import score_text

log = get_logger(__name__)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
SOURCE_WEIGHT = 0.8

# GDELT free tier: ~1 req/sec sustained. Space requests to stay under quota.
_MIN_REQUEST_INTERVAL = 1.2
_last_request_at: float = 0.0


def _pace() -> None:
    """Throttle to at most one GDELT request per interval (process-wide)."""
    global _last_request_at
    now = time.monotonic()
    wait = _MIN_REQUEST_INTERVAL - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def fetch_gdelt_events(symbol: str, maxrecords: int = 25,
                       timespan: str = "24h",
                       storage: Storage | None = None) -> list[dict]:
    """Fetch recent GDELT articles for a symbol and persist scored events."""
    storage = storage or get_storage()
    _pace()
    resp = requests.get(
        GDELT_DOC,
        params={"query": f'"{symbol}"', "mode": "artlist", "maxrecords": maxrecords,
                "timespan": timespan, "format": "json"},
        timeout=30)
    if resp.status_code == 429:
        log.warning("GDELT rate-limited (429) for %s — backing off", symbol)
        time.sleep(5.0)
        return []
    if resp.status_code != 200:
        log.warning("GDELT returned %s for %s", resp.status_code, symbol)
        return []
    articles = resp.json().get("articles", [])
    events = []
    for art in articles:
        title = art.get("title", "")
        score = score_text(title)
        url = art.get("url", "")
        # GDELT artlist provides tone + themes + domain + source timestamp.
        themes = art.get("themes") or []
        tone = _parse_tone(art.get("tone"))
        storage.write_sentiment_event(
            symbol=symbol.upper(), source="GDELT", score=score,
            headline=title, url=url, source_weight=SOURCE_WEIGHT,
            full_text=art.get("description", ""),
            created_at=art.get("seendate", ""),
            tone=tone,
            themes=",".join(themes) if themes else None,
            raw={"domain": art.get("domain"), "language": art.get("language"),
                 "seendate": art.get("seendate"), "themes": themes})
        events.append({"symbol": symbol.upper(), "source": "GDELT",
                       "score": score, "headline": title, "url": url,
                       "themes": themes, "tone": tone})
    log.info("GDELT: %d events for %s", len(events), symbol)
    import asyncio
    for e in events:
        asyncio.run(bus.publish(EVENT_SENTIMENT, e))
    return events


def _parse_tone(tone) -> float | None:
    """GDELT 'tone' field is 'positive,negative,neutral,avg' — take avg."""
    if not tone:
        return None
    try:
        return float(str(tone).split(",")[3])
    except (IndexError, TypeError, ValueError):
        return None


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        fetch_gdelt_events(t)
