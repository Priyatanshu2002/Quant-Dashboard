"""Reddit sentiment fetcher — public JSON endpoints (no auth).

Uses the unauthenticated .json endpoints of the public API. Respect rate
limits (~60 req/min unauthenticated); the module sleeps briefly per request.
"""
from __future__ import annotations

import time

import requests

from core.db import Storage, get_storage
from core.events import EVENT_SENTIMENT, bus
from core.logging import get_logger
from data_ingestion.sentiment_feeds._lexicon import score_text

log = get_logger(__name__)

SUBREDDITS = ["stocks", "investing", "wallstreetbets", "CryptoCurrency"]
SOURCE_WEIGHT = 0.6
HEADERS = {"User-Agent": "agonistes-research/0.1 (paper-trading research)"}


def fetch_reddit_events(symbol: str, limit_per_sub: int = 10,
                        storage: Storage | None = None) -> list[dict]:
    storage = storage or get_storage()
    events = []
    for sub in SUBREDDITS:
        try:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": symbol, "sort": "new", "t": "day", "limit": limit_per_sub},
                headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                continue
            for child in resp.json().get("data", {}).get("children", []):
                post = child.get("data", {})
                text = (post.get("title", "") + " " + (post.get("selftext", "") or ""))[:1000]
                score = score_text(text)
                storage.write_sentiment_event(
                    symbol=symbol.upper(), source="REDDIT", score=score,
                    headline=post.get("title", ""), url=post.get("url", ""),
                    source_weight=SOURCE_WEIGHT)
                events.append({"symbol": symbol.upper(), "source": "REDDIT",
                               "score": score, "headline": post.get("title", "")})
            time.sleep(1.2)
        except Exception as e:  # noqa: BLE001
            log.debug("Reddit fetch failed for %s/%s: %s", sub, symbol, e)
    import asyncio
    for e in events:
        asyncio.run(bus.publish(EVENT_SENTIMENT, e))
    log.info("Reddit: %d events for %s", len(events), symbol)
    return events


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        fetch_reddit_events(t)
