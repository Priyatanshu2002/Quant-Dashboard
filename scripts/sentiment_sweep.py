#!/usr/bin/env python3
"""Sentiment sweep — news sentiment for the full universe (plan Phase 1/2).

Sources: Yahoo Finance news, Google News RSS, Yahoo RSS, StockTwits.
(GDELT/Reddit are best-effort on networks where they are reachable.)

Usage: .venv/Scripts/python scripts/sentiment_sweep.py [--limit N]
"""
from __future__ import annotations

import argparse
import time

from core.db import get_storage
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

UNIVERSE = [
    # US equities
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    # Indian equities
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS",
    # ETFs
    "SPY", "QQQ", "IWM", "TLT", "GLD", "EEM",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD",
]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(UNIVERSE))
    args = ap.parse_args()

    from data_ingestion.sentiment_feeds.news_aggregator import fetch_news_events

    db = get_storage()
    t0 = time.time()
    total = 0
    for sym in UNIVERSE[: args.limit]:
        try:
            total += len(fetch_news_events(sym, storage=db))
        except Exception as e:  # noqa: BLE001
            log.warning("sentiment sweep %s failed: %s", sym, e)
    log.info("Sentiment sweep complete in %.0fs: %d events for %d symbols",
             time.time() - t0, total, min(args.limit, len(UNIVERSE)))


if __name__ == "__main__":
    main()
