#!/usr/bin/env python3
"""Feature-store rebuild for the full (non-crypto) universe.

Builds labeled SWING feature vectors for every symbol with market data and
persists them (idempotent — the store replaces per symbol+timeframe).

Usage: .venv/Scripts/python scripts/build_features.py [--symbols AAPL MSFT ...]
"""
from __future__ import annotations

import argparse

from core.db import get_storage
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

SKIP_PREFIXES = ("BTC-", "ETH-", "SOL-", "BNB-", "XRP-", "DOGE-")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    from feature_engineering.feature_store import build_feature_frame, write_to_store

    db = get_storage()
    symbols = args.symbols or db.symbols()
    symbols = [s for s in symbols if not s.startswith(SKIP_PREFIXES)]

    total = 0
    for sym in symbols:
        ohlcv = db.query_ohlcv(sym)
        if len(ohlcv) < 250:
            log.info("skip %s: only %d bars", sym, len(ohlcv))
            continue
        frame = build_feature_frame(sym, "EQUITY_US", ohlcv, db=db,
                                    timeframe="SWING", with_labels=True)
        n = write_to_store(frame, sym, "EQUITY_US", "SWING", db)
        total += n
    log.info("Feature build complete: %d vectors for %d symbols", total, len(symbols))


if __name__ == "__main__":
    main()
