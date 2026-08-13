#!/usr/bin/env python3
"""Backfill the full screener universe: 5y prices for every instrument, then
build the labeled feature store. Idempotent — safe to re-run.

Usage: .venv/Scripts/python scripts/backfill_universe.py
"""
from __future__ import annotations

from core.db import get_storage
from core.logging import get_logger, setup_logging
from screener.asset_universe import get_universe

log = get_logger(__name__)

CRYPTO_PREFIXES = ("BTC-", "ETH-", "SOL-", "BNB-", "XRP-", "DOGE-", "ADA-", "AVAX-", "LINK-", "LTC-")


def main() -> None:
    setup_logging()
    universe = get_universe()
    db = get_storage()
    symbols = universe.symbols()
    log.info("Backfilling %d symbols across %d classes", len(symbols), len(universe.weights))

    equities = [s for s in symbols if not s.startswith(CRYPTO_PREFIXES)]
    crypto = [s for s in symbols if s.startswith(CRYPTO_PREFIXES)]

    # 1) prices
    from data_ingestion.price_feeds.equity_ws import backfill_equities
    results = backfill_equities(equities, period="5y", interval="1d", storage=db)
    log.info("Backfilled %d equity/ETF/index/FX instruments", len(results))

    if crypto:
        from data_ingestion.price_feeds.crypto_ws import backfill_klines
        for c in crypto:
            try:
                df = backfill_klines(c, interval="1d", days=1825, storage=db)
                log.info("Backfilled %d bars for %s", 0 if df is None else len(df), c)
            except Exception as e:  # noqa: BLE001
                log.warning("crypto backfill failed %s: %s", c, e)

    # 2) features for non-crypto symbols with enough history
    from feature_engineering.feature_store import build_feature_frame, write_to_store
    built = 0
    for sym in equities:
        try:
            ohlcv = db.query_ohlcv(sym)
            if len(ohlcv) < 250:
                log.info("skip %s: %d bars (<250)", sym, len(ohlcv))
                continue
            asset_class = universe.asset_class_of(sym) or "EQUITY_US"
            frame = build_feature_frame(sym, asset_class, ohlcv, db=db,
                                        timeframe="SWING", with_labels=True)
            if frame is not None and not frame.empty:
                write_to_store(frame, sym, asset_class, "SWING", db=db)
                built += 1
                log.info("features built for %s (%d rows)", sym, len(frame))
        except Exception as e:  # noqa: BLE001
            log.warning("feature build failed %s: %s", sym, e)

    log.info("DONE. prices=%d symbols, features=%d symbols", len(results), built)


if __name__ == "__main__":
    main()
