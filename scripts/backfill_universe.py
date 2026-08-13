#!/usr/bin/env python3
"""Full-universe backfill — every instrument in screener_config.yaml.

  Crypto (6)  → Binance REST (5y daily)
  Equities/ETF/Bond/FX (27) → yfinance (5y daily)
  Plus macro series: ^VIX, DX-Y.NYB, GC=F (stored as market data)

Usage: .venv/Scripts/python scripts/backfill_universe.py [--period 5y]
"""
from __future__ import annotations

import argparse
import time

from core.db import get_storage
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
YFINANCE = [
    # US equities
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    # Indian equities
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS",
    # ETFs
    "SPY", "QQQ", "IWM", "TLT", "GLD", "EEM",
    # Bonds / yields
    "^TNX", "^TYX",
    # Forex
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X",
    # Macro reference series
    "^VIX", "DX-Y.NYB", "GC=F",
]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--skip-crypto", action="store_true")
    args = ap.parse_args()

    db = get_storage()
    t0 = time.time()

    if not args.skip_crypto:
        from data_ingestion.price_feeds.crypto_ws import backfill_klines
        for sym in CRYPTO:
            try:
                df = backfill_klines(sym, interval="1d", days=1825, storage=db)
                log.info("crypto %s: %d bars", sym, len(df))
            except Exception as e:  # noqa: BLE001
                log.warning("crypto %s failed: %s", sym, e)

    from data_ingestion.price_feeds.equity_ws import backfill_equities
    results = backfill_equities(YFINANCE, period=args.period, storage=db)
    log.info("yfinance: %d/%d symbols backfilled", len(results), len(YFINANCE))

    log.info("Universe backfill complete in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
