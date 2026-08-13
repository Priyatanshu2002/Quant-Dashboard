#!/usr/bin/env python3
"""Fundamentals backfill — plan Phase 2 deliverables.

  1. SEC EDGAR XBRL annual snapshots (5y of 10-K data) for the US equity universe
     → persisted to fundamental_snapshots
  2. yfinance .info snapshots for equities + ETFs
  3. Earnings calendar (yfinance) → earnings_calendar table (calendar_features)

Usage: .venv/Scripts/python scripts/backfill_fundamentals.py
"""
from __future__ import annotations

import time

from core.db import get_storage
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

# ticker → CIK (SEC EDGAR)
CIK_MAP = {
    "AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
    "AMZN": "0001018724", "GOOGL": "0001652044", "META": "0001326801",
    "TSLA": "0001318605", "JPM": "0000019617", "XOM": "0000034088",
    "UNH": "0000731766",
}
US_EQUITIES = list(CIK_MAP)
ETFS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM"]
IN_EQUITIES = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]


def backfill_edgar(storage) -> int:
    """Annual fundamental snapshots from SEC XBRL company-facts for US equities."""
    from data_ingestion.fundamental_feeds.three_statement_parser import parse_company_facts

    n = 0
    for ticker, cik in CIK_MAP.items():
        try:
            snap = parse_company_facts(cik, ticker)
            if snap.get("revenue"):
                storage.upsert_fundamental_snapshot(snap)
                n += 1
                log.info("EDGAR %s: FY%s revenue=%s fcf=%s",
                         ticker, snap.get("fiscal_year"),
                         snap.get("revenue"), snap.get("free_cash_flow"))
            else:
                log.warning("EDGAR %s: no revenue extracted", ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("EDGAR %s failed: %s", ticker, e)
        time.sleep(0.5)  # be polite to SEC
    return n


def backfill_yfinance_info(storage, symbols) -> int:
    from data_ingestion.fundamental_feeds.yfinance_earnings import refresh_info_snapshot

    n = 0
    for sym in symbols:
        try:
            if refresh_info_snapshot(sym, storage=storage):
                n += 1
        except Exception as e:  # noqa: BLE001
            log.debug("info %s failed: %s", sym, e)
    return n


def backfill_earnings_calendar(storage, symbols) -> int:
    from data_ingestion.fundamental_feeds.yfinance_earnings import fetch_earnings_dates

    n = 0
    for sym in symbols:
        try:
            df = fetch_earnings_dates(sym, limit=8, storage=storage)
            n += len(df)
        except Exception as e:  # noqa: BLE001
            log.debug("earnings calendar %s failed: %s", sym, e)
    return n


def main() -> None:
    setup_logging()
    db = get_storage()
    t0 = time.time()

    n_edgar = backfill_edgar(db)
    n_info = backfill_yfinance_info(db, US_EQUITIES + ETFS + IN_EQUITIES)
    n_cal = backfill_earnings_calendar(db, US_EQUITIES)

    log.info("Fundamentals backfill complete in %.0fs: %d EDGAR snapshots, "
             "%d yfinance info, %d earnings-calendar rows",
             time.time() - t0, n_edgar, n_info, n_cal)


if __name__ == "__main__":
    main()
