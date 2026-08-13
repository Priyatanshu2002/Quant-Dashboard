"""Data loader — OHLCV + benchmark series from the storage layer."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from core.db import Storage, get_storage
from core.logging import get_logger

log = get_logger(__name__)

BENCHMARKS = {
    "SP500": "SPY",
    "NIFTY50": "^NSEI",
    "BTC": "BTC-USD",
}


def get_ohlcv(symbol: str, start=None, end=None, db: Storage | None = None) -> pd.DataFrame:
    db = db or get_storage()
    df = db.query_ohlcv(symbol, start=start, end=end)
    if df.empty:
        log.warning("No market data for %s — run a backfill first", symbol)
    return df


def get_benchmark(name: str = "SP500", start=None, end=None,
                  db: Storage | None = None) -> pd.Series:
    """Benchmark close series (SPY / ^NSEI / BTC-USD)."""
    db = db or get_storage()
    ticker = BENCHMARKS.get(name, name)
    df = db.query_ohlcv(ticker, start=start, end=end)
    return df["close"] if not df.empty else pd.Series(dtype=float)


def get_multi(symbols: Iterable[str], start=None, end=None,
              db: Storage | None = None) -> dict[str, pd.DataFrame]:
    db = db or get_storage()
    return {s: db.query_ohlcv(s, start=start, end=end) for s in symbols}
