"""Macro fetcher via yfinance — works on any network where yfinance works.

Covers the plan §3.5 macro features without any API keys:
  ^VIX       → VIX (volatility index)
  ^TNX       → US 10Y Treasury yield
  ^2YY       → US 2Y Treasury yield  (→ yield_curve_spread = 10Y - 2Y)
  DX-Y.NYB   → US Dollar Index (DXY)
  GC=F       → Gold futures → 5-day % change
  BTC-USD    → BTC price (for reference; dominance comes from CoinGecko)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf

from core.db import Storage, get_storage
from core.events import EVENT_MACRO, bus
from core.logging import get_logger

log = get_logger(__name__)

TICKERS = ["^VIX", "^TNX", "^2YY", "DX-Y.NYB", "GC=F", "BTC-USD"]
FIELDS = {
    "^VIX": "vix",
    "^TNX": "us_10y_yield",
    "^2YY": "us_2y_yield",
    "DX-Y.NYB": "dxy",
}


def fetch_yfinance_macro(storage: Storage | None = None) -> dict | None:
    """Latest macro snapshot from yfinance quotes + 5d gold change."""
    storage = storage or get_storage()
    try:
        data = yf.download(TICKERS, period="7d", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker",
                           threads=True)
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance macro download failed: %s", e)
        return None

    macro: dict = {"ts": dt.datetime.utcnow()}
    for ticker, field in FIELDS.items():
        try:
            df = data[ticker] if len(TICKERS) > 1 else data
            if df.empty:
                continue
            close = df["Close"].dropna()
            if close.empty:
                continue
            val = float(close.iloc[-1])
            macro[field] = val / 100 if ticker in ("^TNX", "^2YY") else val
        except Exception as e:  # noqa: BLE001
            log.debug("macro %s failed: %s", ticker, e)

    # Gold 5-day % change
    try:
        df = data["GC=F"] if len(TICKERS) > 1 else data
        close = df["Close"].dropna()
        if len(close) >= 2:
            macro["gold_pct_change_5d"] = (close.iloc[-1] / close.iloc[0] - 1) * 100
    except Exception:  # noqa: BLE001
        pass

    if "us_10y_yield" in macro and "us_2y_yield" in macro:
        macro["yield_curve_spread"] = macro["us_10y_yield"] - macro["us_2y_yield"]

    if len(macro) <= 1:
        log.warning("No macro values could be fetched")
        return None

    storage.write_macro_snapshot(macro)
    import asyncio
    asyncio.run(bus.publish(EVENT_MACRO, macro))
    log.info("Macro snapshot: vix=%s yc10-2=%s dxy=%s gold5d=%s",
             macro.get("vix"), macro.get("yield_curve_spread"),
             macro.get("dxy"), macro.get("gold_pct_change_5d"))
    return macro


if __name__ == "__main__":
    print(fetch_yfinance_macro())
