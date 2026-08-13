"""Equity price feed — yfinance polling (US/IN equities, ETFs, indices).

yfinance has no WebSocket, so the "live" path polls the last 1-5 days of
1m/1h bars every minute and publishes newly-closed bars on the event bus.
Backfill downloads multi-year daily history in one call.
"""
from __future__ import annotations

import asyncio
import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from core.db import Storage, get_storage
from core.events import EVENT_PRICE_BAR, bus
from core.logging import get_logger

log = get_logger(__name__)


def asset_class_for(symbol: str) -> str:
    s = symbol.upper()
    if s.startswith("^") or s in ("SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "SLV", "EEM", "VOO", "VTI"):
        return "ETF" if not s.startswith("^") else "ETF" if False else "EQUITY_US"
    if s.endswith(".NS") or s.endswith(".BO"):
        return "EQUITY_IN"
    return "EQUITY_US"


def backfill_equities(symbols: Iterable[str], period: str = "5y",
                      interval: str = "1d", storage: Storage | None = None) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for a list of tickers and persist to storage."""
    storage = storage or get_storage()
    tickers = list(symbols)
    results: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 10):  # chunk to be kind to yfinance
        chunk = tickers[i:i + 10]
        data = yf.download(chunk, period=period, interval=interval,
                           auto_adjust=True, group_by="ticker", progress=False,
                           threads=True)
        for t in chunk:
            try:
                df = data[t] if len(chunk) > 1 else data
                df = df.dropna(subset=["Close"]).rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume"})
                if df.empty:
                    continue
                df = df[["open", "high", "low", "close", "volume"]]
                df.index = pd.to_datetime(df.index)
                storage.write_ohlcv(df, symbol=t.upper(),
                                    asset_class=asset_class_for(t),
                                    source="YAHOO", interval=interval)
                results[t.upper()] = df
                log.info("Backfilled %d bars for %s", len(df), t)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to backfill %s: %s", t, e)
    return results


class EquityLiveFeed:
    """Poll 1m bars for a small watchlist every POLL_SECONDS, publish closed bars."""

    def __init__(self, symbols: Iterable[str], poll_seconds: int = 60,
                 storage: Storage | None = None):
        self.symbols = list(symbols)
        self.poll_seconds = poll_seconds
        self.storage = storage or get_storage()
        self._last_seen: dict[str, pd.Timestamp] = {}

    async def run(self) -> None:
        log.info("Equity live feed polling every %ds for %s",
                 self.poll_seconds, self.symbols)
        while True:
            try:
                data = yf.download(self.symbols, period="2d", interval="1m",
                                   auto_adjust=True, group_by="ticker",
                                   progress=False, threads=True)
                for t in self.symbols:
                    try:
                        df = data[t] if len(self.symbols) > 1 else data
                        if df.empty:
                            continue
                        df = df.dropna(subset=["Close"])
                        last = df.index[-1]
                        if self._last_seen.get(t) == last:
                            continue
                        self._last_seen[t] = last
                        await bus.publish(EVENT_PRICE_BAR, {
                            "symbol": t.upper(), "asset_class": asset_class_for(t),
                            "interval": "1m", "time": int(last.timestamp() * 1000),
                            "close": float(df["Close"].iloc[-1]),
                            "volume": float(df["Volume"].iloc[-1]),
                        })
                    except Exception as e:  # noqa: BLE001
                        log.debug("equity bar %s: %s", t, e)
            except Exception as e:  # noqa: BLE001
                log.warning("yfinance poll failed: %s", e)
            await asyncio.sleep(self.poll_seconds)


async def _main_demo() -> None:
    backfill_equities(["AAPL", "MSFT", "^NSEI"], period="2y")
    feed = EquityLiveFeed(["AAPL"], poll_seconds=30)
    await feed.run()


if __name__ == "__main__":
    asyncio.run(_main_demo())
