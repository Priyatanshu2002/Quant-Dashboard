"""Crypto price feeds — Binance WebSocket (live) + REST (backfill).

Supports spot (BTCUSDT, ETHUSDT, ...) and perp futures (BTCUSDT on fapi).
Normalizes to USD-denominated OHLCV bars persisted via core.db and published
on the event bus (EVENT_PRICE_BAR) for downstream consumers.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Iterable

import pandas as pd
import requests
import websockets

from core.db import Storage, get_storage
from core.events import EVENT_PRICE_BAR, bus
from core.logging import get_logger

log = get_logger(__name__)

SPOT_REST = "https://api.binance.com/api/v3"
SPOT_WS = "wss://stream.binance.com:9443/ws"
FUTURES_REST = "https://fapi.binance.com/fapi/v1"
FUTURES_WS = "wss://fstream.binance.com/ws"

INTERVALS_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
    "4h": 14_400_000, "1d": 86_400_000,
}


def normalize_symbol(symbol: str) -> str:
    """BTC-USD / BTC/USDT → BTCUSDT (Binance convention)."""
    return symbol.replace("-", "").replace("/", "").replace("USD", "USDT") \
        if "USDT" not in symbol.upper() and "USD" in symbol.upper() \
        else symbol.replace("-", "").replace("/", "")


def backfill_klines(symbol: str, interval: str = "1d", days: int = 730,
                    futures: bool = False, storage: Storage | None = None,
                    limit: int = 1000) -> pd.DataFrame:
    """Pull historical OHLCV from Binance REST, paginated, persisted to storage."""
    storage = storage or get_storage()
    base = FUTURES_REST if futures else SPOT_REST
    sym = normalize_symbol(symbol)
    frames: list[pd.DataFrame] = []
    start_ms = int((time.time() - days * 86_400) * 1000)
    end_ms = int(time.time() * 1000)

    while start_ms < end_ms:
        resp = requests.get(
            f"{base}/klines",
            params={"symbol": sym, "interval": interval,
                    "startTime": start_ms, "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore"])
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        frames.append(df[["time", "open", "high", "low", "close", "volume"]])
        start_ms = int(df["open_time"].iloc[-1]) + INTERVALS_MS.get(interval, 86_400_000)
        if len(klines) < limit:
            break

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames).drop_duplicates("time").set_index("time")
    out.index = out.index.tz_convert(None)  # storage normalizes tz; keep naive
    storage.write_ohlcv(out, symbol=symbol.upper(), asset_class="CRYPTO",
                        source="BINANCE", interval=interval)
    log.info("Backfilled %d %s bars for %s", len(out), interval, symbol)
    return out


class BinanceLiveFeed:
    """Stream kline updates and publish closed bars on the event bus."""

    def __init__(self, symbols: Iterable[str], interval: str = "1m",
                 futures: bool = False, storage: Storage | None = None,
                 live_write: bool = True):
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.interval = interval
        self.futures = futures
        self.storage = storage or get_storage()
        self.live_write = live_write
        self._bars: dict[str, dict] = {}

    async def run(self) -> None:
        ws_url = (FUTURES_WS if self.futures else SPOT_WS) + f"/{self.interval}"
        stream = "/".join(f"{s.lower()}@kline_{self.interval}" for s in self.symbols)
        url = ws_url + f"?streams={stream}" if len(self.symbols) > 1 else f"{ws_url}/{self.symbols[0].lower()}@kline_{self.interval}"
        log.info("Connecting %s", url)
        async with websockets.connect(url) as ws:
            while True:
                msg = json.loads(await ws.recv())
                data = msg.get("data", msg)
                k = data.get("k", {})
                if not k:
                    continue
                bar = {
                    "t": k["t"], "o": float(k["o"]), "h": float(k["h"]),
                    "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"]),
                    "closed": k["x"],
                }
                self._on_bar(data["s"], bar)

    def _on_bar(self, symbol: str, bar: dict) -> None:
        if bar["closed"]:
            row = pd.DataFrame([{
                "time": pd.to_datetime(bar["t"], unit="ms", utc=True),
                "open": bar["o"], "high": bar["h"], "low": bar["l"],
                "close": bar["c"], "volume": bar["v"],
            }]).set_index("time")
            row.index = row.index.tz_convert(None)
            if self.live_write:
                self.storage.write_ohlcv(row, symbol=symbol, asset_class="CRYPTO",
                                         source="BINANCE_WS", interval=self.interval)
            asyncio.create_task(bus.publish(EVENT_PRICE_BAR, {
                "symbol": symbol, "asset_class": "CRYPTO", "interval": self.interval,
                "time": bar["t"], "close": bar["c"], "volume": bar["v"],
            }))


async def _main_demo() -> None:
    """Demo: backfill 2y of daily bars for BTC + ETH, then stream 1m live."""
    for sym in ("BTC-USD", "ETH-USD"):
        backfill_klines(sym, interval="1d", days=730)
    feed = BinanceLiveFeed(["BTC-USD", "ETH-USD"], interval="1m")
    await feed.run()


if __name__ == "__main__":
    asyncio.run(_main_demo())
