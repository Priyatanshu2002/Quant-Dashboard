"""Coinbase Exchange WebSocket price feed (plan §1.2 crypto_ws).

Coinbase's protocol differs from Binance: subscribe to the `matches`
(trade) channel and aggregate trades into 1-minute OHLCV bars locally.
Symbols are already USD pairs (BTC-USD), no normalization needed.
"""
from __future__ import annotations

import asyncio
import json
from typing import Iterable

import pandas as pd
import websockets

from core.db import Storage, get_storage
from core.events import EVENT_PRICE_BAR, bus
from core.logging import get_logger

log = get_logger(__name__)

WS_URL = "wss://ws-feed.exchange.coinbase.com"
REST = "https://api.exchange.coinbase.com"


class CoinbaseLiveFeed:
    def __init__(self, symbols: Iterable[str], interval_seconds: int = 60,
                 storage: Storage | None = None, live_write: bool = True):
        self.symbols = list(symbols)
        self.interval_seconds = interval_seconds
        self.storage = storage or get_storage()
        self.live_write = live_write
        self._bars: dict[str, dict] = {}   # symbol → current bar

    async def run(self) -> None:
        log.info("Coinbase WS connecting for %s", self.symbols)
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({
                "type": "subscribe",
                "product_ids": self.symbols,
                "channels": [{"name": "matches", "product_ids": self.symbols}],
            }))
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "match":
                    self._on_trade(msg)

    def _on_trade(self, msg: dict) -> None:
        symbol = msg.get("product_id", "")
        price = float(msg.get("price", 0))
        volume = float(msg.get("size", 0))
        ts = pd.Timestamp(msg.get("time"), tz="UTC").tz_convert(None)

        bar = self._bars.get(symbol)
        if bar is None or (ts - bar["t"]).total_seconds() >= self.interval_seconds:
            # close the previous bar
            if bar is not None and bar["o"] > 0:
                self._close_bar(symbol, bar, ts)
            bar = {"t": ts, "o": price, "h": price, "l": price, "c": price, "v": 0.0}
            self._bars[symbol] = bar
        bar["h"] = max(bar["h"], price)
        bar["l"] = min(bar["l"], price)
        bar["c"] = price
        bar["v"] += volume

    def _close_bar(self, symbol: str, bar: dict, close_ts) -> None:
        row = pd.DataFrame([{
            "time": bar["t"], "open": bar["o"], "high": bar["h"],
            "low": bar["l"], "close": bar["c"], "volume": bar["v"],
        }]).set_index("time")
        if self.live_write:
            self.storage.write_ohlcv(row, symbol=symbol, asset_class="CRYPTO",
                                     source="COINBASE_WS", interval="1m")
        asyncio.create_task(bus.publish(EVENT_PRICE_BAR, {
            "symbol": symbol, "asset_class": "CRYPTO", "interval": "1m",
            "time": int(bar["t"].timestamp() * 1000),
            "close": bar["c"], "volume": bar["v"],
        }))


def backfill_coinbase(symbol: str = "BTC-USD", days: int = 730,
                      granularity: int = 86400,
                      storage: Storage | None = None) -> pd.DataFrame:
    """Historical candles via Coinbase REST (granularity in seconds: 60-86400)."""
    storage = storage or get_storage()
    import requests

    frames = []
    end = int(pd.Timestamp.utcnow().timestamp())
    start = end - days * 86400
    step = 300 * granularity  # 300 candles per request
    cursor = start
    while cursor < end:
        resp = requests.get(
            f"{REST}/products/{symbol}/candles",
            params={"start": cursor, "end": min(cursor + step, end),
                    "granularity": granularity},
            headers={"User-Agent": "agonistes/0.1"}, timeout=30)
        resp.raise_for_status()
        candles = resp.json()  # [ [time, low, high, open, close, volume], ... ]
        if not candles:
            break
        df = pd.DataFrame(candles, columns=["time", "low", "high", "open",
                                            "close", "volume"])
        frames.append(df)
        cursor = int(df["time"].max()) + granularity

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames).drop_duplicates("time").sort_values("time")
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.tz_convert(None)
    out = out.set_index("time")[["open", "high", "low", "close", "volume"]]
    storage.write_ohlcv(out, symbol=symbol, asset_class="CRYPTO",
                        source="COINBASE", interval="1d" if granularity >= 86400 else f"{granularity}s")
    log.info("Coinbase backfill: %d bars for %s", len(out), symbol)
    return out


async def _main_demo() -> None:
    feed = CoinbaseLiveFeed(["BTC-USD", "ETH-USD"])
    await feed.run()


if __name__ == "__main__":
    asyncio.run(_main_demo())
