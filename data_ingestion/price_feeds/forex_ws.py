"""Forex price feed — Frankfurter (ECB reference rates, no API key).

Backfill: https://api.frankfurter.app/<start>..<end>?base=USD&symbols=...
(ECB daily fix, 90-day window per request → paginated).
Live path: re-polls the daily fix once per day and publishes a price-bar event.
"""
from __future__ import annotations

import asyncio
import datetime as dt

import pandas as pd
import requests

from core.db import Storage, get_storage
from core.events import EVENT_PRICE_BAR, bus
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.frankfurter.app"
MAJORS = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "INR", "CNY"]
WINDOW_DAYS = 90


def _pairs(base: str = "USD") -> list[str]:
    return [f"{base}{c}=X" for c in MAJORS if c != base]


def backfill_forex(base: str = "USD", days: int = 730,
                   storage: Storage | None = None) -> dict[str, pd.DataFrame]:
    storage = storage or get_storage()
    results: dict[str, pd.DataFrame] = {}
    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    cursor = start
    while cursor < end:
        win_end = min(cursor + dt.timedelta(days=WINDOW_DAYS), end)
        resp = requests.get(
            f"{API}/{cursor.isoformat()}..{win_end.isoformat()}",
            params={"base": base, "symbols": ",".join(MAJORS)}, timeout=30)
        resp.raise_for_status()
        rates = resp.json().get("rates", {})
        for day, fx in rates.items():
            for c, rate in fx.items():
                key = f"{base}{c}=X"
                results.setdefault(key, []).append((day, rate))
        cursor = win_end + dt.timedelta(days=1)

    for pair, rows in results.items():
        df = pd.DataFrame(rows, columns=["time", "close"]).set_index("time")
        df.index = pd.to_datetime(df.index)
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 0.0
        df = df[["open", "high", "low", "close", "volume"]]
        storage.write_ohlcv(df, symbol=pair, asset_class="FOREX",
                            source="FRANKFURTER", interval="1d")
        log.info("Backfilled %d daily FX bars for %s", len(df), pair)
    return results


async def daily_fx_poller(storage: Storage | None = None) -> None:
    """Check for a fresh ECB fix once an hour; publish when a new one lands."""
    storage = storage or get_storage()
    last_date = None
    while True:
        try:
            resp = requests.get(f"{API}/latest", params={"base": "USD"}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            day = data.get("date")
            if day and day != last_date:
                last_date = day
                for c, rate in data.get("rates", {}).items():
                    await bus.publish(EVENT_PRICE_BAR, {
                        "symbol": f"USD{c}=X", "asset_class": "FOREX",
                        "interval": "1d", "time": day, "close": rate, "volume": 0,
                    })
                log.info("Published FX fix for %s", day)
        except Exception as e:  # noqa: BLE001
            log.warning("FX poll failed: %s", e)
        await asyncio.sleep(3600)


if __name__ == "__main__":
    backfill_forex(days=730)
