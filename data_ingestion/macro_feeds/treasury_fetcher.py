"""US Treasury yield-curve fetcher (no key) — daily par yield rates.

Source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates
Daily Treasury Par Yield Curve Rates CSV, e.g.
  /daily-treasury-rates.csv/2026/all
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

from core.db import Storage, get_storage
from core.events import EVENT_MACRO, bus
from core.logging import get_logger

log = get_logger(__name__)

BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
COLUMNS = {"2 Yr": "us_2y_yield", "10 Yr": "us_10y_yield"}


def fetch_treasury_curve(years_back: int = 2,
                         storage: Storage | None = None) -> dict | None:
    """Fetch the latest daily yield curve and persist a MacroSnapshot."""
    storage = storage or get_storage()
    latest: dict[str, float] = {}
    for year in range(dt.date.today().year, dt.date.today().year - years_back - 1, -1):
        try:
            resp = requests.get(f"{BASE}/{year}/all", timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df = df[df["Date"] <= dt.date.today().isoformat()]
            if df.empty:
                continue
            row = df.iloc[-1]
            for col, field in COLUMNS.items():
                if col in df.columns and pd.notna(row.get(col)):
                    latest[field] = float(row[col]) / 100
            break
        except Exception as e:  # noqa: BLE001
            log.debug("Treasury %d failed: %s", year, e)
    if not latest:
        return None
    if "us_10y_yield" in latest and "us_2y_yield" in latest:
        latest["yield_curve_spread"] = latest["us_10y_yield"] - latest["us_2y_yield"]
    macro = {"ts": dt.datetime.utcnow(), **latest}
    storage.write_macro_snapshot(macro)
    import asyncio
    asyncio.run(bus.publish(EVENT_MACRO, macro))
    log.info("Treasury curve: %s", {k: round(v, 4) for k, v in latest.items()})
    return macro


if __name__ == "__main__":
    print(fetch_treasury_curve())


def fetch_vix_stooq() -> float | None:
    """Latest VIX close from Stooq (free, no key). Returns None on failure."""
    try:
        resp = requests.get("https://stooq.com/q/d/l/?s=^vix&i=d", timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if len(lines) < 2:
            return None
        return float(lines[-1].split(",")[-1])
    except Exception as e:  # noqa: BLE001
        log.debug("Stooq VIX fetch failed: %s", e)
        return None
