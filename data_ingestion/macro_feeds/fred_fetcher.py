"""FRED macro fetcher — VIX, yields, fed funds, dollar index proxy.

Requires FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api).
Degrades gracefully (logs + None values) when the key is missing.
"""
from __future__ import annotations

import datetime as dt

import requests

from core.config import FRED_API_KEY
from core.db import Storage, get_storage
from core.events import EVENT_MACRO, bus
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.stlouisfed.org/fred/series/observations"

# series_id → (macro field, transform)
SERIES = {
    "VIXCLS": ("vix", float),
    "DGS10": ("us_10y_yield", lambda v: float(v) / 100),
    "DGS2": ("us_2y_yield", lambda v: float(v) / 100),
    "DFF": ("fed_funds_rate", lambda v: float(v) / 100),
    "DTWEXBGS": ("dxy", float),          # broad dollar index (FRED proxy for DXY)
    "GOLDPMGBD228NLBM": ("gold_price", float),
}


def fetch_fred_latest(storage: Storage | None = None) -> dict | None:
    """Fetch latest observation of each series → MacroSnapshot dict."""
    if not FRED_API_KEY:
        log.info("FRED_API_KEY not set — skipping FRED fetch")
        return None
    storage = storage or get_storage()
    values: dict[str, float] = {}
    for sid, (field, transform) in SERIES.items():
        try:
            resp = requests.get(
                API,
                params={"series_id": sid, "api_key": FRED_API_KEY,
                        "file_type": "json", "sort_order": "desc", "limit": 1},
                timeout=30)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if obs and obs[0].get("value") not in (None, "."):
                values[field] = transform(obs[0]["value"])
        except Exception as e:  # noqa: BLE001
            log.warning("FRED series %s failed: %s", sid, e)

    if not values:
        return None
    macro = {"ts": dt.datetime.utcnow(), **values}
    if "us_10y_yield" in macro and "us_2y_yield" in macro:
        macro["yield_curve_spread"] = macro["us_10y_yield"] - macro["us_2y_yield"]
    storage.write_macro_snapshot(macro)
    import asyncio
    asyncio.run(bus.publish(EVENT_MACRO, macro))
    log.info("FRED macro snapshot: %s", {k: round(v, 4) for k, v in macro.items()
                                         if isinstance(v, float)})
    return macro


if __name__ == "__main__":
    print(fetch_fred_latest())
