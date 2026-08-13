"""BLS macro fetcher — CPI and unemployment (API v2, key optional).

Public data is also mirrored on FRED (CPIAUCSL, UNRATE), so this module is
a thin wrapper around the BLS API for when a key is configured.
"""
from __future__ import annotations

import requests

from core.config import BLS_API_KEY
from core.logging import get_logger

log = get_logger(__name__)

API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# series_id → friendly name
SERIES = {
    "CUUR0000SA0": "cpi_all_urban",          # CPI-U, all items
    "LNS14000000": "unemployment_rate",      # U-3 unemployment
}


def fetch_bls_latest() -> dict:
    """Latest monthly CPI + unemployment. Returns {} when key missing/unavailable."""
    if not BLS_API_KEY:
        log.info("BLS_API_KEY not set — skipping BLS fetch")
        return {}
    try:
        resp = requests.post(
            API,
            json={"seriesid": list(SERIES), "startyear": "2025", "endyear": "2026",
                  "registrationkey": BLS_API_KEY},
            timeout=30)
        resp.raise_for_status()
        out: dict[str, float] = {}
        for series in resp.json().get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            data = series.get("data", [])
            if data:
                out[SERIES.get(sid, sid)] = float(data[0]["value"])
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("BLS fetch failed: %s", e)
        return {}


if __name__ == "__main__":
    print(fetch_bls_latest())
