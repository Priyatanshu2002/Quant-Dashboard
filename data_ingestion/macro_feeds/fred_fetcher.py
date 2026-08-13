"""FRED macro fetcher — expanded series map + transforms.

Requires FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api).
Without the key, fetch_fred_latest() returns None gracefully (the series map
below is ready to "light up" the moment a key is set — no code change needed).

The map covers inflation, rates, credit, growth, labour, and money:
  * Inflation: CPI (CPIAUCSL), core CPI (CPILFESL), PPI (PPIFIS), PCE (PCEPI)
  * Labour:    unemployment (UNRATE), nonfarm payrolls (PAYEMS), job openings (JTSJOL)
  * Activity:  ISM PMI (ISMPMI), retail sales (RRSFS), housing starts (HOUST)
  * Rates:     full nominal curve (DGS1/2/3/5/7/10/20/30), real yields (DFII10),
               breakevens (T10YIE, T5YIE), spread (T10Y2Y), fed funds (DFF)
  * Credit:    high-yield (BAMLH0A0HYM2), investment-grade (BAMLC0A0CM) spreads
  * Money:     M2 (M2SL)
  * Commodity: WTI (DCOILWTICO), Brent (DCOILBRENTEU)
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


def _pct(v) -> float:
    """Percent → decimal (e.g. '4.68' → 0.0468)."""
    return float(v) / 100


def _bps_to_decimal(v) -> float:
    """Basis points → decimal spread (e.g. '256' → 0.0256)."""
    return float(v) / 10_000


def _index(v) -> float:
    """Index value (CPI/PPI/PCE/M2) — no transform."""
    return float(v)


def _thousands(v) -> float:
    """Thousands of units → raw float (payrolls, starts)."""
    return float(v)


# series_id → (macro field, transform)
SERIES = {
    # ── Inflation ──
    "CPIAUCSL": ("cpi_all_urban", _index),        # CPI-U index
    "CPILFESL": ("core_cpi", _index),             # core CPI index
    "PPIFIS": ("ppi_final", _index),              # PPI final demand index
    "PCEPI": ("pce_all", _index),                 # PCE price index
    "T10YIE": ("t10y_breakeven_ie", _pct),        # 10Y breakeven inflation
    "T5YIE": ("t5y_breakeven_ie", _pct),
    # ── Labour ──
    "UNRATE": ("unemployment_rate", _pct),        # % → decimal
    "PAYEMS": ("nonfarm_payrolls", _thousands),   # thousands of jobs
    "JTSJOL": ("job_openings", _thousands),
    # ── Activity ──
    "ISMPMI": ("ism_pmi", _index),
    "RRSFS": ("retail_sales", _index),
    "HOUST": ("housing_starts", _thousands),
    # ── Rates / curve ──
    "DGS1": ("us_1y_yield", _pct),
    "DGS2": ("us_2y_yield", _pct),
    "DGS3": ("us_3y_yield", _pct),
    "DGS5": ("us_5y_yield", _pct),
    "DGS7": ("us_7y_yield", _pct),
    "DGS10": ("us_10y_yield", _pct),
    "DGS20": ("us_20y_yield", _pct),
    "DGS30": ("us_30y_yield", _pct),
    "DFII10": ("us_10y_real_yield", _pct),        # 10Y TIPS real yield
    "DFF": ("fed_funds_rate", _pct),
    "T10Y2Y": ("t10y2y", _pct),                   # 10Y−2Y curve spread
    "VIXCLS": ("vix", _index),
    # ── Credit ──
    "BAMLH0A0HYM2": ("hy_credit_spread", _bps_to_decimal),   # HY OAS (bps)
    "BAMLC0A0CM": ("ig_credit_spread", _bps_to_decimal),     # IG OAS (bps)
    # ── Money / dollar / commodity ──
    "M2SL": ("m2_supply", _index),
    "DTWEXBGS": ("dxy", _index),                  # broad dollar index
    "DCOILWTICO": ("wti_price", _index),
    "DCOILBRENTEU": ("brent_price", _index),
}

# Series required before we consider the snapshot worth persisting.
_CORE = {"vix", "us_10y_yield", "fed_funds_rate"}


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
    # Derive yield_curve_spread if the two points both landed and FRED's own
    # T10Y2Y is unavailable.
    if "yield_curve_spread" not in macro and \
       "us_10y_yield" in macro and "us_2y_yield" in macro:
        macro["yield_curve_spread"] = macro["us_10y_yield"] - macro["us_2y_yield"]
    # Breakeven from nominal vs real when FRED's own T10YIE is missing.
    if "t10y_breakeven_ie" not in macro and \
       "us_10y_yield" in macro and "us_10y_real_yield" in macro:
        macro["breakeven_inflation"] = macro["us_10y_yield"] - macro["us_10y_real_yield"]
    storage.write_macro_snapshot(macro)
    import asyncio
    asyncio.run(bus.publish(EVENT_MACRO, macro))
    log.info("FRED macro snapshot: %s", {k: round(v, 4) for k, v in macro.items()
                                         if isinstance(v, float)})
    return macro


if __name__ == "__main__":
    print(fetch_fred_latest())
