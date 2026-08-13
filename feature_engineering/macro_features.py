"""Macro features (~plan §3.5 + C8 expansion) from the latest macro snapshot.

Consumes the full Treasury curve (nominal + TIPS real yields), breakeven
inflation, credit spreads, inflation/labour/activity indices, and VIX regime —
every key that the macro layer now captures. All keys are optional: values
absent from the snapshot are simply omitted, so this degrades gracefully on a
thin (keyless) macro path.
"""
from __future__ import annotations

from core.db import Storage, get_storage


def _regime(vix) -> int | None:
    if vix is None:
        return None
    return 0 if vix < 15 else 1 if vix < 25 else 2


def _curve_slope(macro) -> float | None:
    """10Y−2Y from the curve, or 30Y−2Y as a longer-horizon slope."""
    if macro.get("us_10y_yield") is not None and macro.get("us_2y_yield") is not None:
        return macro["us_10y_yield"] - macro["us_2y_yield"]
    if macro.get("us_30y_yield") is not None and macro.get("us_2y_yield") is not None:
        return macro["us_30y_yield"] - macro["us_2y_yield"]
    return None


def compute_macro_features(db: Storage | None = None) -> dict:
    db = db or get_storage()
    macro = db.query_latest_macro()
    if not macro:
        return {}

    out: dict = {}
    # Core rates
    for k in ("us_10y_yield", "us_2y_yield", "us_1y_yield", "us_30y_yield",
              "fed_funds_rate", "dxy", "vix", "gold_pct_change_5d"):
        if macro.get(k) is not None:
            out[k] = macro[k]
    out["yield_curve_spread"] = _curve_slope(macro)
    out["vix_regime"] = _regime(out.get("vix"))

    # Full curve capture (all maturities if present)
    for m in ("us_1m_yield", "us_3m_yield", "us_6m_yield", "us_3y_yield",
              "us_5y_yield", "us_7y_yield", "us_20y_yield"):
        if macro.get(m) is not None:
            out[m] = macro[m]

    # Real yields + breakeven inflation
    for k in ("us_5y_real_yield", "us_10y_real_yield", "us_20y_real_yield",
              "us_30y_real_yield", "breakeven_inflation", "t10y_breakeven_ie"):
        if macro.get(k) is not None:
            out[k] = macro[k]

    # Inflation / labour / activity / credit / money / commodity (FRED depth)
    for k in ("cpi_all_urban", "core_cpi", "ppi_final", "pce_all",
              "unemployment_rate", "nonfarm_payrolls", "ism_pmi", "m2_supply",
              "hy_credit_spread", "ig_credit_spread", "wti_price", "brent_price",
              "t10y2y", "job_openings", "retail_sales", "housing_starts"):
        if macro.get(k) is not None:
            out[k] = macro[k]

    # Crypto-global (when present)
    for k in ("btc_dominance", "crypto_total_mcap_chg_24h"):
        if macro.get(k) is not None:
            out[k] = macro[k]

    return {k: v for k, v in out.items() if v is not None}
