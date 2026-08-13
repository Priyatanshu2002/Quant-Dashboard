"""Macro features (~10, plan §3.5) — from the latest macro snapshot."""
from __future__ import annotations

from core.db import Storage, get_storage


def compute_macro_features(db: Storage | None = None) -> dict:
    db = db or get_storage()
    macro = db.query_latest_macro()
    if not macro:
        return {}

    vix = macro.get("vix")
    out: dict = {
        "us_10y_yield": macro.get("us_10y_yield"),
        "us_2y_yield": macro.get("us_2y_yield"),
        "yield_curve_spread": macro.get("yield_curve_spread"),
        "fed_funds_rate": macro.get("fed_funds_rate"),
        "vix": vix,
        "vix_regime": None if vix is None else (0 if vix < 15 else 1 if vix < 25 else 2),
        "dxy": macro.get("dxy"),
        "gold_pct_change_5d": macro.get("gold_pct_change_5d"),
        "btc_dominance": macro.get("btc_dominance"),
        "crypto_total_mcap_chg_24h": macro.get("crypto_total_mcap_chg_24h"),
    }
    return {k: v for k, v in out.items() if v is not None}
