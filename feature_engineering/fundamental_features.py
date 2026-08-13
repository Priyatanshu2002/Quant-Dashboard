"""Fundamental features (~20, plan §3.3) — built from fundamental_snapshots."""
from __future__ import annotations

from core.db import Storage, get_storage


def compute_fundamental_features(symbol: str, asset_class: str,
                                 db: Storage | None = None) -> dict:
    db = db or get_storage()
    snap = db.query_latest_fundamentals(symbol)
    if not snap:
        return {}
    if asset_class == "CRYPTO":
        return {}  # on-chain features handled by cross_asset / exchange flows

    f: dict = {}

    # Earnings quality
    if snap.get("eps_estimate"):
        f["eps_surprise_pct"] = snap["eps_actual"] / snap["eps_estimate"] - 1
    if snap.get("eps_yoy_growth") is not None:
        f["eps_yoy_growth"] = snap["eps_yoy_growth"]
    if snap.get("revenue_estimate"):
        f["revenue_surprise_pct"] = snap["revenue"] / snap["revenue_estimate"] - 1
    if snap.get("revenue_yoy_growth") is not None:
        f["revenue_yoy_growth"] = snap["revenue_yoy_growth"]

    # Profitability
    if snap.get("revenue"):
        if snap.get("ebitda") is not None:
            f["ebitda_margin"] = snap["ebitda"] / snap["revenue"]
        if snap.get("gross_profit") is not None:
            f["gross_margin"] = snap["gross_profit"] / snap["revenue"]
    if snap.get("market_cap"):
        if snap.get("free_cash_flow") is not None:
            f["fcf_yield"] = snap["free_cash_flow"] / snap["market_cap"]
    if snap.get("roic") is not None:
        f["roic"] = snap["roic"]

    # Valuation
    for k in ("forward_pe", "peg_ratio", "ev_to_ebitda"):
        if snap.get(k) is not None:
            f[k] = snap[k]
    if snap.get("dcf_intrinsic_value") and snap.get("current_price"):
        f["dcf_margin_of_safety"] = snap["dcf_intrinsic_value"] / snap["current_price"] - 1

    # Balance sheet health
    for k in ("debt_to_equity", "current_ratio", "interest_coverage_ratio"):
        if snap.get(k) is not None:
            f[k] = snap[k]

    # Corporate actions
    if snap.get("insider_sell_value"):
        f["insider_buy_sell_ratio"] = (snap.get("insider_buy_value") or 0) / snap["insider_sell_value"]
    if snap.get("institutional_ownership_change_pct") is not None:
        f["inst_ownership_change"] = snap["institutional_ownership_change_pct"]
    if snap.get("transcript_sentiment_score") is not None:
        f["earnings_call_sentiment"] = snap["transcript_sentiment_score"]

    # Historical trend features (from quarterly history)
    history = db.query_fundamental_history(symbol, quarters=8)
    margins = [h.get("ebitda_margin") for h in history if h.get("ebitda_margin") is not None]
    if len(margins) >= 4:
        f["margin_trend"] = sum(margins[:4]) / 4 - sum(margins[4:8]) / 4
    growths = [h.get("revenue_yoy_growth") for h in history if h.get("revenue_yoy_growth") is not None]
    if len(growths) >= 4:
        f["revenue_accel"] = growths[0] - growths[3]

    return f
