"""Relative valuation — valuation multiples + peer/sector comparison.

Computes the standard multiples a professional checks before transacting:
trailing/forward P/E, P/B, P/S, P/CF, EV/EBITDA, EV/EBIT, EV/Sales, EV/FCFF,
PEG, dividend yield, FCF yield. EV = market cap + debt - cash.

Also compares a company's multiples against peer/sector medians (Damodaran:
a comparable firm shares risk, growth, and cash-flow profile) and flags
value/quality mismatches ("cheap + good fundamentals" vs "expensive + poor").
"""
from __future__ import annotations

from typing import Any


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def compute_multiples(metrics: dict, market_cap: float | None = None,
                      price: float | None = None,
                      shares_outstanding: float | None = None,
                      forward_eps: float | None = None) -> dict:
    """Compute valuation multiples from a metrics dict.

    metrics keys used: revenue, net_income, ebitda, ebit, fcff, fcf, equity,
    total_debt, cash, dividends_paid, eps, growth (for PEG).
    """
    mcap = market_cap
    if mcap is None and price is not None and shares_outstanding:
        mcap = price * shares_outstanding
    if mcap is None:
        return {}

    rev = _f(metrics.get("revenue"))
    ni = _f(metrics.get("net_income"))
    ebitda = _f(metrics.get("ebitda"))
    ebit = _f(metrics.get("ebit"))
    fcff = _f(metrics.get("fcff")) or _f(metrics.get("fcf"))
    eq = _f(metrics.get("equity"))
    debt = _f(metrics.get("total_debt")) or 0.0
    cash = _f(metrics.get("cash")) or 0.0
    ev = mcap + debt - cash

    out: dict = {}
    if rev and rev > 0:
        out["market_cap"] = mcap
        out["enterprise_value"] = ev
        out["price_to_sales"] = mcap / rev
        out["ev_to_sales"] = ev / rev
    if ni and ni > 0:
        out["price_to_earnings"] = mcap / ni
        out["earnings_yield"] = ni / mcap
    if ebitda and ebitda > 0:
        out["ev_to_ebitda"] = ev / ebitda
    if ebit and ebit > 0:
        out["ev_to_ebit"] = ev / ebit
    if fcff and fcff > 0:
        out["ev_to_fcff"] = ev / fcff
        out["fcf_yield"] = fcff / mcap
        out["price_to_fcf"] = mcap / fcff
    if eq and eq > 0:
        out["price_to_book"] = mcap / eq
    eps = _f(metrics.get("eps"))
    if eps and eps > 0 and price:
        out["trailing_pe"] = price / eps
    if forward_eps and forward_eps > 0 and price:
        out["forward_pe"] = price / forward_eps
    growth = _f(metrics.get("growth"))
    if growth and growth > 0 and "forward_pe" in out:
        out["peg_ratio"] = out["forward_pe"] / (growth * 100)
    elif growth and growth > 0 and "trailing_pe" in out:
        out["peg_ratio"] = out["trailing_pe"] / (growth * 100)
    dividends = _f(metrics.get("dividends_paid"))
    if dividends is not None and shares_outstanding and price:
        dps = dividends / shares_outstanding
        if price > 0:
            out["dividend_yield"] = dps / price
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def compare_to_peers(company_multiples: dict, peer_multiples: dict[str, dict]) -> dict:
    """Compare a company's multiples to peer/sector medians.

    peer_multiples: {peer_symbol: {multiple_name: value, ...}}. For each multiple
    present in both, compute the percentile rank of the company vs peers and a
    cheap/rich/neutral label.
    """
    if not peer_multiples:
        return {"peers": len(peer_multiples), "comparisons": []}
    peer_symbols = list(peer_multiples.keys())
    keys = {k for pm in peer_multiples.values() for k in pm if k not in ("market_cap", "enterprise_value")}
    comparisons = []
    for k in sorted(keys):
        if k not in company_multiples:
            continue
        comp = company_multiples[k]
        if comp is None:
            continue
        peer_vals = [pm[k] for pm in peer_multiples.values() if pm.get(k) is not None]
        if not peer_vals:
            continue
        peer_vals = [v for v in peer_vals if v is not None and v > 0]
        if not peer_vals:
            continue
        pct_rank = sum(1 for v in peer_vals if v < comp) / len(peer_vals)
        median = sorted(peer_vals)[len(peer_vals) // 2]
        # For most multiples, lower = cheaper (a higher rank → expensive).
        label = "expensive" if pct_rank > 0.66 else "cheap" if pct_rank < 0.33 else "neutral"
        comparisons.append({
            "multiple": k, "value": round(comp, 3), "peer_median": round(median, 3),
            "percentile": round(pct_rank, 2), "label": label,
        })
    return {"peers": len(peer_symbols), "peer_symbols": peer_symbols,
            "comparisons": comparisons}


def mismatch_check(quality: dict, multiples: dict) -> list[dict]:
    """Flag value/quality mismatches (Damodaran Session 15/24).

    cheap + poor fundamentals (likely a value trap) and expensive + weak
    quality. Uses Piotroski/Altman Z where available.
    """
    flags = []
    piot = (quality.get("piotroski") or {}).get("score")
    altz = (quality.get("altman_z") or {}).get("zone")
    pe = multiples.get("trailing_pe") or multiples.get("price_to_earnings")
    ps = multiples.get("price_to_sales")
    pb = multiples.get("price_to_book")

    cheap = (pe is not None and pe < 12) or (ps is not None and ps < 1) or (pb is not None and pb < 1)
    if cheap:
        if piot is not None and piot <= 3:
            flags.append({"signal": "Cheap + weak Piotroski (<=3) — potential value trap", "severity": "warning"})
        if altz == "distress":
            flags.append({"signal": "Cheap + Altman-Z distress — potential value trap", "severity": "warning"})
    expensive = (pe is not None and pe > 30) or (ps is not None and ps > 5)
    if expensive and altz == "distress":
        flags.append({"signal": "Expensive + Altman-Z distress — avoid", "severity": "warning"})
    return flags
