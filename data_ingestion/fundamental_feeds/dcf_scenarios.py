"""DCF scenario grid + persistence helpers (plan §8.2/§8.4).

Extends the core `compute_dcf` with:
  * `dcf_sensitivity_grid` — WACC × terminal-growth matrix of intrinsic values
    and margins of safety (the UI sensitivity table).
  * `apply_dcf_to_snapshot` — compute the base-case DCF and stamp
    dcf_intrinsic_value / dcf_margin_of_safety / wacc_used onto a snapshot
    dict so it persists with the next `upsert_fundamental_snapshot`.
"""
from __future__ import annotations

from typing import Any

from data_ingestion.fundamental_feeds.dcf_calculator import DCFResult, compute_dcf

BASE_WACCS = (0.08, 0.09, 0.10, 0.11, 0.12)
BASE_TERMINAL_GROWTHS = (0.015, 0.02, 0.025, 0.03, 0.035)


def _shares_and_debt(snap: dict) -> tuple[float | None, float]:
    mcap, price = snap.get("market_cap"), snap.get("current_price")
    shares = (mcap / price) if mcap and price else None
    net_debt = snap.get("net_debt") or 0.0
    return shares, net_debt


def dcf_sensitivity_grid(snap: dict, waccs: tuple[float, ...] = BASE_WACCS,
                         terminal_growths: tuple[float, ...] = BASE_TERMINAL_GROWTHS
                         ) -> dict:
    """WACC × terminal-growth matrix → {waccs, terminal_growths, grid[[intrinsic,…]]}."""
    fcf = snap.get("free_cash_flow")
    if not fcf:
        return {"waccs": list(waccs), "terminal_growths": list(terminal_growths),
                "grid": [], "base_wacc": 0.10, "base_terminal_growth": 0.025}
    growth = snap.get("revenue_yoy_growth") or 0.06
    shares, net_debt = _shares_and_debt(snap)
    price = snap.get("current_price")
    grid = []
    for wacc in waccs:
        row = []
        for tg in terminal_growths:
            if wacc <= tg:
                row.append(None)
                continue
            try:
                r = compute_dcf(fcf, growth, terminal_growth_rate=tg, wacc=wacc,
                                shares_outstanding=shares, net_debt=net_debt,
                                current_price=price)
            except ValueError:
                row.append(None)
                continue
            row.append(round(r.intrinsic_value_per_share, 2) if r.intrinsic_value_per_share else None)
        grid.append(row)
    return {"waccs": list(waccs), "terminal_growths": list(terminal_growths),
            "grid": grid, "base_wacc": 0.10, "base_terminal_growth": 0.025}


def apply_dcf_to_snapshot(snap: dict, wacc: float = 0.10,
                          terminal_growth: float = 0.025) -> DCFResult | None:
    """Stamp DCF outputs onto a snapshot dict (in place) and return the result."""
    fcf = snap.get("free_cash_flow")
    if not fcf:
        return None
    growth = snap.get("revenue_yoy_growth") or 0.06
    shares, net_debt = _shares_and_debt(snap)
    result = compute_dcf(fcf, growth, terminal_growth_rate=terminal_growth,
                         wacc=wacc, shares_outstanding=shares, net_debt=net_debt,
                         current_price=snap.get("current_price"))
    if result.intrinsic_value_per_share:
        snap["dcf_intrinsic_value"] = result.intrinsic_value_per_share
        snap["dcf_margin_of_safety"] = result.margin_of_safety
        snap["wacc_used"] = wacc
    return result


def dcf_bundle(snap: dict | None) -> dict | None:
    """Full DCF payload for the UI: base result + sensitivity grid."""
    if not snap or not snap.get("free_cash_flow"):
        return None
    result = apply_dcf_to_snapshot(snap)
    grid = dcf_sensitivity_grid(snap)
    return {
        "intrinsic_value_per_share": result.intrinsic_value_per_share if result else None,
        "margin_of_safety": result.margin_of_safety if result else None,
        "wacc": result.wacc if result else None,
        "terminal_growth": result.terminal_growth if result else None,
        "enterprise_value": result.enterprise_value if result else None,
        "pv_of_projected_fcf": result.pv_of_projected_fcf if result else None,
        "pv_of_terminal_value": result.pv_of_terminal_value if result else None,
        "inputs": {
            "ttm_free_cash_flow": snap.get("free_cash_flow"),
            "revenue_growth_rate": snap.get("revenue_yoy_growth") or 0.06,
            "net_debt": snap.get("net_debt") or 0.0,
            "market_cap": snap.get("market_cap"),
            "current_price": snap.get("current_price"),
        },
        "sensitivity": grid,
    }
