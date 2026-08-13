"""DCF scenario grid + persistence helpers (plan §8.2/§8.4).

Extends the core `compute_dcf` with:
  * `dcf_sensitivity_grid` — WACC × terminal-growth matrix of intrinsic values
    and margins of safety (the UI sensitivity table).
  * `apply_dcf_to_snapshot` — compute the base-case DCF and stamp
    dcf_intrinsic_value / dcf_margin_of_safety / wacc_used onto a snapshot
    dict so it persists with the next `upsert_fundamental_snapshot`.
"""
from __future__ import annotations

import math

from data_ingestion.fundamental_feeds.dcf_calculator import DCFResult, compute_dcf, dcf_from_snapshot

BASE_WACCS = (0.08, 0.09, 0.10, 0.11, 0.12)
BASE_TERMINAL_GROWTHS = (0.015, 0.02, 0.025, 0.03, 0.035)

# Sane growth bound for the projection. Snapshot revenue_yoy_growth values that
# are non-finite or beyond ±50% are almost always bad data (e.g. a bad quarter
# comparison) — using them would explode the terminal value into a nonsensical
# intrinsic figure. Fall back to a conservative default.
_MIN_G, _MAX_G = -0.5, 1.0
_DEFAULT_G = 0.06


def _growth(snap: dict) -> float:
    try:
        g = float(snap.get("revenue_yoy_growth"))
    except (TypeError, ValueError):
        return _DEFAULT_G
    if not math.isfinite(g) or not (_MIN_G <= g <= _MAX_G):
        return _DEFAULT_G
    return g


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
    growth = _growth(snap)
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


def apply_dcf_to_snapshot(snap: dict, wacc: float | None = None,
                          terminal_growth: float = 0.025) -> DCFResult | None:
    """Stamp DCF outputs onto a snapshot dict (in place) and return the result.

    WACC is estimated from market data when not explicitly provided (see
    dcf_from_snapshot); share count uses real reported shares.
    """
    fcf = snap.get("free_cash_flow")
    if not fcf:
        return None
    result = dcf_from_snapshot(snap, wacc=wacc, terminal_growth=terminal_growth,
                               estimate_wacc_flag=True)
    if result is None or not result.intrinsic_value_per_share:
        return result
    snap["dcf_intrinsic_value"] = result.intrinsic_value_per_share
    snap["dcf_margin_of_safety"] = result.margin_of_safety
    snap["wacc_used"] = result.wacc
    if result.wacc_detail is not None:
        snap["wacc_detail"] = {
            "cost_of_equity": result.wacc_detail.cost_of_equity,
            "cost_of_debt": result.wacc_detail.cost_of_debt,
            "beta": result.wacc_detail.beta,
            "equity_weight": result.wacc_detail.equity_weight,
            "debt_weight": result.wacc_detail.debt_weight,
            "coverage": result.wacc_detail.coverage,
        }
    return result


# Scenario deltas applied on top of the base case for Bull / Bear.
SCENARIOS = {
    "base": {"growth_delta": 0.00, "wacc_delta": 0.000, "tg_delta": 0.0000},
    "bull": {"growth_delta": +0.03, "wacc_delta": -0.010, "tg_delta": +0.0050},
    "bear": {"growth_delta": -0.03, "wacc_delta": +0.010, "tg_delta": -0.0050},
}


def dcf_scenarios(snap: dict | None) -> dict | None:
    """Base / Bull / Bear intrinsic values + margin of safety for the UI.

    Bull = faster revenue growth, lower WACC, higher terminal growth.
    Bear = slower revenue growth, higher WACC, lower terminal growth.
    Returns None when there is no free cash flow to project.
    """
    if not snap or not snap.get("free_cash_flow"):
        return None
    base = dcf_from_snapshot(snap, estimate_wacc_flag=True)
    if base is None:
        return None
    base_wacc = base.wacc
    base_growth = _growth(snap)
    base_tg = base.terminal_growth
    shares, net_debt = _shares_and_debt(snap)
    price = snap.get("current_price")
    fcf = snap.get("free_cash_flow")

    out = {}
    for name, d in SCENARIOS.items():
        r = compute_dcf(
            fcf, max(-0.5, base_growth + d["growth_delta"]),
            terminal_growth_rate=max(0.0, base_tg + d["tg_delta"]),
            wacc=base_wacc + d["wacc_delta"],
            shares_outstanding=shares, net_debt=net_debt, current_price=price,
            minority_interest=snap.get("minority_interest") or 0,
            preferred_stock=snap.get("preferred_stock") or 0,
        )
        out[name] = {
            "intrinsic_value_per_share": r.intrinsic_value_per_share,
            "margin_of_safety": r.margin_of_safety,
            "wacc": r.wacc,
            "terminal_growth": r.terminal_growth,
            "revenue_growth_rate": max(-0.5, base_growth + d["growth_delta"]),
        }
    return out


def dcf_bundle(snap: dict | None) -> dict | None:
    """Full DCF payload for the UI: base result + sensitivity grid + scenarios."""
    if not snap or not snap.get("free_cash_flow"):
        return None
    result = apply_dcf_to_snapshot(snap)
    grid = dcf_sensitivity_grid(snap)
    detail = snap.get("wacc_detail") or {}
    return {
        "intrinsic_value_per_share": result.intrinsic_value_per_share if result else None,
        "margin_of_safety": result.margin_of_safety if result else None,
        "wacc": result.wacc if result else None,
        "wacc_detail": detail,
        "terminal_growth": result.terminal_growth if result else None,
        "enterprise_value": result.enterprise_value if result else None,
        "equity_value": result.equity_value if result else None,
        "pv_of_projected_fcf": result.pv_of_projected_fcf if result else None,
        "pv_of_terminal_value": result.pv_of_terminal_value if result else None,
        "inputs": {
            "ttm_free_cash_flow": snap.get("free_cash_flow"),
            "revenue_growth_rate": _growth(snap),
            "net_debt": snap.get("net_debt") or 0.0,
            "market_cap": snap.get("market_cap"),
            "current_price": snap.get("current_price"),
        },
        "sensitivity": grid,
        "scenarios": dcf_scenarios(snap),
    }
