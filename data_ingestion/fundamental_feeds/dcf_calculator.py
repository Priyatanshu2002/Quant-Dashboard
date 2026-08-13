"""DCF intrinsic value calculator — fading-growth two-stage model.

Fading growth DCF:
- Growth fades from revenue_growth_rate to terminal_growth_rate over N years
- Discount all FCF projections at WACC
- Terminal value = (FCF_N * (1 + g_terminal)) / (WACC - g_terminal)
- Intrinsic equity value = EV - net_debt
- Intrinsic value per share = equity_value / shares_outstanding
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DCFResult:
    intrinsic_value_per_share: float | None
    enterprise_value: float
    pv_of_projected_fcf: float
    pv_of_terminal_value: float
    wacc: float
    terminal_growth: float
    margin_of_safety: float | None = None  # (intrinsic - price) / price


def compute_dcf(
    ttm_free_cash_flow: float,
    revenue_growth_rate: float,
    terminal_growth_rate: float = 0.025,
    wacc: float = 0.10,
    projection_years: int = 10,
    shares_outstanding: float | None = None,
    net_debt: float = 0,
    current_price: float | None = None,
) -> DCFResult:
    """
    >>> r = compute_dcf(1000, 0.10, shares_outstanding=100, net_debt=500)
    >>> r.intrinsic_value_per_share > 0
    True
    """
    if wacc <= terminal_growth_rate:
        raise ValueError("WACC must exceed terminal growth rate")

    projected_fcf: list[float] = []
    last_fcf = ttm_free_cash_flow

    for year in range(1, projection_years + 1):
        fade = year / projection_years
        growth = revenue_growth_rate * (1 - fade) + terminal_growth_rate * fade
        last_fcf *= (1 + growth)
        projected_fcf.append(last_fcf / (1 + wacc) ** year)

    terminal_value = (last_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / (1 + wacc) ** projection_years
    ev = sum(projected_fcf) + pv_terminal
    equity_value = ev - net_debt
    intrinsic_ps = equity_value / shares_outstanding if shares_outstanding else None

    margin = None
    if intrinsic_ps and current_price:
        margin = intrinsic_ps / current_price - 1

    return DCFResult(
        intrinsic_value_per_share=intrinsic_ps,
        enterprise_value=ev,
        pv_of_projected_fcf=sum(projected_fcf),
        pv_of_terminal_value=pv_terminal,
        wacc=wacc,
        terminal_growth=terminal_growth_rate,
        margin_of_safety=margin,
    )


def dcf_from_snapshot(snap: dict, wacc: float = 0.10,
                      terminal_growth: float = 0.025) -> DCFResult | None:
    """Run DCF off a fundamental snapshot (free_cash_flow, growth, price)."""
    fcf = snap.get("free_cash_flow")
    if not fcf:
        return None
    growth = snap.get("revenue_yoy_growth") or 0.06
    shares = None
    mcap, price = snap.get("market_cap"), snap.get("current_price")
    if mcap and price:
        shares = mcap / price
    return compute_dcf(
        ttm_free_cash_flow=fcf,
        revenue_growth_rate=growth,
        terminal_growth_rate=terminal_growth,
        wacc=wacc,
        shares_outstanding=shares,
        net_debt=snap.get("net_debt") or 0,
        current_price=price,
    )


if __name__ == "__main__":
    import json
    import sys
    import yfinance as yf
    from data_ingestion.fundamental_feeds.yfinance_earnings import refresh_info_snapshot
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    snap = refresh_info_snapshot(ticker)
    result = dcf_from_snapshot(snap)
    print(json.dumps(result.__dict__, indent=2, default=str))
