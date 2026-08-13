"""DCF intrinsic value calculator — fading-growth two-stage model.

Fading growth DCF:
- Growth fades from revenue_growth_rate to terminal_growth_rate over N years
- Discount all FCF projections at WACC
- Terminal value = (FCF_N * (1 + g_terminal)) / (WACC - g_terminal)
- Intrinsic equity value = EV - net_debt - minority_interest - preferred_stock
- Intrinsic value per share = equity_value / shares_outstanding

Production-readiness:
- WACC can be supplied directly (backward compatible) OR estimated from market
  data via valuation/wacc.estimate_wacc (CAPM) when estimate_wacc=True.
- Share count uses the real reported shares (SEC/yfinance) when available,
  falling back to market_cap / price.
"""
from __future__ import annotations

from dataclasses import dataclass

from valuation.wacc import WACCEstimate, estimate_wacc


@dataclass
class DCFResult:
    intrinsic_value_per_share: float | None
    enterprise_value: float
    pv_of_projected_fcf: float
    pv_of_terminal_value: float
    wacc: float
    terminal_growth: float
    margin_of_safety: float | None = None  # (intrinsic - price) / price
    equity_value: float | None = None
    wacc_detail: WACCEstimate | None = None


def _fade_growth(start_growth: float, terminal_growth: float,
                 projection_years: int) -> list[float]:
    return [start_growth * (1 - y / projection_years)
            + terminal_growth * (y / projection_years)
            for y in range(1, projection_years + 1)]


def compute_dcf(
    ttm_free_cash_flow: float,
    revenue_growth_rate: float,
    terminal_growth_rate: float = 0.025,
    wacc: float = 0.10,
    projection_years: int = 10,
    shares_outstanding: float | None = None,
    net_debt: float = 0,
    current_price: float | None = None,
    minority_interest: float = 0,
    preferred_stock: float = 0,
    wacc_detail: WACCEstimate | None = None,
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
    for year, growth in enumerate(_fade_growth(revenue_growth_rate, terminal_growth_rate,
                                               projection_years), start=1):
        last_fcf *= (1 + growth)
        projected_fcf.append(last_fcf / (1 + wacc) ** year)

    terminal_value = (last_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / (1 + wacc) ** projection_years
    ev = sum(projected_fcf) + pv_terminal
    equity_value = ev - net_debt - minority_interest - preferred_stock
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
        equity_value=equity_value,
        wacc_detail=wacc_detail,
    )


def dcf_from_snapshot(snap: dict, wacc: float | None = None,
                      terminal_growth: float = 0.025,
                      estimate_wacc_flag: bool = True,
                      beta: float | None = None) -> DCFResult | None:
    """Run DCF off a fundamental snapshot (free_cash_flow, growth, price).

    Uses the real reported share count when present (shares_outstanding /
    shares_outstanding_diluted), else market_cap / price. When estimate_wacc_flag
    is True and the snapshot carries enough data, WACC is derived from the market
    (valuation.wacc.estimate_wacc); otherwise the explicit `wacc` (or 0.10) is used.
    """
    fcf = snap.get("free_cash_flow")
    if not fcf:
        return None
    growth = snap.get("revenue_yoy_growth") or 0.06

    # Real share count > derived
    shares = snap.get("shares_outstanding_diluted") or snap.get("shares_outstanding")
    if not shares:
        mcap, price = snap.get("market_cap"), snap.get("current_price")
        if mcap and price:
            shares = mcap / price

    wacc_detail = None
    if estimate_wacc_flag and wacc is None:
        wacc_detail = estimate_wacc(snap, beta=beta)
        wacc_used = wacc_detail.wacc
    else:
        wacc_used = wacc if wacc is not None else 0.10

    return compute_dcf(
        ttm_free_cash_flow=fcf,
        revenue_growth_rate=growth,
        terminal_growth_rate=terminal_growth,
        wacc=wacc_used,
        shares_outstanding=shares,
        net_debt=snap.get("net_debt") or 0,
        current_price=snap.get("current_price"),
        minority_interest=snap.get("minority_interest") or 0,
        preferred_stock=snap.get("preferred_stock") or 0,
        wacc_detail=wacc_detail,
    )


if __name__ == "__main__":
    import json
    import sys

    from data_ingestion.fundamental_feeds.yfinance_earnings import refresh_info_snapshot
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    snap = refresh_info_snapshot(ticker)
    result = dcf_from_snapshot(snap)
    print(json.dumps(result.__dict__, indent=2, default=str))
