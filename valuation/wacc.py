"""WACC estimation for DCF — CAPM-based, from the latest fundamental snapshot.

Industry-standard WACC = (E/V)·re + (D/V)·rd·(1−t), where:
  * cost of equity  re = rf + β·ERP            (Capital Asset Pricing Model)
  * cost of debt    rd = effective interest rate = interest_expense / total_debt
        (fallback: rf + credit-spread when interest expense is missing)
  * market value of equity E = market_cap; market value of debt D = total_debt
  * tax rate t derived from the statements (income_tax / pretax_income), else 21%.

Everything degrades gracefully: if the snapshot lacks the inputs, a conservative
default WACC is returned rather than raising. Beta is optional (callers who have
price history can supply a regression beta; default 1.0 = market-average).
"""
from __future__ import annotations

from dataclasses import dataclass

# Market assumptions (configurable). 10Y UST ~4.2% + mature-market ERP ~4.5-5%.
DEFAULT_RISK_FREE_RATE = 0.042
DEFAULT_EQUITY_RISK_PREMIUM = 0.045
DEFAULT_CREDIT_SPREAD = 0.015
DEFAULT_BETA = 1.0
DEFAULT_WACC = 0.10


@dataclass
class WACCEstimate:
    wacc: float
    cost_of_equity: float
    cost_of_debt: float
    equity_weight: float | None
    debt_weight: float | None
    beta: float
    tax_rate: float
    # Data-coverage flags so callers can tell how much was estimated vs measured
    beta_measured: bool = False
    cost_of_debt_measured: bool = False
    coverage: str = ""  # "full" | "partial" | "default"


def estimate_wacc(
    snap: dict,
    beta: float | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    credit_spread: float = DEFAULT_CREDIT_SPREAD,
    default_wacc: float = DEFAULT_WACC,
) -> WACCEstimate:
    """Estimate WACC from a fundamental snapshot. Never raises."""
    tax = _tax_rate(snap)

    # Cost of equity via CAPM
    beta_used = beta if beta is not None else DEFAULT_BETA
    beta_measured = beta is not None
    cost_of_equity = risk_free_rate + beta_used * equity_risk_premium

    # Cost of debt: effective rate if we can measure it, else rf + spread
    ie, td = snap.get("interest_expense"), snap.get("total_debt")
    cost_of_debt_measured = bool(ie and td and td > 0)
    if cost_of_debt_measured:
        cost_of_debt = ie / td
    else:
        cost_of_debt = risk_free_rate + credit_spread

    # Weights from market values
    mcap = snap.get("market_cap")
    equity_weight = debt_weight = None
    if mcap and td is not None and (mcap + td) > 0:
        equity_weight = mcap / (mcap + td)
        debt_weight = td / (mcap + td)

    coverage = "default"
    if mcap and td is not None and cost_of_debt_measured:
        coverage = "full"
    elif mcap and td is not None:
        coverage = "partial"

    if equity_weight is not None and debt_weight is not None:
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax)
    else:
        # No capital-structure data → conservative default
        wacc = default_wacc
        coverage = "default"

    return WACCEstimate(
        wacc=wacc,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        beta=beta_used,
        tax_rate=tax,
        beta_measured=beta_measured,
        cost_of_debt_measured=cost_of_debt_measured,
        coverage=coverage,
    )


def _tax_rate(snap: dict) -> float:
    pretax, tax = snap.get("pretax_income"), snap.get("income_tax")
    if pretax and tax is not None:
        try:
            t = tax / pretax
            if 0 <= t <= 1:
                return float(t)
        except (TypeError, ZeroDivisionError):
            pass
    return 0.21
