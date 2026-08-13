"""Pro-grade FCFF discounted-cash-flow model (Damodaran fcffsimple template).

Projects the full firm value over an explicit horizon using the driver-based
model a professional uses:

  Revenue(t)   = Revenue(t-1) * (1 + g(t))            [growth fades to stable]
  EBIT(t)      = Revenue(t) * OperatingMargin(t)      [margin moves to target]
  NOPAT(t)     = EBIT(t) * (1 - Tax(t))               [effective -> marginal tax]
  InvestCap(t) = InvestCap(t-1) + Reinvestment(t)
  Reinvestment = ΔRevenue / Sales-to-Capital ratio    [efficiency of growth]
  FCFF(t)      = NOPAT(t) - Reinvestment(t)
  ROC(t)       = NOPAT(t) / InvestCap(t)              [value-creation check vs WACC]
  WACC(t)      = fades from initial to stable (≈ rf + 4.5%) in years 6-10

  Terminal value (Gordon, stable-growth discipline):
    stable_reinvest = g_stable / ROC_stable
    TerminalCF = NOPAT(10) * (1 - stable_reinvest)
    TV = TerminalCF / (WACC_stable - g_stable)         [g_stable = risk-free]

  Equity bridge:
    OpAssets = PV(FCFF 1..10) + PV(TV)
    Equity   = OpAssets - Debt - Minority - Preferred + Cash + Non-op assets
             - options - (failure adjustment)
    Value/share = Equity / shares_outstanding

The projection returns a full table (growth, revenue, margin, EBIT, tax,
reinvestment, FCFF, ROC, WACC, discount factor, PV) plus terminal value and the
equity bridge — the same shape as the reference spreadsheet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class FCFFInputs:
    revenue: float
    ebit: float                     # operating income (EBIT), base year
    tax_rate: float = 0.21          # effective tax rate
    marginal_tax_rate: float = 0.25
    invested_capital: float | None = None
    sales_to_capital: float | None = None
    debt: float = 0.0
    cash: float = 0.0
    non_operating_assets: float = 0.0
    minority_interest: float = 0.0
    preferred_stock: float = 0.0
    shares_outstanding: float | None = None
    price: float | None = None
    riskfree: float = 0.0395
    initial_wacc: float = 0.10
    stable_wacc: float | None = None     # default rf + 0.045
    stable_growth: float | None = None   # default riskfree
    stable_roc: float | None = None      # default = stable_wacc (excess returns fade)
    growth_next: float = 0.06
    growth_2_5: float = 0.06
    target_margin: float | None = None   # default = base margin
    margin_convergence_year: int = 5
    projection_years: int = 10
    probability_of_failure: float = 0.0
    failure_proceeds_pct: float = 0.5
    options_value: float = 0.0

    @property
    def base_margin(self) -> float:
        return (self.ebit / self.revenue) if self.revenue else 0.0

    def eff_sales_to_capital(self) -> float:
        if self.sales_to_capital:
            return self.sales_to_capital
        ic = self.eff_invested_capital()
        return (self.revenue / ic) if ic else 1.5

    def eff_invested_capital(self) -> float:
        if self.invested_capital:
            return self.invested_capital
        return self.debt + (self.revenue * (1 - self.base_margin)) - self.cash


def project_fcff(inp: FCFFInputs) -> dict:
    """Run the FCFF projection. Returns the full model dict (or an error dict)."""
    if not inp.revenue or inp.revenue <= 0:
        return {"error": "no revenue"}
    stable_g = inp.stable_growth if inp.stable_growth is not None else inp.riskfree
    stable_w = inp.stable_wacc if inp.stable_wacc is not None else (inp.riskfree + 0.045)
    if stable_w <= stable_g:
        return {"error": "stable WACC must exceed stable growth"}
    target_margin = inp.target_margin if inp.target_margin is not None else inp.base_margin
    sales_cap = inp.eff_sales_to_capital()

    # Revenue growth path: years 1..5 explicit, years 6..10 fade to stable.
    growth = {}
    for y in range(1, inp.projection_years + 1):
        if y <= 5:
            growth[y] = inp.growth_next if y == 1 else inp.growth_2_5
        else:
            t = (y - 5) / max(1, (inp.projection_years - 5))
            growth[y] = inp.growth_2_5 * (1 - t) + stable_g * t

    # Operating margin path: converge base -> target by margin_convergence_year.
    def _margin(y: int) -> float:
        c = inp.margin_convergence_year
        if c <= 1:
            return target_margin
        return inp.base_margin + (target_margin - inp.base_margin) * min(1.0, (y - 1) / max(1, c - 1))

    def _tax(y: int) -> float:
        # effective -> marginal, converging over years 6..10
        if y <= 5:
            return inp.tax_rate
        t = (y - 5) / 5.0
        return inp.tax_rate * (1 - t) + inp.marginal_tax_rate * t

    def _wacc(y: int) -> float:
        if y <= 5:
            return inp.initial_wacc
        t = (y - 5) / 5.0
        return inp.initial_wacc * (1 - t) + stable_w * t

    revenue = inp.revenue
    invested_cap = inp.eff_invested_capital()
    rows = []
    cum_df = 1.0
    pv_sum = 0.0
    nopat_last = None

    for y in range(1, inp.projection_years + 1):
        g = growth[y]
        revenue_prev = revenue
        revenue = revenue_prev * (1 + g)
        ebit = revenue * _margin(y)
        tax = _tax(y)
        nopat = ebit * (1 - tax)
        d_rev = revenue - revenue_prev
        reinvestment = d_rev / sales_cap if sales_cap else 0.0
        invested_cap += reinvestment
        fcff = nopat - reinvestment
        roc = (nopat / invested_cap) if invested_cap else None
        wacc_y = _wacc(y)
        cum_df *= (1 + wacc_y)
        df = 1.0 / cum_df
        pv = fcff * df
        pv_sum += pv
        nopat_last = nopat
        rows.append({
            "year": y, "growth": round(g, 4), "revenue": round(revenue, 2),
            "ebit_margin": round(ebit / revenue, 4), "ebit": round(ebit, 2),
            "tax_rate": round(tax, 4), "nopat": round(nopat, 2),
            "reinvestment": round(reinvestment, 2),
            "invested_capital": round(invested_cap, 2), "fcff": round(fcff, 2),
            "roc": round(roc, 4) if roc else None, "wacc": round(wacc_y, 4),
            "discount_factor": round(df, 4), "pv": round(pv, 2),
        })

    # Terminal value (stable-growth discipline)
    stable_roc = inp.stable_roc if inp.stable_roc is not None else stable_w
    stable_reinvest = (stable_g / stable_roc) if stable_roc else 0.0
    terminal_cf = (nopat_last * (1 - stable_reinvest)) if nopat_last else 0.0
    tv = terminal_cf / (stable_w - stable_g)
    pv_tv = tv / cum_df
    op_assets = pv_sum + pv_tv

    # Failure adjustment (distress): value = DCF*(1-pi) + proceeds*pi
    if inp.probability_of_failure and inp.probability_of_failure > 0:
        proceeds = op_assets * inp.failure_proceeds_pct
        op_assets = op_assets * (1 - inp.probability_of_failure) + proceeds * inp.probability_of_failure

    equity = (op_assets - inp.debt - inp.minority_interest - inp.preferred_stock
              + inp.cash + inp.non_operating_assets - inp.options_value)
    value_per_share = (equity / inp.shares_outstanding) if inp.shares_outstanding else None
    margin = (value_per_share / inp.price - 1) if (value_per_share and inp.price) else None

    return {
        "base_revenue": round(inp.revenue, 2), "base_ebit": round(inp.ebit, 2),
        "base_margin": round(inp.base_margin, 4),
        "sales_to_capital": round(sales_cap, 4),
        "riskfree": inp.riskfree, "initial_wacc": inp.initial_wacc,
        "stable_wacc": round(stable_w, 4), "stable_growth": round(stable_g, 4),
        "stable_roc": round(stable_roc, 4), "stable_reinvestment": round(stable_reinvest, 4),
        "projection": rows,
        "pv_of_projected_fcff": round(pv_sum, 2),
        "terminal_cash_flow": round(terminal_cf, 2),
        "terminal_value": round(tv, 2),
        "pv_of_terminal_value": round(pv_tv, 2),
        "value_of_operating_assets": round(op_assets, 2),
        "equity_bridge": {
            "operating_assets": round(op_assets, 2), "debt": round(inp.debt, 2),
            "minority_interest": round(inp.minority_interest, 2),
            "preferred_stock": round(inp.preferred_stock, 2),
            "cash": round(inp.cash, 2), "non_operating_assets": round(inp.non_operating_assets, 2),
            "options_value": round(inp.options_value, 2),
            "equity_value": round(equity, 2),
            "probability_of_failure": inp.probability_of_failure,
        },
        "shares_outstanding": inp.shares_outstanding,
        "current_price": inp.price,
        "intrinsic_value_per_share": round(value_per_share, 2) if value_per_share else None,
        "margin_of_safety": round(margin, 4) if margin is not None else None,
    }


def fcff_scenarios(inp: FCFFInputs) -> dict:
    """Base / Bull / Bear scenarios on growth, margin, sales-to-capital, WACC."""
    base = project_fcff(inp)
    out = {"base": base}
    for label, d in (("bull", {"growth": 0.02, "margin": 0.02, "sales_cap": 0.0, "wacc": -0.01}),
                     ("bear", {"growth": -0.02, "margin": -0.02, "sales_cap": 0.0, "wacc": 0.01})):
        i2 = FCFFInputs(
            **{k: getattr(inp, k) for k in FCFFInputs.__dataclass_fields__})
        i2.growth_next = max(0.0, inp.growth_next + d["growth"])
        i2.growth_2_5 = max(0.0, inp.growth_2_5 + d["growth"])
        i2.target_margin = (inp.target_margin if inp.target_margin is not None else inp.base_margin) + d["margin"]
        i2.initial_wacc = inp.initial_wacc + d["wacc"]
        if i2.stable_wacc is None:
            i2.stable_wacc = inp.riskfree + 0.045
        out[label] = project_fcff(i2)
    return out


def fcff_sensitivity(inp: FCFFInputs, waccs=(0.08, 0.09, 0.10, 0.11, 0.12),
                     growths=(0.03, 0.05, 0.07, 0.09, 0.11)) -> dict:
    """WACC × revenue-growth grid of intrinsic value per share."""
    grid = []
    for w in waccs:
        row = []
        for g in growths:
            i2 = FCFFInputs(**{k: getattr(inp, k) for k in FCFFInputs.__dataclass_fields__})
            i2.initial_wacc = w
            i2.growth_next = g
            i2.growth_2_5 = g
            r = project_fcff(i2)
            row.append(r.get("intrinsic_value_per_share"))
        grid.append(row)
    return {"waccs": list(waccs), "growths": list(growths), "grid": grid}
