"""Pro-grade financial ratio engine (CFA Book 2 framework).

Computes the full ratio catalogue a professional checks before transacting:
activity/turnover, liquidity, solvency, profitability, cash-flow ratios,
growth, risk, and the DuPont (3-step + 5-step) decomposition of ROE.

Conventions (CFA): balance-sheet items in turnover ratios use the AVERAGE of
beginning and ending balances; turnover days use a 365-day year. Flow items
(income/cash-flow) are TTM (sum of the last `n` quarters).

Inputs: three statement lists, oldest → newest, each row a dict with the
canonical keys produced by yfinance_financials / three_statement_parser
(total_revenue, operating_income, ebitda, net_income, interest_expense, ...).
All inputs optional → missing ratios are simply omitted.
"""
from __future__ import annotations

from typing import Any

_DAYS = 365.0


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _ttm(rows: list[dict], key: str, n: int = 4) -> float | None:
    vals = [_f(r.get(key)) for r in rows[-n:]]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _latest(rows: list[dict], key: str) -> float | None:
    for r in reversed(rows):
        v = _f(r.get(key))
        if v is not None:
            return v
    return None


def _avg(latest: float | None, prior: float | None) -> float | None:
    if latest is None and prior is None:
        return None
    vals = [v for v in (latest, prior) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_balance(rows: list[dict], key: str) -> float | None:
    """Average of the latest balance period and the same-period-last-year."""
    latest = _latest(rows, key)
    prior = _latest(rows[:-4], key) if len(rows) >= 5 else None
    return _avg(latest, prior)


def compute_ratios(income: list[dict], balance: list[dict], cashflow: list[dict],
                   market_cap: float | None = None, price: float | None = None,
                   shares_outstanding: float | None = None) -> dict:
    """Compute the full ratio catalogue. Returns a dict of ratios (missing → absent)."""
    rev = _ttm(income, "total_revenue")
    cogs = _ttm(income, "cost_of_revenue")
    gp = _ttm(income, "gross_profit")
    ebit = _ttm(income, "operating_income") or _ttm(income, "ebitda")
    ebitda = _ttm(income, "ebitda")
    ni = _ttm(income, "net_income")
    pretax = _ttm(income, "pretax_income")
    tax = _ttm(income, "income_tax")
    int_exp = _ttm(income, "interest_expense")
    cfo = _ttm(cashflow, "operating_cash_flow")
    capex = _ttm(cashflow, "capex")
    da = _ttm(cashflow, "depreciation")
    dividends = _ttm(cashflow, "dividends_paid")
    fcff = _ttm(cashflow, "free_cash_flow")

    # Balance (latest + prior-year average)
    ta = _avg_balance(balance, "total_assets")
    ca = _latest(balance, "current_assets")
    cl = _latest(balance, "current_liabilities")
    cash = _latest(balance, "cash_and_equivalents")
    ar = _avg_balance(balance, "accounts_receivable")
    inv = _avg_balance(balance, "inventory")
    ap = _avg_balance(balance, "accounts_payable")
    td = _latest(balance, "total_debt")
    eq = _avg_balance(balance, "shareholders_equity")
    nfa = _avg_balance(balance, "net_ppe")

    out: dict = {}

    # ── Activity / turnover ──
    if rev and ar:
        out["receivables_turnover"] = rev / ar
        out["days_sales_outstanding"] = _DAYS / out["receivables_turnover"]
    if cogs and inv:
        out["inventory_turnover"] = cogs / inv
        out["days_inventory_hand"] = _DAYS / out["inventory_turnover"]
    if cogs and ap:
        out["payables_turnover"] = cogs / ap
        out["days_payables_outstanding"] = _DAYS / out["payables_turnover"]
    if rev and ta:
        out["total_asset_turnover"] = rev / ta
    if rev and nfa:
        out["fixed_asset_turnover"] = rev / nfa
    if rev and ca is not None and cl is not None:
        wc = ca - cl
        out["working_capital_turnover"] = rev / wc if wc else None
    # Cash conversion cycle
    if all(k in out for k in ("days_sales_outstanding", "days_inventory_hand",
                              "days_payables_outstanding")):
        out["cash_conversion_cycle"] = (out["days_sales_outstanding"]
                                        + out["days_inventory_hand"]
                                        - out["days_payables_outstanding"])

    # ── Liquidity ──
    if ca is not None and cl:
        out["current_ratio"] = ca / cl
    quick = (cash or 0.0) + (ar or 0.0)
    if cl:
        out["quick_ratio"] = quick / cl
        out["cash_ratio"] = (cash or 0.0) / cl
    # Defensive interval ≈ (cash+AR) / avg daily (COGS+SG&A+R&D+dep)
    sga_rd = (_ttm(income, "selling_general_admin") or 0.0) + (_ttm(income, "research_development") or 0.0)
    daily_exp = ((cogs or 0.0) + sga_rd + (da or 0.0)) / _DAYS
    if daily_exp > 0 and (cash is not None or ar is not None):
        out["defensive_interval_days"] = ((cash or 0.0) + (ar or 0.0)) / daily_exp

    # ── Solvency ──
    if td is not None and eq:
        out["debt_to_equity"] = td / eq
    if td is not None and ta:
        out["debt_to_assets"] = td / ta
    if td is not None and eq and td + eq > 0:
        out["debt_to_capital"] = td / (td + eq)
    if ta and eq:
        out["financial_leverage"] = ta / eq
    if ebit and int_exp:
        out["interest_coverage"] = ebit / int_exp
    if td is not None and ebitda:
        out["debt_to_ebitda"] = td / ebitda
    if td is not None and cash is not None:
        out["net_debt"] = td - cash

    # ── Profitability ──
    if rev:
        if gp is not None:
            out["gross_margin"] = gp / rev
        if ebit is not None:
            out["operating_margin"] = ebit / rev
        if ebitda is not None:
            out["ebitda_margin"] = ebitda / rev
        if pretax is not None:
            out["pretax_margin"] = pretax / rev
        if ni is not None:
            out["net_margin"] = ni / rev
    if ni is not None and ta:
        out["roa"] = ni / ta
    if ebit is not None and ta:
        out["operating_roa"] = ebit / ta
    if eq:
        if ni is not None:
            out["roe"] = ni / eq
        if cfo is not None:
            out["cash_roe"] = cfo / eq
    # ROIC = NOPAT / avg long-term capital (debt + equity)
    if ebit is not None and tax is not None and pretax and pretax > 0:
        eff_tax = tax / pretax
        nopat = ebit * (1 - eff_tax)
        lt_capital = (td or 0.0) + (eq or 0.0)
        if lt_capital > 0:
            out["roic"] = nopat / lt_capital
        out["nopat"] = nopat
        out["effective_tax_rate"] = eff_tax
    elif ebit is not None:
        nopat = ebit * (1 - 0.21)
        lt_capital = (td or 0.0) + (eq or 0.0)
        if lt_capital > 0:
            out["roic"] = nopat / lt_capital
        out["nopat"] = nopat

    # ── Cash-flow ratios ──
    if rev and cfo is not None:
        out["cash_flow_to_revenue"] = cfo / rev
    if ta and cfo is not None:
        out["cash_return_on_assets"] = cfo / ta
    if ni and cfo is not None:
        out["cash_to_income"] = cfo / ni
    if td and cfo is not None:
        out["debt_coverage"] = cfo / td
    if cfo is not None and capex is not None:
        out["capex_pct_cfo"] = capex / cfo if cfo else None

    # ── Growth ──
    if eq and ni is not None and dividends is not None and eq > 0:
        roe = ni / eq
        payout = (dividends / ni) if ni else None
        retention = (1 - payout) if payout is not None else None
        if retention is not None:
            out["sustainable_growth"] = roe * retention
            out["dividend_payout"] = payout
    # YoY revenue growth (newest 4 quarters vs prior 4)
    revs = [_f(r.get("total_revenue")) for r in income]
    revs = [v for v in revs if v is not None]
    if len(revs) >= 8:
        prior, recent = sum(revs[-8:-4]), sum(revs[-4:])
        if prior and prior > 0:
            out["revenue_yoy_growth"] = recent / prior - 1

    # ── DuPont ──
    dupont = _dupont(ni, rev, ebit, pretax, tax, ta, eq)
    if dupont:
        out["dupont"] = dupont

    # ── Valuation multiples (need market data) ──
    if market_cap is not None or (price is not None and shares_outstanding is not None):
        mcap = market_cap or (price * shares_outstanding)
        if rev and mcap:
            out["price_to_sales"] = mcap / rev
        if ni and mcap and ni > 0:
            out["price_to_earnings"] = mcap / ni
            out["earnings_yield"] = ni / mcap
        if eq is not None and mcap:
            out["price_to_book"] = mcap / eq
        if fcff and mcap:
            out["fcf_yield"] = fcff / mcap
            out["price_to_fcf"] = mcap / fcff
        if dividends is not None and price:
            dps = dividends / shares_outstanding if shares_outstanding else None
            if dps:
                out["dividend_yield"] = dps / price

    # Clean out any accidental None values
    return {k: v for k, v in out.items() if v is not None}


def _dupont(ni, rev, ebit, pretax, tax, ta, eq):
    """3-step and 5-step DuPont decomposition of ROE."""
    if not all(v is not None for v in (ni, rev, ta, eq)) or not rev or not ta or not eq:
        return None
    margin = ni / rev
    turnover = rev / ta
    leverage = ta / eq
    roe_3 = margin * turnover * leverage
    d3 = {"net_margin": margin, "asset_turnover": turnover,
          "financial_leverage": leverage, "roe": roe_3}

    d5 = None
    if ebit is not None and pretax and ebit and pretax > 0:
        tax_burden = ni / pretax
        interest_burden = pretax / ebit
        ebit_margin = ebit / rev
        roe_5 = tax_burden * interest_burden * ebit_margin * turnover * leverage
        d5 = {"tax_burden": tax_burden, "interest_burden": interest_burden,
              "ebit_margin": ebit_margin, "asset_turnover": turnover,
              "financial_leverage": leverage, "roe": roe_5}
    return {"3_step": d3, "5_step": d5}


HEALTHY_THRESHOLDS = {
    "current_ratio": ("1.5 - 2.0 healthy", 1.5, 2.0),
    "quick_ratio": (">= 1 healthy", 1.0, None),
    "debt_to_equity": ("< 1 healthy (higher OK for capital-intensive)", 0.0, 1.0),
    "interest_coverage": ("higher better", 2.0, None),
    "debt_to_ebitda": ("< 3.5x reasonable", 0.0, 3.5),
    "roe": ("> 15% good", 0.15, None),
    "roa": ("> 5% good", 0.05, None),
    "price_to_earnings": ("15-25 reasonable (mature)", 15.0, 25.0),
    "price_to_book": ("< 3 acceptable", 0.0, 3.0),
    "price_to_sales": ("< 1 favorable", 0.0, 1.0),
    "fcf_yield": ("> 5% attractive", 0.05, None),
}


def flag_health(ratios: dict) -> list[dict]:
    """Flag ratios against healthy benchmarks. Returns a list of {ratio, value, healthy, note}."""
    flags = []
    for key, (note, lo, hi) in HEALTHY_THRESHOLDS.items():
        v = ratios.get(key)
        if v is None:
            continue
        healthy = True
        if lo is not None and v < lo:
            healthy = False
        if hi is not None and v > hi:
            healthy = False
        flags.append({"ratio": key, "value": round(v, 3), "healthy": healthy, "benchmark": note})
    return flags
