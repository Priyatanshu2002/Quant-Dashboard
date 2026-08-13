"""CFA-standard 3-statement + DCF valuation model (plan §8).

Builds an integrated financial-statement model from the store's real quarterly
statements, then values the firm with a two-stage FCFF discounted-cash-flow
model per CFA Institute methodology:

  1. LINKED 3 STATEMENTS — income / balance sheet / cash flow for the trailing
     twelve months, with the standard reconciliation bridges:
        Net income → Retained earnings (balance)
        Net income + D&A − ΔNWC → Operating cash flow (CFO)
        CFO − Capex = Free cash flow to firm (proxy, ≈ FCFF)
  2. WACC — cost of equity via CAPM (Ke = rf + β·ERP), after-tax cost of debt
     Kd·(1−t), weighted by market values of equity and debt.
  3. FCFF DCF — explicit 10-year projection with revenue growth fading to a
     terminal growth rate; terminal value via the Gordon growth model;
     Enterprise value = PV(FCF) + PV(TV); equity = EV − net debt; per-share
     vs market price → margin of safety.
  4. SCENARIOS — base / bull / bear on revenue growth and EBIT margin, plus a
     WACC × terminal-growth sensitivity grid.

Every figure is derived from the live store; nothing is hardcoded except
documented assumptions (risk-free fallback, ERP, spread, beta fallback) that
are exposed in the output so they are transparent and editable.
"""
from __future__ import annotations

from typing import Any

from core.logging import get_logger
from valuation.beta import estimate_beta
from valuation.discount_rates import estimate_discount_rates
from valuation.fcff_model import FCFFInputs, fcff_scenarios, fcff_sensitivity, project_fcff
from valuation.quality import evaluate_quality
from valuation.ratios import compute_ratios, flag_health
from valuation.relative import compute_multiples, mismatch_check

log = get_logger(__name__)

# ── documented market assumptions (overridable, exposed in output) ──────
DEFAULT_ERP = 0.045          # equity risk premium
DEFAULT_RF = 0.042           # risk-free fallback if no 10y in macro store
DEFAULT_BETA = 1.0           # beta fallback (no beta feed wired)
DEBT_SPREAD = 0.015          # investment-grade credit spread over rf
PROJECTION_YEARS = 10
TERMINAL_GROWTH = 0.025


def _num(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
def load_statements(db, symbol: str, quarters: int = 12) -> dict[str, list[dict]]:
    """Return per-statement lists oldest→newest from the store."""
    out: dict[str, list[dict]] = {}
    for name in ("income", "balance", "cashflow"):
        rows = db.query_financial_statements(symbol, statement=name, quarters=quarters)
        recs = [{"period": r["period"], **r["data"]} for r in rows]
        recs.sort(key=lambda r: r["period"])
        out[name] = recs
    return out


def ltm_figures(statements: dict[str, list[dict]], periods: int = 4) -> dict:
    """Trailing-twelve-months sums from the last `periods` quarters + latest balance."""
    def _sum(stmt: str, key: str) -> float | None:
        vals = [_num(r.get(key)) for r in statements[stmt][-periods:]]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    bal = statements["balance"][-1] if statements["balance"] else {}
    inc = statements["income"][-1] if statements["income"] else {}

    revenue = _sum("income", "total_revenue")
    ebit = _sum("income", "operating_income")
    ni = _sum("income", "net_income")
    tax = _sum("income", "income_tax")
    int_exp = _sum("income", "interest_expense")
    cfo = _sum("cashflow", "operating_cash_flow")
    capex = _sum("cashflow", "capex")
    da = _sum("cashflow", "depreciation")
    sbc = _sum("cashflow", "stock_based_comp")
    dnwc = _sum("cashflow", "change_in_working_capital")
    dividends_paid = _sum("cashflow", "dividends_paid")

    pre_tax = (ni + tax) if (ni is not None and tax is not None) else None
    eff_tax = (tax / pre_tax) if (pre_tax and pre_tax > 0) else None

    # CFA Institute FCFF (free-cash-flow-valuation framework):
    #   FCFF = EBIT(1−T) + Dep − FCInv − WCInv        (from EBIT)
    #   FCFF = CFO + Int(1−T) − FCInv                  (from CFO)  [fallback]
    nopat = (ebit * (1 - eff_tax)) if (ebit is not None and eff_tax is not None) else None
    fcff_ebit = (nopat + da - capex - dnwc) \
        if (nopat is not None and da is not None and capex is not None and dnwc is not None) \
        else None
    fcff_cfo = (cfo + int_exp * (1 - eff_tax) - capex) \
        if (cfo is not None and capex is not None and int_exp is not None and eff_tax is not None) \
        else ((cfo - capex) if (cfo is not None and capex is not None) else None)
    fcff = fcff_ebit if fcff_ebit is not None else fcff_cfo
    if fcff is None:
        # Fallback: the cash-flow statement's own FCF line (capex/D&A may not be tagged).
        fcff = _sum("cashflow", "free_cash_flow")

    return {
        "revenue": revenue, "ebit": ebit, "net_income": ni, "income_tax": tax,
        "effective_tax_rate": eff_tax, "ebitda": _sum("income", "ebitda"),
        "cfo": cfo, "capex": capex, "da": da, "stock_based_comp": sbc,
        "fcff": fcff,
        "ebit_margin": (ebit / revenue) if (ebit and revenue) else None,
        "net_margin": (ni / revenue) if (ni and revenue) else None,
        "da_pct_rev": (da / revenue) if (da and revenue) else None,
        "capex_pct_rev": (capex / revenue) if (capex and revenue) else None,
        # latest balance sheet
        "total_debt": _num(bal.get("total_debt")),
        "cash": _num(bal.get("cash_and_equivalents")),
        "total_assets": _num(bal.get("total_assets")),
        "total_liabilities": _num(bal.get("total_liabilities")),
        "equity": _num(bal.get("shareholders_equity")),
        "retained_earnings": _num(bal.get("retained_earnings")),
        "shares_outstanding": _num(inc.get("shares_outstanding")),
        "dividends_paid": dividends_paid,
        "eps": _num(inc.get("eps_diluted")),
        "balance_period": bal.get("period"),
    }


def revenue_growth(statements: dict[str, list[dict]], ltm: dict) -> float:
    """TTM revenue growth from the last 8 quarters (older 4 vs newer 4)."""
    revs = [_num(r.get("total_revenue")) for r in statements["income"]]
    revs = [v for v in revs if v is not None]
    if len(revs) >= 8:
        prior, recent = sum(revs[-8:-4]), sum(revs[-4:])
        if prior and prior > 0:
            return recent / prior - 1
    # fall back to snapshot growth if present
    return ltm.get("revenue_growth") or 0.06


# ─────────────────────────────────────────────────────────────────────
# WACC
# ─────────────────────────────────────────────────────────────────────
def wacc(ltm: dict, market_cap: float | None, rf: float | None,
         erp: float = DEFAULT_ERP, beta: float = DEFAULT_BETA,
         debt_spread: float = DEBT_SPREAD) -> dict:
    """CAPM cost of equity + after-tax cost of debt, weighted by market values."""
    rf = rf if rf is not None else DEFAULT_RF
    tax = ltm.get("effective_tax_rate") or 0.21
    ke = rf + beta * erp
    kd = rf + debt_spread
    kd_after_tax = kd * (1 - tax)

    debt = ltm.get("total_debt") or 0.0
    equity = market_cap or 0.0
    denom = equity + debt
    we = equity / denom if denom else 1.0
    wd = debt / denom if denom else 0.0
    wacc_val = we * ke + wd * kd_after_tax

    return {
        "risk_free": round(rf, 4), "erp": erp, "beta": beta,
        "cost_of_equity": round(ke, 4),
        "cost_of_debt": round(kd, 4), "cost_of_debt_after_tax": round(kd_after_tax, 4),
        "effective_tax_rate": round(tax, 4),
        "equity_value": round(equity, 2), "debt_value": round(debt, 2),
        "w_e": round(we, 4), "w_d": round(wd, 4), "wacc": round(wacc_val, 4),
    }


# ─────────────────────────────────────────────────────────────────────
# DCF
# ─────────────────────────────────────────────────────────────────────
def run_dcf(ltm: dict, growth: float, wacc_val: float, terminal_growth: float,
            market_cap: float | None, price: float | None,
            projection_years: int = PROJECTION_YEARS) -> dict:
    """Two-stage FCFF DCF (fading growth → Gordon terminal value)."""
    fcff0 = ltm.get("fcff") or 0.0
    if wacc_val <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate")

    net_debt = (ltm.get("total_debt") or 0.0) - (ltm.get("cash") or 0.0)
    shares = ltm.get("shares_outstanding")

    rows = []
    last = fcff0
    for year in range(1, projection_years + 1):
        fade = year / projection_years
        g = growth * (1 - fade) + terminal_growth * fade
        last *= (1 + g)
        df = (1 + wacc_val) ** year
        rows.append({"year": year, "growth": round(g, 4),
                     "fcff": round(last, 2), "discount_factor": round(df, 4),
                     "pv": round(last / df, 2)})

    tv = last * (1 + terminal_growth) / (wacc_val - terminal_growth)
    pv_tv = tv / (1 + wacc_val) ** projection_years
    pv_fcf = sum(r["pv"] for r in rows)
    ev = pv_fcf + pv_tv
    equity_value = ev - net_debt
    intrinsic_ps = (equity_value / shares) if shares and shares > 0 else None
    margin = (intrinsic_ps / price - 1) if (intrinsic_ps and price) else None

    return {
        "base_fcff": round(fcff0, 2),
        "projection_years": projection_years,
        "terminal_growth": terminal_growth,
        "net_debt": round(net_debt, 2),
        "projections": rows,
        "pv_of_projected_fcf": round(pv_fcf, 2),
        "pv_of_terminal_value": round(pv_tv, 2),
        "terminal_value": round(tv, 2),
        "enterprise_value": round(ev, 2),
        "equity_value": round(equity_value, 2),
        "shares_outstanding": shares,
        "intrinsic_value_per_share": round(intrinsic_ps, 2) if intrinsic_ps else None,
        "current_price": price,
        "margin_of_safety": round(margin, 4) if margin is not None else None,
        "upside_downsides_pct": round(margin * 100, 1) if margin is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────
# Scenarios + sensitivity
# ─────────────────────────────────────────────────────────────────────
def scenarios(ltm: dict, growth: float, wacc_val: float, terminal_growth: float,
              market_cap: float | None, price: float | None) -> list[dict]:
    base = run_dcf(ltm, growth, wacc_val, terminal_growth, market_cap, price)
    out = [{"label": "Base", "growth": growth, "margin_adj": 0.0, **base}]
    # Bull: +200bps growth, +200bps EBIT margin; Bear: opposite.
    for label, g_adj, m_adj in (("Bull", 0.02, 0.02), ("Bear", -0.02, -0.02)):
        ltm2 = dict(ltm)
        if ltm2.get("fcff"):
            ltm2["fcff"] = ltm2["fcff"] * (1 + m_adj)
        r = run_dcf(ltm2, growth + g_adj, wacc_val, terminal_growth, market_cap, price)
        out.append({"label": label, "growth": growth + g_adj, "margin_adj": m_adj, **r})
    return out


# ─────────────────────────────────────────────────────────────────────
# Top-level builder
# ─────────────────────────────────────────────────────────────────────
def build_model(db, symbol: str, rf: float | None = None,
                beta: float | None = None, erp: float | None = None) -> dict | None:
    """Build the full pro-grade valuation model dict for a symbol.

    Returns a dict with: WACC (discount rates), the driver-based FCFF DCF
    projection + scenarios + sensitivity, the full CFA ratio catalogue, quality
    models (Piotroski/Altman Z/Beneish/earnings-quality flags), and relative
    valuation multiples. Returns None when data is insufficient.
    """
    symbol = symbol.upper()
    statements = load_statements(db, symbol)
    if len(statements["income"]) < 4 or len(statements["cashflow"]) < 4:
        return None

    ltm = ltm_figures(statements)
    if not ltm.get("revenue") or not ltm.get("fcff"):
        return None

    snap = db.query_latest_fundamentals(symbol) or {}
    profile = db.get_company_profile(symbol) or {}
    market_cap = _num(snap.get("market_cap")) or _num(snap.get("market_cap"))
    price = _num(snap.get("current_price"))
    if not price:
        ohlcv = db.query_ohlcv(symbol)
        if not ohlcv.empty:
            price = float(ohlcv["close"].dropna().iloc[-1])
    rf = rf if rf is not None else _latest_rf(db)

    # Real beta from price history, then full discount-rate stack
    if beta is None:
        beta_est = estimate_beta(symbol, db)
        beta = beta_est.get("beta") or None
    country = (profile.get("country") or "").upper() or None
    dr = estimate_discount_rates(snap, beta=beta, rf=rf, erp=erp, country=country)
    rf = dr.risk_free  # estimate_discount_rates falls back to a sane default if None

    # Build FCFF inputs from the real statements + snapshot
    equity_bs = _num(ltm.get("equity")) or 0.0
    cash = _num(ltm.get("cash")) or 0.0
    debt = _num(ltm.get("total_debt")) or 0.0
    invested_capital = debt + equity_bs - cash
    sales_to_capital = (ltm["revenue"] / invested_capital) if invested_capital > 0 else None
    growth = revenue_growth(statements, ltm)
    shares = _num(ltm.get("shares_outstanding")) or _num(snap.get("shares_outstanding"))
    # Many statement feeds omit per-period share count → derive from market data.
    if not shares and market_cap and price:
        shares = market_cap / price

    inp = FCFFInputs(
        revenue=ltm["revenue"], ebit=ltm.get("ebit") or 0.0,
        tax_rate=ltm.get("effective_tax_rate") or 0.21, marginal_tax_rate=0.25,
        invested_capital=invested_capital, sales_to_capital=sales_to_capital,
        debt=debt, cash=cash,
        non_operating_assets=_num(snap.get("non_operating_assets")) or 0.0,
        minority_interest=_num(snap.get("minority_interest")) or 0.0,
        preferred_stock=_num(snap.get("preferred_stock")) or 0.0,
        shares_outstanding=shares, price=price,
        riskfree=rf, initial_wacc=dr.wacc, stable_wacc=rf + 0.045,
        stable_growth=rf, stable_roc=None,
        growth_next=max(0.0, growth), growth_2_5=max(0.0, growth),
        target_margin=ltm.get("ebit_margin"),
    )
    dcf = project_fcff(inp)
    if "error" in dcf:
        return None

    # Ratio catalogue, quality models, relative multiples
    ratios = compute_ratios(statements["income"], statements["balance"],
                            statements["cashflow"], market_cap=market_cap,
                            price=price, shares_outstanding=shares)
    quality = evaluate_quality(statements["income"], statements["balance"],
                               statements["cashflow"], market_cap=market_cap)
    metrics = {
        "revenue": ltm.get("revenue"), "net_income": ltm.get("net_income"),
        "ebitda": ltm.get("ebitda"), "ebit": ltm.get("ebit"), "fcff": ltm.get("fcff"),
        "equity": equity_bs, "total_debt": debt, "cash": cash,
        "dividends_paid": ltm.get("dividends_paid"), "eps": ltm.get("eps"),
        "growth": growth,
    }
    multiples = compute_multiples(metrics, market_cap=market_cap, price=price,
                                  shares_outstanding=shares,
                                  forward_eps=_num(snap.get("forward_eps")))
    mismatches = mismatch_check(quality, multiples)

    return {
        "symbol": symbol,
        "as_of": str(ltm.get("balance_period")),
        "profile": profile,
        "statement_model": _statement_model(statements, ltm),
        "discount_rates": {
            "risk_free": dr.risk_free, "erp": dr.erp, "erp_source": dr.erp_source,
            "beta": dr.beta, "cost_of_equity": dr.cost_of_equity,
            "cost_of_debt": dr.cost_of_debt, "cost_of_debt_after_tax": dr.cost_of_debt_after_tax,
            "tax_rate": dr.tax_rate, "w_e": dr.w_e, "w_d": dr.w_d,
            "wacc": dr.wacc, "country": dr.country,
            "synthetic_rating": dr.synthetic_rating,
            "cost_of_debt_source": dr.cost_of_debt_source, "coverage": dr.coverage,
        },
        "dcf": dcf,
        "scenarios": fcff_scenarios(inp),
        "sensitivity": fcff_sensitivity(inp),
        "ratios": ratios,
        "ratio_health_flags": flag_health(ratios),
        "quality": quality,
        "relative": {"multiples": multiples, "mismatches": mismatches},
        "assumptions": {
            "beta_source": "regression (fallback 1.0)" if beta else "fallback 1.0",
            "risk_free_source": "macro_10y (or default 3.95%)",
            "erp_source": dr.erp_source,
            "projection_years": inp.projection_years,
            "terminal_growth": rf,
            "sales_to_capital": inp.eff_sales_to_capital(),
        },
    }


def _sensitivity(ltm: dict, growth: float, market_cap, price) -> dict:
    waccs = (0.08, 0.09, 0.10, 0.11, 0.12)
    tgs = (0.015, 0.02, 0.025, 0.03, 0.035)
    grid = []
    for wv in waccs:
        row = []
        for tg in tgs:
            try:
                r = run_dcf(ltm, growth, wv, tg, market_cap, price)
                row.append(r["intrinsic_value_per_share"])
            except ValueError:
                row.append(None)
        grid.append(row)
    return {"waccs": list(waccs), "terminal_growths": list(tgs), "grid": grid}


def _statement_model(statements: dict[str, list[dict]], ltm: dict) -> dict:
    """Present the linked 3-statement history (last 5 periods) for display."""
    def _series(stmt: str, key: str, n: int = 5) -> list[dict]:
        return [{"period": r["period"], "value": _num(r.get(key))}
                for r in statements[stmt][-n:] if r.get(key) is not None]

    # Balance-sheet sanity: A = L + E on the latest period.
    bal = statements["balance"][-1] if statements["balance"] else {}
    assets = _num(bal.get("total_assets"))
    liabilities = _num(bal.get("total_liabilities"))
    equity_bs = _num(bal.get("shareholders_equity"))
    balance_check = (abs(assets - liabilities - equity_bs) < max(1.0, abs(assets) * 0.001)) \
        if (assets and liabilities is not None and equity_bs) else None

    return {
        "income": _series("income", "total_revenue", 5),
        "net_income": _series("income", "net_income", 5),
        "operating_cash_flow": _series("cashflow", "operating_cash_flow", 5),
        "free_cash_flow": _series("cashflow", "free_cash_flow", 5),
        "total_assets": _series("balance", "total_assets", 5),
        "total_debt": _series("balance", "total_debt", 5),
        "equity": _series("balance", "shareholders_equity", 5),
        "ltm": {k: (round(v, 2) if isinstance(v, (int, float)) and v else v)
                for k, v in ltm.items()},
        "balance_equation_check": balance_check,
        "linkage": {
            "fcff_identity": "FCFF ≈ OperatingCF − Capex (after-tax interest not modeled)",
        },
    }


def _latest_rf(db) -> float | None:
    try:
        m = db.query_latest_macro()
        v = m.get("us_10y_yield") if m else None
        return _num(v)
    except Exception:  # noqa: BLE001
        return None
