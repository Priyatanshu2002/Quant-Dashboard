"""Tests for the pro-grade fundamental analysis layer (ratios + quality)."""
from __future__ import annotations

import pytest

from valuation.quality import altman_z, beneish_m, earnings_quality_flags, evaluate_quality, piotroski
from valuation.ratios import compute_ratios, flag_health


def _income_rows(n=8, rev=1000.0):
    rows = []
    for i in range(n):
        r = 100.0 * (i + 1)
        gross = 0.5 * r
        op = 0.25 * r
        interest = 0.02 * r
        pretax = op - interest
        tax = 0.21 * pretax
        rows.append({
            "period": f"2026-{(3 - i):02d}-30",
            "total_revenue": r, "cost_of_revenue": r - gross, "gross_profit": gross,
            "research_development": 0.05 * r, "selling_general_admin": 0.2 * r,
            "operating_income": op, "ebitda": op + 0.1 * r, "interest_expense": interest,
            "pretax_income": pretax, "income_tax": tax, "net_income": pretax - tax,
            "eps_diluted": pretax / 10, "shares_outstanding": 100.0,
        })
    return rows


def _balance_rows(n=5, ta=1000.0):
    rows = []
    for i in range(n):
        rows.append({
            "period": f"2026-{(3 - i):02d}-30",
            "total_assets": ta, "current_assets": 0.5 * ta,
            "cash_and_equivalents": 0.15 * ta, "accounts_receivable": 0.15 * ta,
            "inventory": 0.1 * ta, "net_ppe": 0.4 * ta,
            "total_liabilities": 0.5 * ta, "current_liabilities": 0.25 * ta,
            "accounts_payable": 0.1 * ta, "total_debt": 0.3 * ta,
            "shareholders_equity": 0.5 * ta, "retained_earnings": 0.2 * ta,
        })
    return rows


def _cashflow_rows(n=4):
    rows = []
    for i in range(n):
        r = 100.0 * (i + 1)
        rows.append({
            "period": f"2026-{(3 - i):02d}-30",
            "operating_cash_flow": 0.35 * r, "capex": 0.08 * r,
            "depreciation": 0.10 * r, "free_cash_flow": 0.27 * r,
            "change_in_working_capital": -0.02 * r, "dividends_paid": 0.05 * r,
        })
    return rows


# ── Ratio engine ──────────────────────────────────────────────────────
def test_ratios_margins_and_profitability():
    r = compute_ratios(_income_rows(), _balance_rows(), _cashflow_rows())
    assert r["gross_margin"] == pytest.approx(0.5)
    assert r["operating_margin"] == pytest.approx(0.25)
    assert r["net_margin"] > 0
    assert r["roa"] > 0
    assert r["roe"] > 0
    assert r["current_ratio"] == pytest.approx(2.0)      # 0.5/0.25
    assert r["debt_to_equity"] == pytest.approx(0.6)     # 0.3/0.5


def test_ratios_turnover_and_ccc():
    r = compute_ratios(_income_rows(), _balance_rows(), _cashflow_rows())
    assert r["receivables_turnover"] > 0
    assert r["days_sales_outstanding"] > 0
    assert r["inventory_turnover"] > 0
    assert "cash_conversion_cycle" in r
    # CCC = DSO + DIO - DPO
    assert r["cash_conversion_cycle"] == pytest.approx(
        r["days_sales_outstanding"] + r["days_inventory_hand"] - r["days_payables_outstanding"])


def test_ratios_dupont():
    r = compute_ratios(_income_rows(), _balance_rows(), _cashflow_rows())
    d = r["dupont"]
    # 3-step product equals ROE
    assert d["3_step"]["roe"] == pytest.approx(r["roe"], rel=1e-6)
    assert d["3_step"]["net_margin"] * d["3_step"]["asset_turnover"] * d["3_step"]["financial_leverage"] == pytest.approx(r["roe"])
    # 5-step product equals ROE
    d5 = d["5_step"]
    assert d5["roe"] == pytest.approx(r["roe"], rel=1e-6)


def test_ratios_valuation_multiples():
    r = compute_ratios(_income_rows(), _balance_rows(), _cashflow_rows(),
                       market_cap=50_000, price=500, shares_outstanding=100)
    assert r["price_to_earnings"] > 0
    assert r["price_to_sales"] > 0
    assert r["price_to_book"] > 0
    assert r["fcf_yield"] > 0


def test_health_flags_reports_healthy_unhealthy():
    # distressed firm: heavy debt, tiny current ratio
    inc = [{"total_revenue": 100, "cost_of_revenue": 70, "gross_profit": 30,
            "operating_income": 5, "ebitda": 6, "interest_expense": 4,
            "pretax_income": 1, "income_tax": 0.2, "net_income": 0.8,
            "shares_outstanding": 10}] * 4
    bal = [{"total_assets": 200, "current_assets": 40, "cash_and_equivalents": 5,
            "accounts_receivable": 20, "inventory": 10, "net_ppe": 120,
            "total_liabilities": 160, "current_liabilities": 60,
            "accounts_payable": 15, "total_debt": 120, "shareholders_equity": 40,
            "retained_earnings": 10}] * 4
    cf = [{"operating_cash_flow": 8, "capex": 10, "depreciation": 5,
           "free_cash_flow": -2, "change_in_working_capital": 2, "dividends_paid": 0}] * 4
    r = compute_ratios(inc, bal, cf)
    flags = flag_health(r)
    flagged = {f["ratio"]: f["healthy"] for f in flags}
    assert flagged.get("debt_to_equity") is False       # 120/40 = 3 > 1
    assert flagged.get("current_ratio") is False         # 40/60 < 1.5
    assert flagged.get("interest_coverage") is False     # 5/4 = 1.25


# ── Quality models ────────────────────────────────────────────────────
def test_piotroski_strong_firm_scores_high():
    # Improving fundamentals: newer quarters show higher ROA, positive CFO>NI,
    # lower debt, higher gross margin, higher asset turnover → high F-score.
    def _period(revenue, ni, gross, debt, assets, cfo, shares=100):
        return {"total_revenue": revenue, "gross_profit": gross, "net_income": ni,
                "operating_income": ni * 1.3, "shares_outstanding": shares}
    inc = [_period(100, 8, 50, 100, 500, 40) for _ in range(4)]  # prior 4 quarters
    inc += [_period(200, 40, 130, 40, 600, 55) for _ in range(4)]  # current 4 quarters
    bal = [{"total_assets": 500, "current_assets": 250, "current_liabilities": 150,
            "total_debt": 100, "shareholders_equity": 300, "retained_earnings": 100,
            "accounts_receivable": 80, "inventory": 40}] * 4
    bal += [{"total_assets": 600, "current_assets": 330, "current_liabilities": 150,
             "total_debt": 40, "shareholders_equity": 400, "retained_earnings": 200,
             "accounts_receivable": 90, "inventory": 40}] * 4
    cf = [{"operating_cash_flow": 40}] * 4 + [{"operating_cash_flow": 55}] * 4
    res = piotroski(inc, bal, cf)
    assert res["score"] is not None
    assert res["score"] >= 7


def test_altman_z_safe_zone():
    inc = _income_rows()
    bal = _balance_rows()
    z = altman_z(inc, bal, market_cap=50_000)
    assert z["z"] is not None
    assert z["zone"] in ("safe", "grey", "distress")


def test_altman_z_distressed():
    inc = [{"total_revenue": 50, "operating_income": -10, "net_income": -15}] * 4
    bal = [{"total_assets": 300, "current_assets": 40, "current_liabilities": 200,
            "retained_earnings": -100, "total_liabilities": 260,
            "shareholders_equity": 40}] * 2
    z = altman_z(inc, bal, market_cap=10)
    assert z["zone"] == "distress"


def test_beneish_m_returns_structure():
    res = beneish_m(_income_rows(), _balance_rows(), _cashflow_rows())
    assert "m_score" in res
    assert "manipulator" in res


def test_quality_flags_detect_cfo_under_ni():
    inc = [{"total_revenue": 100, "net_income": 20, "operating_income": 30,
            "cost_of_revenue": 50}] * 8
    bal = [{"total_assets": 500, "accounts_receivable": 100, "current_assets": 200,
            "current_liabilities": 150, "inventory": 40}] * 6
    cf = [{"operating_cash_flow": 5}] * 4   # CFO far below NI
    flags = earnings_quality_flags(inc, bal, cf)
    assert any("CFO" in f["flag"] for f in flags)


def test_evaluate_quality_returns_all():
    res = evaluate_quality(_income_rows(), _balance_rows(), _cashflow_rows(), market_cap=50_000)
    assert set(res.keys()) == {"piotroski", "altman_z", "beneish_m", "earnings_quality_flags"}


# ── Relative valuation ────────────────────────────────────────────────
from valuation.relative import compare_to_peers, compute_multiples, mismatch_check

METRICS = {"revenue": 1000, "net_income": 100, "ebitda": 180, "ebit": 150,
           "fcff": 120, "equity": 400, "total_debt": 200, "cash": 50,
           "dividends_paid": 20, "eps": 2.5, "growth": 0.10}


def test_compute_multiples():
    m = compute_multiples(METRICS, market_cap=2000, price=50, shares_outstanding=40,
                          forward_eps=3.0)
    assert m["enterprise_value"] == pytest.approx(2150)          # 2000+200-50
    assert m["price_to_earnings"] == pytest.approx(20)
    assert m["ev_to_ebitda"] == pytest.approx(2150 / 180, rel=1e-3)
    assert m["ev_to_sales"] == pytest.approx(2150 / 1000, rel=1e-3)
    assert m["fcf_yield"] == pytest.approx(120 / 2000, rel=1e-3)
    assert m["trailing_pe"] == pytest.approx(20)
    assert m["forward_pe"] == pytest.approx(50 / 3, rel=1e-3)
    assert m["peg_ratio"] == pytest.approx((50 / 3) / 10, rel=1e-3)
    assert m["dividend_yield"] == pytest.approx((20 / 40) / 50, rel=1e-3)


def test_compare_to_peers_flags_expensive():
    peers = {
        "P1": {"price_to_earnings": 10, "ev_to_ebitda": 5, "price_to_sales": 1.0},
        "P2": {"price_to_earnings": 12, "ev_to_ebitda": 6, "price_to_sales": 1.2},
        "P3": {"price_to_earnings": 11, "ev_to_ebitda": 5.5, "price_to_sales": 1.1},
    }
    comp = {"price_to_earnings": 25, "ev_to_ebitda": 12, "price_to_sales": 3.0}
    res = compare_to_peers(comp, peers)
    labels = {c["multiple"]: c["label"] for c in res["comparisons"]}
    assert all(labels[k] == "expensive" for k in labels)
    assert res["peers"] == 3


def test_mismatch_check_flags_value_trap():
    q = {"piotroski": {"score": 2}, "altman_z": {"zone": "distress"}}
    m = {"trailing_pe": 8, "price_to_book": 0.8}
    flags = mismatch_check(q, m)
    assert any("value trap" in f["signal"].lower() for f in flags)


# ── FCFF projection DCF ───────────────────────────────────────────────
from valuation.fcff_model import FCFFInputs, fcff_scenarios, fcff_sensitivity, project_fcff


def test_fcff_projection_structure():
    inp = FCFFInputs(revenue=1000, ebit=250, debt=200, cash=50, shares_outstanding=100,
                     price=10, sales_to_capital=1.5, growth_next=0.05, growth_2_5=0.05)
    r = project_fcff(inp)
    assert "error" not in r
    assert len(r["projection"]) == 10
    assert r["intrinsic_value_per_share"] is not None
    assert r["intrinsic_value_per_share"] > 0
    # Equity bridge ties out
    b = r["equity_bridge"]
    assert b["equity_value"] == pytest.approx(b["operating_assets"] - b["debt"]
                                              + b["cash"] - b["minority_interest"] - b["preferred_stock"] - b["options_value"])
    assert r["intrinsic_value_per_share"] == pytest.approx(b["equity_value"] / 100, rel=1e-2)
    assert r["terminal_value"] > 0
    assert r["pv_of_terminal_value"] > 0


def test_fcff_reproduces_reference_template():
    # Almarai (Damodaran fcffsimple) → expected value/share ≈ 7.19, hugely below price
    inp = FCFFInputs(
        revenue=21765.4, ebit=3060.9, tax_rate=0.175, marginal_tax_rate=0.25,
        invested_capital=36730.8, sales_to_capital=1.7085,
        debt=45063, cash=19000, non_operating_assets=21119, minority_interest=1558,
        shares_outstanding=4315, price=72.28, riskfree=0.0458,
        initial_wacc=0.07055, stable_wacc=0.0881, stable_growth=0.0458, stable_roc=0.0881,
        growth_next=0.05, growth_2_5=0.05, target_margin=0.140631,
    )
    r = project_fcff(inp)
    assert r["intrinsic_value_per_share"] == pytest.approx(7.19, abs=0.5)
    assert r["margin_of_safety"] < 0            # extremely overvalued


def test_fcff_scenarios_ordering():
    inp = FCFFInputs(revenue=1000, ebit=250, debt=200, cash=50, shares_outstanding=100,
                     price=10, sales_to_capital=1.5, growth_next=0.05, growth_2_5=0.05)
    sc = fcff_scenarios(inp)
    assert sc["bull"]["intrinsic_value_per_share"] > sc["base"]["intrinsic_value_per_share"]
    assert sc["bear"]["intrinsic_value_per_share"] < sc["base"]["intrinsic_value_per_share"]


def test_fcff_sensitivity_shape():
    inp = FCFFInputs(revenue=1000, ebit=250, debt=200, cash=50, shares_outstanding=100,
                     price=10, sales_to_capital=1.5)
    s = fcff_sensitivity(inp)
    assert len(s["waccs"]) == 5 and len(s["growths"]) == 5
    assert len(s["grid"]) == 5 and all(len(row) == 5 for row in s["grid"])
    # higher WACC → lower value
    for col in range(5):
        vals = [s["grid"][r][col] for r in range(5)]
        for a, b in zip(vals, vals[1:]):
            assert b <= a


# ── End-to-end build_model against the live dev DB (integration smoke) ─
import pytest as _pt


def test_build_model_live_smoke():
    """Build the pro model for a symbol in the dev DB; skip if no data."""
    try:
        from core.db import get_storage
        from valuation.cfa_model import build_model
    except Exception:  # pragma: no cover
        _pt.skip("storage unavailable")
    db = get_storage()
    m = None
    for sym in ("MSFT", "AAPL", "NVDA"):
        m = build_model(db, sym)
        if m:
            break
    if m is None:
        _pt.skip("no symbol with 3-statement data in dev DB")
    assert "discount_rates" in m and "wacc" in m["discount_rates"]
    assert "dcf" in m and "intrinsic_value_per_share" in m["dcf"]
    assert "ratios" in m and len(m["ratios"]) > 5
    assert "quality" in m and "piotroski" in m["quality"]
    assert "relative" in m and "multiples" in m["relative"]
    assert "scenarios" in m and set(m["scenarios"].keys()) == {"base", "bull", "bear"}
