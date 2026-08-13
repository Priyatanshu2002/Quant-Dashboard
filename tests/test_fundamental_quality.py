"""Tests for the data-integrity core (plan Phase 1/2 hardening).

Covers the correctness fixes: real EBITDA (EBIT + D&A), proper total debt,
NOPAT/ROIC, the balance-sheet identity check, CAPM-based WACC, real share
count, minority/preferred in equity value, and Base/Bull/Bear scenarios.
"""
from __future__ import annotations

import pytest

from data_ingestion.fundamental_feeds.dcf_calculator import dcf_from_snapshot
from data_ingestion.fundamental_feeds.dcf_scenarios import apply_dcf_to_snapshot, dcf_bundle, dcf_scenarios
from data_ingestion.fundamental_feeds.three_statement_parser import _finalize_snapshot, _total_debt
from valuation.wacc import estimate_wacc


# ── SEC parser correctness (pure, no network) ──────────────────────────
def test_ebitda_is_ebit_plus_da():
    snap = _finalize_snapshot({"operating_income": 40_000, "depreciation_amortization": 10_000,
                               "revenue": 200_000, "gross_profit": 90_000})
    assert snap["ebitda"] == 50_000                    # 40k + 10k, NOT gross profit
    assert snap["ebitda_margin"] == pytest.approx(0.25)
    assert snap["operating_margin"] == pytest.approx(0.20)


def test_ebitda_falls_back_to_ebit_only_when_no_da():
    snap = _finalize_snapshot({"operating_income": 40_000, "revenue": 200_000})
    assert snap["ebitda"] == 40_000


def test_total_debt_sums_components_not_liabilities():
    snap = _finalize_snapshot({
        "short_term_debt": 5_000, "current_portion_long_term_debt": 3_000,
        "long_term_debt": 60_000, "finance_lease": 4_000, "operating_lease": 2_000,
        "cash_and_equivalents": 20_000, "shareholders_equity": 100_000,
        "operating_income": 15_000, "revenue": 100_000,
    })
    assert snap["total_debt"] == 74_000                # 5k+3k+60k+4k+2k
    assert snap["net_debt"] == 54_000                  # 74k - 20k
    assert snap["debt_to_equity"] == pytest.approx(0.74)


def test_total_debt_none_when_no_components():
    assert _total_debt({"total_liabilities": 500}) is None  # liabilities is NOT debt


def test_roic_from_nopat_over_invested_capital():
    snap = _finalize_snapshot({
        "operating_income": 20_000, "pretax_income": 20_000, "income_tax": 5_000,
        "total_debt": 50_000, "shareholders_equity": 150_000, "cash_and_equivalents": 20_000,
        "revenue": 100_000, "gross_profit": 60_000,
    })
    # tax rate = 5000/20000 = 0.25; NOPAT = 20000 * 0.75 = 15000
    assert snap["nopat"] == pytest.approx(15_000)
    # invested capital = 50k + 150k - 20k = 180k
    assert snap["roic"] == pytest.approx(15000 / 180_000)


def test_balance_sheet_identity_checked():
    ok = _finalize_snapshot({"total_assets": 500_000, "total_liabilities": 200_000,
                             "shareholders_equity": 300_000, "operating_income": 1,
                             "revenue": 1})
    assert ok["balance_ok"] is True
    bad = _finalize_snapshot({"total_assets": 500_000, "total_liabilities": 200_000,
                              "shareholders_equity": 200_000, "operating_income": 1,
                              "revenue": 1})
    assert bad["balance_ok"] is False


# ── WACC (CAPM) ────────────────────────────────────────────────────────
def test_wacc_full_coverage_capm():
    snap = {"market_cap": 900_000, "total_debt": 100_000,
            "interest_expense": 5_000, "pretax_income": 50_000, "income_tax": 10_500}
    w = estimate_wacc(snap, beta=1.2)
    assert w.coverage == "full"
    assert w.beta == 1.2
    # re = 4.2% + 1.2*4.5% = 9.6%; rd = 5%; weights 0.9/0.1; tax 0.21
    expected = 0.9 * 0.096 + 0.1 * 0.05 * (1 - 0.21)
    assert w.wacc == pytest.approx(expected, abs=1e-9)
    assert w.cost_of_debt_measured is True


def test_wacc_default_when_no_capital_structure():
    w = estimate_wacc({"market_cap": 500})
    assert w.coverage == "default"
    assert w.wacc == pytest.approx(0.10)


def test_wacc_uses_statement_tax_rate():
    snap = {"market_cap": 800, "total_debt": 200, "interest_expense": 10,
            "pretax_income": 100, "income_tax": 20}
    w = estimate_wacc(snap, beta=1.0)
    assert w.tax_rate == pytest.approx(0.20)


# ── DCF share count / minority / preferred ─────────────────────────────
def test_dcf_uses_reported_shares_not_derived():
    # Real shares outrank market_cap/price
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 50_000, "current_price": 100,  # implies 500 shares
            "shares_outstanding": 1_000, "net_debt": 0}
    r = dcf_from_snapshot(snap, wacc=0.10, estimate_wacc_flag=False)
    # EV ≈ FCF/growth DCF; per-share must be ~EV/1000 not /500
    assert r.equity_value > 0
    assert r.intrinsic_value_per_share == pytest.approx(r.equity_value / 1_000)


def test_dcf_subtracts_minority_and_preferred():
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 50_000, "current_price": 100, "net_debt": 200,
            "minority_interest": 300, "preferred_stock": 400}
    r = dcf_from_snapshot(snap, wacc=0.10, estimate_wacc_flag=False)
    # equity_value = EV - 200 - 300 - 400
    expected_ev = r.enterprise_value
    assert r.equity_value == pytest.approx(expected_ev - 200 - 300 - 400)


def test_dcf_uses_estimated_wacc_from_market():
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 900_000, "total_debt": 100_000,
            "interest_expense": 5_000, "pretax_income": 50_000, "income_tax": 10_500,
            "current_price": 100}
    r = dcf_from_snapshot(snap, estimate_wacc_flag=True)
    assert r.wacc_detail is not None
    assert r.wacc_detail.coverage == "full"
    # Estimated WACC (~8.9%) should differ from the 10% default
    assert r.wacc < 0.10


# ── Scenarios ──────────────────────────────────────────────────────────
def test_dcf_scenarios_ordering():
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 50_000, "current_price": 100, "net_debt": 200}
    sc = dcf_scenarios(snap)
    assert sc is not None
    assert sc["bull"]["intrinsic_value_per_share"] > sc["base"]["intrinsic_value_per_share"]
    assert sc["bear"]["intrinsic_value_per_share"] < sc["base"]["intrinsic_value_per_share"]


def test_apply_dcf_to_snapshot_stamps_wacc_detail():
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 900_000, "total_debt": 100_000, "interest_expense": 5_000,
            "pretax_income": 50_000, "income_tax": 10_500, "current_price": 100}
    result = apply_dcf_to_snapshot(snap)
    assert snap["wacc_used"] == pytest.approx(result.wacc)
    assert "wacc_detail" in snap
    assert snap["wacc_detail"]["coverage"] == "full"


def test_dcf_bundle_includes_scenarios():
    snap = {"free_cash_flow": 1_000, "revenue_yoy_growth": 0.08,
            "market_cap": 50_000, "current_price": 100, "net_debt": 200}
    bundle = dcf_bundle(snap)
    assert "scenarios" in bundle
    assert set(bundle["scenarios"].keys()) == {"base", "bull", "bear"}
