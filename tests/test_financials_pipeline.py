"""Tests for the fundamentals pipeline: 3-statement storage, ratio derivation,
DCF scenario grid, and company profiles (plan §8)."""
from __future__ import annotations

import pytest

from core.db import SQLiteStorage
from data_ingestion.fundamental_feeds.dcf_scenarios import (
    apply_dcf_to_snapshot, dcf_bundle, dcf_sensitivity_grid)
from data_ingestion.fundamental_feeds.yfinance_financials import (
    _extract, derive_ratios)


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


# ── storage round-trips ─────────────────────────────────────────────────
def test_financial_statements_roundtrip(db):
    db.write_financial_statements("AAPL", "income", [
        {"period": "2026-06-30", "total_revenue": 1e6, "net_income": 2e5},
        {"period": "2026-03-31", "total_revenue": 9e5, "net_income": 1e5},
    ])
    rows = db.query_financial_statements("AAPL", statement="income", quarters=8)
    assert len(rows) == 2
    assert rows[0]["period"] == "2026-06-30"
    assert rows[0]["data"]["total_revenue"] == 1e6
    # symbol normalization + per-statement filtering
    assert db.query_financial_statements("aapl", statement="balance") == []


def test_financial_statements_upsert_same_period(db):
    db.write_financial_statements("AAPL", "income",
                                  [{"period": "2026-06-30", "total_revenue": 1e6}])
    db.write_financial_statements("AAPL", "income",
                                  [{"period": "2026-06-30", "total_revenue": 2e6}])
    rows = db.query_financial_statements("AAPL", statement="income")
    assert len(rows) == 1
    assert rows[0]["data"]["total_revenue"] == 2e6


def test_company_profile_roundtrip(db):
    db.upsert_company_profile("MSFT", {"company_name": "Microsoft Corp.",
                                       "sector": "Technology"})
    prof = db.get_company_profile("msft")
    assert prof["company_name"] == "Microsoft Corp."
    assert prof["sector"] == "Technology"


def test_company_profile_depth_meta_roundtrip(db):
    """C6: segments/executives/news fold into the profile's JSON meta blob."""
    db.upsert_company_profile("MSFT", {
        "company_name": "Microsoft Corp.",
        "segments": [{"name": "Cloud", "revenue": 1000}],
        "executives": [{"name": "CEO", "title": "Chief Executive Officer"}],
        "news": [{"title": "MSFT beats", "publisher": "Reuters"}],
    })
    prof = db.get_company_profile("MSFT")
    assert prof["meta"]["segments"][0]["revenue"] == 1000
    assert prof["meta"]["executives"][0]["title"] == "Chief Executive Officer"
    assert prof["meta"]["news"][0]["publisher"] == "Reuters"
    # symbolic keys live in dedicated columns, not duplicated in meta
    assert "company_name" not in prof["meta"]


def test_llm_analysis_roundtrip(db):
    db.upsert_llm_analysis("AAPL", "NEWS", {"score": 0.4, "label": "bullish"}, "m1")
    db.upsert_llm_analysis("AAPL", "FUNDAMENTAL", {"rating": "HOLD"}, "m2")
    news = db.query_latest_llm_analysis("AAPL", kind="NEWS")
    assert news["verdict"]["label"] == "bullish"
    assert news["model"] == "m1"
    latest = db.query_latest_llm_analysis("AAPL")
    assert latest["kind"] == "FUNDAMENTAL"
    # kind-filtered lookup must be case-insensitive on symbol
    assert db.query_latest_llm_analysis("aapl", kind="news")["verdict"]["score"] == 0.4


def test_earnings_results_roundtrip(db):
    db.write_earnings_results("AAPL", [
        {"earnings_date": "2026-07-30", "eps_actual": 1.5, "eps_estimate": 1.3,
         "eps_surprise_pct": 0.1538},
    ])
    rows = db.query_earnings_results("AAPL")
    assert len(rows) == 1
    assert rows[0]["eps_surprise_pct"] == pytest.approx(0.1538, abs=1e-3)


def test_sentiment_series_buckets(db, monkeypatch):
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    monkeypatch.setattr("core.db._utc_now", lambda: base)
    for i, score in enumerate([0.5, -0.5, 0.1]):
        db.write_sentiment_event("X", "NEWS", score, source_weight=0.9,
                                 ts=base - timedelta(minutes=i * 2))
    series = db.sentiment_series("X", hours=1, bucket_minutes=60)
    assert len(series) == 1
    assert series[0]["volume"] == 3
    assert series[0]["score"] == pytest.approx((0.5 - 0.5 + 0.1) / 3, abs=1e-5)
    # separate buckets when window spans the hour boundary
    db.write_sentiment_event("X", "NEWS", 1.0, source_weight=0.9,
                             ts=base - timedelta(minutes=70))
    assert len(db.sentiment_series("X", hours=3, bucket_minutes=60)) == 2


# ── ratio derivation (pure logic) ───────────────────────────────────────
def test_derive_ratios_full_quarters():
    income = [
        {"period": "2026-06-30", "total_revenue": 120, "gross_profit": 60,
         "ebitda": 40, "net_income": 30, "eps_diluted": 1.2,
         "operating_income": 35, "interest_expense": 5},
        {"period": "2026-03-31", "total_revenue": 110, "gross_profit": 55,
         "ebitda": 35, "net_income": 25, "eps_diluted": 1.0,
         "operating_income": 30, "interest_expense": 5},
        {"period": "2025-12-31", "total_revenue": 105, "gross_profit": 52,
         "ebitda": 33, "net_income": 23, "eps_diluted": 0.9,
         "operating_income": 28, "interest_expense": 5},
        {"period": "2025-09-30", "total_revenue": 100, "gross_profit": 50,
         "ebitda": 30, "net_income": 20, "eps_diluted": 0.8,
         "operating_income": 25, "interest_expense": 5},
        {"period": "2025-06-30", "total_revenue": 100, "gross_profit": 50,
         "ebitda": 30, "net_income": 20, "eps_diluted": 0.8,
         "operating_income": 25, "interest_expense": 5},
    ]
    balance = [
        {"period": "2026-06-30", "total_assets": 500, "total_debt": 100,
         "cash_and_equivalents": 40, "shareholders_equity": 200,
         "current_assets": 150, "current_liabilities": 100},
        {"period": "2026-03-31", "total_assets": 480, "total_debt": 90,
         "cash_and_equivalents": 30, "shareholders_equity": 190,
         "current_assets": 140, "current_liabilities": 100},
        {"period": "2025-12-31", "total_assets": 460, "total_debt": 85,
         "cash_and_equivalents": 25, "shareholders_equity": 185,
         "current_assets": 135, "current_liabilities": 100},
        {"period": "2025-09-30", "total_assets": 450, "total_debt": 80,
         "cash_and_equivalents": 20, "shareholders_equity": 180,
         "current_assets": 130, "current_liabilities": 100},
        {"period": "2025-06-30", "total_assets": 450, "total_debt": 80,
         "cash_and_equivalents": 20, "shareholders_equity": 180,
         "current_assets": 130, "current_liabilities": 100},
    ]
    cashflow = [
        {"period": "2026-06-30", "operating_cash_flow": 50, "free_cash_flow": 40},
        {"period": "2026-03-31", "operating_cash_flow": 45, "free_cash_flow": 35},
        {"period": "2025-12-31", "operating_cash_flow": 42, "free_cash_flow": 32},
        {"period": "2025-09-30", "operating_cash_flow": 40, "free_cash_flow": 30},
        {"period": "2025-06-30", "operating_cash_flow": 40, "free_cash_flow": 30},
    ]
    rows = derive_ratios(income, balance, cashflow)
    assert [r["period"] for r in rows] == \
        ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
    latest = rows[-1]
    assert latest["revenue_yoy_growth"] == pytest.approx(0.2)      # 120/100-1
    assert latest["eps_yoy_growth"] == pytest.approx(0.5)          # 1.2/0.8-1
    assert latest["gross_margin"] == pytest.approx(0.5)
    assert latest["ebitda_margin"] == pytest.approx(40 / 120)
    assert latest["net_margin"] == pytest.approx(0.25)
    assert latest["debt_to_equity"] == pytest.approx(0.5)
    assert latest["current_ratio"] == pytest.approx(1.5)
    assert latest["interest_coverage_ratio"] == pytest.approx(7.0)
    assert latest["net_debt"] == pytest.approx(60.0)
    assert latest["fcf_conversion"] == pytest.approx(0.8)
    # first period has no prior-year → growth None
    assert rows[0]["revenue_yoy_growth"] is None


def test_extract_first_matching_label():
    import pandas as pd
    frame = pd.DataFrame({"2026-06-30": [10.0, 20.0]},
                         index=["Total Revenue", "Net Income"])
    out = _extract(frame, [("total_revenue", ("Total Revenue", "Revenue")),
                           ("net_income", ("Net Income",))])
    assert out == {"total_revenue": 10.0, "net_income": 20.0}


# ── DCF scenarios ───────────────────────────────────────────────────────
SNAP = {"free_cash_flow": 1000, "revenue_yoy_growth": 0.08,
        "market_cap": 50000, "current_price": 100, "net_debt": 200}


def test_dcf_sensitivity_grid_shape():
    grid = dcf_sensitivity_grid(SNAP)
    assert len(grid["waccs"]) == 5
    assert len(grid["terminal_growths"]) == 5
    assert len(grid["grid"]) == 5 and all(len(r) == 5 for r in grid["grid"])
    # higher WACC → lower intrinsic value everywhere
    for col in range(5):
        vals = [grid["grid"][r][col] for r in range(5)]
        assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_dcf_sensitivity_grid_no_fcf():
    grid = dcf_sensitivity_grid({"free_cash_flow": None})
    assert grid["grid"] == []


def test_apply_dcf_to_snapshot_stamps_fields():
    snap = dict(SNAP)
    result = apply_dcf_to_snapshot(snap)
    assert result is not None
    assert snap["dcf_intrinsic_value"] == result.intrinsic_value_per_share
    assert snap["dcf_margin_of_safety"] == result.margin_of_safety
    assert snap["wacc_used"] == 0.10
    # margin of safety sign sanity: intrinsic far below 100 → negative
    assert snap["dcf_margin_of_safety"] < 0


def test_dcf_bundle_shape():
    bundle = dcf_bundle(SNAP)
    assert bundle["intrinsic_value_per_share"] is not None
    assert "sensitivity" in bundle and "inputs" in bundle
    assert bundle["inputs"]["ttm_free_cash_flow"] == 1000
    assert dcf_bundle({"free_cash_flow": None}) is None
