"""Tests for the DCF calculator (plan §8.2)."""
import pytest

from data_ingestion.fundamental_feeds.dcf_calculator import (
    compute_dcf, dcf_from_snapshot)


def test_dcf_basic():
    r = compute_dcf(ttm_free_cash_flow=1000, revenue_growth_rate=0.10,
                    shares_outstanding=100, net_debt=500)
    assert r.intrinsic_value_per_share is not None
    assert r.intrinsic_value_per_share > 0
    assert r.enterprise_value > 0
    assert r.pv_of_projected_fcf > 0
    assert r.pv_of_terminal_value > 0
    assert r.pv_of_terminal_value > r.pv_of_projected_fcf  # TV dominates at 10y


def test_dcf_margin_of_safety():
    r = compute_dcf(1000, 0.10, shares_outstanding=100, net_debt=0,
                    current_price=10)
    assert r.margin_of_safety is not None
    assert r.margin_of_safety > 0            # intrinsic > price → positive MoS
    r2 = compute_dcf(1000, 0.10, shares_outstanding=100, net_debt=0,
                     current_price=1e9)
    assert r2.margin_of_safety < 0           # overvalued → negative MoS


def test_dcf_growth_fades_to_terminal():
    """Growth must fade from revenue_growth_rate to terminal_growth_rate."""
    r = compute_dcf(1000, 0.20, terminal_growth_rate=0.03, wacc=0.10,
                    projection_years=10)
    assert r.terminal_growth == 0.03
    assert r.intrinsic_value_per_share is None  # no shares → None


def test_dcf_wacc_must_exceed_terminal_growth():
    with pytest.raises(ValueError):
        compute_dcf(1000, 0.10, terminal_growth_rate=0.12, wacc=0.10)


def test_dcf_from_snapshot():
    snap = {"free_cash_flow": 1000, "revenue_yoy_growth": 0.08,
            "market_cap": 50000, "current_price": 100, "net_debt": 200}
    r = dcf_from_snapshot(snap)
    assert r is not None
    assert r.intrinsic_value_per_share is not None


def test_dcf_from_snapshot_missing_fcf_returns_none():
    assert dcf_from_snapshot({"revenue_yoy_growth": 0.1}) is None
