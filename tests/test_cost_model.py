"""Tests for the cost model (plan §9.1)."""
from backtesting.cost_model import CostModel


def test_round_trip_cost_positive_and_reasonable():
    cm = CostModel()
    cost = cm.total_round_trip_cost("EQUITY_US", 10_000, adv_usd=1e9,
                                    bid=99.95, ask=100.05, order_type="MARKET")
    assert 0 < cost < 0.05          # < 5% round trip


def test_crypto_taker_more_expensive_than_maker():
    cm = CostModel()
    args = dict(order_size_usd=10_000, adv_usd=1e9, bid=99.99, ask=100.01)
    taker = cm.total_round_trip_cost("CRYPTO", order_type="MARKET", **args)
    maker = cm.total_round_trip_cost("CRYPTO", order_type="LIMIT", **args)
    assert taker > maker


def test_slippage_grows_with_participation():
    cm = CostModel()
    small = cm.compute_slippage(1_000, adv_usd=1e6)
    large = cm.compute_slippage(100_000, adv_usd=1e6)
    assert large > small
    assert large < 0.1  # sqrt model stays bounded


def test_half_spread():
    cm = CostModel()
    assert cm.compute_half_spread(100, 101) == pytest_approx(0.004975)


def test_round_trip_cost_pct_convenience():
    cm = CostModel()
    cost = cm.round_trip_cost_pct("EQUITY_US", 10_000, 1e9, spread_pct=0.0005)
    assert cost > 0


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-3)
