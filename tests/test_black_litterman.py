"""Tests for Black-Litterman and the risk budgeter (plan §7.2)."""
import numpy as np
import pytest

from portfolio_manager.black_litterman import (
    black_litterman_allocate, black_litterman_posterior, equilibrium_returns)
from portfolio_manager.risk_budgeter import RiskBudgeter, position_var


def test_equilibrium_returns():
    caps = np.array([600, 300, 100], dtype=float)
    cov = np.eye(3) * 0.04
    pi = equilibrium_returns(caps, cov, risk_aversion=2.5)
    assert pi.shape == (3,)
    assert pi[0] > pi[1] > pi[2]      # bigger cap weight → higher implied return


def test_black_litterman_posterior_shape():
    n = 3
    eq = np.array([0.06, 0.05, 0.04])
    cov = np.eye(n) * 0.04
    mu = black_litterman_posterior(eq, cov, views=np.array([0.12]),
                                   view_assets=np.array([0]),
                                   confidences=np.array([0.8]))
    assert mu.shape == (n,)
    # View raises the posterior mean for the viewed asset
    assert mu[0] > eq[0]


def test_allocate_respects_position_cap():
    # 3 assets with a binding 15% cap: each weight ≤ cap, remainder in cash.
    w = black_litterman_allocate(
        ["A", "B", "C"],
        views={"B": 0.30}, confidence={"B": 0.9},
        nav_usd=1_000_000)
    assert set(w) == {"A", "B", "C"}
    assert sum(w.values()) <= 1.0 + 1e-6
    assert sum(w.values()) > 0
    assert all(0 <= v <= 0.15 + 1e-6 for v in w.values())  # 15% cap


def test_allocate_with_loose_cap_normalizes_to_one():
    # With a non-binding cap the weights must sum to ~1 (fully invested).
    w = black_litterman_allocate(["A", "B"], views={}, confidence={},
                                 max_position=0.9)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v <= 0.9 + 1e-6 for v in w.values())


def test_allocate_single_ticker_capped():
    w = black_litterman_allocate(["ONLY"], views={}, confidence={})
    assert w["ONLY"] == pytest.approx(0.15, abs=1e-3)


def test_allocate_no_views_falls_back_gracefully():
    w = black_litterman_allocate(["A", "B"], views={}, confidence={})
    assert sum(w.values()) <= 1.0 + 1e-6
    assert all(v <= 0.15 + 1e-6 for v in w.values())


def test_risk_budgeter_allocates():
    rb = RiskBudgeter(daily_var_limit=0.02, nav_usd=1_000_000)
    used = rb.allocate("AAPL", 8_000)
    assert 0 < used < 1
    assert rb.utilization() == pytest.approx(8000 / 20_000)


def test_risk_budgeter_raises_over_limit():
    rb = RiskBudgeter(daily_var_limit=0.02, nav_usd=1_000_000)
    rb.allocate("A", 15_000)
    with pytest.raises(ValueError):
        rb.allocate("B", 10_000)      # 25k > 20k limit


def test_position_var_scales_with_notional():
    v1 = position_var(1_000_000, 0.20)
    v2 = position_var(2_000_000, 0.20)
    assert v2 == pytest.approx(2 * v1, rel=1e-6)
    assert v1 > 0
