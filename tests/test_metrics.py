"""Tests for the backtesting metrics (plan §9.4)."""
import numpy as np
import pandas as pd
import pytest

from backtesting.metrics.alpha_metrics import (
    alpha_vs_benchmark, cagr, information_ratio)
from backtesting.metrics.drawdown import (
    calmar_ratio, drawdown_series, max_drawdown, max_drawdown_duration_days)
from backtesting.metrics.sharpe import sharpe_ratio, sortino_ratio
from backtesting.metrics.trade_stats import trade_statistics
from backtesting.engine import Trade


def _returns(positive: bool = True) -> pd.Series:
    rng = np.random.default_rng(42)
    vals = rng.normal(0.001 if positive else -0.001, 0.01, 500)
    return pd.Series(vals)


def test_sharpe_positive_for_positive_returns():
    assert sharpe_ratio(_returns(True)) > 0
    assert sharpe_ratio(_returns(False)) < 0
    assert sharpe_ratio(pd.Series([0.0])) == 0.0


def test_sortino_bounds():
    s = sortino_ratio(_returns(True))
    assert s > 0


def test_drawdown_series_and_max():
    equity = pd.Series([100, 110, 90, 95, 130])
    dd = drawdown_series(equity)
    assert dd.iloc[0] == 0
    assert max_drawdown(equity) == pytest.approx(90 / 110 - 1, abs=1e-9)
    assert dd.max() <= 0


def test_calmar():
    assert calmar_ratio(0.20, -0.10) == pytest.approx(2.0)
    assert calmar_ratio(0.20, 0.0) == 0.0


def test_cagr():
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    equity = pd.Series(np.linspace(100, 121, 365), index=idx)
    assert cagr(equity) == pytest.approx(0.21, abs=0.01)


def test_alpha_and_information_ratio():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    bench = pd.Series(np.linspace(100, 110, 300), index=idx)
    strat = pd.Series(np.linspace(100, 120, 300), index=idx)
    assert alpha_vs_benchmark(strat, bench) > 0
    assert information_ratio(strat, bench) > 0


def _trade(symbol, pnl, pct, hours=24):
    return Trade(symbol=symbol, direction="LONG",
                 entry_time=pd.Timestamp("2024-01-01"),
                 entry_price=100.0, exit_time=pd.Timestamp("2024-01-02"),
                 exit_price=100.0, pnl_usd=pnl, pnl_pct=pct)


def test_trade_statistics():
    trades = [_trade("A", 100, 0.01), _trade("B", 100, 0.02),
              _trade("C", -50, -0.01)]
    s = trade_statistics(trades)
    assert s["total_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["profit_factor"] == pytest.approx(200 / 50)
    assert s["expectancy_per_trade_usd"] == pytest.approx(150 / 3)
    assert s["avg_holding_period_hours"] == pytest.approx(24.0)


def test_trade_statistics_empty():
    s = trade_statistics([])
    assert s["total_trades"] == 0
    assert s["win_rate"] == 0.0
