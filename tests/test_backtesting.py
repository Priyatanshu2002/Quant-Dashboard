"""Tests for the backtesting engine + strategies (plan §9)."""
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.strategies import (
    MACrossStrategy, MomentumStrategy, RsiMeanReversionStrategy, make_strategy)


def _ohlcv(n=300, drift=0.0008, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)


def test_engine_returns_report_and_equity():
    ohlcv = _ohlcv()
    strat = MACrossStrategy(fast=10, slow=30)
    strat.fit(ohlcv)
    result = BacktestEngine().run("TEST", "EQUITY_US", ohlcv,
                                  strat.generate_signals(),
                                  strategy_name="ma")
    assert len(result.equity_curve) == len(ohlcv)
    assert result.report.total_trades >= 1
    assert result.report.period_start == ohlcv.index[0].date()
    assert len(result.trades) == result.report.total_trades


def test_engine_trending_market_long_profits():
    ohlcv = _ohlcv(drift=0.0015, seed=3)
    strat = MACrossStrategy(fast=10, slow=30)
    strat.fit(ohlcv)
    result = BacktestEngine().run("TEST", "EQUITY_US", ohlcv,
                                  strat.generate_signals())
    assert result.report.total_return_pct > 0


def test_engine_flat_signal_no_trades():
    ohlcv = _ohlcv()
    signals = pd.Series(0.0, index=ohlcv.index)
    result = BacktestEngine().run("TEST", "EQUITY_US", ohlcv, signals)
    assert result.report.total_trades == 0
    assert result.report.total_return_pct == pytest.approx(0.0, abs=1e-6)


def test_engine_costs_reduce_returns():
    ohlcv = _ohlcv(drift=0.0015, seed=3)
    strat = MACrossStrategy(fast=10, slow=30)
    strat.fit(ohlcv)
    sig = strat.generate_signals()
    free = BacktestEngine(spread_pct=0.0).run("TEST", "EQUITY_US", ohlcv, sig)
    costly = BacktestEngine(spread_pct=0.005).run("TEST", "EQUITY_US", ohlcv, sig)
    assert costly.report.total_return_pct <= free.report.total_return_pct + 1e-9


def test_strategy_registry():
    for name in ("ma_cross", "rsi_reversion", "momentum"):
        s = make_strategy(name)
        ohlcv = _ohlcv(n=300)
        s.fit(ohlcv)
        sig = s.generate_signals()
        assert len(sig) == len(ohlcv)
        assert sig.abs().max() <= 1.0


def test_rsi_reversion_signals():
    ohlcv = _ohlcv(n=300, seed=5)
    s = RsiMeanReversionStrategy()
    s.fit(ohlcv)
    sig = s.generate_signals()
    assert set(sig.unique()) <= {0.0, 1.0}
    assert sig.sum() >= 0


def test_momentum_needs_history():
    ohlcv = _ohlcv(n=50)   # too little for 50-bar MA
    s = MomentumStrategy()
    s.fit(ohlcv)
    sig = s.generate_signals()
    assert sig.notna().all()
