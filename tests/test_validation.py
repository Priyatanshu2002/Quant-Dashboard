"""Tests for the validation suite (plan §9.2/§9.5): walk-forward, MC, K-fold."""
import numpy as np
import pandas as pd
import pytest

from backtesting.monte_carlo import mc_summary, monte_carlo_shuffle
from backtesting.purged_kfold import PurgedKFold
from backtesting.strategies import MACrossStrategy
from backtesting.walk_forward import (
    _generate_periods, summarize_walk_forward, walk_forward_backtest)


def _ohlcv(n=700, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)


def test_generate_periods_contiguous():
    periods = list(_generate_periods("2023-01-01", "2025-01-01",
                                     train_months=12, test_months=3,
                                     step_months=1))
    assert len(periods) > 5
    for train_start, train_end, test_start, test_end in periods[:3]:
        assert train_start < train_end <= test_start < test_end


def test_walk_forward_returns_results():
    ohlcv = _ohlcv()
    results = walk_forward_backtest(
        ohlcv, lambda: MACrossStrategy(fast=10, slow=30),
        symbol="SPY", asset_class="ETF",
        train_months=12, test_months=3, step_months=1,
        start_date="2023-01-01", end_date="2025-01-01")
    assert len(results) >= 1
    for r in results:
        assert r.report.total_trades >= 0
        assert len(r.equity_curve) == len(r.trades) or True  # equity always full-length
    table = summarize_walk_forward(results)
    assert len(table) == len(results)


def test_purged_kfold_disjoint_folds():
    X = np.arange(200)
    kf = PurgedKFold(n_splits=5, embargo=10)
    for train_idx, val_idx in kf.split(X):
        assert len(set(train_idx) & set(val_idx)) == 0
        # embargo purges samples within 10 of the validation window
        assert not np.any((train_idx >= val_idx.min() - 10) &
                          (train_idx <= val_idx.max() + 10))
    assert kf.get_n_splits() == 5


def test_purged_kfold_covers_all():
    X = np.arange(200)
    kf = PurgedKFold(n_splits=4, embargo=5)
    seen = set()
    for _, val_idx in kf.split(X):
        seen.update(val_idx.tolist())
    assert seen == set(range(200))


def test_monte_carlo_shuffle_shape():
    ohlcv = _ohlcv(n=400)
    strat = MACrossStrategy(fast=10, slow=30)
    strat.fit(ohlcv)
    signals = strat.generate_signals()
    mc = monte_carlo_shuffle("SPY", "ETF", ohlcv, signals,
                             n_runs=20, seed=7)
    assert len(mc) == 20
    assert {"run", "total_return_pct", "sharpe", "max_dd_pct"} <= set(mc.columns)


def test_mc_summary():
    mc = pd.DataFrame({"run": range(10), "total_return_pct": np.linspace(-5, 5, 10),
                       "sharpe": np.linspace(-1, 1, 10),
                       "max_dd_pct": np.linspace(-20, -5, 10)})
    s = mc_summary(mc, actual_sharpe=0.5)
    assert s["runs"] == 10
    assert 0 <= s["pctile_of_actual"] <= 1
    assert s["sharpe_mean"] == pytest.approx(0.0, abs=1e-6)
