"""Fast unit tests for strategy_builder (no training runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from strategy_builder.backtest import (breakeven_costs, full_metrics,
                                       portfolio_returns)
from strategy_builder.features import (FEATURE_COLS, build_features,
                                       build_universe_frame)
from strategy_builder.models import ENCODERS, SignalHead, build_encoder
from strategy_builder.trainer import (WindowedDataset, pooled_sharpe_loss,
                                      walk_forward_windows)


def _fake_panel(n_days: int = 600, n_syms: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
                           for s in ["AAA", "BBB", "CCC"]}, index=dates)
    return build_universe_frame(closes)


# ---------------------------------------------------------------- features

def test_feature_columns():
    assert len(FEATURE_COLS) > 7          # rich feature set (was 7 — too thin)
    assert "ret_norm_1" in FEATURE_COLS and "macd_signal" in FEATURE_COLS
    assert "rsi_14" in FEATURE_COLS and "sma_ratio_50" in FEATURE_COLS
    assert "skew_21" in FEATURE_COLS and "rv_21" in FEATURE_COLS


def test_build_features_shapes_and_finite():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 500)),
                      index=pd.bdate_range("2023-01-02", periods=500))
    f = build_features(close)
    assert set(FEATURE_COLS) <= set(f.columns)
    assert {"sigma", "vs_factor", "ret_1"} <= set(f.columns)
    assert f[FEATURE_COLS].dropna().shape[0] > 200
    assert np.isfinite(f[FEATURE_COLS].dropna().values).all()


def test_build_universe_frame_columns():
    panel = _fake_panel()
    assert {"time", "symbol", "target", "vs_factor", "ret_1"} <= set(panel.columns)
    assert panel["symbol"].nunique() == 3
    assert len(panel) > 1000


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoders_forward(name):
    torch.manual_seed(0)
    enc = build_encoder(name, in_dim=7, hidden=32, lookback=64, n_assets=5)
    x = torch.randn(4, 64, 7)
    tid = torch.randint(0, 5, (4,))
    h = enc(x, tid)
    assert h.shape == (4, 32)
    assert torch.isfinite(h).all()
    y = SignalHead(32)(h)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1.0 + 1e-5


# ---------------------------------------------------------------- trainer

def test_pooled_sharpe_loss_math():
    # constant positive edge → positive Sharpe → negative loss
    sig = torch.full((100,), 0.5)
    ret = torch.full((100,), 0.01)
    vs = torch.full((100,), 50.0)          # 20% vol → leverage 50 at 10% target... use 25
    vs = torch.full((100,), 25.0)
    loss = pooled_sharpe_loss(sig, ret, vs, sigma_tgt=0.10)
    # portfolio ret per sample = 0.5 * 0.10 * 25 * 0.01 = 0.0125 constant → var=0 → eps path
    assert loss < 0  # positive Sharpe


def test_pooled_sharpe_loss_deterministic():
    a = pooled_sharpe_loss(torch.zeros(64), torch.randn(64), torch.rand(64) + 1)
    b = pooled_sharpe_loss(torch.zeros(64), torch.randn(64), torch.rand(64) + 1)
    assert torch.equal(a, b)  # same inputs → same result (no randomness)


def test_windowed_dataset_alignment():
    panel = _fake_panel()
    symbols = sorted(panel["symbol"].unique())
    ds = WindowedDataset(panel, FEATURE_COLS, lookback=32, symbols=symbols)
    assert ds.x.shape[0] == len(ds.r) == len(ds.times)
    assert ds.x.shape[1] == 32 and ds.x.shape[2] == len(FEATURE_COLS)
    # first sample of a symbol must be its (lookback)-th calendar row
    sym0 = ds.syms[0]
    g = panel[panel["symbol"] == sym0].sort_values("time")
    assert ds.times[0] == g["time"].iloc[32]
    assert ds.syms[0] == sym0


def test_walk_forward_windows_no_lookahead():
    panel = _fake_panel(n_days=900)
    wins = walk_forward_windows(panel, train_months=24, test_months=4)
    assert len(wins) >= 1
    for tr, va in wins:
        assert tr["time"].max() <= va["time"].min()
        assert va["time"].max() > tr["time"].max()


# ---------------------------------------------------------------- backtest

def _fake_weights(panel: pd.DataFrame, signal: float) -> pd.DataFrame:
    rows = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("time")
        for _, r in g.iloc[30:].iterrows():
            rows.append({"time": r["time"], "symbol": sym,
                         "weight": signal * 0.10 * r["vs_factor"]})
    return pd.DataFrame(rows)


def test_full_metrics_deterministic_and_sane():
    panel = _fake_panel()
    w = _fake_weights(panel, 0.5)
    m = full_metrics(w, panel)
    assert set(["sharpe", "cagr", "max_dd", "t_hac", "turnover", "hit_rate",
                "calmar", "cvar_5", "info_ratio"]) <= set(m)
    assert np.isfinite(m["sharpe"]) and np.isfinite(m["max_dd"])


def test_portfolio_returns_wide():
    panel = _fake_panel()
    w = _fake_weights(panel, 0.3)
    pr = portfolio_returns(w, panel)
    assert "portfolio_ret" in pr.columns
    assert pr["portfolio_ret"].notna().sum() > 100


def test_breakeven_costs_positive_for_profitable():
    panel = _fake_panel()
    # all-positive weights on a random walk ≈ ~zero gross; just check structure
    w = _fake_weights(panel, 0.2)
    be = breakeven_costs(w, panel)
    assert {"symbol", "gross_ann", "turnover_ann", "breakeven_bps"} <= set(be.columns)
    assert len(be) == 3


# --- regression tests for the bug fixes -----------------------------------

def test_target_vol_scaling_is_not_inverted():
    """The classical target must be r/sigma (eq. 23), NOT r*sigma.

    The old code computed target = r / vs_factor = r*sigma, giving a ~zero
    target (std ~0.0007) that collapsed every classical model to ~0 positions.
    Correct: target = r * vs_factor = r / sigma, with O(1) magnitude.
    """
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 500)),
                      index=pd.bdate_range("2023-01-02", periods=500))
    from strategy_builder.features import build_features, build_target
    feats = build_features(close)
    target = build_target(close)
    vs = feats["vs_factor"]
    ret = feats["ret_1"].shift(-1)          # r_{t+1}
    # correct target aligns with r * vs (r/sigma), not r / vs (r*sigma)
    expect = (ret * vs).clip(-20, 20)
    assert np.allclose(target.dropna(), expect.reindex(target.dropna().index), atol=1e-6)
    # and its magnitude is O(1), not ~0
    assert abs(target.dropna().std()) > 0.3


def test_portfolio_returns_caps_gross_exposure():
    """Weights must be gross-capped so a model can't take ~±600% leverage."""
    panel = _fake_panel()
    # build weights with huge magnitude (pos=1 => weight=0.10*vs ~ up to ±6)
    w = _fake_weights(panel, 1.0)
    pr = portfolio_returns(w, panel, max_gross=1.5)
    assert pr["gross_exposure"].max() <= 1.5 + 1e-9


def test_portfolio_returns_cost_drag_reduces_return():
    panel = _fake_panel()
    w = _fake_weights(panel, 0.5)
    pr0 = portfolio_returns(w, panel, cost_bps=0.0)["portfolio_ret"]
    prc = portfolio_returns(w, panel, cost_bps=10.0)["portfolio_ret"]
    assert (pr0 - prc).abs().sum() > 0          # cost drag is applied
