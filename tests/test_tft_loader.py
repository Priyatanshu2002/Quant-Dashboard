"""Tests for the TFT training-panel loader (transformer_model/loader.py).

The split/build logic is pure (given a panel), so these run without a DB or
without touching the network — a small synthetic panel stands in for the store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformer_model.loader import build_tft_train_val


def _synthetic_panel(n_symbols: int = 3, n_days: int = 400,
                     start: str = "2021-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    frames = []
    for s in range(n_symbols):
        rng = np.random.default_rng(s)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n_days))
        ret = pd.Series(close).pct_change()
        frame = pd.DataFrame({
            "time": dates,
            "symbol": f"SYM{s}",
            "asset_class": "EQUITY_US",
            "rsi_14": rng.uniform(20, 80, n_days),
            "macd_histogram": rng.normal(0, 1, n_days),
            "bb_pct_b": rng.uniform(0, 1, n_days),
            "realized_vol_20": rng.uniform(0.005, 0.05, n_days),
            "vix": rng.uniform(12, 30, n_days),
            "return_1bar": ret,
            "future_return_5d": pd.Series(close).pct_change(5).shift(-5),
            "day_of_week": dates.dayofweek,
            "month_end_effect": 0.0,
            "quarter_end_effect": 0.0,
            "days_to_earnings": 5,
            "days_to_expiry": 10,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_load_panel_missing_symbol_column_raises():
    with pytest.raises(ValueError):
        build_tft_train_val(pd.DataFrame({"x": [1]}), target="future_return_5d")


def test_build_tft_train_val_returns_train_and_val_datasets():
    panel = _synthetic_panel()
    train_ds, val_ds = build_tft_train_val(panel, val_frac=0.2,
                                           max_encoder_length=30,
                                           max_prediction_length=5)
    # Both are pytorch-forecasting TimeSeriesDataSet objects with samples.
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert train_ds.target == "future_return_5d"
    # Groups (symbols) are preserved in both splits.
    assert set(train_ds.group_ids) == {"symbol"}


def test_build_tft_train_val_empty_panel_raises():
    with pytest.raises(ValueError):
        build_tft_train_val(pd.DataFrame(), val_frac=0.2)


def test_dataloader_emits_tensors():
    panel = _synthetic_panel(n_symbols=2, n_days=300)
    train_ds, _ = build_tft_train_val(panel, val_frac=0.2,
                                      max_encoder_length=30,
                                      max_prediction_length=5, batch_size=8)
    loader = train_ds.to_dataloader(batch_size=8, train=True)
    batch = next(iter(loader))
    # pytorch-forecasting 1.8 yields a tuple: (x, y, weight, target_scale);
    # x["encoder_cont"] is (batch, enc_len, n_feat).
    x = batch[0]
    assert x["encoder_cont"].ndim == 3
    assert x["encoder_cont"].shape[0] <= 8
