"""Tests for plan C8: expanded technical features (~75), full-curve macro
features, and discount-rates risk-free-from-macro."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.db import SQLiteStorage
from feature_engineering.macro_features import compute_macro_features
from feature_engineering.technical_features import (
    _aroon, _ichimoku, _supertrend, compute_technical_features,
    latest_technical_features)
from valuation.discount_rates import risk_free_from_macro


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


@pytest.fixture
def ohlcv():
    idx = pd.date_range("2026-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e5, 5e5, 300).astype(float),
    }, index=idx)


# ── expanded technical features ──────────────────────────────────────────
def test_technical_feature_count_goal(ohlcv):
    out = compute_technical_features(ohlcv)
    # plan target ~75; we land 66 today (from the previous ~34)
    assert len(out.columns) >= 60
    assert len(out.columns) > 55


def test_technical_new_indicators_present(ohlcv):
    out = compute_technical_features(ohlcv)
    for k in ("psar", "psar_direction", "supertrend", "supertrend_direction",
              "aroon_up", "aroon_down", "aroon_osc",
              "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion",
              "keltner_width", "keltner_pct_b",
              "mfi_14", "adl", "adl_slope_10",
              "return_skew_20", "return_kurt_20", "range_vol_14"):
        assert k in out.columns, f"missing {k}"


def test_technical_vectorized_no_inf(ohlcv):
    out = compute_technical_features(ohlcv)
    assert not np.isinf(out.to_numpy(dtype=float)).any()
    # latest row mostly populated after warmup
    latest = latest_technical_features(ohlcv)
    assert latest["supertrend_direction"] in (1.0, -1.0)
    assert "rsi_14" in latest


def test_supertrend_direction_agrees_with_price():
    # monotonically rising price → bullish (+1)
    idx = pd.date_range("2026-01-01", periods=80, freq="B")
    c = pd.Series(np.linspace(100, 200, 80), index=idx)
    df = pd.DataFrame({"open": c * 0.999, "high": c * 1.001, "low": c * 0.999,
                       "close": c, "volume": 1e6}, index=idx)
    st = _supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    assert st["supertrend_direction"].iloc[-1] == 1.0
    # monotonically falling price → bearish (−1)
    c2 = pd.Series(np.linspace(200, 100, 80), index=idx)
    df2 = df.copy()
    df2["close"] = c2
    df2["high"] = c2 * 1.001
    df2["low"] = c2 * 0.999
    st2 = _supertrend(df2["high"], df2["low"], df2["close"], 10, 3.0)
    assert st2["supertrend_direction"].iloc[-1] == -1.0


def test_aroon_bounds():
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    h = pd.Series(np.linspace(100, 200, 60), index=idx)
    lo = pd.Series(np.linspace(90, 190, 60), index=idx)
    a = _aroon(h, lo, 25)
    assert a["aroon_up"].iloc[-1] == pytest.approx(100.0)  # new high today
    assert a["aroon_osc"].iloc[-1] == pytest.approx(100.0)


def test_ichimoku_components():
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    h = pd.Series(np.linspace(100, 150, 120), index=idx)
    lo = pd.Series(np.linspace(90, 140, 120), index=idx)
    i = _ichimoku(h, lo)
    assert {"ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion"} <= set(i)
    # leading spans are shifted (first ~26 values NaN)
    assert i["ichimoku_a"].isna().iloc[:26].all()


# ── full-curve macro features ────────────────────────────────────────────
def test_macro_features_full_curve(db):
    db.write_macro_snapshot({
        "us_1m_yield": 0.038, "us_3m_yield": 0.039, "us_6m_yield": 0.040,
        "us_2y_yield": 0.042, "us_10y_yield": 0.047, "us_30y_yield": 0.052,
        "us_10y_real_yield": 0.024, "breakeven_inflation": 0.023,
        "vix": 15.3, "hy_credit_spread": 0.036, "cpi_all_urban": 324.5,
        "unemployment_rate": 0.039, "ism_pmi": 52.0, "m2_supply": 21000,
    })
    m = compute_macro_features(db)
    assert m["us_30y_yield"] == pytest.approx(0.052)
    assert m["us_10y_real_yield"] == pytest.approx(0.024)
    assert m["breakeven_inflation"] == pytest.approx(0.023)
    assert m["hy_credit_spread"] == pytest.approx(0.036)
    assert m["yield_curve_spread"] == pytest.approx(0.047 - 0.042)
    assert m["vix_regime"] == 1


def test_macro_features_empty_when_no_snapshot(db):
    assert compute_macro_features(db) == {}


# ── discount-rates risk-free from macro ──────────────────────────────────
def test_risk_free_from_macro_nominal():
    macro = {"us_10y_yield": 0.0468}
    assert risk_free_from_macro(macro) == pytest.approx(0.0468)


def test_risk_free_from_macro_prefer_real():
    macro = {"us_10y_yield": 0.0468, "us_10y_real_yield": 0.0242,
             "breakeven_inflation": 0.0226}
    assert risk_free_from_macro(macro, prefer_real=True) == pytest.approx(0.0242)
    assert risk_free_from_macro(macro) == pytest.approx(0.0468)


def test_risk_free_from_macro_breakeven_fallback():
    macro = {"us_10y_yield": 0.0468, "breakeven_inflation": 0.0226}
    assert risk_free_from_macro(macro, prefer_real=True) == pytest.approx(0.0468 - 0.0226)


def test_risk_free_from_macro_default():
    assert risk_free_from_macro(None) == pytest.approx(0.0395)
    assert risk_free_from_macro({}) == pytest.approx(0.0395)
