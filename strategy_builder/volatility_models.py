"""Volatility models for the strategy builder benchmark.

Two distinct roles
------------------
1. **Volatility forecasting**: Predict next-period volatility (GARCH family,
   HAR-RV). Used as an improved sigma_t in vol-targeting and as a standalone
   predictive signal.

2. **Volatility-regime signals**: Convert vol forecasts into long/short/flat
   position signals.

Model roster
------------
Forecasters (fit per-asset, walk-forward):
  garch11      - GARCH(1,1) Normal innovations
  egarch       - EGARCH(1,1): asymmetric vol (leverage effect)
  gjr_garch    - GJR-GARCH(1,1): asymmetric, threshold GARCH
  har_rv       - HAR-RV: Corsi (2009) linear model on daily/weekly/monthly RV
  ewma_vol     - RiskMetrics EWMA (lambda=0.94) baseline

Signal strategies (produce weights on the standard benchmark interface):
  vol_timing       - Long when GARCH vol < long-run avg, flat/short otherwise
  vol_carry        - Long assets with falling GARCH vol (vol carry)
  vol_momentum     - Long when realized vol > GARCH forecast (vol surprise)
  vol_regime_ml    - LightGBM on GARCH + HAR features -> position signal
  har_signal       - Pure HAR-RV direction: short rising vol, long falling vol
  vol_timing_egarch/gjr - Same as vol_timing but with EGARCH / GJR-GARCH sigma

All signal strategies share the standard interface:
  fn(panel, feature_cols, symbols, **kw) -> pd.DataFrame[time, symbol, weight]
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

from strategy_builder.trainer import walk_forward_windows

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Core volatility estimators (per-asset, pure numpy / optional arch lib)
# ---------------------------------------------------------------------------

def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA variance: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r2_{t-1}."""
    var = np.empty(len(returns))
    var[0] = returns[0] ** 2
    for t in range(1, len(returns)):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return var


def garch11_variance(
    returns: np.ndarray,
    omega: float | None = None,
    alpha: float = 0.05,
    beta: float = 0.90,
    fit: bool = True,
) -> np.ndarray:
    """GARCH(1,1) conditional variance via MLE (arch lib) or moment-matching fallback."""
    if fit:
        try:
            from arch import arch_model
            am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            return res.conditional_volatility.values ** 2 / 1e4
        except Exception:
            pass
    lr_var = np.var(returns)
    omega = lr_var * (1 - alpha - beta) if omega is None else omega
    var = np.empty(len(returns))
    var[0] = lr_var
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return var


def egarch_variance(returns: np.ndarray) -> np.ndarray:
    """EGARCH(1,1) via arch library; falls back to GARCH(1,1) on failure."""
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="EGARCH", p=1, q=1, dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        return res.conditional_volatility.values ** 2 / 1e4
    except Exception:
        return garch11_variance(returns, fit=False)


def gjr_garch_variance(returns: np.ndarray) -> np.ndarray:
    """GJR-GARCH(1,1) via arch library; falls back to GARCH(1,1) on failure."""
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="GARCH", p=1, o=1, q=1, dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        return res.conditional_volatility.values ** 2 / 1e4
    except Exception:
        return garch11_variance(returns, fit=False)


def har_rv_forecast(returns: np.ndarray, train_end: int) -> np.ndarray:
    """HAR-RV: 1-step OOS forecasts for returns[train_end:].
    Corsi (2009): RV_{t+1} = c + b_d*RV_d_t + b_w*RV_w_t + b_m*RV_m_t.
    """
    rv_d = returns ** 2
    rv_w = np.convolve(rv_d, np.ones(5) / 5, mode="full")[:len(rv_d)]
    rv_m = np.convolve(rv_d, np.ones(21) / 21, mode="full")[:len(rv_d)]
    start = 21
    X_tr = np.column_stack([
        np.ones(train_end - start),
        rv_d[start - 1: train_end - 1],
        rv_w[start - 1: train_end - 1],
        rv_m[start - 1: train_end - 1],
    ])
    y_tr = rv_d[start: train_end]
    try:
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    except np.linalg.LinAlgError:
        coef = np.array([np.mean(y_tr), 0.0, 0.0, 0.0])
    n_oos = len(returns) - train_end
    forecasts = np.full(n_oos, np.mean(rv_d[:train_end]))
    for i in range(n_oos):
        t = train_end + i
        x = np.array([1.0, rv_d[t - 1], rv_w[t - 1], rv_m[t - 1]])
        forecasts[i] = max(float(x @ coef), 1e-10)
    return forecasts


# ---------------------------------------------------------------------------
# Helper: enrich a panel slice with GARCH + HAR vol features
# ---------------------------------------------------------------------------

def _build_vol_features(
    panel_slice: pd.DataFrame, symbols: list[str], model: str = "garch11",
) -> pd.DataFrame:
    """Add garch_var, har_var, ewma_var, vol_surprise, vol_regime, vol_trend."""
    frames = []
    for sym in symbols:
        g = panel_slice[panel_slice["symbol"] == sym].sort_values("time").copy()
        rets = g["ret_1"].fillna(0.0).to_numpy()
        n = len(rets)
        if n < 60:
            continue
        g["ewma_var"] = ewma_variance(rets)
        if model == "garch11":
            g["garch_var"] = garch11_variance(rets)
        elif model == "egarch":
            g["garch_var"] = egarch_variance(rets)
        elif model == "gjr":
            g["garch_var"] = gjr_garch_variance(rets)
        else:
            g["garch_var"] = garch11_variance(rets)
        har_var = np.full(n, np.nan)
        for t in range(252, n):
            fcast = har_rv_forecast(rets[:t + 1], train_end=t)
            if len(fcast) > 0:
                har_var[t] = fcast[-1]
        g["har_var"] = har_var
        lr_mean = g["garch_var"].rolling(252, min_periods=60).mean()
        g["vol_surprise"] = g["ewma_var"] - g["garch_var"]
        g["vol_regime"] = (g["garch_var"] < lr_mean).astype(float)
        g["vol_trend"] = g["garch_var"].pct_change(5)
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Signal strategies
# ---------------------------------------------------------------------------

def _vol_signal_runner(
    signal_fn: Callable,
    panel: pd.DataFrame,
    symbols: list[str],
    garch_model: str = "garch11",
    train_months: int = 36,
    test_months: int = 6,
    sigma_tgt: float = 0.10,
) -> pd.DataFrame:
    windows = walk_forward_windows(panel, train_months, test_months)
    frames = []
    for tr, va in windows:
        full = pd.concat([tr, va]).sort_values("time")
        with_vol = _build_vol_features(full, symbols, model=garch_model)
        va_vol = with_vol[with_vol["time"].isin(va["time"].values)]
        if va_vol.empty:
            continue
        sig = signal_fn(va_vol)
        for i, (idx, row) in enumerate(va_vol.iterrows()):
            s = float(sig.iloc[i]) if hasattr(sig, "iloc") else float(sig.get(idx, 0.0))
            vs = 1.0 / max(float(row.get("garch_var", row.get("ewma_var", 1e-4))) ** 0.5, 1e-6)
            frames.append({"time": row["time"], "symbol": row["symbol"],
                           "weight": float(np.tanh(s) * sigma_tgt * vs)})
    return pd.DataFrame(frames) if frames else pd.DataFrame(
        columns=["time", "symbol", "weight"])


def vol_timing(panel, feature_cols, symbols, garch_model="garch11", **kw):
    """Long in calm vol regime scaled by recent momentum, flat/short in turbulent."""
    def _sig(va):
        calm = va["vol_regime"]
        mom = va.get("ret_norm_21", pd.Series(0.0, index=va.index)).fillna(0.0)
        return (calm * 1.0 + (1 - calm) * -0.5) * mom
    return _vol_signal_runner(_sig, panel, symbols, garch_model=garch_model, **kw)


def vol_carry(panel, feature_cols, symbols, garch_model="garch11", **kw):
    """Long assets with falling GARCH vol (negative vol_trend = compression)."""
    def _sig(va):
        return -va["vol_trend"].fillna(0.0)
    return _vol_signal_runner(_sig, panel, symbols, garch_model=garch_model, **kw)


def vol_momentum(panel, feature_cols, symbols, garch_model="garch11", **kw):
    """Long when realized > GARCH forecast (vol surprise) with momentum confirmation."""
    def _sig(va):
        surprise = np.sign(va["vol_surprise"])
        mom = np.sign(va.get("ret_norm_21", pd.Series(0.0, index=va.index)).fillna(0.0))
        return surprise * mom
    return _vol_signal_runner(_sig, panel, symbols, garch_model=garch_model, **kw)


def har_signal(panel, feature_cols, symbols, train_months=36, test_months=6,
               sigma_tgt=0.10, **kw):
    """Pure HAR-RV: short rising vol, long falling vol, scaled by momentum."""
    windows = walk_forward_windows(panel, train_months, test_months)
    frames = []
    for tr, va in windows:
        full = pd.concat([tr, va]).sort_values("time")
        with_vol = _build_vol_features(full, symbols, model="garch11")
        va_vol = with_vol[with_vol["time"].isin(va["time"].values)].copy()
        if va_vol.empty:
            continue
        sig = -np.sign(va_vol["vol_trend"].fillna(0.0))
        if "ret_norm_21" in va_vol.columns:
            sig = sig * np.sign(va_vol["ret_norm_21"].fillna(0.0))
        for i, (_, row) in enumerate(va_vol.iterrows()):
            vs = 1.0 / max(float(row.get("har_var", row.get("ewma_var", 1e-4))) ** 0.5, 1e-6)
            frames.append({"time": row["time"], "symbol": row["symbol"],
                           "weight": float(float(sig.iloc[i]) * sigma_tgt * vs)})
    return pd.DataFrame(frames) if frames else pd.DataFrame(
        columns=["time", "symbol", "weight"])


def vol_regime_ml(panel, feature_cols, symbols, garch_model="garch11",
                  train_months=36, test_months=6, sigma_tgt=0.10, **kw):
    """LightGBM on GARCH + HAR-RV vol features (last step) -> position signal."""
    import lightgbm as lgb
    vol_feat_cols = ["garch_var", "har_var", "ewma_var",
                     "vol_surprise", "vol_regime", "vol_trend"]
    windows = walk_forward_windows(panel, train_months, test_months)
    frames = []
    for tr, va in windows:
        full = pd.concat([tr, va]).sort_values("time")
        with_vol = _build_vol_features(full, symbols, model=garch_model)
        tr_vol = with_vol[with_vol["time"].isin(tr["time"].values)].dropna(subset=vol_feat_cols)
        va_vol = with_vol[with_vol["time"].isin(va["time"].values)].dropna(subset=vol_feat_cols)
        if tr_vol.empty or va_vol.empty:
            continue
        avail = [c for c in feature_cols + vol_feat_cols if c in tr_vol.columns]
        Xtr = tr_vol[avail].fillna(0.0).to_numpy()
        ytr = np.clip(
            tr_vol["ret_1"].shift(-1).fillna(0.0).to_numpy()
            / np.clip(tr_vol["ewma_var"].to_numpy() ** 0.5, 1e-6, None),
            -20, 20)
        Xva = va_vol[avail].fillna(0.0).to_numpy()
        m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               n_jobs=4, verbose=-1)
        m.fit(Xtr, ytr)
        pred = np.tanh(m.predict(Xva))
        for i, (_, row) in enumerate(va_vol.iterrows()):
            vs = 1.0 / max(float(row.get("garch_var", 1e-4)) ** 0.5, 1e-6)
            frames.append({"time": row["time"], "symbol": row["symbol"],
                           "weight": float(pred[i] * sigma_tgt * vs)})
    return pd.DataFrame(frames) if frames else pd.DataFrame(
        columns=["time", "symbol", "weight"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VOLATILITY_REGISTRY: dict[str, callable] = {
    "vol_timing":         vol_timing,
    "vol_carry":          vol_carry,
    "vol_momentum":       vol_momentum,
    "vol_regime_ml":      vol_regime_ml,
    "har_signal":         har_signal,
    "vol_timing_egarch":  lambda p, f, s, **kw: vol_timing(p, f, s, garch_model="egarch", **kw),
    "vol_timing_gjr":     lambda p, f, s, **kw: vol_timing(p, f, s, garch_model="gjr", **kw),
}
