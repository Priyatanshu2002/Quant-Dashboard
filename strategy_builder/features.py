"""Benchmark features (Oxford DL-for-finance protocol, arXiv:2603.01820).

Implements Appendix A.4-A.5:
  * Volatility-normalized multi-horizon returns (1d, 1w, 1m, 3m, 6m, 1y):
        r_norm_{t,h} = r_{t,h} / (sigma_t * sqrt(h))
  * Volatility-normalized, regime-adjusted MACD signal:
        MACD = EWMA(P, short) - EWMA(P, long)
        q    = MACD / Std63(P)
        signal = q / Std252(q)
  * EWMA conditional volatility sigma_t (half-life per Appendix A) and
    volatility-scaling factor vs_factor = 1 / sigma_t
  * Learning target: clipped volatility-scaled next-day return:
        target_t = clip(r_{t+1} / sigma_t, -20, 20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EWMA_HALF_LIFE = 60          # days (paper: ~60d EWMA for sigma)
LONG_EWMA_HALF_LIFE = 252    # MACD long window
SHORT_EWMA_HALF_LIFE = 12    # MACD short window
RETURN_HORIZONS = (1, 5, 21, 63, 126, 252)
TARGET_CLIP = 20.0
FEATURE_COLS = [f"ret_norm_{h}" for h in RETURN_HORIZONS] + ["macd_signal"]


def _ewma_halflife(series: pd.Series, half_life: float) -> pd.Series:
    """Exponentially weighted mean with a half-life (days), forward-filled."""
    return series.ewm(halflife=half_life, min_periods=1).mean()


def _ewma_std(series: pd.Series, half_life: float) -> pd.Series:
    """EWMA standard deviation (population) with a half-life."""
    mean = series.ewm(halflife=half_life, min_periods=5).mean()
    var = ((series - mean) ** 2).ewm(halflife=half_life, min_periods=5).mean()
    return np.sqrt(var)


def build_features(close: pd.Series, vol_half_life: float = EWMA_HALF_LIFE) -> pd.DataFrame:
    """Build the benchmark feature frame for one asset.

    Returns a DataFrame indexed like `close` with columns:
      ret_norm_1, ret_norm_5, ret_norm_21, ret_norm_63, ret_norm_126, ret_norm_252,
      macd_signal, and helper columns sigma, vs_factor, ret_1 (raw 1-day return).
    """
    close = close.astype(float)
    ret = close.pct_change()
    sigma = _ewma_std(ret, vol_half_life)
    vs_factor = 1.0 / sigma.replace(0.0, np.nan)

    cols: dict[str, pd.Series] = {}
    for h in RETURN_HORIZONS:
        r_h = close.pct_change(h)
        cols[f"ret_norm_{h}"] = r_h / (sigma * np.sqrt(h))

    # MACD signal (eqs. 19-21)
    macd = (_ewma_halflife(close, SHORT_EWMA_HALF_LIFE)
            - _ewma_halflife(close, LONG_EWMA_HALF_LIFE))
    q = macd / close.rolling(63, min_periods=20).std()
    cols["macd_signal"] = q / q.rolling(252, min_periods=63).std()

    frame = pd.DataFrame(cols, index=close.index)
    frame["sigma"] = sigma
    frame["vs_factor"] = vs_factor
    frame["ret_1"] = ret
    return frame


def build_target(close: pd.Series, vol_half_life: float = EWMA_HALF_LIFE) -> pd.Series:
    """Volatility-scaled next-day return, clipped (eq. 23)."""
    ret = close.pct_change()
    sigma = _ewma_std(ret, vol_half_life)
    target = ret.shift(-1) / sigma
    return target.clip(-TARGET_CLIP, TARGET_CLIP)


def build_universe_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Feature panel for a cross-section of assets.

    prices: DataFrame, columns = symbols, index = DatetimeIndex (close prices).
    Returns long-format DataFrame with columns
      [time, symbol, <features>, sigma, vs_factor, ret_1, target].
    Rows where any required feature is NaN are dropped.
    """
    frames = []
    for sym in prices.columns:
        close = prices[sym].dropna()
        if len(close) < 300:
            continue
        feats = build_features(close)
        target = build_target(close)
        df = feats.join(target.rename("target"))
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise ValueError("no asset has enough history")
    panel = pd.concat(frames).sort_index()
    panel = panel.reset_index().rename(columns={"index": "time"})
    panel = panel.dropna(subset=[c for c in panel.columns
                                 if c.startswith("ret_norm") or c in ("macd_signal",)])
    return panel[["time", "symbol", "ret_1", "sigma", "vs_factor", "target"]
                 + [c for c in panel.columns if c.startswith("ret_norm")]
                 + ["macd_signal"]]
