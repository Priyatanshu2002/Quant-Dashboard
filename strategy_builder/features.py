"""Benchmark features (Oxford DL-for-finance protocol, arXiv:2603.01820).

Implements Appendix A.4-A.5 plus a substantially richer feature set so models
actually receive enough signal (the prior 7-feature set was too thin).

Per-asset features (computed inside `build_features`):
  * Volatility-normalized multi-horizon returns:
        r_norm_{h} = r_{t,h} / (sigma_t * sqrt(h))   for h in RETURN_HORIZONS
  * EWMA conditional vol sigma_t, vol-scaling factor vs_factor = 1/sigma_t
  * Trend / momentum: close vs SMA ratios, MACD + histogram, RSI
  * Realized vol at multiple horizons, vol ratio, downside vol
  * Skewness / kurtosis of returns (rolling)
  * Volume z-score

Cross-sectional features (added by `add_cross_sectional_features` at the panel
level, i.e. across assets at each timestamp):
  * Cross-sectional percentile rank of each momentum feature (rank in [0,1])
  * Cross-sectional z-score of momentum returns

Learning target: clipped volatility-scaled next-day return
        target_t = clip(r_{t+1} / sigma_t, -20, 20)     (eq. 23)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EWMA_HALF_LIFE = 60          # days (paper: ~60d EWMA for sigma)
LONG_EWMA_HALF_LIFE = 252    # MACD long window
SHORT_EWMA_HALF_LIFE = 12    # MACD short window
RETURN_HORIZONS = (1, 5, 21, 63, 126, 252)
TARGET_CLIP = 20.0

# --- per-asset feature columns ---------------------------------------------
FEATURE_COLS = (
    [f"ret_norm_{h}" for h in RETURN_HORIZONS]          # 6 momentum
    + ["macd_signal", "macd_hist", "rsi_14"]            # trend / momentum
    + ["sma_ratio_20", "sma_ratio_50", "sma_ratio_200"]  # close vs trend
    + ["rv_10", "rv_21", "rv_60", "rv_ratio_10_60", "downside_vol_21"]  # vol
    + ["skew_21", "kurt_21"]                            # higher moments
    + ["vol_zscore"]                                    # volume
    # OHLC-based indicators (computed from daily high/low/close/volume)
    + ["atr_14", "adx_14", "stoch_k_14", "stoch_d_14", "williams_r_14", "cci_20",
       "roc_12", "boll_bandwidth_20", "boll_pct_b_20", "parkinson_vol_20",
       "gk_vol_20", "cmf_20", "mfi_14", "choppiness_14", "kaufman_er_20",
       "close_loc"]
)


def _ewma_halflife(series: pd.Series, half_life: float) -> pd.Series:
    """Exponentially weighted mean with a half-life (days), forward-filled."""
    return series.ewm(halflife=half_life, min_periods=1).mean()


def _ewma_std(series: pd.Series, half_life: float) -> pd.Series:
    """EWMA standard deviation (population) with a half-life."""
    mean = series.ewm(halflife=half_life, min_periods=5).mean()
    var = ((series - mean) ** 2).ewm(halflife=half_life, min_periods=5).mean()
    return np.sqrt(var)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_features(close: pd.Series, vol: pd.Series | None = None,
                   high: pd.Series | None = None, low: pd.Series | None = None,
                   vol_half_life: float = EWMA_HALF_LIFE) -> pd.DataFrame:
    """Build the per-asset benchmark feature frame for one asset.

    close: close-price Series (DatetimeIndex). vol/high/low: optional volume /
    high / low Series of the same index. Returns a DataFrame indexed like `close`
    with FEATURE_COLS plus helper columns sigma, vs_factor, ret_1.
    """
    close = close.astype(float)
    ret = close.pct_change()
    sigma = _ewma_std(ret, vol_half_life)
    vs_factor = 1.0 / sigma.replace(0.0, np.nan)

    cols: dict[str, pd.Series] = {}

    # 1) vol-normalized momentum at multiple horizons
    for h in RETURN_HORIZONS:
        r_h = close.pct_change(h)
        cols[f"ret_norm_{h}"] = r_h / (sigma * np.sqrt(h))

    # 2) MACD signal (eqs. 19-21) + histogram
    macd = (_ewma_halflife(close, SHORT_EWMA_HALF_LIFE)
            - _ewma_halflife(close, LONG_EWMA_HALF_LIFE))
    q = macd / close.rolling(63, min_periods=20).std()
    cols["macd_signal"] = q / q.rolling(252, min_periods=63).std()
    cols["macd_hist"] = macd - _ewma_halflife(macd, 9)

    # 3) RSI + close vs SMA ratios
    cols["rsi_14"] = _rsi(close, 14)
    for w in (20, 50, 200):
        sma = close.rolling(w, min_periods=min(w, 10)).mean()
        cols[f"sma_ratio_{w}"] = close / sma - 1.0

    # 4) realized vol at several horizons, vol ratio, downside vol
    for w in (10, 21, 60):
        cols[f"rv_{w}"] = ret.rolling(w, min_periods=min(w, 5)).std() * np.sqrt(252)
    cols["rv_ratio_10_60"] = cols["rv_10"] / cols["rv_60"].replace(0.0, np.nan)
    neg = ret.clip(upper=0.0)
    cols["downside_vol_21"] = neg.rolling(21, min_periods=5).std() * np.sqrt(252)

    # 5) higher moments
    cols["skew_21"] = ret.rolling(21, min_periods=10).skew()
    cols["kurt_21"] = ret.rolling(21, min_periods=10).kurt()

    # 6) volume z-score
    if vol is not None and vol.notna().any():
        v = vol.astype(float)
        vmean = v.ewm(halflife=20, min_periods=5).mean()
        vstd = v.ewm(halflife=20, min_periods=5).std()
        cols["vol_zscore"] = (v - vmean) / vstd.replace(0.0, np.nan)
    else:
        cols["vol_zscore"] = pd.Series(0.0, index=close.index)

    # ---- OHLC-based indicators (master features list, daily-computable) ----
    h = high.astype(float) if high is not None else close
    l = low.astype(float) if low is not None else close
    tr = pd.concat([h - l, (h - close.shift()).abs(), (l - close.shift()).abs()],
                   axis=1).max(axis=1)
    cols["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=5).mean()

    # ADX(14)
    up = h.diff(); dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
    atr_e = tr.ewm(alpha=1 / 14, min_periods=5).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=5).mean() / atr_e.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=5).mean() / atr_e.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    cols["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=5).mean()

    # Stochastic %K/%D
    ll14 = l.rolling(14).min(); hh14 = h.rolling(14).max()
    k = 100 * (close - ll14) / (hh14 - ll14).replace(0, np.nan)
    cols["stoch_k_14"] = k
    cols["stoch_d_14"] = k.rolling(3).mean()

    # Williams %R
    cols["williams_r_14"] = -100 * (hh14 - close) / (hh14 - ll14).replace(0, np.nan)

    # CCI(20)
    tp = (h + l + close) / 3
    tp_ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda s: np.abs(s - s.mean()).mean(), raw=True)
    cols["cci_20"] = (tp - tp_ma) / (0.015 * md.replace(0, np.nan))

    # ROC(12)
    cols["roc_12"] = close.pct_change(12)

    # Bollinger bandwidth + %B (20, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    cols["boll_bandwidth_20"] = 4 * std20 / sma20.replace(0, np.nan)
    cols["boll_pct_b_20"] = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # Parkinson + Garman-Klass vol (20)
    if (h > l).any():
        park = (np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))
        cols["parkinson_vol_20"] = np.sqrt(park.clip(lower=0)) * np.sqrt(252)
        hl = (np.log(h / l) ** 2) / 2 - (2 * np.log(2) - 1) * (np.log(close.shift() / close) ** 2)
        cols["gk_vol_20"] = np.sqrt(hl.rolling(20).mean().clip(lower=0)) * np.sqrt(252)
    else:
        cols["parkinson_vol_20"] = cols["rv_20"] if "rv_20" in cols else cols["rv_21"]
        cols["gk_vol_20"] = cols["parkinson_vol_20"]

    # Chaikin Money Flow(20) + Money Flow Index(14)
    if vol is not None and vol.notna().any():
        mfm = ((close - l) - (h - close)) / (h - l).replace(0, np.nan)
        mfv = mfm * vol
        cols["cmf_20"] = mfv.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
        rng = h - l
        typical = tp
        pos_mf = (typical * vol).where(typical > typical.shift(), 0.0)
        neg_mf = (typical * vol).where(typical < typical.shift(), 0.0)
        mfi = 100 - 100 / (1 + pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan))
        cols["mfi_14"] = mfi
    else:
        cols["cmf_20"] = pd.Series(0.0, index=close.index)
        cols["mfi_14"] = pd.Series(50.0, index=close.index)

    # Choppiness Index(14)
    atr_sum = tr.rolling(14).sum()
    hi = h.rolling(14).max(); lo = l.rolling(14).min()
    cols["choppiness_14"] = 100 * np.log(atr_sum / (hi - lo).replace(0, np.nan)) / np.log(14)

    # Kaufman Efficiency Ratio(20)
    net = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    cols["kaufman_er_20"] = net / path.replace(0, np.nan)

    # Close location value within recent high-low range (20)
    cols["close_loc"] = (close - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)

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


def add_cross_sectional_features(panel: pd.DataFrame,
                                 mom_cols: list[str] | None = None) -> pd.DataFrame:
    """Add cross-sectional ranks/z-scores of momentum features per timestamp.

    panel: long-format [time, symbol, <features>]. For each timestamp, rank the
    given momentum columns across symbols into [0,1] and add z-scores. This is
    what makes a cross-sectional momentum strategy learnable (features are
    comparable across assets).
    """
    mom_cols = mom_cols or [f"ret_norm_{h}" for h in RETURN_HORIZONS]
    out = panel.copy()
    for c in mom_cols:
        if c not in out.columns:
            continue
        rk = out.groupby("time")[c].rank(pct=True)
        out[f"{c}_cs_rank"] = rk
        z = out.groupby("time")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
        out[f"{c}_cs_z"] = z
    return out


def build_universe_frame(prices: pd.DataFrame, volumes: pd.DataFrame | None = None,
                         highs: pd.DataFrame | None = None,
                         lows: pd.DataFrame | None = None,
                         cs_rank: bool = True) -> pd.DataFrame:
    """Feature panel for a cross-section of assets.

    prices: DataFrame, columns = symbols, index = DatetimeIndex (close prices).
    volumes/highs/lows: optional DataFrames of the same shape for volume / high /
    low. Returns long-format DataFrame with columns
      [time, symbol, <features>, sigma, vs_factor, ret_1, target] (+ cs features).
    Rows where any required feature is NaN are dropped.
    """
    frames = []
    for sym in prices.columns:
        close = prices[sym].dropna()
        if len(close) < 300:
            continue
        vol = None
        if volumes is not None and sym in volumes.columns:
            vol = volumes[sym].reindex(close.index)
        high = lows_df = None
        if highs is not None and sym in highs.columns:
            high = highs[sym].reindex(close.index)
        if lows is not None and sym in lows.columns:
            lows_df = lows[sym].reindex(close.index)
        feats = build_features(close, vol=vol, high=high, low=lows_df)
        target = build_target(close)
        df = feats.join(target.rename("target"))
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise ValueError("no asset has enough history")
    panel = pd.concat(frames).sort_index()
    panel = panel.reset_index().rename(columns={"index": "time"})

    feat_cols = [c for c in panel.columns
                 if c in FEATURE_COLS or c in ("sigma", "vs_factor")]
    panel = panel.dropna(subset=feat_cols)
    if cs_rank:
        panel = add_cross_sectional_features(panel)

    out_cols = ["time", "symbol", "ret_1", "sigma", "vs_factor", "target"] \
        + [c for c in FEATURE_COLS if c in panel.columns] \
        + [c for c in panel.columns if c.endswith("_cs_rank") or c.endswith("_cs_z")]
    return panel[out_cols]


# Full feature list actually emitted (incl. cross-sectional columns)
ALL_FEATURE_COLS = FEATURE_COLS + [
    f"{c}_cs_rank" for c in [f"ret_norm_{h}" for h in RETURN_HORIZONS]
] + [
    f"{c}_cs_z" for c in [f"ret_norm_{h}" for h in RETURN_HORIZONS]
]
