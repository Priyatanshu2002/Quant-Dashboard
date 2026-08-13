"""Technical feature engineering (~30 features) — vectorized per-bar.

Returns a DataFrame with one row per bar so the same code serves both
live inference (take .iloc[-1]) and historical feature-store backfills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the plan §3.2 feature set for the full OHLCV frame."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    vol = ohlcv["volume"].astype(float)

    f = pd.DataFrame(index=ohlcv.index)

    # ── Momentum oscillators ──
    f["rsi_14"] = ta.momentum.RSIIndicator(close, 14).rsi()
    f["rsi_7"] = ta.momentum.RSIIndicator(close, 7).rsi()
    f["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    f["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    f["roc_10"] = ta.momentum.ROCIndicator(close, 10).roc()

    # ── Trend ──
    macd = ta.trend.MACD(close)
    f["macd_line"] = macd.macd()
    f["macd_signal_val"] = macd.macd_signal()
    f["macd_histogram"] = macd.macd_diff()
    for n in (9, 21, 50, 200):
        f[f"ema_{n}"] = ta.trend.EMAIndicator(close, n).ema_indicator()
    f["price_vs_ema200_pct"] = (close / f["ema_200"] - 1) * 100
    f["adx_14"] = ta.trend.ADXIndicator(high, low, close, 14).adx()
    f["cci_20"] = ta.trend.CCIIndicator(high, low, close, 20).cci()

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(close, 20, 2)
    f["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    f["bb_pct_b"] = bb.bollinger_pband()
    f["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    f["atr_pct"] = f["atr_14"] / close
    f["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # ── Volume ──
    vol_ma20 = vol.rolling(20).mean()
    vol_std20 = vol.rolling(20).std()
    f["volume_z_score"] = (vol - vol_ma20) / vol_std20.replace(0, np.nan)
    f["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    vwap = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()
    f["vwap_pct"] = (close / vwap - 1) * 100

    # ── Multi-timeframe returns ──
    for n, label in [(1, "1bar"), (5, "5bar"), (20, "20bar"), (60, "60bar")]:
        f[f"return_{label}"] = (close / close.shift(n) - 1) * 100

    # Replace inf with NaN, drop leading warmup rows lacking enough history
    return f.replace([np.inf, -np.inf], np.nan)


def latest_technical_features(ohlcv: pd.DataFrame) -> dict:
    """Single latest feature dict (live inference)."""
    frame = compute_technical_features(ohlcv)
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
