"""Technical feature engineering (~75 features) — vectorized per-bar.

Returns a DataFrame with one row per bar so the same code serves both live
inference (take .iloc[-1]) and historical feature-store backfills.

Covers momentum, trend (incl. PSAR/Supertrend/Ichimoku/Aroon), volatility
(incl. Keltner/range-vol/realized-vol/skew/kurt), and volume/money-flow
(incl. MFI/ADL). All indicators come from the `ta` library; vectorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the full (~75) technical feature set for an OHLCV frame.

    Requires columns: open, high, low, close, volume. Warm-up NaNs are left
    in place (consumers drop or carry them); inf is replaced with NaN.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    vol = ohlcv["volume"].astype(float)
    typ_price = (high + low + close) / 3

    f = pd.DataFrame(index=ohlcv.index)

    # ── Momentum oscillators ──
    f["rsi_14"] = ta.momentum.RSIIndicator(close, 14).rsi()
    f["rsi_7"] = ta.momentum.RSIIndicator(close, 7).rsi()
    f["rsi_2"] = ta.momentum.RSIIndicator(close, 2).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()
    f["stoch_rsi"] = ta.momentum.StochRSIIndicator(close, 14, 3).stochrsi()
    f["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r()
    f["roc_10"] = ta.momentum.ROCIndicator(close, 10).roc()
    f["roc_21"] = ta.momentum.ROCIndicator(close, 21).roc()
    f["tsi"] = ta.momentum.TSIIndicator(close).tsi()
    f["uo"] = ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator()
    f["awesome"] = ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()
    f["mfi_14"] = ta.volume.MFIIndicator(high, low, close, vol, 14).money_flow_index()

    # ── Trend ──
    macd = ta.trend.MACD(close)
    f["macd_line"] = macd.macd()
    f["macd_signal_val"] = macd.macd_signal()
    f["macd_histogram"] = macd.macd_diff()
    for n in (9, 21, 50, 200):
        f[f"ema_{n}"] = ta.trend.EMAIndicator(close, n).ema_indicator()
    f["price_vs_ema200_pct"] = (close / f["ema_200"] - 1) * 100
    f["price_vs_ema50_pct"] = (close / f["ema_50"] - 1) * 100
    f["adx_14"] = ta.trend.ADXIndicator(high, low, close, 14).adx()
    f["adx_pos"] = ta.trend.ADXIndicator(high, low, close, 14).adx_pos()
    f["adx_neg"] = ta.trend.ADXIndicator(high, low, close, 14).adx_neg()
    f["cci_20"] = ta.trend.CCIIndicator(high, low, close, 20).cci()
    psar = ta.trend.PSARIndicator(high, low, close)
    f["psar"] = psar.psar()
    f["psar_direction"] = psar.psar_up()  # 1.0 bullish / 0.0 bearish
    ichi = _ichimoku(high, low)
    for k, v in ichi.items():
        f[k] = v
    aroon = _aroon(high, low, 25)
    f["aroon_up"] = aroon["aroon_up"]
    f["aroon_down"] = aroon["aroon_down"]
    f["aroon_osc"] = aroon["aroon_osc"]
    st = _supertrend(high, low, close, 10, 3.0)
    f["supertrend"] = st["supertrend"]
    f["supertrend_direction"] = st["supertrend_direction"]  # 1 / -1

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(close, 20, 2)
    f["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    f["bb_pct_b"] = bb.bollinger_pband()
    f["bb_bandwidth"] = bb.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(high, low, close, 14)
    f["atr_14"] = atr.average_true_range()
    f["atr_pct"] = f["atr_14"] / close
    kc = ta.volatility.KeltnerChannel(high, low, close, 20, 2)
    kc_band = kc.keltner_channel_hband() - kc.keltner_channel_lband()
    f["keltner_width"] = kc_band / kc.keltner_channel_mband()
    f["keltner_pct_b"] = (close - kc.keltner_channel_lband()) / kc_band.replace(0, np.nan)
    don = ta.volatility.DonchianChannel(high, low, close, 20)
    f["donchian_up"] = don.donchian_channel_hband()
    f["donchian_down"] = don.donchian_channel_lband()
    f["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    f["realized_vol_10"] = close.pct_change().rolling(10).std() * np.sqrt(252)
    f["range_vol_14"] = (high - low).rolling(14).mean() / close
    f["intraday_range_pct"] = (high - low) / close * 100
    ret = close.pct_change()
    f["return_skew_20"] = ret.rolling(20).skew()
    f["return_kurt_20"] = ret.rolling(20).kurt()
    f["return_skew_60"] = ret.rolling(60).skew()
    f["return_kurt_60"] = ret.rolling(60).kurt()

    # ── Volume / money flow ──
    vol_ma20 = vol.rolling(20).mean()
    vol_std20 = vol.rolling(20).std()
    f["volume_z_score"] = (vol - vol_ma20) / vol_std20.replace(0, np.nan)
    f["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    f["obv_slope_10"] = f["obv"].diff(10)
    f["adl"] = ta.volume.AccDistIndexIndicator(high, low, close, vol).acc_dist_index()
    ad = (f["adl"] - f["adl"].shift(1)).rolling(10).mean()
    f["adl_slope_10"] = ad
    f["vwap_pct"] = (close / (vol * typ_price).cumsum().div(vol.cumsum()) - 1) * 100
    f["volume_ratio_5"] = vol / vol.rolling(5).mean().replace(0, np.nan)

    # ── Multi-timeframe returns ──
    for n, label in [(1, "1bar"), (5, "5bar"), (10, "10bar"), (20, "20bar"),
                     (60, "60bar")]:
        f[f"return_{label}"] = (close / close.shift(n) - 1) * 100

    # Replace inf with NaN
    return f.replace([np.inf, -np.inf], np.nan)


def _supertrend(high, low, close, period: int = 10, mult: float = 3.0):
    """Vectorized Supertrend (version-independent implementation).

    Returns {supertrend, supertrend_direction} where direction is +1 (bullish,
    price above the band) or −1 (bearish). ATR uses the classic Wilder method.
    """
    hl2 = (high + low) / 2
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()

    basic_ub = (hl2 + mult * atr).to_numpy()
    basic_lb = (hl2 - mult * atr).to_numpy()
    close_np = close.to_numpy()
    n = len(close)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    direction = np.zeros(n)
    band = np.full(n, np.nan)

    for i in range(n):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
            # Start bullish iff price is above the first lower band.
            direction[i] = 1.0 if not np.isnan(basic_lb[i]) and close_np[i] >= basic_lb[i] else -1.0
            band[i] = final_lb[i] if direction[i] == 1.0 else final_ub[i]
            continue
        pu, pl = final_ub[i - 1], final_lb[i - 1]
        # Upper band carry: use new basic unless a prior (non-NaN) band holds
        # and price hasn't closed above it.
        if not np.isnan(pu) and not (basic_ub[i] < pu or close_np[i - 1] > pu):
            final_ub[i] = pu
        else:
            final_ub[i] = basic_ub[i]
        if not np.isnan(pl) and not (basic_lb[i] > pl or close_np[i - 1] < pl):
            final_lb[i] = pl
        else:
            final_lb[i] = basic_lb[i]
        # Skip while the band is still NaN (ATR warmup).
        if np.isnan(final_lb[i]) or np.isnan(final_ub[i]):
            direction[i] = direction[i - 1]
            band[i] = band[i - 1]
            continue
        # Direction flips
        if direction[i - 1] == 1.0 and close_np[i] < final_lb[i]:
            direction[i] = -1.0
        elif direction[i - 1] == -1.0 and close_np[i] > final_ub[i]:
            direction[i] = 1.0
        else:
            direction[i] = direction[i - 1]
        band[i] = final_lb[i] if direction[i] == 1.0 else final_ub[i]

    return {"supertrend": pd.Series(band, index=close.index),
            "supertrend_direction": pd.Series(direction, index=close.index)}


def _ichimoku(high, low):
    """Ichimoku cloud components (manual — ta 0.11's Ichimoku is broken)."""
    hh, ll = high, low
    conv = (hh.rolling(9).max() + ll.rolling(9).min()) / 2
    base = (hh.rolling(26).max() + ll.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((hh.rolling(52).max() + ll.rolling(52).min()) / 2).shift(26)
    return {"ichimoku_a": span_a, "ichimoku_b": span_b,
            "ichimoku_base": base, "ichimoku_conversion": conv}


def _aroon(high, low, window: int = 25):
    """Aroon up/down/oscillator (manual — ta 0.11 signature differs).

    Aroon-up = (window − periods-since-highest)/window × 100; the rolling
    argmax over the trailing window+1 bars gives the periods-since index.
    """
    highs = high.rolling(window + 1, min_periods=1).apply(
        lambda x: np.argmax(x), raw=True)
    lows = low.rolling(window + 1, min_periods=1).apply(
        lambda x: np.argmin(x), raw=True)
    aroon_up = (highs / window) * 100
    aroon_down = (lows / window) * 100
    return {"aroon_up": aroon_up, "aroon_down": aroon_down,
            "aroon_osc": aroon_up - aroon_down}


def latest_technical_features(ohlcv: pd.DataFrame) -> dict:
    """Single latest feature dict (live inference)."""
    frame = compute_technical_features(ohlcv)
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}


# Technical features the composite scorer actually consumes. Computing just
# these (instead of the full ~75) keeps a full-universe screener pass fast.
_SCORE_NEEDED = ("rsi_14", "price_vs_ema200_pct", "macd_histogram",
                 "adx_14", "return_20bar", "return_5bar")


def latest_scoring_features(ohlcv: pd.DataFrame) -> dict:
    """Latest dict of only the technical features the screener scorer uses."""
    close = ohlcv["close"]
    high, low = ohlcv["high"], ohlcv["low"]
    out: dict[str, float | None] = {}
    rsi = ta.momentum.RSIIndicator(close, 14).rsi()
    out["rsi_14"] = _last(rsi)
    ema200 = ta.trend.EMAIndicator(close, 200).ema_indicator()
    out["price_vs_ema200_pct"] = _last((close / ema200 - 1) * 100)
    macd = ta.trend.MACD(close)
    out["macd_histogram"] = _last(macd.macd_diff())
    adx = ta.trend.ADXIndicator(high, low, close, 14).adx()
    out["adx_14"] = _last(adx)
    for n, key in ((20, "return_20bar"), (5, "return_5bar")):
        out[key] = _last((close / close.shift(n) - 1) * 100)
    return {k: v for k, v in out.items() if v is not None}


def _last(s) -> float | None:
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)
