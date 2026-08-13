"""Built-in strategy signal generators for backtesting and smoke tests.

Each strategy exposes:
  fit(ohlcv)          — optional warmup/parameter estimation
  generate_signals()  — pd.Series of target position fraction [-1, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MACrossStrategy:
    """Fast EMA vs slow EMA crossover, long-only with optional short side."""

    def __init__(self, fast: int = 20, slow: int = 100, allow_short: bool = False):
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.signals: pd.Series | None = None

    def fit(self, ohlcv: pd.DataFrame) -> "MACrossStrategy":
        close = ohlcv["close"]
        fast_ema = close.ewm(span=self.fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow, adjust=False).mean()
        pos = np.where(fast_ema > slow_ema, 1.0, -1.0 if self.allow_short else 0.0)
        self.signals = pd.Series(pos, index=ohlcv.index)
        return self

    def generate_signals(self) -> pd.Series:
        assert self.signals is not None, "call fit() first"
        return self.signals


class RsiMeanReversionStrategy:
    """RSI mean reversion: buy dips (RSI<30), exit at RSI>55; inverse for shorts."""

    def __init__(self, period: int = 14, oversold: float = 30.0,
                 exit_level: float = 55.0, allow_short: bool = False):
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.allow_short = allow_short
        self.signals: pd.Series | None = None

    def fit(self, ohlcv: pd.DataFrame) -> "RsiMeanReversionStrategy":
        import ta
        rsi = ta.momentum.RSIIndicator(ohlcv["close"], self.period).rsi()
        pos = np.zeros(len(ohlcv))
        state = 0.0
        for i in range(len(ohlcv)):
            r = rsi.iloc[i]
            if np.isnan(r):
                continue
            if state == 0 and r < self.oversold:
                state = 1.0
            elif state == 1 and r > self.exit_level:
                state = 0.0
            elif self.allow_short and state == 0 and r > 100 - self.oversold:
                state = -1.0
            elif self.allow_short and state == -1 and r < 100 - self.exit_level:
                state = 0.0
            pos[i] = state
        self.signals = pd.Series(pos, index=ohlcv.index)
        return self

    def generate_signals(self) -> pd.Series:
        assert self.signals is not None, "call fit() first"
        return self.signals


class MomentumStrategy:
    """Breakout momentum: long when 20d return strong and above 50d MA."""

    def __init__(self, lookback: int = 20, threshold: float = 0.03,
                 ma: int = 50):
        self.lookback = lookback
        self.threshold = threshold
        self.ma = ma
        self.signals: pd.Series | None = None

    def fit(self, ohlcv: pd.DataFrame) -> "MomentumStrategy":
        close = ohlcv["close"]
        ret = close.pct_change(self.lookback)
        above_ma = close > close.rolling(self.ma).mean()
        pos = np.where((ret > self.threshold) & above_ma, 1.0, 0.0)
        self.signals = pd.Series(pos, index=ohlcv.index)
        return self

    def generate_signals(self) -> pd.Series:
        assert self.signals is not None, "call fit() first"
        return self.signals


STRATEGIES = {
    "ma_cross": MACrossStrategy,
    "rsi_reversion": RsiMeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def make_strategy(name: str, **kwargs):
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. Choose from {list(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
