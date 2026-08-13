"""Drawdown metrics (plan §9.4 metric 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    dd = drawdown_series(equity)
    return float(dd.min())


def max_drawdown_duration_days(equity: pd.Series) -> int:
    """Longest underwater streak in calendar days."""
    if len(equity) < 2:
        return 0
    dd = drawdown_series(equity)
    underwater = (dd < 0).astype(int)
    longest = 0
    current = 0
    for v in underwater.values:
        current = current + 1 if v else 0
        longest = max(longest, current)
    if longest == 0:
        return 0
    start = underwater[underwater == 1].index[0]
    end = underwater[underwater == 1].index[longest - 1] if longest <= len(underwater) else underwater.index[-1]
    return int((end - start).days) if longest <= len(underwater) else int((underwater.index[-1] - underwater.index[0]).days)


def calmar_ratio(cagr: float, max_dd: float) -> float:
    if max_dd >= 0:
        return 0.0
    return float(cagr / abs(max_dd))


def daily_var_95(returns: pd.Series) -> float:
    if len(returns) < 30:
        return 0.0
    return float(-np.percentile(returns.dropna(), 5))
