"""Sharpe / Sortino ratios (plan §9.4 metrics 1-2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def volatility_annualized(returns: pd.Series,
                          periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
