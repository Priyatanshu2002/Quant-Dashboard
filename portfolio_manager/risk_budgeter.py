"""Risk budgeter — VaR estimation and per-position budget accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parametric_var(returns: pd.Series, confidence: float = 0.95,
                   horizon_days: int = 1) -> float:
    """Parametric VaR (fraction of value) at the given confidence."""
    mu = returns.mean()
    sigma = returns.std()
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)
    return float(-(mu * horizon_days + z * sigma * np.sqrt(horizon_days)))


def historical_var(returns: pd.Series, confidence: float = 0.95,
                   horizon_days: int = 1) -> float:
    if len(returns) < 30:
        return 0.0
    horizon_returns = returns.rolling(horizon_days).sum().dropna()
    return float(-np.percentile(horizon_returns, (1 - confidence) * 100))


def position_var(position_notional: float, volatility_annual: float,
                 confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Position-level VaR in USD: notional × vol × z × √(horizon/252).

    VaR is reported as a POSITIVE loss magnitude — the z-score is taken
    in absolute value (norm.ppf(1-c) is negative for c > 0.5).
    """
    from scipy.stats import norm
    z = abs(norm.ppf(1 - confidence))
    return abs(position_notional) * volatility_annual * z * np.sqrt(horizon_days / 252)


class RiskBudgeter:
    """Tracks risk budget usage across positions (plan §7.2: total VaR < 2%/day)."""

    def __init__(self, daily_var_limit: float = 0.02, nav_usd: float = 1_000_000.0):
        self.daily_var_limit = daily_var_limit
        self.nav_usd = nav_usd
        self.used: dict[str, float] = {}

    def allocate(self, symbol: str, var_usd: float) -> float:
        """Reserve budget; returns budget used (0-1). Raises if over limit."""
        total_var = sum(self.used.values()) + var_usd
        if total_var > self.daily_var_limit * self.nav_usd:
            raise ValueError(
                f"VaR budget exceeded: {total_var:,.0f} > "
                f"{self.daily_var_limit * self.nav_usd:,.0f}")
        self.used[symbol] = var_usd
        return var_usd / (self.daily_var_limit * self.nav_usd)

    def utilization(self) -> float:
        return sum(self.used.values()) / (self.daily_var_limit * self.nav_usd)
