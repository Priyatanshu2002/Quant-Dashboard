"""Composite reward (plan §6.1) — Sharpe contribution − drawdown penalty − costs."""
from __future__ import annotations

import numpy as np


def compute_reward(raw_return: float, position_pct: float,
                   portfolio_volatility: float, current_drawdown: float,
                   slippage: float, commission_rate: float = 0.001,
                   max_drawdown_limit: float = 0.15) -> float:
    """
    Composite reward:
      + Sharpe contribution of this trade
      - Drawdown penalty if we exceed drawdown limit
      - Slippage cost
      - Transaction cost
    """
    commission = commission_rate * position_pct
    sharpe_term = raw_return / (portfolio_volatility + 1e-8)
    dd_penalty = max(0.0, current_drawdown - max_drawdown_limit)

    return sharpe_term * 1.0 - dd_penalty * 2.0 - slippage * 0.5 - commission * 0.5


def estimate_slippage(position_pct: float, order_type: str = "MARKET") -> float:
    """Slippage grows with size; LIMIT/TWAP orders slip less than MARKET."""
    base = 0.0005 + 0.003 * position_pct
    if order_type == "LIMIT":
        return base * 0.4
    if order_type == "TWAP":
        return base * 0.6
    return base
