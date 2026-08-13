"""Alpha metrics (plan §9.4 metrics 8-10)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    return float((1 + total_return) ** (1 / years) - 1)


def align_returns(equity: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([equity.pct_change().rename("strategy"),
                    benchmark.pct_change().rename("bench")], axis=1).dropna()
    return df["strategy"], df["bench"]


def alpha_vs_benchmark(equity: pd.Series, benchmark: pd.Series) -> float:
    """Annualized alpha: mean(strat) - mean(bench), annualized."""
    s, b = align_returns(equity, benchmark)
    if len(s) < 2:
        return 0.0
    return float((s.mean() - b.mean()) * 252)


def information_ratio(equity: pd.Series, benchmark: pd.Series) -> float:
    s, b = align_returns(equity, benchmark)
    if len(s) < 2:
        return 0.0
    active = s - b
    te = active.std()
    if te == 0 or np.isnan(te):
        return 0.0
    return float(active.mean() / te * np.sqrt(252))


def tracking_error(equity: pd.Series, benchmark: pd.Series) -> float:
    s, b = align_returns(equity, benchmark)
    if len(s) < 2:
        return 0.0
    return float((s - b).std() * np.sqrt(252))
