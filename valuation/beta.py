"""Beta estimation — regression, adjusted, and unlever/relever.

Professional practice (Damodaran): a raw OLS regression beta against a broad
market index (Rj = a + b·Rm) has high standard error and reflects the firm's
past business mix + leverage. So:
  * adjusted beta = (2/3)·raw + (1/3)·1.0  (Blume-adjust toward market mean)
  * bottom-up approach: unlever the peer beta with the peer's D/E, then
    relever with the firm's own D/E.

Functions here are pure (take return series), plus a db-backed helper that
pulls daily returns for a symbol + benchmark from the store.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BENCHMARK = "SPY"
MIN_OBS = 60


def regression_beta(symbol_returns: pd.Series, benchmark_returns: pd.Series,
                    min_obs: int = MIN_OBS) -> dict:
    """OLS beta of symbol returns on benchmark returns over common dates."""
    df = pd.concat([symbol_returns.rename("s"), benchmark_returns.rename("m")], axis=1)
    df = df.dropna()
    if len(df) < min_obs:
        return {"beta": None, "r_squared": None, "std_error": None, "obs": len(df)}
    cov = np.cov(df["s"], df["m"])
    var_m = cov[1, 1]
    if var_m <= 0:
        return {"beta": None, "r_squared": None, "std_error": None, "obs": len(df)}
    beta = cov[0, 1] / var_m
    corr = np.corrcoef(df["s"], df["m"])[0, 1]
    r_squared = corr ** 2
    # standard error of the slope
    resid = df["s"] - (df["m"].mean() + beta * (df["m"] - df["m"].mean()))
    dof = len(df) - 2
    se = float(np.sqrt((resid ** 2).sum() / dof) / np.sqrt(var_m * len(df))) if dof > 0 else None
    return {"beta": float(beta), "r_squared": float(r_squared),
            "std_error": se, "obs": len(df)}


def adjusted_beta(raw_beta: float | None) -> float | None:
    """Blume adjustment: adjusted = (2/3)·raw + (1/3)·1.0."""
    if raw_beta is None:
        return None
    return float((2.0 / 3.0) * raw_beta + (1.0 / 3.0) * 1.0)


def unlever_beta(levered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)


def relever_beta(unlevered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)


def estimate_beta(symbol: str, db, benchmark: str = DEFAULT_BENCHMARK,
                  min_obs: int = MIN_OBS) -> dict:
    """Db-backed regression + adjusted beta for a symbol vs a benchmark."""
    sym = db.query_ohlcv(symbol)
    bench = db.query_ohlcv(benchmark)
    if sym.empty or bench.empty:
        return {"beta": None, "adjusted_beta": None, "r_squared": None,
                "std_error": None, "obs": 0, "benchmark": benchmark}
    s_ret = sym["close"].astype(float).pct_change().dropna()
    b_ret = bench["close"].astype(float).pct_change().dropna()
    reg = regression_beta(s_ret, b_ret, min_obs=min_obs)
    reg["adjusted_beta"] = adjusted_beta(reg["beta"])
    reg["benchmark"] = benchmark
    return reg


def bottom_up_beta(peer_unlevered_beta: float, debt_to_equity: float,
                   tax_rate: float) -> float:
    """Relever an average peer unlevered beta with the firm's own D/E."""
    return relever_beta(peer_unlevered_beta, debt_to_equity, tax_rate)
