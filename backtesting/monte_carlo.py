"""Monte Carlo simulation (plan §9.5) — randomize entries/exits, N runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics.sharpe import sharpe_ratio
from core.logging import get_logger

log = get_logger(__name__)


def monte_carlo_shuffle(symbol: str, asset_class: str, ohlcv: pd.DataFrame,
                        signals: pd.Series, n_runs: int = 1000,
                        engine: BacktestEngine | None = None,
                        seed: int = 42) -> pd.DataFrame:
    """
    Randomize signal timing (block-shuffle) N times and re-run the backtest.

    A strategy with genuine edge should beat most shuffled variants;
    the 95th percentile of shuffled Sharpe approximates the "luck" baseline.
    """
    rng = np.random.default_rng(seed)
    engine = engine or BacktestEngine()
    rows = []
    n = len(signals)
    block = max(5, n // 20)

    for run in range(n_runs):
        # Block-shuffle the signal series to destroy temporal structure
        shuffled = signals.copy()
        idx = np.arange(n)
        n_blocks = int(np.ceil(n / block))
        perm = rng.permutation(n_blocks)
        blocks = np.array_split(idx, n_blocks)
        order = np.concatenate([blocks[p] for p in perm])
        shuffled = shuffled.iloc[order]
        shuffled.index = signals.index  # keep original dates

        try:
            result = engine.run(symbol, asset_class, ohlcv, shuffled,
                                regime="MONTE_CARLO",
                                strategy_name="shuffled")
            rows.append({
                "run": run,
                "total_return_pct": result.report.total_return_pct,
                "sharpe": result.report.sharpe_ratio,
                "max_dd_pct": result.report.max_drawdown_pct,
            })
        except Exception:  # noqa: BLE001
            continue

    df = pd.DataFrame(rows)
    log.info("Monte Carlo: %d shuffled runs complete", len(df))
    return df


def mc_summary(mc_df: pd.DataFrame, actual_sharpe: float) -> dict:
    """Percentile summary + where the real strategy ranks."""
    if mc_df.empty:
        return {}
    return {
        "runs": len(mc_df),
        "sharpe_mean": float(mc_df["sharpe"].mean()),
        "sharpe_p95": float(mc_df["sharpe"].quantile(0.95)),
        "sharpe_max": float(mc_df["sharpe"].max()),
        "max_dd_mean_pct": float(mc_df["max_dd_pct"].mean()),
        "actual_sharpe": actual_sharpe,
        "pctile_of_actual": float((mc_df["sharpe"] < actual_sharpe).mean()),
    }
