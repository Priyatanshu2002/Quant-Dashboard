"""Regime-based testing (plan §9.3) — validate across all historical regimes."""
from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics.all_metrics import performance_by_regime
from core.logging import get_logger

log = get_logger(__name__)

HISTORICAL_REGIMES = {
    "COVID_CRASH":        ("2020-02-20", "2020-03-23"),
    "COVID_RECOVERY":     ("2020-03-24", "2021-01-01"),
    "BULL_2021":          ("2021-01-01", "2021-11-01"),
    "CRYPTO_WINTER_2022": ("2022-01-01", "2022-12-31"),
    "RATES_SHOCK_2022":   ("2022-03-01", "2022-12-31"),
    "BULL_2023_2024":     ("2023-01-01", "2024-12-31"),
    "AI_BUBBLE_2025":     ("2025-01-01", "2025-12-31"),
}


def run_regime_tests(symbol: str, asset_class: str, ohlcv: pd.DataFrame,
                     signals: pd.Series, engine: BacktestEngine | None = None,
                     regimes: dict[str, tuple[str, str]] | None = None) -> dict[str, BacktestResult]:
    """Run the strategy separately inside each historical regime window."""
    engine = engine or BacktestEngine()
    regimes = regimes or HISTORICAL_REGIMES
    results: dict[str, BacktestResult] = {}
    for name, (start, end) in regimes.items():
        window = ohlcv.loc[start:end]
        if len(window) < 20:
            continue
        sig = signals.reindex(window.index).fillna(0.0)
        result = engine.run(symbol, asset_class, window, sig,
                            regime=name, strategy_name="regime_test")
        results[name] = result
        log.info("Regime %s: sharpe=%.2f ret=%.1f%% trades=%d", name,
                 result.report.sharpe_ratio, result.report.total_return_pct,
                 result.report.total_trades)
    return results


def regime_sharpe_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = [{
        "regime": name,
        "total_return_pct": round(r.report.total_return_pct, 2),
        "sharpe": round(r.report.sharpe_ratio, 2),
        "max_dd_pct": round(r.report.max_drawdown_pct, 2),
        "trades": r.report.total_trades,
    } for name, r in results.items()]
    return pd.DataFrame(rows)


def full_period_regime_breakdown(equity: pd.Series) -> dict[str, float]:
    """Sharpe per regime for one continuous equity curve."""
    return performance_by_regime(equity, HISTORICAL_REGIMES)
