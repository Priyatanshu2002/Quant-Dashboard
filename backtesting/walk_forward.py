"""Walk-forward validation (plan §9.2) — retrain per window, test strictly OOS."""
from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics.all_metrics import BacktestReport
from core.logging import get_logger

log = get_logger(__name__)


def _generate_periods(start_date: str, end_date: str, train_months: int,
                      test_months: int, step_months: int):
    """Yield (train_start, train_end, test_start, test_end) tuples."""
    cursor = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while cursor + pd.DateOffset(months=train_months + test_months) <= end:
        train_start = cursor
        train_end = cursor + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        yield train_start, train_end, test_start, test_end
        cursor += pd.DateOffset(months=step_months)


def walk_forward_backtest(
    ohlcv: pd.DataFrame,
    strategy_factory,
    symbol: str = "SYM",
    asset_class: str = "EQUITY_US",
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 1,
    start_date: str = "2020-01-01",
    end_date: str = "2026-01-01",
    engine: BacktestEngine | None = None,
) -> list[BacktestResult]:
    """
    Walk-Forward Validation (prevents look-ahead bias):

    Window 1:  Train [Jan 2020 – Dec 2020] → Test [Jan–Mar 2021]
    Window 2:  Train [Feb 2020 – Jan 2021] → Test [Feb–Apr 2021]
    ...

    The strategy is RETRAINED on each window — no data from the test period
    is ever used in training for that window.
    """
    engine = engine or BacktestEngine()
    results = []
    for train_start, train_end, test_start, test_end in _generate_periods(
            start_date, end_date, train_months, test_months, step_months):
        train_df = ohlcv.loc[train_start:train_end]
        test_df = ohlcv.loc[test_start:test_end]
        if len(train_df) < 60 or len(test_df) < 20:
            continue
        strategy = strategy_factory()
        strategy.fit(train_df)                      # learn params on TRAIN ONLY
        # Compute signals over train+test history (indicators at time t only use
        # data up to t — no look-ahead), then keep the test slice.
        combined = pd.concat([train_df, test_df])
        combined = combined[~combined.index.duplicated(keep="first")]
        signals = strategy.fit(combined).generate_signals().loc[test_start:test_end]
        result = engine.run(symbol, asset_class, test_df, signals,
                            regime=f"WF {test_start.date()}→{test_end.date()}",
                            strategy_name=strategy.__class__.__name__)
        results.append(result)
    log.info("Walk-forward: %d windows evaluated", len(results))
    return results


def summarize_walk_forward(results: list[BacktestResult]) -> pd.DataFrame:
    """Sharpe / return / drawdown per window → DataFrame."""
    rows = []
    for r in results:
        rep: BacktestReport = r.report
        rows.append({
            "window": f"{rep.period_start}→{rep.period_end}",
            "total_return_pct": round(rep.total_return_pct, 2),
            "sharpe": round(rep.sharpe_ratio, 2),
            "max_dd_pct": round(rep.max_drawdown_pct, 2),
            "trades": rep.total_trades,
        })
    return pd.DataFrame(rows)
