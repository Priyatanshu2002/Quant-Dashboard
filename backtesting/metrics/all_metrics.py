"""BacktestReport — the full 10-metric report (plan §9.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtesting.metrics.alpha_metrics import (
    alpha_vs_benchmark, cagr, information_ratio, tracking_error)
from backtesting.metrics.drawdown import (
    calmar_ratio, daily_var_95, drawdown_series, max_drawdown, max_drawdown_duration_days)
from backtesting.metrics.sharpe import sharpe_ratio, sortino_ratio, volatility_annualized
from backtesting.metrics.trade_stats import trade_statistics

if TYPE_CHECKING:  # avoid circular import (engine imports all_metrics)
    from backtesting.engine import Trade


@dataclass
class BacktestReport:
    strategy_name: str
    period_start: date
    period_end: date
    regime: str

    # Core performance
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    daily_var_95: float = 0.0
    volatility_annualized: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_per_trade_usd: float = 0.0
    avg_holding_period_hours: float = 0.0

    # Alpha
    alpha_vs_sp500: float = 0.0
    alpha_vs_nifty50: float = 0.0
    alpha_vs_btc: float = 0.0

    # Cost impact
    total_commission_paid_usd: float = 0.0
    total_slippage_usd: float = 0.0
    total_funding_cost_usd: float = 0.0
    cost_drag_pct: float = 0.0

    # Per-regime performance
    performance_by_regime: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def summary_row(self) -> list:
        return [self.strategy_name, self.regime, f"{self.total_return_pct:.1f}%",
                f"{self.cagr:.1f}%", f"{self.sharpe_ratio:.2f}",
                f"{self.sortino_ratio:.2f}", f"{self.calmar_ratio:.2f}",
                f"{self.max_drawdown_pct:.1f}%", f"{self.win_rate:.0%}",
                f"{self.profit_factor:.2f}", f"{self.information_ratio:.2f}"]


def compute_report(strategy_name: str, equity_curve: pd.Series,
                   trades: list[Any], benchmark: pd.Series | None,
                   period_start: date, period_end: date, regime: str,
                   initial_cash: float, total_commission: float = 0.0,
                   total_slippage: float = 0.0) -> BacktestReport:
    rets = equity_curve.pct_change().dropna()
    stats = trade_statistics(trades)
    total_return = equity_curve.iloc[-1] / initial_cash - 1
    cagr_val = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    dd_duration = max_drawdown_duration_days(equity_curve)
    costs = total_commission + total_slippage

    report = BacktestReport(
        strategy_name=strategy_name, period_start=period_start,
        period_end=period_end, regime=regime,
        total_return_pct=total_return * 100,
        cagr=cagr_val * 100,
        sharpe_ratio=sharpe_ratio(rets),
        sortino_ratio=sortino_ratio(rets),
        calmar_ratio=calmar_ratio(cagr_val, max_dd),
        max_drawdown_pct=max_dd * 100,
        max_drawdown_duration_days=dd_duration,
        daily_var_95=daily_var_95(rets) * 100,
        volatility_annualized=volatility_annualized(rets) * 100,
        total_trades=stats["total_trades"],
        win_rate=stats["win_rate"],
        avg_win_pct=stats["avg_win_pct"] * 100,
        avg_loss_pct=stats["avg_loss_pct"] * 100,
        profit_factor=stats["profit_factor"],
        expectancy_per_trade_usd=stats["expectancy_per_trade_usd"],
        avg_holding_period_hours=stats["avg_holding_period_hours"],
        total_commission_paid_usd=total_commission,
        total_slippage_usd=total_slippage,
        cost_drag_pct=(costs / initial_cash) * 100 if initial_cash else 0.0,
    )

    if benchmark is not None and len(benchmark) > 1:
        report.alpha_vs_sp500 = alpha_vs_benchmark(equity_curve, benchmark) * 100
        report.information_ratio = information_ratio(equity_curve, benchmark)
    return report


def performance_by_regime(equity: pd.Series, regimes: dict[str, tuple[str, str]]) -> dict[str, float]:
    """Sharpe per historical regime (plan §9.3)."""
    out: dict[str, float] = {}
    for name, (start, end) in regimes.items():
        window = equity.loc[start:end]
        if len(window) > 20:
            out[name] = sharpe_ratio(window.pct_change().dropna())
    return out
