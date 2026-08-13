"""Trade statistics (plan §9.4 metrics 5-7, 11)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid circular import (engine → all_metrics → trade_stats)
    from backtesting.engine import Trade


def trade_statistics(trades: list[Any]) -> dict:
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "win_rate": 0.0, "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0, "profit_factor": 0.0,
                "expectancy_per_trade_usd": 0.0, "avg_holding_period_hours": 0.0}

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))

    avg_holding = sum(t.holding_hours() for t in trades) / n
    return {
        "total_trades": n,
        "win_rate": len(wins) / n,
        "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0,
        "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "expectancy_per_trade_usd": sum(t.pnl_usd for t in trades) / n,
        "avg_holding_period_hours": avg_holding,
    }
