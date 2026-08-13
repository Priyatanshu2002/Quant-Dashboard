"""Backtesting engine — signal series → equity curve + trades + full report.

Execution model (daily bars):
  * Signals are target position fractions [-1, 1] aligned to the OHLCV index.
  * When the target changes, the new position is filled at the NEXT bar open,
    with spread + slippage + commission from the CostModel.
  * Equity = cash + position_value at each bar close.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from backtesting.cost_model import CostModel
from backtesting.metrics.all_metrics import BacktestReport, compute_report
from core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Trade:
    symbol: str
    direction: str            # LONG / SHORT / FLAT
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    quantity: float = 0.0
    notional_usd: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    commission_usd: float = 0.0
    slippage_usd: float = 0.0
    exit_reason: str = "signal"

    def holding_hours(self) -> float:
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestResult:
    report: BacktestReport
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    def __init__(self, cost_model: CostModel | None = None,
                 initial_cash: float = 1_000_000.0,
                 spread_pct: float = 0.0005,
                 avg_daily_volume_usd: float = 1e9):
        self.cost_model = cost_model or CostModel()
        self.initial_cash = initial_cash
        self.spread_pct = spread_pct
        self.avg_daily_volume_usd = avg_daily_volume_usd

    def run(self, symbol: str, asset_class: str, ohlcv: pd.DataFrame,
            signals: pd.Series, benchmark: pd.Series | None = None,
            regime: str = "FULL", strategy_name: str = "strategy") -> BacktestResult:
        """Run the backtest. signals: target position fraction per bar."""
        df = ohlcv.copy()
        sig = signals.reindex(df.index).fillna(0.0).clip(-1, 1)
        n = len(df)

        cash = self.initial_cash
        position = 0.0                      # signed units
        trades: list[Trade] = []
        equity = np.empty(n)
        open_trade: Trade | None = None
        total_commission = 0.0
        total_slippage = 0.0

        for t in range(n):
            target = sig.iloc[t]
            price_close = float(df["close"].iloc[t])

            # Execute target changes at the next bar open (no look-ahead)
            if t > 0 and target != position:
                fill = float(df["open"].iloc[t])
                delta = target - position
                notional = abs(delta) * self.initial_cash * 0.5  # scale by half NAV
                qty = delta * (notional / fill)
                round_trip = self.cost_model.round_trip_cost_pct(
                    asset_class, notional, self.avg_daily_volume_usd,
                    self.spread_pct)
                cost_usd = notional * round_trip / 2      # half the round trip now

                if open_trade is not None:
                    open_trade.exit_time = df.index[t]
                    open_trade.exit_price = fill
                    open_trade.pnl_usd = (fill - open_trade.entry_price) * open_trade.quantity
                    if open_trade.direction == "SHORT":
                        open_trade.pnl_usd = -open_trade.pnl_usd
                    open_trade.pnl_usd -= open_trade.commission_usd + open_trade.slippage_usd
                    open_trade.pnl_pct = (open_trade.pnl_usd / open_trade.notional_usd
                                          if open_trade.notional_usd else 0.0)
                    cash += open_trade.pnl_usd
                    trades.append(open_trade)
                    open_trade = None

                if delta != 0:
                    direction = "LONG" if delta > 0 else "SHORT"
                    open_trade = Trade(
                        symbol=symbol, direction=direction,
                        entry_time=df.index[t], entry_price=fill,
                        quantity=qty, notional_usd=notional,
                        commission_usd=cost_usd, slippage_usd=0.0)
                    total_commission += cost_usd
                    cash -= cost_usd
                position = target

            equity[t] = cash + position * price_close

        # Close any remaining position at the last close
        if open_trade is not None:
            fill = float(df["close"].iloc[-1])
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = fill
            open_trade.pnl_usd = (fill - open_trade.entry_price) * open_trade.quantity
            if open_trade.direction == "SHORT":
                open_trade.pnl_usd = -open_trade.pnl_usd
            open_trade.pnl_usd -= open_trade.commission_usd + open_trade.slippage_usd
            open_trade.pnl_pct = (open_trade.pnl_usd / open_trade.notional_usd
                                  if open_trade.notional_usd else 0.0)
            trades.append(open_trade)

        equity_curve = pd.Series(equity, index=df.index, name="equity")
        report = compute_report(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            trades=trades,
            benchmark=benchmark,
            period_start=df.index[0].date(),
            period_end=df.index[-1].date(),
            regime=regime,
            initial_cash=self.initial_cash,
            total_commission=total_commission,
            total_slippage=total_slippage,
        )
        return BacktestResult(report=report, equity_curve=equity_curve,
                              trades=trades,
                              drawdown_curve=drawdown_series(equity_curve))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater curve: (equity / running_peak) - 1."""
    peak = equity.cummax()
    return equity / peak - 1.0
