"""Multi-asset ledger (plan §7.1) — Position dataclass + in-memory ledger ops."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Position:
    symbol: str
    asset_class: Literal["CRYPTO", "EQUITY_US", "EQUITY_IN", "ETF", "BOND", "FOREX", "FNO"]
    direction: Literal["LONG", "SHORT"]
    timeframe: Literal["INTRADAY", "SWING", "LONGTERM"]

    entry_price: float
    quantity: float
    notional_usd: float
    entry_timestamp: datetime
    cycle_id: str = ""

    current_price: float = 0.0
    unrealized_pnl_usd: float = 0.0
    unrealized_pnl_pct: float = 0.0

    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_pct: float = 0.0

    position_var: float = 0.0
    portfolio_weight: float = 0.0
    risk_budget_used: float = 0.0

    total_commission: float = 0.0
    total_slippage: float = 0.0
    funding_rate_cost: float = 0.0

    def mark(self, price: float, portfolio_nav: float = 0.0) -> None:
        self.current_price = price
        pnl = (price - self.entry_price) * self.quantity
        if self.direction == "SHORT":
            pnl = -pnl
        self.unrealized_pnl_usd = pnl
        self.unrealized_pnl_pct = pnl / self.notional_usd if self.notional_usd else 0.0
        if portfolio_nav > 0:
            self.portfolio_weight = abs(self.notional_usd) / portfolio_nav

    def close(self, price: float, commission: float = 0.0,
              slippage: float = 0.0, funding: float = 0.0) -> float:
        self.mark(price)
        realized = self.unrealized_pnl_usd - commission - slippage - funding
        self.total_commission += commission
        self.total_slippage += slippage
        self.funding_rate_cost += funding
        return realized


@dataclass
class MultiAssetLedger:
    nav_usd: float = 1_000_000.0
    cash_usd: float = 1_000_000.0
    positions: dict[str, Position] = field(default_factory=dict)

    def open_position(self, symbol: str, asset_class: str, direction: str,
                      price: float, notional_usd: float, timeframe: str = "SWING",
                      cycle_id: str = "") -> Position:
        quantity = notional_usd / price if price else 0.0
        pos = Position(
            symbol=symbol, asset_class=asset_class,  # type: ignore[arg-type]
            direction=direction, timeframe=timeframe,  # type: ignore[arg-type]
            entry_price=price, quantity=quantity, notional_usd=notional_usd,
            entry_timestamp=datetime.utcnow(), cycle_id=cycle_id,
            current_price=price)
        self.positions[symbol] = pos
        self.cash_usd -= notional_usd
        return pos

    def mark_all(self, prices: dict[str, float]) -> dict[str, float]:
        pnl: dict[str, float] = {}
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.mark(prices[sym], self.nav_usd)
                pnl[sym] = pos.unrealized_pnl_usd
        return pnl

    def close_position(self, symbol: str, price: float, commission: float = 0.0,
                       slippage: float = 0.0) -> float:
        pos = self.positions.pop(symbol)
        realized = pos.close(price, commission, slippage)
        self.cash_usd += pos.quantity * price
        self.nav_usd += realized
        return realized

    def summary(self) -> dict:
        gross = sum(abs(p.notional_usd) for p in self.positions.values())
        unreal = sum(p.unrealized_pnl_usd for p in self.positions.values())
        return {
            "nav_usd": self.nav_usd,
            "cash_usd": self.cash_usd,
            "gross_exposure_usd": gross,
            "unrealized_pnl_usd": unreal,
            "position_count": len(self.positions),
            "exposure_ratio": gross / self.nav_usd if self.nav_usd else 0.0,
        }
