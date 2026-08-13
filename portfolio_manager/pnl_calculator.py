"""PnL calculator — realized/unrealized with full cost attribution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PnLBreakdown:
    gross_pnl_usd: float
    commission_usd: float
    slippage_usd: float
    funding_usd: float
    net_pnl_usd: float
    net_pnl_pct: float
    cost_drag_pct: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def compute_trade_pnl(direction: str, entry_price: float, exit_price: float,
                      quantity: float, commission: float = 0.0,
                      slippage: float = 0.0, funding: float = 0.0) -> PnLBreakdown:
    """Full-cost PnL for one closed trade."""
    gross = (exit_price - entry_price) * quantity
    if direction == "SHORT":
        gross = -gross
    net = gross - commission - slippage - funding
    notional = entry_price * quantity
    return PnLBreakdown(
        gross_pnl_usd=gross,
        commission_usd=commission,
        slippage_usd=slippage,
        funding_usd=funding,
        net_pnl_usd=net,
        net_pnl_pct=net / notional if notional else 0.0,
        cost_drag_pct=(commission + slippage + funding) / notional if notional else 0.0,
    )


def portfolio_pnl(closed_trades: list[PnLBreakdown]) -> dict:
    gross = sum(t.gross_pnl_usd for t in closed_trades)
    costs = sum(t.commission_usd + t.slippage_usd + t.funding_usd for t in closed_trades)
    return {
        "gross_pnl_usd": gross,
        "total_cost_usd": costs,
        "net_pnl_usd": gross - costs,
        "n_trades": len(closed_trades),
    }
