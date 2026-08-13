"""Position tracker — bridges the in-memory ledger with the paper trade log."""
from __future__ import annotations

import datetime as dt

from core.db import Storage, get_storage
from core.logging import get_logger
from portfolio_manager.multi_asset_ledger import MultiAssetLedger

log = get_logger(__name__)


class PositionTracker:
    def __init__(self, nav_usd: float = 1_000_000.0, db: Storage | None = None):
        self.ledger = MultiAssetLedger(nav_usd=nav_usd)
        self.db = db or get_storage()
        self._prices: dict[str, float] = {}

    def update_prices(self, prices: dict[str, float]) -> dict[str, float]:
        self._prices.update(prices)
        return self.ledger.mark_all(self._prices)

    def open_from_order(self, order: dict, fill_price: float) -> None:
        """Open a paper position from a Node-H execution order."""
        direction = "LONG" if order.get("side") == "BUY" else "SHORT"
        notional = order.get("notional_usd", 0.0)
        self.ledger.open_position(
            symbol=order["symbol"],
            asset_class=order.get("asset_class", "EQUITY_US"),
            direction=direction,
            price=fill_price,
            notional_usd=notional,
            timeframe=order.get("timeframe", "SWING"),
            cycle_id=order.get("cycle_id", ""),
        )
        self.db.write_trade({
            "time": dt.datetime.utcnow(),
            "trade_id": order.get("trade_id", "paper-open"),
            "cycle_id": order.get("cycle_id"),
            "symbol": order["symbol"],
            "asset_class": order.get("asset_class", "EQUITY_US"),
            "direction": direction, "timeframe": order.get("timeframe", "SWING"),
            "entry_price": fill_price,
            "quantity": notional / fill_price if fill_price else 0.0,
            "notional_usd": notional,
            "entry_time": dt.datetime.utcnow().isoformat(),
            "strategy": "debate_gated",
        })
        log.info("Opened paper %s %s @ %.4f", direction, order["symbol"], fill_price)

    def snapshot(self) -> dict:
        summary = self.ledger.summary()
        summary["time"] = dt.datetime.utcnow()
        self.db.write_portfolio_snapshot(summary)
        return summary
