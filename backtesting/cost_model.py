"""Cost model — all market frictions (plan §9.1, verbatim)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    # ── Crypto ──────────────────────────────────────────────────────────
    crypto_maker_fee:     float = 0.0010   # 0.10%
    crypto_taker_fee:     float = 0.0015   # 0.15%
    funding_rate_annual:  float = 0.10     # ~10% annualized (bull market est.)

    # ── US Equities ──────────────────────────────────────────────────────
    us_equity_sec_fee:    float = 0.0000278  # $27.80 per $1M traded

    # ── Indian Equities (NSE) ────────────────────────────────────────────
    in_equity_brokerage:   float = 0.0003    # 0.03% or flat ₹20
    in_stt_delivery:       float = 0.001     # STT 0.1% on delivery
    in_stt_intraday:       float = 0.00025   # STT 0.025% intraday sell
    in_gst_on_brokerage:   float = 0.18      # GST on brokerage
    in_stamp_duty:         float = 0.00015   # 0.015% on buy side
    in_exchange_txn:       float = 0.0000325 # NSE transaction charge

    # ── Indian F&O ───────────────────────────────────────────────────────
    in_fno_stt:            float = 0.000625  # On sell side
    in_fno_stamp:          float = 0.00003

    def compute_slippage(self, order_size_usd: float, adv_usd: float) -> float:
        """Almgren-Chriss square-root market impact model."""
        participation_rate = order_size_usd / max(adv_usd, 1)
        return 0.1 * (participation_rate ** 0.5)

    def compute_half_spread(self, bid: float, ask: float) -> float:
        return (ask - bid) / (2 * ((ask + bid) / 2))

    def total_round_trip_cost(
        self, asset_class: str, order_size_usd: float,
        adv_usd: float, bid: float, ask: float,
        order_type: str = "MARKET"
    ) -> float:
        """Total cost of entry + exit (round trip), as a fraction of notional."""
        spread = self.compute_half_spread(bid, ask) * 2
        slip = self.compute_slippage(order_size_usd, adv_usd) * 2

        if asset_class == "CRYPTO":
            commission = (self.crypto_taker_fee if order_type == "MARKET"
                          else self.crypto_maker_fee) * 2
        elif asset_class == "EQUITY_IN":
            commission = (self.in_equity_brokerage + self.in_stt_intraday
                          + self.in_stamp_duty) * 2
        elif asset_class == "EQUITY_US":
            commission = self.us_equity_sec_fee * 2
        else:
            commission = 0.002

        return spread + slip + commission

    def round_trip_cost_pct(self, asset_class: str, order_size_usd: float,
                            adv_usd: float, spread_pct: float = 0.0005,
                            order_type: str = "MARKET") -> float:
        """Convenience: compute cost from a spread fraction directly."""
        bid, ask = 1.0 - spread_pct / 2, 1.0 + spread_pct / 2
        return self.total_round_trip_cost(asset_class, order_size_usd, adv_usd,
                                          bid, ask, order_type)
