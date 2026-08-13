"""Shared dataclasses used across layers (storage-facing shapes)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

AssetClass = Literal["CRYPTO", "EQUITY_US", "EQUITY_IN", "ETF", "BOND", "FOREX", "FNO"]
Direction = Literal["LONG", "SHORT"]
Timeframe = Literal["INTRADAY", "SWING", "LONGTERM"]


@dataclass
class Asset:
    symbol: str
    asset_class: AssetClass
    name: str = ""
    exchange: str = ""
    sector: str = ""
    currency: str = "USD"
    isin: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FundamentalSnapshot:
    symbol: str
    asset_class: AssetClass = "EQUITY_US"
    period_type: str = "QUARTERLY"
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    reported_date: date | None = None

    revenue: float | None = None
    revenue_estimate: float | None = None
    revenue_yoy_growth: float | None = None
    gross_profit: float | None = None
    ebitda: float | None = None
    net_income: float | None = None
    eps_actual: float | None = None
    eps_estimate: float | None = None
    eps_yoy_growth: float | None = None

    total_assets: float | None = None
    total_debt: float | None = None
    cash_and_equivalents: float | None = None
    net_debt: float | None = None
    shareholders_equity: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    interest_coverage_ratio: float | None = None

    operating_cash_flow: float | None = None
    capex: float | None = None
    free_cash_flow: float | None = None

    roic: float | None = None
    gross_margin: float | None = None
    ebitda_margin: float | None = None
    fcf_yield: float | None = None

    market_cap: float | None = None
    current_price: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    ev_to_ebitda: float | None = None

    dcf_intrinsic_value: float | None = None
    dcf_margin_of_safety: float | None = None
    wacc_used: float | None = None

    insider_buy_value: float | None = None
    insider_sell_value: float | None = None
    institutional_ownership_change_pct: float | None = None

    transcript_sentiment_score: float | None = None
    transcript_summary: str = ""
    filing_url: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.reported_date:
            d["time"] = datetime.combine(self.reported_date, datetime.min.time())
        return d


@dataclass
class SentimentEvent:
    symbol: str
    source: str            # GDELT / REDDIT / NEWS
    score: float           # [-1, +1]
    source_weight: float = 1.0
    headline: str = ""
    url: str = ""
    ts: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MacroSnapshot:
    ts: datetime = field(default_factory=datetime.utcnow)
    us_10y_yield: float | None = None
    us_2y_yield: float | None = None
    fed_funds_rate: float | None = None
    vix: float | None = None
    dxy: float | None = None
    gold_pct_change_5d: float | None = None
    btc_dominance: float | None = None
    crypto_total_mcap_chg_24h: float | None = None
