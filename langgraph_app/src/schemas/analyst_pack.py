"""Pydantic schemas shared by the debate nodes (plan §5.2)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalystPack(BaseModel):
    """Identical data package fed to both Bull and Bear agents (plan §5.2)."""

    cycle_id: str
    symbol: str
    asset_class: str

    # Screener output
    composite_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score_value: float

    # TFT signal
    tft_direction: str                  # LONG / SHORT / NEUTRAL
    tft_conviction: float               # [0, 1]
    return_1d_p50: float | None = None
    return_5d_p50: float | None = None
    return_20d_p50: float | None = None
    tft_top_features: list[tuple[str, float]] = []

    # Technical snapshot
    rsi_14: float | None = None
    macd_histogram: float | None = None
    price_vs_ema200_pct: float | None = None
    atr_pct: float | None = None
    volume_z_score: float | None = None
    bb_pct_b: float | None = None
    realized_vol_20: float | None = None

    # Fundamental (equities/ETF)
    eps_surprise_pct: float | None = None
    revenue_yoy_growth: float | None = None
    dcf_margin_of_safety: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    roic: float | None = None
    fcf_yield: float | None = None
    debt_to_equity: float | None = None
    insider_buy_sell_ratio: float | None = None
    inst_ownership_change: float | None = None
    earnings_call_sentiment: float | None = None
    margin_trend: float | None = None
    revenue_accel: float | None = None

    # On-chain (crypto)
    tvl_change_30d: float | None = None
    protocol_revenue_30d: float | None = None
    exchange_inflow_24h: float | None = None

    # Macro
    vix: float | None = None
    yield_curve_spread: float | None = None
    fed_funds_rate: float | None = None
    macro_regime: str = "UNKNOWN"   # EXPANSION / LATE_CYCLE / CONTRACTION / RECOVERY

    # Sentiment
    sentiment_24h: float = 0.0
    sentiment_momentum: float = 0.0
    reddit_sentiment: float = 0.0
    gdelt_sentiment: float = 0.0

    # Calendar
    days_to_earnings: int | None = None
    earnings_week: bool = False

    # GraphRAG context (from Neo4j)
    graphrag_key_relationships: list[str] = []
    risk_factors: list[str] = []

    # Reflection injection
    yesterday_reflection: str | None = None

    def to_prompt_block(self) -> str:
        """Compact text rendering of the pack for LLM prompts."""
        lines = [
            f"SYMBOL: {self.symbol} ({self.asset_class})",
            f"SCREENER: composite={self.composite_score:.1f} "
            f"(tech={self.technical_score:.1f}, fund={self.fundamental_score:.1f}, "
            f"sent={self.sentiment_score_value:.1f})",
            f"TFT: direction={self.tft_direction} conviction={self.tft_conviction:.2f} "
            f"e1d={self.return_1d_p50} e5d={self.return_5d_p50}",
            f"TECH: rsi14={self.rsi_14}, macd_hist={self.macd_histogram}, "
            f"vs_ema200={self.price_vs_ema200_pct}%, atr%={self.atr_pct}, "
            f"vol_z={self.volume_z_score}, bb%={self.bb_pct_b}, rv20={self.realized_vol_20}",
        ]
        if self.forward_pe is not None:
            lines.append(f"FUND: fwd_pe={self.forward_pe}, peg={self.peg_ratio}, "
                         f"roic={self.roic}, fcf_yield={self.fcf_yield}, "
                         f"d/e={self.debt_to_equity}, dcf_mos={self.dcf_margin_of_safety}, "
                         f"rev_growth={self.revenue_yoy_growth}")
        lines.append(f"MACRO: vix={self.vix}, yc_spread={self.yield_curve_spread}, "
                     f"ffr={self.fed_funds_rate}, regime={self.macro_regime}")
        lines.append(f"SENTIMENT: 24h={self.sentiment_24h}, mom={self.sentiment_momentum}, "
                     f"reddit={self.reddit_sentiment}, gdelt={self.gdelt_sentiment}")
        if self.days_to_earnings is not None:
            lines.append(f"CALENDAR: days_to_earnings={self.days_to_earnings} "
                         f"(week={self.earnings_week})")
        if self.yesterday_reflection:
            lines.append(f"LESSONS FROM YESTERDAY: {self.yesterday_reflection}")
        if self.graphrag_key_relationships:
            lines.append("KNOWLEDGE GRAPH (relationships): "
                         + " | ".join(self.graphrag_key_relationships))
        return "\n".join(lines)


class ThesisOutput(BaseModel):
    """Structured thesis from Bull or Bear agent (plan §5.3)."""

    overall_confidence: float = Field(ge=0.0, le=1.0,
                                      description="Agent's overall conviction in its side")
    thesis_summary: str = Field(description="2-3 sentence thesis")
    key_points: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    target_price: float | None = None
    time_horizon_days: int = 5
    key_metrics_flagged: list[str] = Field(default_factory=list)


class GatingDecision(BaseModel):
    decision: str                     # TRADE / NO_TRADE
    confidence_delta: float
    adaptive_threshold: float
    dominant_side: str                # BULL / BEAR / NEUTRAL
    tft_aligned: bool
    rationale: str = ""
