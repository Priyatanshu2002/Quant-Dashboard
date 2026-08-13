"""Node A — Ingestion: assemble the AnalystPack for the cycle."""
from __future__ import annotations

import uuid

from langgraph_app.src.schemas.analyst_pack import AnalystPack
from langgraph_app.src.state import DebateState
from langgraph_app.src.utils.db_clients import latest_snapshot_for
from core.db import get_storage
from core.logging import get_logger
from feature_engineering.technical_features import latest_technical_features
from screener.signal_scorer import build_signal

log = get_logger(__name__)


def node_a_ingestion(state: DebateState) -> DebateState:
    db = get_storage()
    symbol = state["symbol"]
    asset_class = state.get("asset_class", "EQUITY_US")
    cycle_id = state.get("cycle_id") or f"cyc-{uuid.uuid4().hex[:12]}"

    ohlcv = db.query_ohlcv(symbol)
    tech = latest_technical_features(ohlcv) if not ohlcv.empty else {}
    data = latest_snapshot_for(symbol, db)
    macro = data["macro"]
    sentiment = data["sentiment"]

    signal = build_signal(symbol, asset_class, tech, data["fundamental"],
                          sentiment, macro)
    b = signal.breakdown()

    vix = macro.get("vix") if macro else None
    regime = "UNKNOWN"
    if vix is not None:
        regime = ("CONTRACTION" if vix >= 30 else "LATE_CYCLE" if vix >= 20
                  else "EXPANSION")

    pack = AnalystPack(
        cycle_id=cycle_id,
        symbol=symbol,
        asset_class=asset_class,
        composite_score=b["composite"],
        technical_score=b["technical"],
        fundamental_score=b["fundamental"],
        sentiment_score_value=b["sentiment"],
        tft_direction="NEUTRAL",
        tft_conviction=0.0,
        rsi_14=tech.get("rsi_14"),
        macd_histogram=tech.get("macd_histogram"),
        price_vs_ema200_pct=tech.get("price_vs_ema200_pct"),
        atr_pct=tech.get("atr_pct"),
        volume_z_score=tech.get("volume_z_score"),
        bb_pct_b=tech.get("bb_pct_b"),
        realized_vol_20=tech.get("realized_vol_20"),
        eps_surprise_pct=data["fundamental"].get("eps_surprise_pct"),
        revenue_yoy_growth=data["fundamental"].get("revenue_yoy_growth"),
        dcf_margin_of_safety=data["fundamental"].get("dcf_margin_of_safety"),
        forward_pe=data["fundamental"].get("forward_pe"),
        peg_ratio=data["fundamental"].get("peg_ratio"),
        roic=data["fundamental"].get("roic"),
        fcf_yield=data["fundamental"].get("fcf_yield"),
        debt_to_equity=data["fundamental"].get("debt_to_equity"),
        margin_trend=data["fundamental"].get("margin_trend"),
        revenue_accel=data["fundamental"].get("revenue_accel"),
        vix=vix,
        yield_curve_spread=macro.get("yield_curve_spread") if macro else None,
        fed_funds_rate=macro.get("fed_funds_rate") if macro else None,
        macro_regime=regime,
        sentiment_24h=sentiment.get("sentiment_score", 0.0),
        sentiment_momentum=sentiment.get("sentiment_momentum", 0.0),
        reddit_sentiment=sentiment.get("reddit_sentiment", 0.0),
        gdelt_sentiment=sentiment.get("gdelt_sentiment", 0.0),
        yesterday_reflection=data.get("latest_reflection"),
    )

    log.info("Node A: AnalystPack ready for %s (cycle %s, composite %.1f)",
             symbol, cycle_id, pack.composite_score)
    return {"cycle_id": cycle_id, "analyst_pack": pack}
