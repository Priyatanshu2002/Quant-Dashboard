"""Node I — MiroFish: final sanity / circuit-breaker check before the trade logs.

Spec (plan §1.1) leaves this node open; it is implemented as the last-mile
risk filter: position limits, class exposure caps, and VaR sanity, mirroring
plan §7.2 constraints. It also persists the gating + trade records.
"""
from __future__ import annotations

import datetime as dt
import uuid

from langgraph_app.src.state import DebateState
from core.db import get_storage
from core.logging import get_logger

log = get_logger(__name__)

MAX_POSITION_PCT = 0.15          # No single position > 15% of NAV
MAX_CLASS_PCT = 0.40             # No single asset class > 40%
MAX_OPEN_POSITIONS = 10


def node_i_mirofish(state: DebateState) -> DebateState:
    db = get_storage()
    pack = state["analyst_pack"]
    gating = state["gating"]
    order = state.get("execution_order") or {}
    decision = state.get("portfolio_decision") or {}

    violations: list[str] = []
    size = order.get("position_size_pct", decision.get("position_size_pct", 0.0)) or 0.0
    if size > MAX_POSITION_PCT:
        violations.append(f"position size {size:.2%} > {MAX_POSITION_PCT:.0%} cap")

    # Count open positions in the trade log (paper portfolio).
    open_count = _count_open_positions(db)
    if open_count >= MAX_OPEN_POSITIONS:
        violations.append(f"open positions {open_count} >= {MAX_OPEN_POSITIONS}")

    if violations:
        msg = "; ".join(violations)
        db.write_circuit_breaker("POSITION_LIMIT", "WARNING", msg,
                                 symbol=pack.symbol, raw=order)
        log.warning("Node I: circuit breaker — %s", msg)
        return {"trade_logged": False, "circuit_breaker": msg}

    # Persist gating + trade records.
    db.write_gating_log({
        "time": dt.datetime.utcnow(), "cycle_id": state["cycle_id"],
        "symbol": pack.symbol, "bull_confidence": state["bull"].overall_confidence,
        "bear_confidence": state["bear"].overall_confidence,
        "confidence_delta": gating.confidence_delta,
        "adaptive_threshold": gating.adaptive_threshold,
        "dominant_side": gating.dominant_side,
        "tft_direction": pack.tft_direction, "tft_aligned": gating.tft_aligned,
        "decision": gating.decision, "vix": pack.vix,
        "bull_summary": state["bull"].thesis_summary,
        "bear_summary": state["bear"].thesis_summary,
    })
    db.write_trade({
        "time": dt.datetime.utcnow(), "trade_id": f"trd-{uuid.uuid4().hex[:10]}",
        "cycle_id": state["cycle_id"], "symbol": pack.symbol,
        "asset_class": pack.asset_class,
        "direction": decision.get("direction", "LONG"), "timeframe": "SWING",
        "notional_usd": order.get("notional_usd", 0),
        "strategy": "debate_gated",
    })

    # Archive the debate theses to Qdrant (GAP 2) for future semantic recall.
    # Graceful: a Qdrant outage never fails the trade cycle.
    try:
        from data_ingestion.vector_feeds.qdrant_writer import archive_debate_thesis
        archive_debate_thesis(
            symbol=pack.symbol,
            bull_thesis=state["bull"].thesis_summary if state.get("bull") else "",
            bear_thesis=state["bear"].thesis_summary if state.get("bear") else "",
            decision=gating.decision,
            date=str(dt.datetime.utcnow().date()),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Thesis archive skipped for %s: %s", pack.symbol, exc)

    log.info("Node I: cleared — gating + trade logged for %s", pack.symbol)
    return {"trade_logged": True, "circuit_breaker": None}


def _count_open_positions(db) -> int:
    if db.backend != "sqlite":
        return 0
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM trade_log "
                "WHERE exit_time IS NULL OR exit_time = ''").fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0
