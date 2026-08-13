"""Node G — Daily Reflection: review trades, extract lessons for the next cycle."""
from __future__ import annotations

from langgraph_app.src.state import DebateState
from core.db import get_storage
from core.logging import get_logger

log = get_logger(__name__)


def node_g_reflection(state: DebateState) -> DebateState:
    db = get_storage()
    symbol = state["symbol"]
    gating = state.get("gating")

    lesson = (
        f"[{state['cycle_id']}] {symbol}: debate gated to {gating.decision if gating else '?'} "
        f"(bull={state.get('bull').overall_confidence if state.get('bull') else '?'}, "
        f"bear={state.get('bear').overall_confidence if state.get('bear') else '?'}, "
        f"delta={gating.confidence_delta if gating else '?'}). "
        f"Standing aside preserves capital; reassess on next screener cycle."
    )
    db.write_reflection(state["cycle_id"], lesson)
    log.info("Node G: reflection stored")
    return {"reflection": lesson}
