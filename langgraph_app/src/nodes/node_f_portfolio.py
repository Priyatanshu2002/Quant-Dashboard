"""Node F — Portfolio Manager: Black-Litterman sizing for the approved trade."""
from __future__ import annotations

from langgraph_app.src.state import DebateState
from core.logging import get_logger

log = get_logger(__name__)


def node_f_portfolio(state: DebateState) -> DebateState:
    pack = state["analyst_pack"]
    gating = state["gating"]

    # View strength from the confidence delta (plan §7.2).
    view_strength = gating.confidence_delta
    direction = 1.0 if gating.dominant_side == "BULL" else -1.0

    try:
        from portfolio_manager.black_litterman import black_litterman_allocate
        weights = black_litterman_allocate(
            tickers=[pack.symbol], views={pack.symbol: view_strength * direction},
            confidence={pack.symbol: min(view_strength * 2.0, 1.0)},
            nav_usd=1_000_000.0,  # paper NAV
        )
        size_pct = weights.get(pack.symbol, 0.0)
    except Exception as e:  # noqa: BLE001
        log.warning("Black-Litterman failed (%s) — proportional sizing fallback", e)
        size_pct = min(0.10 * view_strength * 4.0, 0.15)

    decision = {
        "symbol": pack.symbol,
        "direction": "LONG" if direction > 0 else "SHORT",
        "position_size_pct": round(min(size_pct, 0.15), 4),   # 15% cap (plan §7.2)
        "notional_usd": round(1_000_000.0 * min(size_pct, 0.15), 2),
        "view_strength": view_strength,
    }
    log.info("Node F: %s", decision)
    return {"portfolio_decision": decision}
