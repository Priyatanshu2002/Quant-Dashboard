"""LangGraph pipeline definition (plan §1.1 Layer 3 + §5).

Flow: A(ingest) → B(analyst) → {C(bull), D(bear)} → E(gating)
      E → TRADE:   F(portfolio) → H(execution) → I(mirofish) → END
      E → NO_TRADE: G(reflection) → END
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langgraph_app.src.state import DebateState
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

# Make langgraph_app/src importable when running from the repo root.
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_graph():
    """Build the compiled LangGraph (lazy import so the app can run without it)."""
    from langgraph.graph import END, START, StateGraph

    from langgraph_app.src.nodes.node_a_ingestion import node_a_ingestion
    from langgraph_app.src.nodes.node_b_analyst import node_b_analyst
    from langgraph_app.src.nodes.node_c_bull import node_c_bull
    from langgraph_app.src.nodes.node_d_bear import node_d_bear
    from langgraph_app.src.nodes.node_e_gating import node_e_gating
    from langgraph_app.src.nodes.node_f_portfolio import node_f_portfolio
    from langgraph_app.src.nodes.node_g_reflection import node_g_reflection
    from langgraph_app.src.nodes.node_h_execution import node_h_execution
    from langgraph_app.src.nodes.node_i_mirofish import node_i_mirofish

    g = StateGraph(DebateState)
    g.add_node("ingestion", node_a_ingestion)
    g.add_node("analyst", node_b_analyst)
    g.add_node("bull", node_c_bull)
    g.add_node("bear", node_d_bear)
    g.add_node("gating", node_e_gating)
    g.add_node("portfolio", node_f_portfolio)
    g.add_node("execution", node_h_execution)
    g.add_node("reflection", node_g_reflection)
    g.add_node("mirofish", node_i_mirofish)

    g.add_edge(START, "ingestion")
    g.add_edge("ingestion", "analyst")
    g.add_edge("analyst", "bull")
    g.add_edge("analyst", "bear")
    g.add_edge("bull", "gating")
    g.add_edge("bear", "gating")
    g.add_conditional_edges(
        "gating",
        lambda s: "portfolio" if s.get("gating") and s["gating"].decision == "TRADE"
        else "reflection",
        {"portfolio": "portfolio", "reflection": "reflection"},
    )
    g.add_edge("portfolio", "execution")
    g.add_edge("execution", "mirofish")
    g.add_edge("mirofish", END)
    g.add_edge("reflection", END)

    return g.compile()


async def run_cycle(symbol: str, asset_class: str = "EQUITY_US",
                    mock: bool = False) -> dict:
    """Run one full debate cycle for a symbol; returns the final state dict."""
    if mock:
        # Deterministic dry run without LangGraph (tests node logic directly).
        from langgraph_app.src.nodes.node_a_ingestion import node_a_ingestion
        from langgraph_app.src.nodes.node_b_analyst import node_b_analyst
        from langgraph_app.src.nodes.node_c_bull import node_c_bull
        from langgraph_app.src.nodes.node_d_bear import node_d_bear
        from langgraph_app.src.nodes.node_e_gating import node_e_gating

        state: DebateState = {"symbol": symbol, "asset_class": asset_class}
        state.update(node_a_ingestion(state))
        state.update(node_b_analyst(state))
        state.update(await node_c_bull(state))
        state.update(await node_d_bear(state))
        state.update(node_e_gating(state))
        return state

    graph = build_graph()
    result = await graph.ainvoke({"symbol": symbol, "asset_class": asset_class})
    return dict(result)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Run one debate cycle")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--asset-class", default="EQUITY_US")
    ap.add_argument("--mock", action="store_true",
                    help="run node logic without LangGraph (no extra deps)")
    args = ap.parse_args()

    state = asyncio.run(run_cycle(args.symbol, args.asset_class, mock=args.mock))
    gating = state.get("gating")
    print(f"\n=== Debate cycle {state.get('cycle_id')} for {args.symbol} ===")
    if state.get("bull"):
        print(f"BULL  conf={state['bull'].overall_confidence:.2f} — {state['bull'].thesis_summary[:80]}")
    if state.get("bear"):
        print(f"BEAR  conf={state['bear'].overall_confidence:.2f} — {state['bear'].thesis_summary[:80]}")
    if gating:
        print(f"GATE  decision={gating.decision} delta={gating.confidence_delta:.2f} "
              f"threshold={gating.adaptive_threshold:.2f} dominant={gating.dominant_side}")
    if state.get("portfolio_decision"):
        print(f"PM    {state['portfolio_decision']}")
    if state.get("circuit_breaker"):
        print(f"BREAKER {state['circuit_breaker']}")


if __name__ == "__main__":
    main()
