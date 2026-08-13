"""LangGraph shared state for the debate pipeline."""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph_app.src.schemas.analyst_pack import AnalystPack, GatingDecision, ThesisOutput


class DebateState(TypedDict, total=False):
    cycle_id: str
    symbol: str
    asset_class: str

    # Node A/B outputs
    analyst_pack: AnalystPack
    analyst_summary: str
    tft_signal: Any                 # TFTSignal (avoid hard import of ML pkg)

    # Node C/D outputs
    bull: ThesisOutput | None
    bear: ThesisOutput | None

    # Node E output
    gating: GatingDecision | None

    # Node F output
    portfolio_decision: dict | None

    # Node H output
    execution_order: dict | None

    # Node G output
    reflection: str | None

    # Node I output
    trade_logged: bool
    circuit_breaker: str | None

    error: str | None
