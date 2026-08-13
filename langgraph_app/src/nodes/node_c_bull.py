"""Node C — Bull agent: construct the strongest LONG case (LLM or heuristic)."""
from __future__ import annotations

from langgraph_app.src.schemas.analyst_pack import AnalystPack, ThesisOutput
from langgraph_app.src.state import DebateState
from langgraph_app.src.utils.llm_client import call_openrouter_structured, llm_available
from core.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are the BULL agent in an adversarial trading debate for a paper-trading "
    "system. Your ONLY job is to construct the strongest possible LONG case for the "
    "given asset, using the provided data. Be quantitative and honest: cite the "
    "numbers, flag what supports longs, and state the risks that would invalidate "
    "your thesis. overall_confidence must reflect how strong YOUR side is, "
    "0.0 (no case) to 1.0 (extremely strong long case)."
)


def _heuristic_bull(pack: AnalystPack) -> ThesisOutput:
    conf = 0.25
    points = []
    if pack.rsi_14 is not None and 40 <= pack.rsi_14 <= 65:
        conf += 0.15
        points.append(f"RSI {pack.rsi_14:.0f} in constructive zone (40-65)")
    if pack.price_vs_ema200_pct is not None and pack.price_vs_ema200_pct > 0:
        conf += 0.20
        points.append(f"Price {pack.price_vs_ema200_pct:.1f}% above 200-day EMA (uptrend)")
    if pack.macd_histogram is not None and pack.macd_histogram > 0:
        conf += 0.15
        points.append("MACD histogram positive (momentum up)")
    if pack.revenue_yoy_growth is not None and pack.revenue_yoy_growth > 0.05:
        conf += 0.15
        points.append(f"Revenue growing {pack.revenue_yoy_growth*100:.0f}% YoY")
    if pack.dcf_margin_of_safety is not None and pack.dcf_margin_of_safety > 0:
        conf += 0.15
        points.append(f"DCF margin of safety +{pack.dcf_margin_of_safety*100:.0f}%")
    if pack.tft_direction == "LONG":
        conf += 0.15
        points.append(f"TFT model agrees (conviction {pack.tft_conviction:.2f})")
    conf = min(conf, 0.95)
    return ThesisOutput(
        overall_confidence=round(conf, 2),
        thesis_summary=("Momentum, trend and (where available) fundamentals align "
                        "for a LONG position; the model and data agree on upside."),
        key_points=points,
        risks=["Unexpected macro shock", "Earnings miss could invalidate thesis"],
        time_horizon_days=5,
    )


async def node_c_bull(state: DebateState) -> DebateState:
    pack = state["analyst_pack"]
    try:
        if llm_available():
            thesis = await call_openrouter_structured(
                SYSTEM_PROMPT,
                f"Build the strongest LONG case.\n\n{pack.to_prompt_block()}",
                ThesisOutput)
        else:
            thesis = _heuristic_bull(pack)
    except Exception as e:  # noqa: BLE001
        log.warning("Bull LLM failed (%s) — falling back to heuristic", e)
        thesis = _heuristic_bull(pack)

    log.info("Node C: BULL confidence %.2f", thesis.overall_confidence)
    return {"bull": thesis}
