"""Node D — Bear agent: construct the strongest SHORT case (LLM or heuristic)."""
from __future__ import annotations

from langgraph_app.src.schemas.analyst_pack import AnalystPack, ThesisOutput
from langgraph_app.src.state import DebateState
from langgraph_app.src.utils.llm_client import call_openrouter_structured, llm_available
from core.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are the BEAR agent in an adversarial trading debate for a paper-trading "
    "system. Your ONLY job is to construct the strongest possible SHORT case for "
    "the given asset, using the provided data. Be quantitative and honest: cite "
    "the numbers, flag what supports shorts, and state what would invalidate your "
    "thesis. overall_confidence must reflect how strong YOUR side is, "
    "0.0 (no case) to 1.0 (extremely strong short case)."
)


def _heuristic_bear(pack: AnalystPack) -> ThesisOutput:
    conf = 0.25
    points = []
    if pack.rsi_14 is not None and pack.rsi_14 > 65:
        conf += 0.15
        points.append(f"RSI {pack.rsi_14:.0f} overbought (>65)")
    if pack.price_vs_ema200_pct is not None and pack.price_vs_ema200_pct < 0:
        conf += 0.20
        points.append(f"Price {abs(pack.price_vs_ema200_pct):.1f}% BELOW 200-day EMA (downtrend)")
    if pack.macd_histogram is not None and pack.macd_histogram < 0:
        conf += 0.15
        points.append("MACD histogram negative (momentum down)")
    if pack.revenue_yoy_growth is not None and pack.revenue_yoy_growth < 0.02:
        conf += 0.15
        points.append(f"Revenue growth weak ({pack.revenue_yoy_growth*100:.0f}% YoY)")
    if pack.forward_pe is not None and pack.forward_pe > 30:
        conf += 0.15
        points.append(f"Forward P/E rich at {pack.forward_pe:.1f}x")
    if pack.tft_direction == "SHORT":
        conf += 0.15
        points.append(f"TFT model agrees (conviction {pack.tft_conviction:.2f})")
    if pack.vix is not None and pack.vix > 25:
        conf += 0.10
        points.append(f"Elevated VIX {pack.vix:.0f} — fragile tape")
    conf = min(conf, 0.95)
    return ThesisOutput(
        overall_confidence=round(conf, 2),
        thesis_summary=("Overbought/weak-trend conditions plus rich valuation (where "
                        "available) argue for a SHORT position or standing aside."),
        key_points=points,
        risks=["Short squeeze", "Positive catalyst could reverse the setup"],
        time_horizon_days=5,
    )


async def node_d_bear(state: DebateState) -> DebateState:
    pack = state["analyst_pack"]
    try:
        if llm_available():
            thesis = await call_openrouter_structured(
                SYSTEM_PROMPT,
                f"Build the strongest SHORT case.\n\n{pack.to_prompt_block()}",
                ThesisOutput)
        else:
            thesis = _heuristic_bear(pack)
    except Exception as e:  # noqa: BLE001
        log.warning("Bear LLM failed (%s) — falling back to heuristic", e)
        thesis = _heuristic_bear(pack)

    log.info("Node D: BEAR confidence %.2f", thesis.overall_confidence)
    return {"bear": thesis}
