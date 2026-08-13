"""Node E — Confidence Delta Gating (plan §5.3, verbatim logic)."""
from __future__ import annotations

from langgraph_app.src.schemas.analyst_pack import GatingDecision, ThesisOutput
from langgraph_app.src.state import DebateState
from core.logging import get_logger

log = get_logger(__name__)

BASE_DELTA_THRESHOLD = 0.25    # Minimum |Bull - Bear| to trade
HIGH_VIX_MULTIPLIER = 1.5      # VIX > 25
EXTREME_VIX_MULTIPLIER = 2.0   # VIX > 35
MIN_INDIVIDUAL_CONF = 0.40     # Neither agent below this floor
TFT_ALIGNMENT_BONUS = 0.05     # Reduces threshold if TFT agrees with dominant agent


def compute_gating(bull: ThesisOutput, bear: ThesisOutput,
                   tft_direction: str, vix: float | None) -> GatingDecision:
    # Adaptive threshold
    if vix is not None and vix > 35:
        threshold = BASE_DELTA_THRESHOLD * EXTREME_VIX_MULTIPLIER
    elif vix is not None and vix > 25:
        threshold = BASE_DELTA_THRESHOLD * HIGH_VIX_MULTIPLIER
    else:
        threshold = BASE_DELTA_THRESHOLD

    bull_strength = bull.overall_confidence
    bear_strength = bear.overall_confidence
    dominant = ("BULL" if bull_strength > bear_strength
                else "BEAR" if bear_strength > bull_strength else "NEUTRAL")

    # Reduce threshold if TFT confirms the dominant direction
    tft_aligned = ((dominant == "BULL" and tft_direction == "LONG") or
                   (dominant == "BEAR" and tft_direction == "SHORT"))
    if tft_aligned:
        threshold -= TFT_ALIGNMENT_BONUS

    delta = abs(bull_strength - bear_strength)
    quality_pass = (bull_strength >= MIN_INDIVIDUAL_CONF and
                    bear_strength >= MIN_INDIVIDUAL_CONF)
    should_trade = delta > threshold and quality_pass and dominant != "NEUTRAL"

    return GatingDecision(
        decision="TRADE" if should_trade else "NO_TRADE",
        confidence_delta=delta,
        adaptive_threshold=threshold,
        dominant_side=dominant,
        tft_aligned=tft_aligned,
        rationale=(f"delta={delta:.2f} vs threshold={threshold:.2f} "
                   f"(vix={vix}, tft_aligned={tft_aligned}, "
                   f"quality_pass={quality_pass})"),
    )


def node_e_gating(state: DebateState) -> DebateState:
    bull = state["bull"]
    bear = state["bear"]
    pack = state["analyst_pack"]
    vix = pack.vix

    decision = compute_gating(bull, bear, pack.tft_direction, vix)
    log.info("Node E: %s (%s)", decision.decision, decision.rationale)
    return {"gating": decision}
