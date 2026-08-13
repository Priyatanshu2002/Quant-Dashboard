"""Tests for the confidence-delta gating node (plan §5.3)."""
from langgraph_app.src.nodes.node_e_gating import (
    BASE_DELTA_THRESHOLD, compute_gating)
from langgraph_app.src.schemas.analyst_pack import ThesisOutput


def _thesis(conf: float) -> ThesisOutput:
    return ThesisOutput(overall_confidence=conf, thesis_summary="t")


def test_trade_when_delta_above_threshold():
    g = compute_gating(_thesis(0.70), _thesis(0.40), "NEUTRAL", vix=15)
    assert g.decision == "TRADE"
    assert g.dominant_side == "BULL"
    assert g.confidence_delta == pytest_approx(0.30)


def test_no_trade_when_delta_below_threshold():
    g = compute_gating(_thesis(0.55), _thesis(0.45), "NEUTRAL", vix=15)
    assert g.decision == "NO_TRADE"


def test_high_vix_raises_threshold():
    low_vix = compute_gating(_thesis(0.70), _thesis(0.40), "NEUTRAL", vix=15)
    high_vix = compute_gating(_thesis(0.70), _thesis(0.40), "NEUTRAL", vix=30)
    assert low_vix.decision == "TRADE"
    assert high_vix.decision == "NO_TRADE"
    assert high_vix.adaptive_threshold == pytest_approx(
        BASE_DELTA_THRESHOLD * 1.5)


def test_extreme_vix_multiplier():
    g = compute_gating(_thesis(0.80), _thesis(0.20), "NEUTRAL", vix=40)
    assert g.adaptive_threshold == pytest_approx(BASE_DELTA_THRESHOLD * 2.0)


def test_tft_alignment_reduces_threshold():
    no_align = compute_gating(_thesis(0.60), _thesis(0.35), "NEUTRAL", vix=15)
    aligned = compute_gating(_thesis(0.60), _thesis(0.35), "LONG", vix=15)
    assert aligned.adaptive_threshold < no_align.adaptive_threshold
    # TFT alignment can flip a marginal case to TRADE
    assert no_align.decision == "NO_TRADE" or aligned.decision == "TRADE"


def test_min_individual_confidence_floor():
    # delta is big but bear is below the floor → NO_TRADE
    g = compute_gating(_thesis(0.90), _thesis(0.10), "NEUTRAL", vix=15)
    assert g.decision == "NO_TRADE"


def test_neutral_dominant_never_trades():
    g = compute_gating(_thesis(0.50), _thesis(0.50), "NEUTRAL", vix=15)
    assert g.decision == "NO_TRADE"
    assert g.dominant_side == "NEUTRAL"


def test_bear_dominant():
    g = compute_gating(_thesis(0.40), _thesis(0.70), "SHORT", vix=15)
    assert g.dominant_side == "BEAR"
    assert g.decision == "TRADE"
    assert g.tft_aligned is True


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-6)
