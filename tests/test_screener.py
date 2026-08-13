"""Tests for the screener (plan §2.2/§2.3)."""
import numpy as np
import pytest

from screener.signal_scorer import (
    DEFAULT_WEIGHTS, AssetSignal, build_signal, score_fundamental,
    score_macro_alignment, score_momentum, score_sentiment, score_technical)
from screener.top_n_selector import select_top_n


def _sig(symbol, composite_parts=None):
    s = AssetSignal(symbol=symbol, asset_class="EQUITY_US")
    if composite_parts is None:
        return s
    for k, v in composite_parts.items():
        setattr(s, k, v)
    return s


def test_composite_weights_equity():
    s = AssetSignal(symbol="X", asset_class="EQUITY_US",
                    technical_score=100, fundamental_score=0,
                    sentiment_score=0, macro_alignment_score=0, momentum_score=0)
    w = DEFAULT_WEIGHTS["EQUITY_US"]
    assert s.composite_score == pytest.approx(100 * w["technical"])


def test_composite_in_range():
    rng = np.random.default_rng(0)
    for cls in DEFAULT_WEIGHTS:
        s = AssetSignal(symbol="X", asset_class=cls,
                        technical_score=float(rng.uniform(0, 100)),
                        fundamental_score=float(rng.uniform(0, 100)),
                        sentiment_score=float(rng.uniform(0, 100)),
                        macro_alignment_score=float(rng.uniform(0, 100)),
                        momentum_score=float(rng.uniform(0, 100)))
        assert 0 <= s.composite_score <= 100


def test_score_technical_ranges():
    assert 0 <= score_technical({"rsi_14": 50}) <= 100
    assert 0 <= score_technical({}) == 50.0
    assert 0 <= score_technical({"rsi_14": float("nan"), "adx_14": None}) <= 100


def test_score_momentum():
    assert score_momentum({"return_20bar": 0.2}) > score_momentum({"return_20bar": -0.2})
    assert score_momentum({}) == 50.0
    assert 0 <= score_momentum({"return_20bar": 5.0}) <= 100


def test_score_fundamental_neutral_without_data():
    assert score_fundamental(None) == 50.0
    assert score_fundamental({}) == 50.0
    # cheap + growing → above neutral
    assert score_fundamental({"forward_pe": 10, "revenue_yoy_growth": 0.2}) > 50.0


def test_score_sentiment():
    assert score_sentiment(None) == 50.0
    assert score_sentiment({"sentiment_volume": 0}) == 50.0
    assert score_sentiment({"sentiment_volume": 5, "sentiment_score": 0.8}) > 50.0
    assert score_sentiment({"sentiment_volume": 5, "sentiment_score": -0.8}) < 50.0


def test_score_macro_alignment():
    assert 0 <= score_macro_alignment(None, "EQUITY_US") <= 100
    assert score_macro_alignment({"vix": 12}, "EQUITY_US") > \
           score_macro_alignment({"vix": 45}, "EQUITY_US")


def test_build_signal_never_raises():
    s = build_signal("TEST", "CRYPTO")
    assert s.composite_score == pytest.approx(50.0)
    s2 = build_signal("TEST", "EQUITY_US",
                      features={"rsi_14": 55, "return_20bar": 1.0},
                      snapshot={"forward_pe": 12},
                      sentiment={"sentiment_volume": 3, "sentiment_score": 0.5},
                      macro={"vix": 18, "yield_curve_spread": 0.01})
    assert 0 <= s2.composite_score <= 100


def test_select_top_n_basic():
    sigs = [_sig(f"S{i}", {"technical_score": 80, "fundamental_score": 80,
                           "sentiment_score": 80, "macro_alignment_score": 80,
                           "momentum_score": 80}) for i in range(5)]
    selected = select_top_n(sigs, n=3)
    assert len(selected) == 3


def test_select_top_n_threshold():
    low = _sig("LOW", {"technical_score": 10, "fundamental_score": 10,
                       "sentiment_score": 10, "macro_alignment_score": 10,
                       "momentum_score": 10})
    high = _sig("HIGH", {"technical_score": 100, "fundamental_score": 100,
                         "sentiment_score": 100, "macro_alignment_score": 100,
                         "momentum_score": 100})
    selected = select_top_n([low, high], min_score=60)
    assert [s.symbol for s in selected] == ["HIGH"]


def test_select_top_n_class_cap():
    sigs = [_sig(f"C{i}", {"technical_score": 100, "fundamental_score": 100,
                           "sentiment_score": 100, "macro_alignment_score": 100,
                           "momentum_score": 100}) for i in range(6)]
    # all EQUITY_US → cap 3
    assert len(select_top_n(sigs, max_per_class=3)) == 3


def test_select_top_n_ranks_by_composite():
    a = _sig("A", {"technical_score": 90, "fundamental_score": 90,
                   "sentiment_score": 90, "macro_alignment_score": 90,
                   "momentum_score": 90})
    b = _sig("B", {"technical_score": 70, "fundamental_score": 70,
                   "sentiment_score": 70, "macro_alignment_score": 70,
                   "momentum_score": 70})
    selected = select_top_n([b, a])
    assert selected[0].symbol == "A"
