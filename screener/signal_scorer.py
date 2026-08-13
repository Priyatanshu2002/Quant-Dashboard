"""Composite signal scorer (plan §2.2) + per-component scoring functions.

Each component maps raw data (features, fundamentals, sentiment, macro) into
a 0–100 score; the weighted composite is the AssetSignal.composite_score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.models import AssetClass


@dataclass
class AssetSignal:
    symbol: str
    asset_class: AssetClass

    # Component scores (0–100 each)
    technical_score: float = 50.0
    fundamental_score: float = 50.0
    sentiment_score: float = 50.0
    macro_alignment_score: float = 50.0
    momentum_score: float = 50.0

    weights: dict[str, float] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        w = self.weights or DEFAULT_WEIGHTS.get(self.asset_class, DEFAULT_WEIGHTS["EQUITY_US"])
        return (
            self.technical_score * w["technical"] +
            self.fundamental_score * w["fundamental"] +
            self.sentiment_score * w["sentiment"] +
            self.macro_alignment_score * w["macro"] +
            self.momentum_score * w["momentum"]
        )

    def breakdown(self) -> dict[str, float]:
        return {
            "technical": self.technical_score,
            "fundamental": self.fundamental_score,
            "sentiment": self.sentiment_score,
            "macro": self.macro_alignment_score,
            "momentum": self.momentum_score,
            "composite": self.composite_score,
        }


DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "CRYPTO":    {"technical": 0.35, "fundamental": 0.10, "sentiment": 0.30, "macro": 0.15, "momentum": 0.10},
    "EQUITY_US": {"technical": 0.25, "fundamental": 0.35, "sentiment": 0.20, "macro": 0.10, "momentum": 0.10},
    "EQUITY_IN": {"technical": 0.25, "fundamental": 0.35, "sentiment": 0.20, "macro": 0.10, "momentum": 0.10},
    "ETF":       {"technical": 0.20, "fundamental": 0.30, "sentiment": 0.15, "macro": 0.25, "momentum": 0.10},
    "BOND":      {"technical": 0.10, "fundamental": 0.20, "sentiment": 0.10, "macro": 0.50, "momentum": 0.10},
    "FOREX":     {"technical": 0.30, "fundamental": 0.10, "sentiment": 0.20, "macro": 0.30, "momentum": 0.10},
    "FNO":       {"technical": 0.35, "fundamental": 0.25, "sentiment": 0.15, "macro": 0.10, "momentum": 0.15},
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def score_technical(features: dict[str, Any]) -> float:
    """RSI positioning + trend + momentum blend → 0–100."""
    scores = []
    rsi = features.get("rsi_14")
    if rsi is not None and not np.isnan(rsi):
        # 50 → 50; 30 → ~85 (oversold bounce zone); 70 → ~15 (overbought fade zone)
        scores.append(50 + (50 - rsi) * 1.0)
    pct_ema200 = features.get("price_vs_ema200_pct")
    if pct_ema200 is not None and not np.isnan(pct_ema200):
        scores.append(50 + float(np.clip(pct_ema200, -20, 20)) * 2.5)
    macd_hist = features.get("macd_histogram")
    if macd_hist is not None and not np.isnan(macd_hist):
        scores.append(50 + float(np.clip(macd_hist * 2000, -25, 25)))
    adx = features.get("adx_14")
    if adx is not None and not np.isnan(adx):
        scores.append(50 + float(np.clip((adx - 25) / 2, -20, 20)))
    if not scores:
        return 50.0
    return float(np.clip(np.mean(scores), 0, 100))


def score_momentum(features: dict[str, Any]) -> float:
    """Return-based momentum: 20-bar return scaled to 0–100."""
    r20 = features.get("return_20bar")
    if r20 is None or np.isnan(r20):
        r5 = features.get("return_5bar")
        r20 = r5 if r5 is not None and not np.isnan(r5) else 0.0
    return float(np.clip(50 + r20 * 100, 0, 100))


def score_fundamental(snap: dict | None) -> float:
    """Valuation + growth + quality blend → 0–100 (neutral 50 when no data)."""
    if not snap:
        return 50.0
    scores = []
    fpe = snap.get("forward_pe")
    if fpe and fpe > 0:
        scores.append(50 + float(np.clip((15 - fpe) * 2, -35, 35)))
    peg = snap.get("peg_ratio")
    if peg and peg > 0:
        scores.append(50 + float(np.clip((1 - peg) * 25, -30, 30)))
    growth = snap.get("revenue_yoy_growth")
    if growth is not None:
        scores.append(50 + float(np.clip(growth * 150, -30, 30)))
    roic = snap.get("roic")
    if roic is not None:
        scores.append(50 + float(np.clip((roic - 0.10) * 150, -25, 25)))
    margin = snap.get("dcf_margin_of_safety")
    if margin is not None:
        scores.append(50 + float(np.clip(margin * 100, -40, 40)))
    if not scores:
        return 50.0
    return float(np.clip(np.mean(scores), 0, 100))


def score_sentiment(sentiment: dict[str, Any] | None) -> float:
    """[-1, +1] sentiment → 0–100 (neutral 50 when no events)."""
    if not sentiment or not sentiment.get("sentiment_volume"):
        return 50.0
    s = sentiment.get("sentiment_score", 0.0)
    return float(np.clip(50 + s * 50, 0, 100))


def score_macro_alignment(macro: dict | None, asset_class: str) -> float:
    """How well the macro regime supports this asset class → 0–100."""
    if not macro:
        return 50.0
    scores = []
    vix = macro.get("vix")
    if vix is not None:
        if asset_class in ("CRYPTO", "EQUITY_US", "EQUITY_IN", "FNO"):
            scores.append(float(np.clip(75 - (vix - 15) * 2.5, 10, 90)))
        else:
            scores.append(float(np.clip(40 + (vix - 15) * 2.0, 10, 90)))  # bonds/fx love stress
    spread = macro.get("yield_curve_spread")
    if spread is not None:
        if asset_class == "BOND":
            scores.append(float(np.clip(50 + (spread + 0.5) * 50, 10, 90)))
        else:
            scores.append(float(np.clip(60 + spread * 60, 10, 90)))  # positive curve = risk-on
    if not scores:
        return 50.0
    return float(np.clip(np.mean(scores), 0, 100))


def build_signal(symbol: str, asset_class: AssetClass,
                 features: dict[str, Any] | None = None,
                 snapshot: dict | None = None,
                 sentiment: dict[str, Any] | None = None,
                 macro: dict | None = None,
                 weights: dict[str, dict[str, float]] | None = None) -> AssetSignal:
    """Assemble one AssetSignal from all available data (missing → neutral 50)."""
    features = features or {}
    return AssetSignal(
        symbol=symbol,
        asset_class=asset_class,
        technical_score=score_technical(features),
        fundamental_score=score_fundamental(snapshot),
        sentiment_score=score_sentiment(sentiment),
        macro_alignment_score=score_macro_alignment(macro, asset_class),
        momentum_score=score_momentum(features),
        weights=(weights or DEFAULT_WEIGHTS).get(asset_class, DEFAULT_WEIGHTS["EQUITY_US"]),
    )
