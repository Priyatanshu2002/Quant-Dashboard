"""Compact sentiment lexicon (AFFIN-111 subset) — no external dependency.

Scores run roughly in [-1, +1]: score = tanh(sum(weights) / sqrt(n_words)).
"""
from __future__ import annotations

import math
import re

_POSITIVE = {
    "beat": 2, "beats": 2, "record": 2, "surge": 2, "surges": 2, "rally": 2,
    "rallies": 2, "gain": 1, "gains": 1, "profit": 2, "profits": 2, "growth": 2,
    "upgrade": 2, "upgrades": 2, "buy": 1, "strong": 2, "outperform": 3,
    "bullish": 2, "breakout": 3, "all-time high": 3, "ath": 3, "adoption": 2,
    "partnership": 2, "partners": 2, "launch": 1, "launches": 1, "approval": 2,
    "approved": 2, "win": 1, "wins": 1, "positive": 1, "innovation": 1,
    "momentum": 1, "expansion": 1, "dividend": 1, "buyback": 2, "up": 1,
}
_NEGATIVE = {
    "miss": -2, "misses": -2, "plunge": -2, "plunges": -2, "crash": -3,
    "crashes": -3, "selloff": -2, "sell-off": -2, "drop": -1, "drops": -1,
    "loss": -2, "losses": -2, "downgrade": -2, "downgrades": -2, "sell": -1,
    "weak": -2, "bearish": -2, "lawsuit": -2, "lawsuits": -2, "fraud": -3,
    "scandal": -3, "investigation": -2, "probe": -2, "recall": -2, "recalls": -2,
    "layoff": -2, "layoffs": -2, "guidance cut": -2, "warning": -1,
    "warns": -1, "recession": -2, "inflation": -1, "delist": -3, "bankrupt": -3,
    "bankruptcy": -3, "negative": -1, "downturn": -2, "below": -1, "down": -1,
}
_WORDS = {**_POSITIVE, **_NEGATIVE}
_TOKEN_RE = re.compile(r"[a-z0-9'\-]+")


def score_text(text: str) -> float:
    """Lexicon sentiment in [-1, +1]."""
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    total = 0.0
    for tok in tokens:
        total += _WORDS.get(tok, 0)
    # Normalize by length so long headlines don't inflate scores.
    return float(math.tanh(total / math.sqrt(max(len(tokens), 1))))
