"""Top-N selection (plan §2.3) — ranks qualified signals with class caps."""
from __future__ import annotations

from screener.signal_scorer import AssetSignal

N_CANDIDATES = 10
MIN_SCORE_THRESHOLD = 60.0
MAX_PER_ASSET_CLASS = 3      # Diversification cap


def select_top_n(signals: list[AssetSignal], n: int = N_CANDIDATES,
                 min_score: float = MIN_SCORE_THRESHOLD,
                 max_per_class: int = MAX_PER_ASSET_CLASS) -> list[AssetSignal]:
    qualified = [s for s in signals if s.composite_score >= min_score]
    qualified.sort(key=lambda s: s.composite_score, reverse=True)

    selected = []
    class_counts: dict[str, int] = {}

    for signal in qualified:
        if len(selected) >= n:
            break
        count = class_counts.get(signal.asset_class, 0)
        if count < max_per_class:
            selected.append(signal)
            class_counts[signal.asset_class] = count + 1

    return selected
