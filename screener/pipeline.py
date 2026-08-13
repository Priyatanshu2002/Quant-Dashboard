"""Screener pipeline — score the universe and emit Top-N (used by CLI + UI)."""
from __future__ import annotations

import pandas as pd

from core.db import get_storage
from core.logging import get_logger
from feature_engineering.fundamental_features import compute_fundamental_features
from feature_engineering.macro_features import compute_macro_features
from feature_engineering.sentiment_features import compute_sentiment_features
from feature_engineering.technical_features import latest_technical_features
from screener.asset_universe import get_universe
from screener.signal_scorer import AssetSignal, build_signal
from screener.top_n_selector import select_top_n

log = get_logger(__name__)


def score_universe(db=None, symbols: list[str] | None = None) -> list[AssetSignal]:
    """Score every universe asset that has market data in the store."""
    db = db or get_storage()
    universe = get_universe()
    macro = compute_macro_features(db)
    signals: list[AssetSignal] = []

    targets = symbols or universe.symbols()
    for sym in targets:
        asset_class = universe.asset_class_of(sym) or "EQUITY_US"
        ohlcv = db.query_ohlcv(sym)
        if ohlcv.empty or len(ohlcv) < 30:
            continue
        tech = latest_technical_features(ohlcv)
        sentiment = compute_sentiment_features(sym, db)
        snapshot = compute_fundamental_features(sym, asset_class, db)
        signals.append(build_signal(sym, asset_class, tech, snapshot, sentiment,
                                    macro, universe.weights))
    return signals


def run_screener(top_n: int = 10, db=None) -> list[AssetSignal]:
    signals = score_universe(db)
    if not signals:
        log.warning("No scoreable assets — backfill data first (agonistes backfill)")
        return []
    selected = select_top_n(signals, n=top_n)
    log.info("Screener: scored %d assets, selected %d candidates",
             len(signals), len(selected))
    return selected


def screener_table(selected: list[AssetSignal]) -> list[dict]:
    return [{"symbol": s.symbol, "asset_class": s.asset_class, **s.breakdown()}
            for s in selected]


def ranked_table(signals: list[AssetSignal], min_score: float = 60.0) -> list[dict]:
    """All scored assets ranked, with a qualification flag vs the threshold."""
    rows = []
    for s in sorted(signals, key=lambda x: -x.composite_score):
        b = s.breakdown()
        rows.append({
            "symbol": s.symbol, "asset_class": s.asset_class,
            "composite": round(b["composite"], 1),
            "technical": round(b["technical"], 1),
            "fundamental": round(b["fundamental"], 1),
            "sentiment": round(b["sentiment"], 1),
            "macro": round(b["macro"], 1),
            "momentum": round(b["momentum"], 1),
            "qualified": b["composite"] >= min_score,
        })
    return rows
