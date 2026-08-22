"""Screener pipeline — score the universe and emit Top-N (used by CLI + UI)."""
from __future__ import annotations

from core.db import get_storage
from core.logging import get_logger
from feature_engineering.fundamental_features import compute_fundamental_features
from feature_engineering.macro_features import compute_macro_features
from feature_engineering.sentiment_features import compute_sentiment_features
from feature_engineering.technical_features import latest_scoring_features
from screener.asset_universe import get_universe
from screener.signal_scorer import AssetSignal, build_signal
from screener.top_n_selector import select_top_n

log = get_logger(__name__)


def score_universe(db=None, symbols: list[str] | None = None,
                   max_workers: int | None = None) -> list[AssetSignal]:
    """Score every universe asset that has market data in the store.

    Per-symbol scoring only needs the LATEST row, and every indicator has a
    bounded lookback (longest is EMA-200 / 60-bar returns), so we cap the OHLCV
    history to a recent window before computing technical features. This keeps
    a full-universe (~550 names) pass fast instead of recomputing 75 indicators
    over years of bars per symbol.
    """
    db = db or get_storage()
    universe = get_universe()
    macro = compute_macro_features(db)
    targets = symbols or universe.symbols()
    # Longest lookback across the technical indicators (EMA-200, ichimoku,
    # 60-bar returns) + a safety margin, enough to warm every indicator.
    _LOOKBACK = 400

    signals: list[AssetSignal] = []
    for sym in targets:
        try:
            asset_class = universe.asset_class_of(sym) or "EQUITY_US"
            ohlcv = db.query_ohlcv(sym)
            if ohlcv.empty or len(ohlcv) < 30:
                continue
            if len(ohlcv) > _LOOKBACK:
                ohlcv = ohlcv.tail(_LOOKBACK)
            tech = latest_scoring_features(ohlcv)
            sentiment = compute_sentiment_features(sym, db)
            snapshot = compute_fundamental_features(sym, asset_class, db)
            signals.append(build_signal(sym, asset_class, tech, snapshot, sentiment,
                                        macro, universe.weights))
        except Exception:  # noqa: BLE001
            continue
    return signals


def run_screener(top_n: int = 10, db=None) -> list[AssetSignal]:
    signals = score_universe(db)
    if not signals:
        log.warning("No scoreable assets — backfill data first (agonistes backfill)")
        return []
    selected = select_top_n(signals, n=top_n)
    log.info("Screener: scored %d assets, selected %d candidates",
             len(signals), len(selected))
    try:
        from core.api_server import record_screener
        record_screener(len(signals), len(selected))
    except Exception:  # noqa: BLE001
        pass
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
