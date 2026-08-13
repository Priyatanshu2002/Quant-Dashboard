"""Feature store — assembles per-asset feature frames + labels, writes to DB."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.db import FEATURE_COLUMNS, Storage, get_storage
from core.logging import get_logger
from feature_engineering.calendar_features import compute_calendar_features
from feature_engineering.cross_asset_features import compute_cross_asset_features
from feature_engineering.fundamental_features import compute_fundamental_features
from feature_engineering.macro_features import compute_macro_features
from feature_engineering.sentiment_features import compute_sentiment_features
from feature_engineering.technical_features import compute_technical_features

log = get_logger(__name__)

LABEL_HORIZONS = {"future_return_1d": 1, "future_return_5d": 5, "future_return_20d": 20}


def add_labels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Append forward-return labels (plan §3.6) computed at bar time."""
    close = ohlcv["close"]
    out = pd.DataFrame(index=ohlcv.index)
    for col, horizon in LABEL_HORIZONS.items():
        out[col] = close.shift(-horizon) / close - 1
    # Future 5d Sharpe proxy: mean 5d return / std of daily returns
    rets = close.pct_change()
    out["future_sharpe_5d"] = (rets.rolling(5).mean().shift(-5)
                               / rets.rolling(5).std().shift(-5).replace(0, np.nan))
    return out


def build_feature_frame(symbol: str, asset_class: str, ohlcv: pd.DataFrame,
                        db: Storage | None = None, timeframe: str = "SWING",
                        with_labels: bool = True) -> pd.DataFrame:
    """One row per bar: technical frame + static fundamental/macro/sentiment/calendar."""
    db = db or get_storage()
    tech = compute_technical_features(ohlcv)

    # Static (per-latest-snapshot) features broadcast to every bar.
    static: dict = {}
    static.update(compute_fundamental_features(symbol, asset_class, db))
    static.update(compute_sentiment_features(symbol, db))
    static.update(compute_macro_features(db))
    static.update(compute_calendar_features(symbol, db))
    static.update(compute_cross_asset_features(symbol, ohlcv, db))

    frame = tech.join(pd.DataFrame(static, index=tech.index))
    if with_labels:
        frame = frame.join(add_labels(ohlcv))

    # Keep only canonical columns (plus labels) that exist
    keep = [c for c in FEATURE_COLUMNS if c in frame.columns]
    frame = frame[keep]
    return frame


def write_to_store(frame: pd.DataFrame, symbol: str, asset_class: str,
                   timeframe: str, db: Storage | None = None) -> int:
    """Persist a built feature frame to the store; returns row count."""
    db = db or get_storage()
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty:
        return 0
    db.write_feature_vectors(clean, symbol, asset_class, timeframe)
    log.info("Wrote %d feature vectors for %s [%s]", len(clean), symbol, timeframe)
    return len(clean)


def load_training_frame(symbols: list[str] | None = None, timeframe: str = "SWING",
                        db: Storage | None = None) -> pd.DataFrame:
    """Pull labeled feature vectors from the store, ready for TFT training."""
    db = db or get_storage()
    df = db.query_feature_vectors(symbol=symbols[0] if symbols else None,
                                  timeframe=timeframe)
    if symbols and len(symbols) > 1:
        frames = [df]
        for s in symbols[1:]:
            frames.append(db.query_feature_vectors(symbol=s, timeframe=timeframe))
        df = pd.concat(frames)
    return df.sort_index()
