"""TFT training-panel loader — pulls the labeled feature store into a
pytorch-forecasting dataset.

This is the wiring that makes the Phase 3 TFT training path consume REAL data
from the feature store (675k+ rows across the universe) instead of a caller
hand-building a panel. It complements `transformer_model/dataset.py`:

    build_tft_dataset(feature_frame, ...)   -> TimeSeriesDataSet (given a frame)
    load_feature_panel(...)                 -> the frame, from the DB
    build_tft_train_val(panel, ...)         -> (train_ds, val_ds) chronological split

The DB-touching load is kept separate from the pure split/build so the split is
trivially unit-testable with a synthetic panel.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.db import get_storage
from core.logging import get_logger
from transformer_model.dataset import build_tft_dataset

log = get_logger(__name__)

DEFAULT_TARGET = "future_return_5d"


def load_feature_panel(timeframe: str = "SWING",
                       symbols: list[str] | None = None,
                       start: Any = None, end: Any = None) -> pd.DataFrame:
    """Read labeled feature vectors from the store into a long-format panel.

    Returns a DataFrame indexed by DatetimeIndex 'time' with a 'symbol' column
    and every feature column the store carries (technical + fundamental +
    sentiment + macro + labeled future returns).
    """
    db = get_storage()
    frame = db.query_feature_vectors(symbol=None, timeframe=timeframe,
                                     start=start, end=end)
    if symbols:
        keep = set(symbols)
        frame = frame[frame["symbol"].isin(keep)]
    if frame.empty:
        return frame
    if "time" not in frame.index.names:
        frame = frame.set_index("time")
    return frame


def build_tft_train_val(
    panel: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    val_frac: float = 0.2,
    max_encoder_length: int = 120,
    max_prediction_length: int = 5,
    batch_size: int = 64,
    **ds_kwargs: Any,
) -> tuple[Any, Any]:
    """Split a labeled panel chronologically and build train/val TimeSeriesDataSets.

    The split is by time (no leakage): the last `val_frac` of the panel's date
    range becomes validation. Both datasets are built with the same config so a
    TemporalFusionTransformer can be fit on train and validated on val.

    Returns (train_ds, val_ds) — pytorch-forecasting TimeSeriesDataSet objects.
    """
    if panel.empty:
        raise ValueError("panel is empty — nothing to build datasets from")
    if "symbol" not in panel.columns:
        raise ValueError("panel must include a 'symbol' column (group id)")

    df = panel.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    # column is 'time' after reset_index if the index was named 'time'
    time_col = "time" if "time" in df.columns else df.index.name
    if time_col not in df.columns:
        raise ValueError("panel must have a 'time' column (or DatetimeIndex named 'time')")
    df[time_col] = pd.to_datetime(df[time_col])

    cutoff = df[time_col].quantile(1.0 - val_frac)
    train_df = df[df[time_col] <= cutoff]
    val_df = df[df[time_col] > cutoff]

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"split produced empty set: train={len(train_df)} val={len(val_df)} "
            f"(cutoff={cutoff})")

    train_ds = build_tft_dataset(
        train_df.set_index(time_col), target=target,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size, **ds_kwargs)
    val_ds = build_tft_dataset(
        val_df.set_index(time_col), target=target,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size, **ds_kwargs)
    log.info("TFT datasets built: train=%d samples, val=%d samples",
             len(train_ds), len(val_ds))
    return train_ds, val_ds
