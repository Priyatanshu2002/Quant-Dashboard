"""TFT dataset builder — reads the feature store into TimeSeriesDataSet."""
from __future__ import annotations

import pandas as pd

from core.logging import get_logger
from transformer_model.model import AgonistesTFT

log = get_logger(__name__)


def _fill_model_features(df: pd.DataFrame, cols: list[str], group: str) -> None:
    """Per-group ffill then 0-fill the given feature columns, in place.

    pytorch-forecasting rejects NaN/Inf in real-valued features. The feature
    store has sparse fundamentals/sentiment and leading-bar NaNs, so for each
    symbol we forward-fill each column and then backfill any remaining leading
    NaNs with 0 so every sample the model sees is finite.
    """
    if not cols:
        return
    for idx in df.groupby(group).groups.values():
        grp = df.loc[idx, cols]
        df.loc[idx, cols] = grp.ffill().fillna(0.0)


def build_tft_dataset(feature_frame: pd.DataFrame, target: str = "future_return_5d",
                      max_encoder_length: int = 120, max_prediction_length: int = 5,
                      batch_size: int = 64):
    """Create a pytorch-forecasting TimeSeriesDataSet from labeled features.

    feature_frame must contain a 'symbol' column (group id) and a DatetimeIndex.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    df = feature_frame.copy()
    df = df.reset_index()
    if "symbol" not in df.columns:
        raise ValueError("feature_frame must include a 'symbol' column (group id)")
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

    # Add minimal static categorical (default sector/exchange if absent)
    if "asset_class" not in df.columns:
        df["asset_class"] = "EQUITY_US"
    if "sector" not in df.columns:
        df["sector"] = "UNKNOWN"
    if "exchange" not in df.columns:
        df["exchange"] = "UNKNOWN"

    # pytorch-forecasting requires an INTEGER time_idx (monotonic per group),
    # not the raw datetime. Encode each symbol's timeline as 0..n-1.
    df["time"] = pd.to_datetime(df["time"])
    df["time_idx"] = df.groupby("symbol")["time"].cumcount().astype(int)

    available = set(df.columns)
    time_varying_known = [c for c in AgonistesTFT.TIME_VARYING_KNOWN if c in available]
    time_varying_unknown = [c for c in AgonistesTFT.TIME_VARYING_UNKNOWN if c in available]

    # pytorch-forecasting forbids NaN/Inf in real features. The store has
    # sparse fundamental/sentiment columns and leading-bar NaNs, so fill the
    # model-required features per symbol: forward-fill, then 0-fill leftovers.
    _fill_model_features(df, time_varying_known + time_varying_unknown, group="symbol")

    # Rows with no label (trailing forecast window) can't be learned from —
    # drop them. This is a training dataset builder, so unlabeled rows are noise.
    if target in df.columns:
        df = df.dropna(subset=[target]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows have a non-null '{target}' target")

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["asset_class", "sector", "exchange"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def make_dataloaders(feature_frame: pd.DataFrame, target: str = "future_return_5d",
                     max_encoder_length: int = 120, max_prediction_length: int = 5,
                     batch_size: int = 64, validation_fraction: float = 0.15,
                     seed: int = 42):
    """Train/validation split (temporal — last fraction is validation)."""
    from pytorch_forecasting import TimeSeriesDataSet

    dataset = build_tft_dataset(
        feature_frame, target=target,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length, batch_size=batch_size)

    # Temporal split: keep the LAST validation_fraction of each group for validation
    n = len(feature_frame)
    split_idx = int(n * (1 - validation_fraction))
    train_df = feature_frame.iloc[:split_idx]
    val_df = feature_frame.iloc[split_idx:]

    training = TimeSeriesDataSet.from_dataset(
        dataset, train_df, stop_randomization=True)
    validation = TimeSeriesDataSet.from_dataset(
        dataset, val_df, stop_randomization=True, predict=True)
    return training, validation
