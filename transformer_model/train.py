"""TFT training — walk-forward CV loop (plan §9.2 pattern for the model)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from core.logging import get_logger
from feature_engineering.feature_store import load_training_frame
from transformer_model.dataset import build_tft_dataset
from transformer_model.model import AgonistesTFT

log = get_logger(__name__)


def train_one_window(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     config: dict, checkpoint_dir: Path, window_id: str):
    from pytorch_forecasting import TemporalFusionTransformer
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

    dataset = build_tft_dataset(
        train_df,
        target=config.get("target", "future_return_5d"),
        max_encoder_length=config.get("max_encoder_length", 120),
        max_prediction_length=config.get("max_prediction_length", 5),
        batch_size=config.get("batch_size", 64))

    from pytorch_forecasting import TimeSeriesDataSet
    val_ds = TimeSeriesDataSet.from_dataset(
        dataset, val_df, stop_randomization=True, predict=True)

    model = AgonistesTFT().build_model(
        dataset,
        hidden_size=config.get("hidden_size", 128),
        attention_head_size=config.get("attention_head_size", 4),
        dropout=config.get("dropout", 0.1),
        hidden_continuous_size=config.get("hidden_continuous_size", 64),
        learning_rate=config.get("learning_rate", 1e-3),
    )

    class LossRecorder(pl.Callback):
        """Snapshot train/val loss at every epoch end (for loss curves)."""

        def __init__(self) -> None:
            self.epochs: list[int] = []
            self.train_loss: list[float] = []
            self.val_loss: list[float] = []

        def on_validation_epoch_end(self, trainer, pl_module):  # noqa: ANN001
            m = trainer.callback_metrics
            ep = trainer.current_epoch
            self.epochs.append(ep)
            self.train_loss.append(float(m.get("train_loss", float("nan"))))
            self.val_loss.append(float(m.get("val_loss", float("nan"))))

    recorder = LossRecorder()
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 30),
        accelerator="auto",
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor(logging_interval=None),
            recorder,
        ],
        default_root_dir=str(checkpoint_dir / window_id),
    )
    trainer.fit(
        model,
        train_dataloaders=dataset.to_dataloader(train=True),
        val_dataloaders=val_ds.to_dataloader(train=False),
    )
    return model, recorder


def _window_val_metrics(model, val_df: pd.DataFrame, config: dict,
                        target: str) -> dict:
    """Validate the trained window: MAE/MSE on median forecast + quantile coverage."""
    import numpy as np

    dataset = build_tft_dataset(
        val_df,
        target=target,
        max_encoder_length=config.get("max_encoder_length", 120),
        max_prediction_length=config.get("max_prediction_length", 5),
        batch_size=config.get("batch_size", 64))
    from pytorch_forecasting import TimeSeriesDataSet
    val_ds = TimeSeriesDataSet.from_dataset(dataset, val_df,
                                            stop_randomization=True, predict=True)

    pred = model.predict(val_ds, mode="quantiles")
    # shape: (n_samples, max_prediction_length, n_quantiles)
    q = np.asarray(pred)
    horizon = q.shape[1]

    # Actuals straight from the validation batches (decoder targets)
    loader = val_ds.to_dataloader(train=False, batch_size=config.get("batch_size", 64))
    actuals = []
    for b in loader:
        actuals.append(np.asarray(b["decoder_target"].detach().cpu()))
    actual = np.concatenate(actuals, axis=0).squeeze(-1)
    if actual.ndim == 1:  # already squeezed to (n,)
        actual = actual[:, None]
    actual = actual[:, -horizon:]
    q10, q50, q90 = q[..., 0], q[..., 1], q[..., 2]

    mae = float(np.mean(np.abs(q50 - actual)))
    mse = float(np.mean((q50 - actual) ** 2))
    # pinball loss at the 0.1 / 0.9 quantiles (forecast sharpness + calibration)
    pin10 = float(np.mean(np.maximum(0.1 * (actual - q10), 0.9 * (q10 - actual))))
    pin90 = float(np.mean(np.maximum(0.9 * (actual - q90), 0.1 * (q90 - actual))))
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    mae_naive = float(np.mean(np.abs(actual - 0.0)))  # zero-return baseline
    return {
        "val_rows": len(val_df),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "pinball_10": round(pin10, 6),
        "pinball_90": round(pin90, 6),
        "quantile_coverage_pct": round(100 * coverage, 1),
        "mae_vs_zero_baseline": round(mae / mae_naive if mae_naive else 0.0, 3),
    }


def walk_forward_train(feature_frame: pd.DataFrame, config: dict,
                       checkpoint_dir: Path, train_months: int = 12,
                       test_months: int = 3, step_months: int = 1):
    """Retrain per window; returns per-window loss curves + val metrics."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    windows = _temporal_windows(feature_frame, train_months, test_months, step_months)
    target = config.get("target", "future_return_5d")
    for i, (tr, va) in enumerate(windows):
        log.info("Window %d/%d: train %s → %s, val %s → %s", i + 1, len(windows),
                 tr.index.min().date(), tr.index.max().date(),
                 va.index.min().date(), va.index.max().date())
        model, recorder = train_one_window(tr, va, config, checkpoint_dir,
                                           f"window_{i}")
        metrics = _window_val_metrics(model, va, config, target)
        metrics["window"] = i
        metrics["train_start"] = str(tr.index.min().date())
        metrics["train_end"] = str(tr.index.max().date())
        metrics["val_start"] = str(va.index.min().date())
        metrics["val_end"] = str(va.index.max().date())
        metrics["epochs"] = recorder.epochs
        metrics["train_loss_curve"] = recorder.train_loss
        metrics["val_loss_curve"] = recorder.val_loss
        results.append(metrics)
    return results


def _temporal_windows(df: pd.DataFrame, train_months: int, test_months: int,
                      step_months: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling temporal split — no look-ahead (plan §9.2)."""
    start, end = df.index.min(), df.index.max()
    windows = []
    cursor = start
    while cursor + pd.DateOffset(months=train_months + test_months) <= end:
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        tr = df[(df.index >= cursor) & (df.index < tr_end)]
        va = df[(df.index >= tr_end) & (df.index < te_end)]
        if len(tr) > 100 and len(va) > 20:
            windows.append((tr, va))
        cursor += pd.DateOffset(months=step_months)
    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description="TFT walk-forward training")
    ap.add_argument("--config", default="transformer_model/configs/tft_swing.yaml")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--checkpoints", default="data/checkpoints")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    frame = load_training_frame(symbols=args.symbols,
                                timeframe=config.get("timeframe", "SWING"))
    if frame.empty:
        log.error("No feature vectors in store — run a backfill + feature build first")
        return
    results = walk_forward_train(frame, config, Path(args.checkpoints))
    # Persist per-window loss curves + validation metrics for reporting
    out = Path(args.checkpoints) / "walk_forward_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Walk-forward done: %d windows → %s", len(results), out)
    for r in results:
        log.info("window=%d val=%s..%s mae=%.5f mae/zero=%.3f cov=%.1f%% epochs=%d",
                 r["window"], r["val_start"], r["val_end"], r["mae"],
                 r["mae_vs_zero_baseline"], r["quantile_coverage_pct"],
                 len(r["epochs"]))


if __name__ == "__main__":
    main()
