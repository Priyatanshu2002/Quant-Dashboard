"""TFT real-time inference — checkpoint → TFTSignal."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from core.db import Storage, get_storage
from core.logging import get_logger
from feature_engineering.feature_store import build_feature_frame
from transformer_model.dataset import build_tft_dataset
from transformer_model.model import TFTSignal, signal_from_quantiles

log = get_logger(__name__)


def infer_signal(symbol: str, asset_class: str, ohlcv: pd.DataFrame,
                 checkpoint_dir: Path, config: dict,
                 db: Storage | None = None) -> TFTSignal | None:
    from pytorch_forecasting import TemporalFusionTransformer

    db = db or get_storage()
    frame = build_feature_frame(symbol, asset_class, ohlcv, db=db,
                                timeframe=config.get("timeframe", "SWING"),
                                with_labels=False)
    if len(frame) < config.get("max_encoder_length", 120):
        log.warning("Not enough history for %s: %d rows", symbol, len(frame))
        return None

    frame = frame.copy()
    frame["symbol"] = symbol
    frame["sector"] = "UNKNOWN"
    frame["exchange"] = "UNKNOWN"

    dataset = build_tft_dataset(
        frame, target=config.get("target", "future_return_5d"),
        max_encoder_length=config.get("max_encoder_length", 120),
        max_prediction_length=config.get("max_prediction_length", 5),
        batch_size=1)

    ckpt = sorted(Path(checkpoint_dir).glob("*.ckpt"))
    if not ckpt:
        log.error("No checkpoints in %s", checkpoint_dir)
        return None
    best = ckpt[-1]
    model = TemporalFusionTransformer.load_from_checkpoint(str(best))

    loader = dataset.to_dataloader(train=False, batch_size=1)
    raw = model.predict(loader, mode="quantiles")
    row = raw.iloc[0]
    p10, p50, p90 = float(row["0.1"]), float(row["0.5"]), float(row["0.9"])

    signal = signal_from_quantiles(symbol, asset_class,
                                   config.get("timeframe", "SWING"), p10, p50, p90)
    signal.return_5d_p50 = p50
    log.info("TFT %s: p50=%.4f direction=%s conviction=%.2f",
             symbol, p50, signal.direction, signal.conviction)
    return signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="transformer_model/configs/tft_swing.yaml")
    ap.add_argument("--checkpoints", default="data/checkpoints")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--asset-class", default="EQUITY_US")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db = get_storage()
    ohlcv = db.query_ohlcv(args.symbol)
    if ohlcv.empty:
        log.error("No market data for %s — backfill first", args.symbol)
        return
    print(infer_signal(args.symbol, args.asset_class, ohlcv,
                       Path(args.checkpoints), config, db))


if __name__ == "__main__":
    main()
