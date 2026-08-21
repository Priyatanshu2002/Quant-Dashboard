#!/usr/bin/env python3
"""Prepare + verify the TFT training panel from the real labeled feature store.

This does NOT train anything — it proves the Phase 3 gate is wired: the store's
labeled feature vectors load into a pytorch-forecasting TimeSeriesDataSet
(train + validation) with no manual panel-building. Run it before the first GPU
training run.

Usage:
  .venv/Scripts/python scripts/prepare_tft_data.py                # full universe, SWING
  .venv/Scripts/python scripts/prepare_tft_data.py --symbols AAPL MSFT NVDA
  .venv/Scripts/python scripts/prepare_tft_data.py --timeframe SWING --val-frac 0.2
  .venv/Scripts/python scripts/prepare_tft_data.py --dump panel.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformer_model.loader import (
    DEFAULT_TARGET,
    build_tft_train_val,
    load_feature_panel,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeframe", default="SWING")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="restrict to these symbols (default: whole universe)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--dump", default=None, help="optional path to dump the panel")
    args = ap.parse_args()

    print(f"[1/3] Loading labeled feature store (timeframe={args.timeframe})...")
    panel = load_feature_panel(timeframe=args.timeframe, symbols=args.symbols)
    if panel.empty:
        print("  EMPTY panel — run scripts/backfill_universe.py first.")
        return
    labeled = panel[args.target].notna().sum() if args.target in panel.columns else 0
    print(f"  rows={len(panel):,}  symbols={panel['symbol'].nunique():,}  "
          f"labeled[{args.target}]={labeled:,}  "
          f"range={panel.index.min().date()} -> {panel.index.max().date()}")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(args.dump) if args.dump.endswith(".parquet") \
            else panel.to_csv(args.dump)
        print(f"  panel dumped -> {args.dump}")

    print(f"[2/3] Building train/val TimeSeriesDataSets (val_frac={args.val_frac})...")
    train_ds, val_ds = build_tft_train_val(
        panel, target=args.target, val_frac=args.val_frac)
    print(f"  train samples={len(train_ds):,}  val samples={len(val_ds):,}")

    print("[3/3] Sampling one training batch to confirm tensors are valid...")
    loader = train_ds.to_dataloader(batch_size=64, train=True)
    batch = next(iter(loader))
    # pytorch-forecasting 1.8 yields a tuple: (x, y, weight, target_scale).
    x = batch[0]
    print(f"  encoder_cont shape={tuple(x['encoder_cont'].shape)}  "
          f"features={len(train_ds.reals)} ")
    print("\nOK — TFT training panel is wired to the real feature store. "
          "Ready for GPU training.")

    # Sanity: confirm every model-required feature is actually present.
    from transformer_model.model import AgonistesTFT
    missing = [c for c in (AgonistesTFT.TIME_VARYING_KNOWN
                           + AgonistesTFT.TIME_VARYING_UNKNOWN)
               if c not in panel.columns]
    if missing:
        print(f"  WARNING: model-required features missing from panel: {missing}")


if __name__ == "__main__":
    main()
