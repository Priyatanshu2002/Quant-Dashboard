"""Run the pytorch-forecasting TFT on the real labeled feature store, walk-forward.

This is the OTHER untested DL family (transformer_model/). Uses the actual
feature_vectors rows + future_return_5d target, per configs/tft_swing.yaml,
reduced to CPU-feasible settings. Reports MAE vs zero-baseline + quantile coverage.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)

import numpy as np, pandas as pd
from feature_engineering.feature_store import load_training_frame
from transformer_model.dataset import build_tft_dataset

# CPU-feasible config (swing, 5d target)
TARGET="future_return_5d"
ENC=60; PRED=5; BATCH=128
SYMBOLS=["AAPL","MSFT","NVDA","SPY","QQQ","JPM","TSLA","XOM","GLD","TLT",
         "EURUSD=X","^NSEI","INFY.NS","HDFCBANK.NS"]

frame=load_training_frame(symbols=SYMBOLS, timeframe="SWING")
if frame.empty:
    print("EMPTY feature frame"); sys.exit(1)
# ensure datetime index
if not isinstance(frame.index, pd.DatetimeIndex):
    frame=frame.set_index("time")
print(f"frame rows={len(frame):,} symbols={frame['symbol'].nunique()} "
      f"cols={len(frame.columns)} range={frame.index.min().date()}->{frame.index.max().date()}")

from transformer_model.model import AgonistesTFT
need = set(AgonistesTFT.TIME_VARYING_KNOWN + AgonistesTFT.TIME_VARYING_UNKNOWN + [TARGET])
avail = set(frame.columns)
missing = need - avail
print(f"missing model-required cols: {sorted(missing) if missing else 'NONE'}")
frame = frame[[c for c in frame.columns if c in avail]]

# temporal split: last 15% val
df = frame.reset_index()
df["time"] = pd.to_datetime(df["time"])
cutoff = df["time"].quantile(0.85)
tr = df[df["time"] <= cutoff].set_index("time")
va = df[df["time"] > cutoff].set_index("time")
print(f"train rows={len(tr):,} val rows={len(va):,}")

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

tr_ds = build_tft_dataset(tr, target=TARGET, max_encoder_length=ENC,
                          max_prediction_length=PRED, batch_size=BATCH)
# Independent val dataset with the same schema (loader build_tft_train_val pattern)
va_ds = build_tft_dataset(va, target=TARGET, max_encoder_length=ENC,
                          max_prediction_length=PRED, batch_size=BATCH)

model = AgonistesTFT().build_model(
    tr_ds, hidden_size=64, attention_head_size=4, dropout=0.1,
    hidden_continuous_size=32, learning_rate=1e-3)

trainer = pl.Trainer(max_epochs=15, accelerator="cpu", enable_progress_bar=False,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=4, mode="min")],
                     default_root_dir="data/checkpoints/tft_local")
trainer.fit(model, train_dataloaders=tr_ds.to_dataloader(train=True),
            val_dataloaders=va_ds.to_dataloader(train=False))

pred = model.predict(va_ds, mode="quantiles")
q = np.asarray(pred); horizon = q.shape[1]
loader = va_ds.to_dataloader(train=False, batch_size=BATCH)
acts=[]
for b in loader:
    a=np.asarray(b["decoder_target"].detach().cpu()); acts.append(a)
actual=np.concatenate(acts,axis=0).squeeze(-1)
if actual.ndim==1: actual=actual[:,None]
actual=actual[:,-horizon:]
q10,q50,q90=q[...,0],q[...,1],q[...,2]
mae=float(np.mean(np.abs(q50-actual)))
mae_naive=float(np.mean(np.abs(actual)))
pin10=float(np.mean(np.maximum(0.1*(actual-q10),0.9*(q10-actual))))
pin90=float(np.mean(np.maximum(0.9*(actual-q90),0.1*(q90-actual))))
cov=float(np.mean((actual>=q10)&(actual<=q90)))
print("="*70)
print(f"TFT (target={TARGET}, enc={ENC}, pred={PRED}) on real feature store, CPU")
print(f"  MAE={mae:.6f}  zero-baseline MAE={mae_naive:.6f}  ratio={mae/mae_naive:.3f}")
print(f"  quantile coverage(10-90%)={100*cov:.1f}%  pinball10={pin10:.6f} pinball90={pin90:.6f}")
print(f"  horizon={horizon} steps (5d target)")
print("  ratio < 1.0 => beats predicting 0; ratio ~1.0 => no forecast skill")
