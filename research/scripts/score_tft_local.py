"""Score the already-trained TFT checkpoint (no retrain). Loads model, predicts
quantiles on val, extracts actuals via return_y=True, reports forecast skill."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
import torch
from feature_engineering.feature_store import load_training_frame
from transformer_model.dataset import build_tft_dataset

TARGET="future_return_5d"; ENC=60; PRED=5; BATCH=128
SYMBOLS=["AAPL","MSFT","NVDA","SPY","QQQ","JPM","TSLA","XOM","GLD","TLT",
         "EURUSD=X","^NSEI","INFY.NS","HDFCBANK.NS"]
CKPT="data/checkpoints/tft_local/lightning_logs/version_0/checkpoints/epoch=7-step=1792.ckpt"

frame=load_training_frame(symbols=SYMBOLS, timeframe="SWING")
if not isinstance(frame.index, pd.DatetimeIndex): frame=frame.set_index("time")
df=frame.reset_index(); df["time"]=pd.to_datetime(df["time"])
cutoff=df["time"].quantile(0.85)
tr=df[df["time"]<=cutoff].set_index("time")
va=df[df["time"]>cutoff].set_index("time")

tr_ds=build_tft_dataset(tr,target=TARGET,max_encoder_length=ENC,max_prediction_length=PRED,batch_size=BATCH)
va_ds=build_tft_dataset(va,target=TARGET,max_encoder_length=ENC,max_prediction_length=PRED,batch_size=BATCH)

from transformer_model.model import AgonistesTFT
from pytorch_forecasting import TemporalFusionTransformer
import torch.serialization
from pytorch_forecasting.data.encoders import EncoderNormalizer
torch.serialization.add_safe_globals([EncoderNormalizer])
# Load the trained lightning checkpoint properly
model = TemporalFusionTransformer.load_from_checkpoint(CKPT)
model.eval()

res = model.predict(va_ds, mode="quantiles", return_y=True)
# res is a Prediction sequence: res[0] = quantiles (n,horizon,nq), res[4] = (pred, target_scale)
q = np.asarray(res[0])
pair = res[4] if len(res) > 4 and isinstance(res[4], (tuple, list)) else None
y = pair[0] if pair is not None else None
horizon = q.shape[1]
if y is not None:
    yt = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    while yt.ndim > 2:  # collapse trailing singleton dims
        yt = yt.squeeze(-1) if yt.shape[-1] == 1 else yt.reshape(yt.shape[0], -1)
    if yt.ndim == 1: yt = yt[:, None]
    actual = yt[:,-horizon:]
else:
    actual = None
q10,q50,q90=q[...,0],q[...,1],q[...,2]
n=min(len(actual),len(q50))
actual,q50,q10,q90=actual[:n],q50[:n],q10[:n],q90[:n]
mae=float(np.nanmean(np.abs(q50-actual)))
mae_naive=float(np.nanmean(np.abs(actual)))
pin10=float(np.nanmean(np.maximum(0.1*(actual-q10),0.9*(q10-actual))))
pin90=float(np.nanmean(np.maximum(0.9*(actual-q90),0.1*(q90-actual))))
cov=float(np.mean((actual>=q10)&(actual<=q90)))
print("="*70)
print(f"TFT score (target={TARGET}, enc={ENC}, pred={PRED}), epoch-7 checkpoint, CPU")
print(f"  MAE={mae:.6f}  zero-baseline MAE={mae_naive:.6f}  ratio={mae/mae_naive:.3f}")
print(f"  quantile coverage(10-90%)={100*cov:.1f}%  pinball10={pin10:.6f} pinball90={pin90:.6f}")
print(f"  horizon={horizon} steps (5d target)  val rows scored={n}")
print("  ratio < 1.0 => beats predicting 0; ratio ~1.0 => no forecast skill")
