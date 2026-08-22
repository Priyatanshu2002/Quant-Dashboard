"""Decisive diagnosis: is the classical-ML collapse genuine no-signal or a pipeline bug?

Tests: train ridge/xgboost/logistic on the SAME walk-forward window. Report
in-sample R2/AUC and OOS. If in-sample R2 ~ 0 even on training data, the
regression target (next-day return) is simply unpredictable -> collapse is CORRECT
and honest, not a bug. If in-sample R2 is large but OOS collapses -> overfitting/leakage bug.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from core.db import get_storage
from strategy_builder.features import build_universe_frame, ALL_FEATURE_COLS
from strategy_builder.trainer import WindowedDataset, walk_forward_windows

db = get_storage()
UNIVERSE = ["AAPL","AMZN","GOOGL","JPM","META","MSFT","NVDA","TSLA","UNH","XOM",
            "HDFCBANK.NS","INFY.NS","RELIANCE.NS","SBIN.NS","TCS.NS",
            "SPY","QQQ","IWM","EEM","GLD","TLT","EURUSD=X","GBPUSD=X","USDJPY=X",
            "USDINR=X","DX-Y.NYB","GC=F","^NSEI"]
closes={}
for s in UNIVERSE:
    o=db.query_ohlcv(s)
    if o is not None and len(o)>=400: closes[s]=o['close']
prices=pd.DataFrame(closes).sort_index().dropna(axis=1,thresh=int(0.8*len(pd.DataFrame(closes))))
panel=build_universe_frame(prices)
symbols=sorted(panel['symbol'].unique())

# first window
wins = walk_forward_windows(panel, 36, 6)
tr, va = wins[0]
lookback=64
tr_ds=WindowedDataset(tr, ALL_FEATURE_COLS, lookback, symbols)
va_ds=WindowedDataset(va, ALL_FEATURE_COLS, lookback, symbols)

# build matrices (last-step features like the linear models)
def mats(ds):
    X=ds.x[:,-1,:]; 
    y=np.clip(ds.r*ds.v,-20,20)   # regression target
    yc=(y>=0).astype(int)          # classification target (up/down)
    return X, y, yc, ds.r, ds.v
Xtr,ytr,ytrc,rtr,vtr=mats(tr_ds)
Xva,yva,yvac,rva,vva=mats(va_ds)
print(f"train {Xtr.shape} val {Xva.shape}")
print(f"target std train={ytr.std():.3f} val={yva.std():.3f}")

# RIDGE regression: in-sample R2
ridge=Pipeline([('s',StandardScaler()),('m',Ridge(alpha=1.0))])
ridge.fit(Xtr,ytr)
p_tr=ridge.predict(Xtr); p_va=ridge.predict(Xva)
def r2(y,p): ss=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2); return ss
print(f"\nRIDGE  in-sample R2={r2(ytr,p_tr):+.3f}  OOS R2={r2(yva,p_va):+.3f}")
print(f"       in-sample pred std={p_tr.std():.4f}  OOS pred std={p_va.std():.4f}")
print(f"       -> if in-sample R2~0, next-day return is unpredictable (correct collapse)")

# XGBOOST regression
xr=xgb.XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.05,n_jobs=4)
xr.fit(Xtr,ytr)
p_tr=xr.predict(Xtr); p_va=xr.predict(Xva)
print(f"XGBOOST in-sample R2={r2(ytr,p_tr):+.3f}  OOS R2={r2(yva,p_va):+.3f}  pred std OOS={p_va.std():.4f}")

# LOGISTIC classification (why it produced signal): in-sample & OOS AUC
from sklearn.metrics import roc_auc_score
lr=Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=1.0,max_iter=500))])
lr.fit(Xtr,ytrc)
for name,yy,pp in [("in-sample",ytrc,lr.predict_proba(Xtr)[:,1]),("OOS",yvac,lr.predict_proba(Xva)[:,1])]:
    if len(np.unique(yy))==2:
        print(f"LOGISTIC {name} AUC={roc_auc_score(yy,pp):.3f}")

# correlation of raw features with target (any linear signal at all?)
X_=Xtr; 
# standardize
Xs=(X_-X_.mean(0))/(X_.std(0)+1e-9)
corrs=np.abs(np.corrcoef(Xs.T, ytr)[:-1,-1])
print(f"\nMax |corr| of any single feature with next-day target: {corrs.max():.3f}  (0.0-0.1 = essentially no linear signal)")
top=np.argsort(corrs)[-5:][::-1]
print("Top-5 most predictive feature indices:", top, "corrs:", [round(corrs[i],3) for i in top])
