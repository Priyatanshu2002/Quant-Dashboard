"""Diagnose the strategy_builder target pipeline: is the collapse a data bug or genuine no-signal?"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
from core.db import get_storage
from strategy_builder.features import build_universe_frame
from strategy_builder.trainer import WindowedDataset

db = get_storage()
# replicate the benchmark universe
UNIVERSE = ["AAPL","AMZN","GOOGL","JPM","META","MSFT","NVDA","TSLA","UNH","XOM",
            "HDFCBANK.NS","INFY.NS","RELIANCE.NS","SBIN.NS","TCS.NS",
            "SPY","QQQ","IWM","EEM","GLD","TLT","EURUSD=X","GBPUSD=X","USDJPY=X",
            "USDINR=X","DX-Y.NYB","GC=F","^NSEI"]
closes={}
for s in UNIVERSE:
    o=db.query_ohlcv(s)
    if o is not None and len(o)>=400: closes[s]=o['close']
prices=pd.DataFrame(closes).sort_index()
prices=prices.dropna(axis=1,thresh=int(0.8*len(prices)))
panel=build_universe_frame(prices)
print(f"panel rows={len(panel):,} symbols={panel['symbol'].nunique()} range={panel['time'].min().date()}->{panel['time'].max().date()}")

# inspect the LEARNING TARGET the classical models actually fit
for sym in ["SPY","AAPL","NVDA","EURUSD=X"]:
    g=panel[panel['symbol']==sym]
    t=g['target'].dropna()
    print(f"\n{sym:10s} target: n={len(t):,} mean={t.mean():+.4f} std={t.std():.4f} "
          f"p5={t.quantile(.05):+.2f} p95={t.quantile(.95):+.2f} clip%={((t.abs()>=19.99).mean()*100):.1f}")
    print(f"          vs_factor: mean={g['vs_factor'].mean():.2f} p50={g['vs_factor'].median():.2f} "
          f"p90={g['vs_factor'].quantile(.9):.2f}")
    print(f"          ret_1: mean={g['ret_1'].mean():+.5f} std={g['ret_1'].std():.5f}")

# check cross-sectional: how many non-NaN targets per symbol
nz = panel.groupby('symbol')['target'].apply(lambda s:(s.abs()>1e-3).mean())
print(f"\nFraction of NON-zero targets per symbol (p50 across symbols): {nz.median():.3f}")

# What does the WindowedDataset produce as target (ds.r * ds.v)?
from strategy_builder.features import ALL_FEATURE_COLS
symbols=sorted(panel['symbol'].unique())
ds=WindowedDataset(panel,ALL_FEATURE_COLS,64,symbols)
tv=np.clip(ds.r*ds.v,-20,20)
print(f"\nWindowedDataset target = clip(r*v): mean={tv.mean():+.4f} std={tv.std():.4f} "
      f"p5={np.percentile(tv,5):+.2f} p95={np.percentile(tv,95):+.2f} |r|<1e-3 frac={(np.abs(tv)<1e-3).mean():.3f}")
print(f"  raw r: mean={ds.r.mean():+.6f} std={ds.r.std():.6f} | vs: mean={ds.v.mean():.2f} std={ds.v.std():.2f}")
