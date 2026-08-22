"""Compute net-of-cost metrics for whatever dl_weights_*.csv exist on disk."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import glob, os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
from core.db import get_storage
from strategy_builder.backtest import full_metrics, passive_benchmark

UNIVERSE=["AAPL","MSFT","NVDA","GOOGL","AMZN","JPM","TSLA","META","UNH","XOM",
          "SPY","QQQ","GLD","TLT","IWM","EEM","^NSEI","HDFCBANK.NS","INFY.NS",
          "RELIANCE.NS","EURUSD=X","GBPUSD=X","USDJPY=X","USDINR=X","DX-Y.NYB","GC=F"]
COST=10.0
db=get_storage()
closes={}
for s in UNIVERSE:
    o=db.query_ohlcv(s)
    if o is not None and len(o)>=400: closes[s]=o['close']
prices=pd.DataFrame(closes).sort_index().dropna(axis=1,thresh=int(0.8*len(closes)))
rets=prices.pct_change()
panels=[]
for s in rets.columns:
    r=rets[s].dropna(); panels.append(pd.DataFrame({'time':r.index,'symbol':s,'ret_1':r.values}))
panel=pd.concat(panels).sort_values(['symbol','time'])
passive=passive_benchmark(panel)

files=sorted(glob.glob('research/results/dl_weights_*.csv'))
print(f"{'model':14s} {'netSharpe':>9} {'cagr%':>8} {'vol%':>6} {'maxdd%':>7} {'calmar':>6} {'hit%':>6} {'turn':>7} {'infoR':>6} {'valShp':>7}")
rows=[]
for f in files:
    name=os.path.basename(f).replace('dl_weights_','').replace('.csv','')
    w=pd.read_csv(f); w['time']=pd.to_datetime(w['time'])
    m=full_metrics(w,panel,passive,cost_bps=COST,rebalance_days=1)
    m['model']=name; rows.append(m)
    print(f"{name:14s} {m['sharpe']:>+9.3f} {m['cagr']*100:>+8.2f} {m['ann_vol']*100:>6.2f} "
          f"{m['max_dd']*100:>+7.1f} {m['calmar']:>6.2f} {m['hit_rate']*100:>6.1f} "
          f"{m['turnover']:>7.1f} {m['info_ratio']:>+6.2f}")
print(f"\nPassive: sharpe={passive.mean()/passive.std()*np.sqrt(252):+.3f} ann={passive.mean()*252*100:+.1f}%")
print(f"\n({len(files)} models on disk so far; run_dl_models_local.py still training the rest)")
