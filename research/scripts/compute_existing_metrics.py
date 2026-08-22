"""Compute full financial metrics for every model with on-disk weights."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import glob, os
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)

import numpy as np, pandas as pd
from core.db import get_storage
from strategy_builder.backtest import full_metrics, passive_benchmark

COST_BPS = 10.0  # benchmark's realistic net-of-cost value

def load_panel():
    db = get_storage()
    symbols = None
    # derive symbol set from first weights file
    w0 = pd.read_csv(sorted(glob.glob('research/results/weights_*.csv'))[0])
    symbols = sorted(w0['symbol'].unique())
    closes = {}
    for sym in symbols:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv is None or ohlcv.empty:
            continue
        closes[sym] = ohlcv['close']
    prices = pd.DataFrame(closes).sort_index()
    rets = prices.pct_change()
    # build long panel with ret_1
    panels = []
    for sym in rets.columns:
        s = rets[sym].dropna()
        panels.append(pd.DataFrame({'time': s.index, 'symbol': sym, 'ret_1': s.values}))
    panel = pd.concat(panels).sort_values(['symbol','time'])
    return panel, symbols

panel, symbols = load_panel()
print(f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
      f"{panel['time'].min().date()} -> {panel['time'].max().date()}")
passive = passive_benchmark(panel)

rows = []
for f in sorted(glob.glob('research/results/weights_*.csv')):
    name = os.path.basename(f).replace('weights_','').replace('.csv','')
    w = pd.read_csv(f)
    w['time'] = pd.to_datetime(w['time'])
    for cost, tag in [(0.0,'gross'), (COST_BPS,'net10bps')]:
        try:
            m = full_metrics(w, panel, passive, cost_bps=cost, rebalance_days=1)
            rows.append({'model': name, 'cost': tag, **m})
        except Exception as e:
            rows.append({'model': name, 'cost': tag, 'error': str(e)})

df = pd.DataFrame(rows)
cols = ['model','cost','cagr','ann_return','ann_vol','sharpe','t_hac','hit_rate',
        'max_dd','calmar','worst_3m_sharpe','min_ann_sharpe','cvar_5',
        'turnover','xgmv','info_ratio','t_hac_vs_passive','corr_vs_passive','days']
for c in cols:
    if c in df.columns and c not in ('model','cost'):
        df[c] = df[c].map(lambda x: round(x,4) if isinstance(x,(int,float)) else x)
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',200)
print("\n=== FINANCIAL METRICS (gross AND net-of-10bps) ===")
print(df[cols].to_string(index=False))

# passive benchmark for reference
print(f"\nPassive equal-weight benchmark: ann_ret={passive.mean()*252:+.4f} "
      f"ann_vol={passive.std()*np.sqrt(252):.4f} sharpe={passive.mean()/passive.std()*np.sqrt(252):+.3f} "
      f"days={len(passive)}")
