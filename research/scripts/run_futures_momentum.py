"""Futures momentum benchmark: TSMOM baseline vs DL models.

Implements the Momentum Transformer / DMN protocol on real continuous-futures
daily data:
  * features = vol-normalized multi-horizon momentum (ret_norm_h) + MACD
  * position signal z = tanh(encoder) -> vol-targeted weight (sigma_tgt)
  * loss = pooled Sharpe
  * evaluated NET of transaction costs, vs TSMOM and passive baselines.

Data: 1-min Databento futures resampled to daily (ES,NQ,ZN,CL,DX,6E,GC).
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, time
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
import torch
torch.set_num_threads(os.cpu_count())

from data_ingestion.futures_daily import load_daily_futures
from strategy_builder.features import build_universe_frame
from strategy_builder.backtest import full_metrics, passive_benchmark
from strategy_builder.trainer import run_benchmark_model

# ---------------- data ----------------
prices = load_daily_futures()
print(f"Futures panel: {prices.shape}  {prices.index.min().date()}->{prices.index.max().date()}")

# Build long feature panel (cross-sectional rank on) with momentum features
panel = build_universe_frame(prices)
symbols = sorted(panel['symbol'].unique())
print(f"panel rows={len(panel):,} symbols={symbols}")
passive = passive_benchmark(panel)

# TSMOM baseline: sign of 3-month (63d) momentum, vol-scaled, equal across
def tsmom_weights(panel, horizon=63, sigma_tgt=0.10):
    rows=[]
    for sym,g in panel.groupby('symbol',sort=False):
        g=g.sort_values('time')
        mom = g['close'].pct_change(horizon) if 'close' in g else g['ret_norm_63']
        # use ret_norm_63 (vol-normalized momentum) as the TSMOM signal proxy
        sig = np.sign(g['ret_norm_63'].fillna(0))
        w = sig * sigma_tgt * g['vs_factor']
        rows.append(pd.DataFrame({'time':g['time'],'symbol':sym,'weight':w.values}))
    return pd.concat(rows,ignore_index=True)

tsmom = tsmom_weights(panel)
m_tsmom = full_metrics(tsmom, panel, passive, cost_bps=10.0, rebalance_days=1)
print(f"\nTSMOM (sign 63d momentum) net: sharpe={m_tsmom['sharpe']:+.3f} cagr={m_tsmom['cagr']*100:+.1f}% "
      f"maxdd={m_tsmom['max_dd']*100:+.1f}% turn={m_tsmom['turnover']:.0f}")

# ---------------- DL models on futures momentum ----------------
MODELS = ["nlinear","tft","vlstm","lstm","dlinear","itransformer","xlstm"]
LOOKBACK=32; HIDDEN=24; SEEDS=2; TOP=2; EPOCHS=20; BATCH=128
TRAIN_M=30; TEST_M=5; COST=10.0
from strategy_builder.features import ALL_FEATURE_COLS

print(f"\n=== DL models on futures momentum (net of {COST}bps) ===")
print(f"{'model':12s} {'netSharpe':>9} {'cagr%':>8} {'maxdd%':>7} {'hit%':>6} {'turn':>7} {'valShp':>7}")
rows=[]
for m in MODELS:
    t0=time.time()
    try:
        res=run_benchmark_model(m, panel, ALL_FEATURE_COLS, symbols, lookback=LOOKBACK,
            hidden=HIDDEN, seeds=SEEDS, top_seeds=TOP, epochs=EPOCHS, batch_size=BATCH,
            train_months=TRAIN_M, test_months=TEST_M, sigma_tgt=0.10, verbose=False)
        w=res['weights']
        w.to_csv(f"research/results/fut_{m}.csv", index=False)
        mm=full_metrics(w, panel, passive, cost_bps=COST, rebalance_days=1)
        vs=round(np.mean([v['val_sharpe'] for v in res.get('val_log',[])]),3) if res.get('val_log') else float('nan')
        rows.append((m,mm,vs))
        print(f"{m:12s} {mm['sharpe']:>+9.3f} {mm['cagr']*100:>+8.2f} {mm['max_dd']*100:>+7.1f} "
              f"{mm['hit_rate']*100:>6.1f} {mm['turnover']:>7.0f} {vs:>7.3f}")
    except Exception as e:
        print(f"{m:12s} ERROR {e}")

print(f"\nPassive (equal-weight long futures): sharpe={passive.mean()/passive.std()*np.sqrt(252):+.3f} "
      f"ann={passive.mean()*252*100:+.1f}%")
print(f"TSMOM net: sharpe={m_tsmom['sharpe']:+.3f}")
