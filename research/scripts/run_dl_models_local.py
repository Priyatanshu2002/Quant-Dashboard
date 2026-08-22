"""Run the FULL neural/DL model leg locally on CPU, honest net-of-cost metrics.

Runs every encoder in strategy_builder/models.py through the pooled-Sharpe
walk-forward protocol, then computes full metrics (gross + net-of-10bps) and a
passive benchmark. CPU config is reduced (fewer seeds/epochs/symbols) so it
completes on a laptop, but the protocol is identical to the GPU run.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, time
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd
import torch
from core.db import get_storage
from strategy_builder.features import ALL_FEATURE_COLS, build_universe_frame
from strategy_builder.backtest import full_metrics, passive_benchmark
from strategy_builder.trainer import run_benchmark_model
from strategy_builder.models import ENCODERS

torch.set_num_threads(os.cpu_count())

# ---- CPU-reduced config (identical protocol to GPU run, fewer seeds/epochs) ----
UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","JPM","TSLA","META","UNH","XOM",
            "SPY","QQQ","GLD","TLT","IWM","EEM","^NSEI","HDFCBANK.NS","INFY.NS",
            "RELIANCE.NS","EURUSD=X","GBPUSD=X","USDJPY=X","USDINR=X","DX-Y.NYB","GC=F"]
MODELS = ["ar1x","dlinear","nlinear","lstm","xlstm","vlstm","patchtst","itransformer",
          "lpatchtst","tft","mamba2","pslstm"]
LOOKBACK=16; HIDDEN=16; SEEDS=1; TOP_SEEDS=1; EPOCHS=8; BATCH=256
TRAIN_MONTHS=30; TEST_MONTHS=5; SIGMA_TGT=0.10; COST_BPS=10.0
RESULTS=PROJECT_ROOT/"research/results"
RESULTS.mkdir(exist_ok=True)

db=get_storage()
closes={}
for s in UNIVERSE:
    o=db.query_ohlcv(s)
    if o is not None and len(o)>=400: closes[s]=o['close']
prices=pd.DataFrame(closes).sort_index().dropna(axis=1,thresh=int(0.8*len(closes)))
panel=build_universe_frame(prices)
symbols=sorted(panel['symbol'].unique())
passive=passive_benchmark(panel)
print(f"panel rows={len(panel):,} symbols={len(symbols)} {panel['time'].min().date()}->{panel['time'].max().date()}")
print(f"features={len(ALL_FEATURE_COLS)} | models={MODELS} | device=cpu({os.cpu_count()} threads)")
print(f"config lookback={LOOKBACK} hidden={HIDDEN} seeds={SEEDS} epochs={EPOCHS}")

summary=[]
for m in MODELS:
    t0=time.time()
    try:
        res=run_benchmark_model(m, panel, ALL_FEATURE_COLS, symbols, lookback=LOOKBACK,
            hidden=HIDDEN, seeds=SEEDS, top_seeds=TOP_SEEDS, epochs=EPOCHS,
            batch_size=BATCH, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
            sigma_tgt=SIGMA_TGT, use_ticker_emb=True, verbose=False)
        w=res['weights']
        w.to_csv(RESULTS/f"dl_weights_{m}.csv", index=False)
        metrics=full_metrics(w, panel, passive, cost_bps=COST_BPS, rebalance_days=1)
        metrics['model']=m; metrics['seconds']=round(time.time()-t0,1); metrics['windows']=res['windows']
        if res.get('val_log'): metrics['avg_val_sharpe']=round(np.mean([v['val_sharpe'] for v in res['val_log']]),3)
        summary.append(metrics)
        print(f"  {m:14s} done {time.time()-t0:6.1f}s  netSharpe={metrics['sharpe']:+.3f}  wrows={len(w)}  valSharpe={metrics.get('avg_val_sharpe','-')}")
    except Exception as e:
        print(f"  {m:14s} ERROR: {e}")
        summary.append({'model':m,'error':str(e)})

df=pd.DataFrame(summary)
print("\n"+"="*100)
print(f"DL LEADERBOARD  (net of {COST_BPS}bps, daily rebalance, {TRAIN_MONTHS}m/{TEST_MONTHS}m walk-forward, CPU)")
print("="*100)
cols=['model','sharpe','cagr','ann_vol','max_dd','calmar','hit_rate','turnover','info_ratio','avg_val_sharpe','seconds']
for s in df.sort_values('sharpe',ascending=False).to_dict('records'):
    if 'sharpe' not in s: print(f"  {s['model']}: ERROR {s.get('error')}"); continue
    print(f"  {s['model']:14s} netSharpe={s['sharpe']:+7.3f} cagr={s['cagr']*100:+7.2f}% vol={s['ann_vol']*100:5.2f}% "
          f"maxdd={s['max_dd']*100:+6.1f}% calmar={s['calmar']:+5.2f} hit={s['hit_rate']*100:4.1f}% "
          f"turn={s['turnover']:6.1f} infoR={s['info_ratio']:+5.2f} valSharpe={s.get('avg_val_sharpe','-'):>6}")
print(f"\nPassive benchmark: sharpe={passive.mean()/passive.std()*np.sqrt(252):+.3f} ann={passive.mean()*252*100:+.1f}%")
df.to_csv(RESULTS/'dl_neural_leaderboard.csv', index=False)
print(f"\nSaved -> {RESULTS/'dl_neural_leaderboard.csv'}")
