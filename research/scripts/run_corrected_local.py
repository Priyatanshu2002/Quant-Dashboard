"""Corrected benchmark run (local CPU) after root-cause fixes.

Runs all classical models (full config) + a couple neural encoders (reduced
epochs, since CPU is slow) through the FIXED pipeline:
  - correct vol-normalized target (r/sigma, not r*sigma)
  - gross-capped leverage (sum|w| <= 1.5)
  - net-of-cost metrics (10 bps)
  - 48-feature set incl. cross-sectional ranks
Prints a leaderboard sorted by net OOS Sharpe.

Usage: .venv/Scripts/python research/scripts/run_corrected_local.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.db import get_storage  # noqa: E402
from strategy_builder.backtest import full_metrics, passive_benchmark
from strategy_builder.classical import CLASSICAL_REGISTRY
from strategy_builder.features import ALL_FEATURE_COLS, build_universe_frame
from strategy_builder.trainer import run_benchmark_model

COST_BPS = 10.0
UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "TSLA", "JPM", "XOM", "GLD", "TLT"]


def main() -> None:
    db = get_storage()
    closes, volumes, highs, lows = {}, {}, {}, {}
    for s in UNIVERSE:
        o = db.query_ohlcv(s)
        if o is None or len(o) < 400:
            continue
        closes[s] = o["close"]; volumes[s] = o["volume"]
        highs[s] = o["high"]; lows[s] = o["low"]
    prices = pd.DataFrame(closes).sort_index().dropna(axis=1, thresh=int(0.8 * len(closes)))
    idx = prices.index
    panel = build_universe_frame(prices, volumes=pd.DataFrame(volumes).reindex(idx),
                                 highs=pd.DataFrame(highs).reindex(idx),
                                 lows=pd.DataFrame(lows).reindex(idx))
    symbols = sorted(panel["symbol"].unique())
    passive = passive_benchmark(panel)
    print(f"panel: {len(panel)} rows, {len(symbols)} syms, {len(ALL_FEATURE_COLS)} features, "
          f"target_std={panel['target'].std():.2f}")

    summary = []
    # Classical — full config (fast on CPU)
    for name in CLASSICAL_REGISTRY:
        t0 = time.time()
        try:
            w = CLASSICAL_REGISTRY[name](panel, ALL_FEATURE_COLS, symbols,
                                         lookback=32, train_months=24, test_months=4)
            m = full_metrics(w, panel, passive, cost_bps=COST_BPS)
            m["model"] = name; m["type"] = "classical"; m["seconds"] = round(time.time() - t0, 1)
            summary.append(m)
            print(f"  {name:<14} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']*100:+.1f}% "
                  f"MaxDD={m['max_dd']*100:.1f}% hit={m['hit_rate']:.2f} ({m['seconds']}s)")
        except Exception as e:
            summary.append({"model": name, "error": str(e)[:80]})
            print(f"  {name:<14} ERROR {str(e)[:70]}")

    # Neural — reduced epochs on CPU
    for name in ["nlinear", "tft"]:
        t0 = time.time()
        try:
            res = run_benchmark_model(name, panel, ALL_FEATURE_COLS, symbols, lookback=24,
                                      hidden=16, seeds=2, top_seeds=1, epochs=8, batch_size=128,
                                      train_months=24, test_months=4, verbose=False)
            m = full_metrics(res["weights"], panel, passive, cost_bps=COST_BPS)
            m["model"] = name; m["type"] = "neural"; m["seconds"] = round(time.time() - t0, 1)
            summary.append(m)
            print(f"  {name:<14} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']*100:+.1f}% "
                  f"MaxDD={m['max_dd']*100:.1f}% hit={m['hit_rate']:.2f} ({m['seconds']}s)")
        except Exception as e:
            summary.append({"model": name, "error": str(e)[:80]})
            print(f"  {name:<14} ERROR {str(e)[:70]}")

    valid = [s for s in summary if "sharpe" in s]
    valid.sort(key=lambda s: s["sharpe"], reverse=True)
    print("\n===== CORRECTED LEADERBOARD (net-of-cost, sorted by Sharpe) =====")
    print(f"{'model':<14}{'sharpe':>8}{'cagr%':>9}{'maxdd%':>9}{'hit%':>7}{'t_hac':>7}{'turn':>8}")
    for s in valid:
        print(f"{s['model']:<14}{s['sharpe']:>+8.2f}{s['cagr']*100:>9.1f}"
              f"{s['max_dd']*100:>9.1f}{s['hit_rate']*100:>7.1f}{s['t_hac']:>7.2f}{s['turnover']:>8.1f}")


if __name__ == "__main__":
    main()
