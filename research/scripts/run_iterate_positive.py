"""Iterate to find positive Sharpe: sweep rebalance intervals + horizons.

Daily-rebalance turnover + costs killed the models. This script tests the key
low-turnover models (tft, lasso, elasticnet, momentum) at rebalance intervals
of 5/10/20 days and reports net-of-cost Sharpe/CAGR for each, so we find the
combination that produces positive returns.

Usage: .venv/Scripts/python research/scripts/run_iterate_positive.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.db import get_storage  # noqa: E402
from strategy_builder.backtest import full_metrics, passive_benchmark  # noqa: E402
from strategy_builder.classical import (elasticnet_ml, ridge_lasso)  # noqa: E402
from strategy_builder.features import ALL_FEATURE_COLS, build_universe_frame  # noqa: E402
from strategy_builder.trainer import run_benchmark_model  # noqa: E402

COST_BPS = 10.0
UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "TSLA", "JPM", "XOM", "GLD", "TLT"]


def lasso_ml(panel, feature_cols, symbols, **kw):
    return ridge_lasso(panel, feature_cols, symbols, penalty="l1", **kw)


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
    print(f"panel: {len(panel)} rows, {len(symbols)} syms, target_std={panel['target'].std():.2f}")

    # Build model weights once (fast), then evaluate at each rebalance interval.
    models = {}
    for name in ["lasso", "elasticnet"]:
        t0 = time.time()
        w = {"lasso": lasso_ml, "elasticnet": elasticnet_ml}[name](
            panel, ALL_FEATURE_COLS, symbols, lookback=32, train_months=24, test_months=4)
        models[name] = w
        print(f"  built {name}: {len(w)} rows ({time.time()-t0:.1f}s)")

    # momentum cross-sectional strategy: top/bottom decile by ret_norm_21
    def momentum_weights():
        g = panel.groupby("time")
        mom = panel.groupby("time")["ret_norm_21"].rank(pct=True)
        w = pd.DataFrame({"time": panel["time"], "symbol": panel["symbol"],
                          "weight": 2.0 * (mom - 0.5)})  # [-1, 1] long-short
        return w
    models["momentum_cs"] = momentum_weights()

    for name in ["tft"]:
        t0 = time.time()
        try:
            res = run_benchmark_model(name, panel, ALL_FEATURE_COLS, symbols, lookback=24,
                                      hidden=16, seeds=2, top_seeds=1, epochs=8, batch_size=128,
                                      train_months=24, test_months=4, verbose=False)
            models[name] = res["weights"]
            print(f"  built {name}: {len(res['weights'])} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {name} build FAILED: {str(e)[:80]}")

    # Sweep rebalance intervals.
    print(f"\n{'model':<14}{'reb':>5}{'sharpe':>8}{'cagr%':>9}{'maxdd%':>9}{'hit%':>7}{'turn':>8}")
    best = None
    for name, w in models.items():
        for reb in (5, 10, 20):
            try:
                m = full_metrics(w, panel, passive, cost_bps=COST_BPS, rebalance_days=reb)
                print(f"{name:<14}{reb:>5}{m['sharpe']:>+8.2f}{m['cagr']*100:>9.1f}"
                      f"{m['max_dd']*100:>9.1f}{m['hit_rate']*100:>7.1f}{m['turnover']:>8.1f}")
                if best is None or m["sharpe"] > best[0]:
                    best = (m["sharpe"], name, reb, m["cagr"], m["max_dd"])
            except Exception as e:
                print(f"{name:<14}{reb:>5}  ERROR {str(e)[:50]}")

    print(f"\nBEST: {best[1]} @ rebalance {best[2]}d -> Sharpe {best[0]:+.2f}, "
          f"CAGR {best[3]*100:+.1f}%, MaxDD {best[4]*100:.1f}%")


if __name__ == "__main__":
    main()
