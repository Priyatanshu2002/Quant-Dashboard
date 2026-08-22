"""
Research Test #2: Full Model Benchmark
=======================================
Project Agonistes -- Research Lab

Runs the actual ML and DL models from the Oxford benchmark protocol on our
5-year data. Focus: mid-frequency (swing) and low-frequency (position/investing).

Models tested (by published Sharpe, highest first):
  NEURAL:  VLSTM (2.40), TFT (2.20), xLSTM (1.79), LSTM (1.48), LPatchTST (2.31)
  CLASSICAL ML: XGBoost, LightGBM, CatBoost, HMM+LightGBM, Ridge, Logistic

Protocol:
  - Walk-forward: 36-month train, 6-month OOS test (mid/low frequency)
  - Pooled-Sharpe loss for neural models
  - Volatility-normalized features (Oxford Appendix A)
  - Seed ensembling (3 seeds, top 2)
  - Full metrics: Sharpe, Sortino, Calmar, MaxDD, Win Rate, etc.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from core.db import get_storage
from core.logging import get_logger
from strategy_builder.features import ALL_FEATURE_COLS, build_universe_frame
from strategy_builder.backtest import (
    full_metrics, passive_benchmark, breakeven_costs, equity_curve_from_weights
)
from strategy_builder.trainer import run_benchmark_model
from strategy_builder.classical import CLASSICAL_REGISTRY
from strategy_builder.models import ENCODERS, DEFAULT_HIDDEN, DEFAULT_LOOKBACK

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration: Mid/Low Frequency Trading Focus
# ---------------------------------------------------------------------------

# Universe: liquid multi-asset symbols for swing/position trading
UNIVERSE = [
    # US large-cap equities
    "AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "UNH", "XOM",
    # Indian equities
    "HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS",
    # ETFs / indices
    "SPY", "QQQ", "IWM", "EEM", "GLD", "TLT",
    # FX / rates / commodities
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "DX-Y.NYB", "GC=F",
    # Benchmarks
    "^NSEI",
]

# Walk-forward params: appropriate for mid/low frequency
TRAIN_MONTHS = 36    # 3 years training
TEST_MONTHS = 6      # 6 months OOS
LOOKBACK = 64        # ~3 months of daily bars
HIDDEN = 32          # encoder hidden dim
SEEDS = 3            # 3 random seeds
TOP_SEEDS = 2        # ensemble top 2
EPOCHS = 60          # full training
BATCH_SIZE = 256
LR = 1e-3
SIGMA_TGT = 0.10     # 10% annualized vol target
COST_BPS = 10.0      # proportional transaction cost (10 bps per unit turnover)

# Models to run, ordered by priority
NEURAL_MODELS = ["vlstm", "tft", "xlstm", "lstm", "lpatchtst"]
CLASSICAL_MODELS = ["xgboost", "lightgbm", "catboost", "hmm_lgbm", "ridge", "logistic"]

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_panel(db):
    """Load OHLCV, build features, return panel + symbol list."""
    closes, volumes, highs, lows = {}, {}, {}, {}
    for sym in UNIVERSE:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < 400:
            print(f"  SKIP {sym}: insufficient data ({0 if ohlcv is None else len(ohlcv)} bars)")
            continue
        closes[sym] = ohlcv["close"]
        if "volume" in ohlcv.columns:
            volumes[sym] = ohlcv["volume"]
        if "high" in ohlcv.columns:
            highs[sym] = ohlcv["high"]
        if "low" in ohlcv.columns:
            lows[sym] = ohlcv["low"]

    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
    panel = build_universe_frame(
        prices,
        volumes=_df_or_none(volumes, prices.index),
        highs=_df_or_none(highs, prices.index),
        lows=_df_or_none(lows, prices.index))
    symbols = sorted(panel["symbol"].unique())
    print(f"  Panel: {len(panel):,} rows, {len(symbols)} symbols, "
          f"{panel['time'].min().date()} -> {panel['time'].max().date()}")
    print(f"  Features ({len(ALL_FEATURE_COLS)}): {ALL_FEATURE_COLS}")
    return panel, symbols


def _df_or_none(d: dict, index):
    if not d:
        return None
    df = pd.DataFrame(d).reindex(index)
    return df.dropna(axis=1, thresh=int(0.8 * len(df))) if len(df) else None


def run_neural_model(model_name, panel, symbols):
    """Run a single neural model through the full benchmark pipeline."""
    print(f"\n  Running {model_name.upper()} (neural, {SEEDS} seeds, {EPOCHS} epochs)...")
    t0 = time.time()

    try:
        res = run_benchmark_model(
            model_name, panel, ALL_FEATURE_COLS, symbols,
            lookback=LOOKBACK, hidden=HIDDEN,
            seeds=SEEDS, top_seeds=TOP_SEEDS, epochs=EPOCHS,
            batch_size=BATCH_SIZE, lr=LR,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
            sigma_tgt=SIGMA_TGT, use_ticker_emb=True,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  {model_name}: done in {elapsed:.1f}s, {res['windows']} windows")
        return {
            "model": model_name,
            "type": "neural",
            "weights": res["weights"],
            "val_log": res.get("val_log", []),
            "windows": res["windows"],
            "seconds": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR {model_name}: {e}")
        import traceback; traceback.print_exc()
        return {
            "model": model_name,
            "type": "neural",
            "weights": pd.DataFrame(columns=["time", "symbol", "weight"]),
            "error": str(e),
            "seconds": round(elapsed, 1),
        }


def run_classical_model(model_name, panel, symbols):
    """Run a single classical ML model."""
    print(f"\n  Running {model_name.upper()} (classical ML)...")
    t0 = time.time()

    try:
        if model_name not in CLASSICAL_REGISTRY:
            raise ValueError(f"Unknown: {model_name}. Available: {sorted(CLASSICAL_REGISTRY)}")

        fn = CLASSICAL_REGISTRY[model_name]
        weights = fn(panel, ALL_FEATURE_COLS, symbols,
                     lookback=LOOKBACK,
                     train_months=TRAIN_MONTHS,
                     test_months=TEST_MONTHS)
        elapsed = time.time() - t0
        print(f"  {model_name}: done in {elapsed:.1f}s, {len(weights)} weight rows")
        return {
            "model": model_name,
            "type": "classical_ml",
            "weights": weights,
            "seconds": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR {model_name}: {e}")
        import traceback; traceback.print_exc()
        return {
            "model": model_name,
            "type": "classical_ml",
            "weights": pd.DataFrame(columns=["time", "symbol", "weight"]),
            "error": str(e),
            "seconds": round(elapsed, 1),
        }


def compute_all_metrics(results, panel, passive_ret, cost_bps: float = 0.0):
    """Compute full metrics for all models.

    cost_bps: proportional transaction cost (bps) applied so the leaderboard
    shows NET-of-cost performance rather than the misleading gross Sharpe.
    """
    summary = []
    for res in results:
        model_name = res["model"]
        w = res["weights"]
        if w.empty:
            summary.append({
                "model": model_name, "type": res["type"],
                "error": res.get("error", "empty weights"),
                "seconds": res["seconds"],
            })
            continue

        try:
            metrics = full_metrics(w, panel, passive_ret, cost_bps=cost_bps)
            metrics["model"] = model_name
            metrics["type"] = res["type"]
            metrics["seconds"] = res["seconds"]
            if "val_log" in res:
                val_sharpes = [v["val_sharpe"] for v in res.get("val_log", [])]
                if val_sharpes:
                    metrics["avg_val_sharpe"] = round(np.mean(val_sharpes), 4)
                    metrics["max_val_sharpe"] = round(max(val_sharpes), 4)
            summary.append(metrics)
        except Exception as e:
            print(f"  METRICS ERROR {model_name}: {e}")
            summary.append({
                "model": model_name, "type": res["type"],
                "error": str(e), "seconds": res["seconds"],
            })

    return summary


def print_leaderboard(summary):
    """Print a formatted leaderboard."""
    valid = [s for s in summary if "sharpe" in s]
    valid.sort(key=lambda s: s.get("sharpe", -99), reverse=True)

    print("\n" + "=" * 100)
    print("  MODEL LEADERBOARD (sorted by OOS Sharpe)")
    print("=" * 100)
    print(f"  {'#':>3s} {'Model':<18s} {'Type':<14s} {'Sharpe':>8s} {'CAGR':>8s} "
          f"{'MaxDD':>8s} {'Calmar':>8s} {'Hit%':>7s} {'Turnover':>9s} "
          f"{'InfoRat':>8s} {'t(HAC)':>7s} {'Time':>7s}")
    print("  " + "-" * 97)

    for i, s in enumerate(valid, 1):
        print(f"  {i:3d} {s['model']:<18s} {s['type']:<14s} "
              f"{s.get('sharpe', 0):+7.3f} "
              f"{s.get('cagr', 0):+7.2f}% "
              f"{s.get('max_dd', 0):+7.2f}% "
              f"{s.get('calmar', 0):+7.3f} "
              f"{s.get('hit_rate', 0):6.1%} "
              f"{s.get('turnover', 0):8.0f} "
              f"{s.get('info_ratio', 0):+7.3f} "
              f"{s.get('t_hac', 0):+6.2f} "
              f"{s.get('seconds', 0):6.1f}s")

    # Errors
    errors = [s for s in summary if "error" in s]
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for s in errors:
            print(f"    {s['model']}: {s['error']}")


def main():
    print("=" * 100)
    print("  PROJECT AGONISTES -- RESEARCH TEST #2: FULL MODEL BENCHMARK")
    print("  Focus: Mid-Frequency (Swing) + Low-Frequency (Position/Investing)")
    print("=" * 100)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Walk-forward: {TRAIN_MONTHS}m train / {TEST_MONTHS}m test")
    print(f"  Lookback: {LOOKBACK} bars | Hidden: {HIDDEN} | Seeds: {SEEDS}")
    print(f"  Neural models: {NEURAL_MODELS}")
    print(f"  Classical ML: {CLASSICAL_MODELS}")
    print("=" * 100)

    # Load data
    print("\n[1/4] Loading data and building features...")
    db = get_storage()
    panel, symbols = load_panel(db)

    # Compute passive benchmark
    passive = passive_benchmark(panel)
    print(f"  Passive benchmark: {len(passive)} daily returns")

    # Run classical ML first (faster)
    print("\n[2/4] Running Classical ML models...")
    print("=" * 60)
    classical_results = []
    for model_name in CLASSICAL_MODELS:
        res = run_classical_model(model_name, panel, symbols)
        classical_results.append(res)
        # Save weights
        if not res["weights"].empty:
            res["weights"].to_csv(
                RESULTS_DIR / f"weights_{model_name}.csv", index=False)

    # Run neural models (slower)
    print("\n[3/4] Running Neural/DL models...")
    print("=" * 60)
    neural_results = []
    for model_name in NEURAL_MODELS:
        res = run_neural_model(model_name, panel, symbols)
        neural_results.append(res)
        # Save weights
        if not res["weights"].empty:
            res["weights"].to_csv(
                RESULTS_DIR / f"weights_{model_name}.csv", index=False)

    # Compute metrics
    print("\n[4/4] Computing metrics...")
    all_results = classical_results + neural_results
    summary = compute_all_metrics(all_results, panel, passive, cost_bps=COST_BPS)

    # Print leaderboard
    print_leaderboard(summary)

    # Save results
    output = {
        "test_name": "02_full_model_benchmark",
        "run_time": datetime.now().isoformat(),
        "config": {
            "train_months": TRAIN_MONTHS,
            "test_months": TEST_MONTHS,
            "lookback": LOOKBACK,
            "hidden": HIDDEN,
            "seeds": SEEDS,
            "top_seeds": TOP_SEEDS,
            "epochs": EPOCHS,
            "sigma_tgt": SIGMA_TGT,
            "universe": UNIVERSE,
        },
        "panel_info": {
            "rows": len(panel),
            "symbols": symbols,
            "date_range": f"{panel['time'].min().date()} to {panel['time'].max().date()}",
            "features": ALL_FEATURE_COLS,
        },
        "summary": summary,
    }

    json_path = RESULTS_DIR / "02_full_benchmark_results.json"
    json_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n  Results saved to: {json_path}")

    # Save CSV summary
    valid_summary = [s for s in summary if "sharpe" in s]
    if valid_summary:
        df = pd.DataFrame(valid_summary)
        cols = ["model", "type", "sharpe", "cagr", "ann_return", "ann_vol",
                "max_dd", "calmar", "hit_rate", "turnover", "info_ratio",
                "t_hac", "t_hac_vs_passive", "corr_vs_passive",
                "worst_3m_sharpe", "min_ann_sharpe", "cvar_5", "days", "seconds"]
        available = [c for c in cols if c in df.columns]
        df[available].to_csv(RESULTS_DIR / "02_full_benchmark_summary.csv", index=False)

    # Breakeven costs for top models
    top = sorted(valid_summary, key=lambda s: s.get("sharpe", -99), reverse=True)[:4]
    be = {}
    for m in top:
        try:
            w_path = RESULTS_DIR / f"weights_{m['model']}.csv"
            if w_path.exists():
                w = pd.read_csv(w_path)
                w["time"] = pd.to_datetime(w["time"])
                be[m["model"]] = breakeven_costs(w, panel).to_dict("records")
        except Exception as e:
            be[m["model"]] = {"error": str(e)}
    (RESULTS_DIR / "02_breakeven_costs.json").write_text(
        json.dumps(be, indent=2, default=str), encoding="utf-8")

    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_time = sum(s.get("seconds", 0) for s in summary)
    print(f"  Total compute time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print("=" * 100)


if __name__ == "__main__":
    main()
