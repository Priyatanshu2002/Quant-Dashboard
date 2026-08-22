"""Benchmark runner — Oxford DL-for-finance protocol on the Agonistes universe.

Usage:
  python -m strategy_builder.run --models vlstm tft --seeds 3 --quick
  python -m strategy_builder.run --all --seeds 3
  python -m strategy_builder.run --models vlstm --smoke

Outputs to data/benchmark/: per-model weights CSV + results JSON + charts.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from strategy_builder.backtest import (breakeven_costs, full_metrics,
                                       passive_benchmark)
from strategy_builder.features import ALL_FEATURE_COLS, build_universe_frame
from strategy_builder.models import DEFAULT_HIDDEN, DEFAULT_LOOKBACK, ENCODERS
from strategy_builder.trainer import run_benchmark_model

from core.db import get_storage
from core.logging import get_logger

log = get_logger(__name__)

OUT = Path("data/benchmark")

# models to run: all encoders + all classical + all volatility baselines
NEURAL_MODELS = sorted(ENCODERS)
CLASSICAL_MODELS = [
    # Linear
    "ridge", "lasso", "elasticnet", "bayesian_ridge",
    # Tree / boosting
    "random_forest", "extra_trees", "gbm", "xgboost", "lightgbm", "catboost",
    # Kernel
    "svr_rbf",
    # Instance-based
    "knn",
    # Classification-as-signal
    "logistic", "lda",
    # Hybrid
    "hmm_lgbm", "strategy_xgb",
]
VOLATILITY_MODELS = [
    "vol_timing", "vol_carry", "vol_momentum",
    "vol_regime_ml", "har_signal",
    "vol_timing_egarch", "vol_timing_gjr",
]

# universe: liquid, multi-class assets from the store (skip crypto per scope)
UNIVERSE = [
    # US equities
    "AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "UNH", "XOM",
    # India
    "HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS",
    # ETFs / indices
    "SPY", "QQQ", "IWM", "EEM", "GLD", "TLT", "^NSEI",
    # FX / rates / commodities / vol
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "DX-Y.NYB", "^TNX", "^TYX", "GC=F",
]


def load_panel(db, universe: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    universe = universe or UNIVERSE
    closes, volumes, highs, lows = {}, {}, {}, {}
    for sym in universe:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < 400:
            log.info("skip %s: insufficient data", sym)
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
    log.info("panel: %d rows, %d symbols, %d features, %s → %s",
             len(panel), len(symbols), len(ALL_FEATURE_COLS),
             panel["time"].min().date(), panel["time"].max().date())
    return panel, symbols


def _df_or_none(d: dict, index) -> pd.DataFrame | None:
    if not d:
        return None
    df = pd.DataFrame(d).reindex(index)
    return df.dropna(axis=1, thresh=int(0.8 * len(df))) if len(df) else None


def run_neural(model_name: str, panel: pd.DataFrame, symbols: list[str],
               seeds: int, top_seeds: int, quick: bool) -> dict:
    t0 = time.time()
    lookback = 32 if quick else DEFAULT_LOOKBACK[model_name]
    hidden = 16 if quick else DEFAULT_HIDDEN[model_name]
    epochs = 15 if quick else 60
    res = run_benchmark_model(
        model_name, panel, ALL_FEATURE_COLS, symbols, lookback=lookback,
        hidden=hidden, seeds=seeds, top_seeds=top_seeds, epochs=epochs,
        train_months=24 if quick else 36, test_months=4 if quick else 6)
    res["model"] = model_name
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_classical(model_name: str, panel: pd.DataFrame, symbols: list[str],
                  quick: bool) -> dict:
    from strategy_builder.classical import CLASSICAL_REGISTRY
    t0 = time.time()
    lookback = 32 if quick else 64
    tm, tem = (24, 4) if quick else (36, 6)
    if model_name not in CLASSICAL_REGISTRY:
        raise ValueError(f"Unknown classical model: {model_name}. "
                         f"Available: {sorted(CLASSICAL_REGISTRY)}")
    fn = CLASSICAL_REGISTRY[model_name]
    weights = fn(panel, ALL_FEATURE_COLS, symbols,
                 lookback=lookback, train_months=tm, test_months=tem)
    return {"model": model_name, "weights": weights,
            "seconds": round(time.time() - t0, 1)}


def run_volatility(model_name: str, panel: pd.DataFrame, symbols: list[str],
                   quick: bool) -> dict:
    from strategy_builder.volatility_models import VOLATILITY_REGISTRY
    t0 = time.time()
    tm, tem = (24, 4) if quick else (36, 6)
    if model_name not in VOLATILITY_REGISTRY:
        raise ValueError(f"Unknown volatility model: {model_name}. "
                         f"Available: {sorted(VOLATILITY_REGISTRY)}")
    fn = VOLATILITY_REGISTRY[model_name]
    weights = fn(panel, ALL_FEATURE_COLS, symbols, train_months=tm, test_months=tem)
    return {"model": model_name, "weights": weights,
            "seconds": round(time.time() - t0, 1)}


def _worker(args: tuple) -> dict:
    """ProcessPool worker: (model_name, panel, symbols, seeds, top_seeds, quick)."""
    model_name, panel, symbols, seeds, top_seeds, quick = args
    if model_name in NEURAL_MODELS:
        return run_neural(model_name, panel, symbols, seeds, top_seeds, quick)
    if model_name in VOLATILITY_MODELS:
        return run_volatility(model_name, panel, symbols, quick)
    return run_classical(model_name, panel, symbols, quick)


def main() -> None:
    ap = argparse.ArgumentParser(description="DL-for-finance benchmark")
    ap.add_argument("--models", nargs="*", default=None, help="model names to run")
    ap.add_argument("--all", action="store_true", help="run every model")
    ap.add_argument("--smoke", action="store_true", help="1 model, tiny config")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--top-seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="reduced lookback/epochs for faster runs")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    db = get_storage()
    panel, symbols = load_panel(db)

    if args.smoke:
        models = ["vlstm"]
    elif args.all:
        models = NEURAL_MODELS + CLASSICAL_MODELS + VOLATILITY_MODELS
    elif args.models:
        models = args.models
    else:
        models = ["vlstm", "lstm", "tft", "xlstm"]

    quick = args.quick or args.smoke
    jobs = [(m, panel, symbols, args.seeds, args.top_seeds, quick)
            for m in models]
    results: dict[str, dict] = {}
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_worker, jobs):
                results[res["model"]] = res
    else:
        for m, pj, syms, seeds, top, qk in jobs:
            results[m] = _worker((m, pj, syms, seeds, top, qk))

    # metrics
    passive = passive_benchmark(panel)
    summary = []
    for model_name, res in results.items():
        w = res["weights"]
        w.to_csv(OUT / f"weights_{model_name}.csv", index=False)
        try:
            metrics = full_metrics(w, panel, passive)
        except Exception as e:  # noqa: BLE001
            log.error("%s metrics failed: %s", model_name, e)
            metrics = {"error": str(e)}
        metrics["model"] = model_name
        metrics["seconds"] = res["seconds"]
        summary.append(metrics)
        log.info("%s: sharpe=%.3f cagr=%.1f%% dd=%.1f%% t_hac=%.2f (%ss)",
                 model_name, metrics.get("sharpe", 0), 100 * metrics.get("cagr", 0),
                 100 * metrics.get("max_dd", 0), metrics.get("t_hac", 0),
                 res["seconds"])

    (OUT / "results.json").write_text(
        json.dumps({"summary": summary,
                    "val_log": [v for r in results.values() for v in r.get("val_log", [])]},
                   indent=2, default=str), encoding="utf-8")
    # breakeven costs for the best few models
    top = sorted(summary, key=lambda s: s.get("sharpe", -9), reverse=True)[:4]
    be = {}
    for m in top:
        try:
            be[m["model"]] = breakeven_costs(
                pd.read_csv(OUT / f"weights_{m['model']}.csv"), panel).to_dict("records")
        except Exception as e:  # noqa: BLE001
            be[m["model"]] = {"error": str(e)}
    (OUT / "breakeven.json").write_text(json.dumps(be, indent=2, default=str),
                                        encoding="utf-8")
    log.info("done — results in %s", OUT.resolve())


if __name__ == "__main__":
    main()
