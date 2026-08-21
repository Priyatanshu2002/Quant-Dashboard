import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from strategy_builder.backtest import full_metrics, breakeven_costs, passive_benchmark
from strategy_builder.features import build_universe_frame
from core.db import get_storage
from core.logging import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

BENCH_DIR = Path("data/benchmark")

def main():
    db = get_storage()
    from strategy_builder.run import UNIVERSE
    closes = {}
    for sym in UNIVERSE:
        df = db.query_ohlcv(sym)
        if df is not None and not df.empty and len(df) >= 400:
            closes[sym] = df["close"]
    prices = pd.DataFrame(closes).sort_index().dropna(axis=1, thresh=int(0.8 * len(closes[list(closes.keys())[0]])))
    panel = build_universe_frame(prices)
    passive = passive_benchmark(panel)

    weight_files = list(BENCH_DIR.glob("weights_*.csv"))
    log.info("Found %d model weight files in %s", len(weight_files), BENCH_DIR)

    results = []
    breakevens = {}

    for wf in weight_files:
        model_name = wf.stem.replace("weights_", "")
        try:
            w = pd.read_csv(wf)
            m = full_metrics(w, panel, passive)
            m["model"] = model_name
            results.append(m)
            try:
                be = breakeven_costs(w, panel)
                breakevens[model_name] = be.to_dict("records")
            except Exception as be_err:
                log.warning("Breakeven failed for %s: %s", model_name, be_err)
        except Exception as e:
            log.error("Failed to compute metrics for %s: %s", model_name, e)

    results.sort(key=lambda x: x.get("sharpe", -99), reverse=True)

    leaderboard_file = BENCH_DIR / "leaderboard.json"
    leaderboard_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    breakeven_file = BENCH_DIR / "breakeven.json"
    breakeven_file.write_text(json.dumps(breakevens, indent=2), encoding="utf-8")

    log.info("Leaderboard compiled with %d models -> %s", len(results), leaderboard_file)
    print("\n" + "="*80)
    print(f"{'RANK':<5} {'MODEL':<20} {'SHARPE':<8} {'CAGR %':<10} {'MAX DD %':<10} {'CALMAR':<8} {'HIT %':<8} {'T-HAC':<8}")
    print("="*80)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['model']:<20} {r.get('sharpe', 0):<8.3f} {r.get('cagr', 0)*100:<10.1f} {r.get('max_dd', 0)*100:<10.1f} {r.get('calmar', 0):<8.2f} {r.get('hit_rate', 0)*100:<8.1f} {r.get('t_hac', 0):<8.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
