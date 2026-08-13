#!/usr/bin/env python3
"""Project Agonistes — CLI entry point.

Subcommands:
  backfill   Fetch OHLCV history (yfinance / Binance) into the store
  features   Build + persist labeled feature vectors
  screen     Score the universe, print Top-N candidates
  backtest   Run a strategy backtest with the full report
  smoke      Full Phase-1 pipeline end-to-end
  serve      Minimal JSON API for the UI (stdlib http.server)
"""
from __future__ import annotations

import argparse
import json

from core.logging import setup_logging


def cmd_backfill(args) -> None:
    from data_ingestion.price_feeds.crypto_ws import backfill_klines
    from data_ingestion.price_feeds.equity_ws import backfill_equities

    tickers = args.tickers or ["AAPL", "MSFT", "SPY", "BTC-USD"]
    equities = [t for t in tickers if not t.startswith(("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"))]
    crypto = [t for t in tickers if t not in equities]
    if equities:
        backfill_equities(equities, period=args.period)
    for c in crypto:
        backfill_klines(c, interval="1d", days=_period_days(args.period))
    print(f"Backfilled {tickers}")


def cmd_features(args) -> None:
    from core.db import get_storage
    from feature_engineering.feature_store import build_feature_frame, write_to_store

    db = get_storage()
    symbols = args.tickers or db.symbols()
    for sym in symbols:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv.empty:
            print(f"skip {sym}: no data")
            continue
        frame = build_feature_frame(sym, "EQUITY_US", ohlcv, db=db, timeframe="SWING")
        n = write_to_store(frame, sym, "EQUITY_US", "SWING", db)
        print(f"{sym}: {n} feature vectors")


def cmd_screen(args) -> None:
    from screener.pipeline import ranked_table, run_screener, score_universe

    selected = run_screener(top_n=args.n)
    print(f"\n=== Screener — {len(selected)} qualified candidates (threshold 60.0) ===")
    for s in selected:
        b = s.breakdown()
        print(f"{s.symbol:10s} {s.asset_class:10s} composite={b['composite']:6.1f}  "
              f"tech={b['technical']:5.1f} fund={b['fundamental']:5.1f} "
              f"sent={b['sentiment']:5.1f} macro={b['macro']:5.1f} mom={b['momentum']:5.1f}")

    print(f"\n=== Full ranked universe ({len(selected) and 'top-N cap' or 'all scored'}) ===")
    rows = ranked_table(score_universe())
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("(no assets with market data — run: agonistes backfill)")


def cmd_backtest(args) -> None:
    from backtesting.data_loader import get_benchmark, get_ohlcv
    from backtesting.engine import BacktestEngine
    from backtesting.regime_tester import run_regime_tests
    from backtesting.strategies import make_strategy
    from core.db import get_storage

    db = get_storage()
    ohlcv = get_ohlcv(args.symbol, db=db)
    if ohlcv.empty:
        return
    strategy = make_strategy(args.strategy)
    strategy.fit(ohlcv)
    engine = BacktestEngine()
    benchmark = get_benchmark("SP500", start=ohlcv.index[0], end=ohlcv.index[-1], db=db)
    result = engine.run(args.symbol, "EQUITY_US", ohlcv, strategy.generate_signals(),
                        benchmark=benchmark if not benchmark.empty else None,
                        strategy_name=args.strategy)
    print("\n" + json.dumps(result.report.to_dict(), indent=2, default=str))
    regimes = run_regime_tests(args.symbol, "EQUITY_US", ohlcv,
                               strategy.generate_signals(), engine=engine)
    for name, r in regimes.items():
        print(f"  {name:20s} sharpe={r.report.sharpe_ratio:6.2f} "
              f"ret={r.report.total_return_pct:7.2f}% maxDD={r.report.max_drawdown_pct:6.2f}%")


def cmd_smoke(args) -> None:
    from scripts.smoke_test import run_smoke_test

    run_smoke_test()


def cmd_serve(args) -> None:
    from core.api_server import serve

    serve(host=args.host, port=args.port)


def cmd_orchestrate(args) -> None:
    """Launch the autonomous orchestrator (runs forever)."""
    from orchestrator import run_scheduler, run_job, ALL_JOBS

    if args.now:
        if args.now not in ALL_JOBS:
            print(f"Unknown job: {args.now}. Available: {', '.join(ALL_JOBS)}")
            return
        run_job(ALL_JOBS[args.now], dry_run=args.dry_run)
    else:
        run_scheduler(dry_run=args.dry_run)


def _period_days(period: str) -> int:
    return {"1y": 365, "2y": 730, "5y": 1825}.get(period, 730)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(prog="agonistes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backfill", help="fetch OHLCV history")
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--period", default="2y")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("features", help="build + store feature vectors")
    p.add_argument("--tickers", nargs="*", default=None)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("screen", help="run the screener")
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("backtest", help="run a strategy backtest")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--strategy", default="ma_cross",
                   choices=["ma_cross", "rsi_reversion", "momentum"])
    p.set_defaults(func=cmd_backtest)

    sub.add_parser("smoke", help="full Phase-1 pipeline").set_defaults(func=cmd_smoke)

    p = sub.add_parser("serve", help="JSON API for the UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("orchestrate",
                        help="start the autonomous orchestrator (runs forever)")
    p.add_argument("--dry-run", action="store_true",
                   help="print schedule, execute nothing")
    p.add_argument("--now", metavar="JOB", default=None,
                   help="force a specific job immediately")
    p.set_defaults(func=cmd_orchestrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
