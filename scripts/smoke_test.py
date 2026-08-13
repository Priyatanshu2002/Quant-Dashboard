#!/usr/bin/env python3
"""End-to-end Phase-1 smoke test on REAL market data.

Pipeline exercised:
  backfill → technical features → labeled feature store →
  screener scoring → Top-N → strategy backtest (full cost model) →
  walk-forward → regime tests → Monte Carlo → persistence

If the network is unavailable, falls back to synthetic GBM data (flagged).
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from core.db import get_storage
from core.logging import get_logger, setup_logging

log = get_logger(__name__)

EQUITY_UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "TLT", "^NSEI"]
CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD"]


def _synthetic_ohlcv(symbol: str, days: int = 730, seed: int = 0) -> pd.DataFrame:
    """GBM fallback used only when live data cannot be fetched."""
    rng = np.random.default_rng(seed)
    n = days
    drift = 0.0004 + 0.0002 * (seed % 5)
    vol = 0.015 + 0.01 * (seed % 3)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range(end=dt.date.today(), periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 - 0.001), "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)


def backfill_all(db) -> tuple[bool, dict[str, pd.DataFrame]]:
    """Fetch real data; returns (used_synthetic, {symbol: ohlcv})."""
    ohlcv_map: dict[str, pd.DataFrame] = {}
    synthetic = False

    try:
        from data_ingestion.price_feeds.equity_ws import backfill_equities
        results = backfill_equities(EQUITY_UNIVERSE, period="2y", storage=db)
        ohlcv_map.update(results)
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance backfill failed: %s", e)

    try:
        from data_ingestion.price_feeds.crypto_ws import backfill_klines
        for i, c in enumerate(CRYPTO_UNIVERSE):
            df = backfill_klines(c, interval="1d", days=730, storage=db)
            if not df.empty:
                ohlcv_map[c] = df
    except Exception as e:  # noqa: BLE001
        log.warning("Binance backfill failed: %s", e)

    # Fallback for anything missing
    for i, sym in enumerate(EQUITY_UNIVERSE + CRYPTO_UNIVERSE):
        if sym not in ohlcv_map:
            df = _synthetic_ohlcv(sym, seed=i)
            db.write_ohlcv(df, symbol=sym, asset_class=_class_of(sym),
                           source="SYNTHETIC", interval="1d")
            ohlcv_map[sym] = df
            synthetic = True
            log.warning("No live data for %s — using SYNTHETIC GBM", sym)

    return synthetic, ohlcv_map


def _class_of(symbol: str) -> str:
    if symbol.startswith(("BTC", "ETH", "SOL", "XRP")):
        return "CRYPTO"
    if symbol.endswith((".NS", ".BO")):
        return "EQUITY_IN"
    if symbol in ("SPY", "QQQ", "TLT", "GLD", "IWM", "EEM"):
        return "ETF"
    if symbol.startswith("^"):
        return "BOND" if "TNX" in symbol or "TYX" in symbol else "ETF"
    return "EQUITY_US"


def build_features(db, ohlcv_map: dict[str, pd.DataFrame]) -> None:
    from feature_engineering.feature_store import build_feature_frame, write_to_store

    for sym, ohlcv in ohlcv_map.items():
        if len(ohlcv) < 250:
            continue
        frame = build_feature_frame(sym, _class_of(sym), ohlcv, db=db,
                                    timeframe="SWING", with_labels=True)
        write_to_store(frame, sym, _class_of(sym), "SWING", db)


def fetch_context_feeds(db) -> None:
    """Best-effort no-key context: macro snapshot, equity fundamentals, sentiment."""
    from data_ingestion.macro_feeds.treasury_fetcher import (
        fetch_treasury_curve, fetch_vix_stooq)
    from data_ingestion.onchain_feeds.exchange_flow_fetcher import fetch_crypto_global
    from data_ingestion.fundamental_feeds.yfinance_earnings import refresh_info_snapshot

    # Macro (no API keys required)
    try:
        macro = fetch_treasury_curve(storage=db) or {}
        vix = fetch_vix_stooq()
        if vix is not None:
            import datetime as _dt
            db.write_macro_snapshot({"ts": _dt.datetime.utcnow(), "vix": vix})
            macro["vix"] = vix
        log.info("Macro snapshot: vix=%s yc=%s", macro.get("vix"),
                 macro.get("yield_curve_spread"))
    except Exception as e:  # noqa: BLE001
        log.warning("Macro fetch failed: %s", e)
    try:
        fetch_crypto_global(storage=db)
    except Exception as e:  # noqa: BLE001
        log.warning("CoinGecko fetch failed: %s", e)

    # Equity fundamentals via yfinance (one call per ticker, cached in store)
    for sym in EQUITY_UNIVERSE:
        try:
            refresh_info_snapshot(sym, storage=db)
        except Exception as e:  # noqa: BLE001
            log.debug("info snapshot %s failed: %s", sym, e)

    # Light sentiment pass for a few symbols (GDELT + news, no keys)
    from data_ingestion.sentiment_feeds.gdelt_fetcher import fetch_gdelt_events
    from data_ingestion.sentiment_feeds.news_aggregator import fetch_news_events
    for sym in ["AAPL", "MSFT", "BTC-USD"][:2]:
        for fetcher in (fetch_gdelt_events, fetch_news_events):
            try:
                fetcher(sym, maxrecords=10, storage=db) if fetcher.__name__ == "fetch_gdelt_events" \
                    else fetcher(sym, storage=db)
            except Exception as e:  # noqa: BLE001
                log.debug("%s for %s failed: %s", fetcher.__name__, sym, e)


def run_screener(db) -> list:
    from screener.pipeline import run_screener, screener_table
    selected = run_screener(top_n=8, db=db)
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  SCREENER — composite signal scores & Top-N candidates            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    rows = screener_table(selected)
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("(no candidates above threshold)")
    return selected


def run_backtests(db, ohlcv_map: dict[str, pd.DataFrame]) -> None:
    from backtesting.engine import BacktestEngine
    from backtesting.strategies import MACrossStrategy, MomentumStrategy

    engine = BacktestEngine()
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  BACKTESTS — full cost model (spread + slippage + commission)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    header = (f"{'strategy':18s} {'regime':12s} {'ret%':>8s} {'sharpe':>7s} "
              f"{'maxDD%':>7s} {'win%':>6s} {'trades':>6s}")
    print(header)
    print("-" * len(header))

    for sym in ["SPY", "BTC-USD", "AAPL", "^NSEI"]:
        ohlcv = ohlcv_map.get(sym)
        if ohlcv is None or len(ohlcv) < 120:
            continue
        for strat in (MACrossStrategy(fast=20, slow=100),
                      MomentumStrategy(lookback=20, threshold=0.03)):
            strat.fit(ohlcv)
            r = engine.run(sym, _class_of(sym), ohlcv, strat.generate_signals(),
                           strategy_name=type(strat).__name__)
            rep = r.report
            print(f"{type(strat).__name__ + ' ' + sym:18s} {rep.regime:12s} "
                  f"{rep.total_return_pct:8.2f} {rep.sharpe_ratio:7.2f} "
                  f"{rep.max_drawdown_pct:7.2f} {rep.win_rate:6.0%} "
                  f"{rep.total_trades:6d}")


def run_walk_forward(db, ohlcv_map) -> None:
    from backtesting.strategies import MACrossStrategy
    from backtesting.walk_forward import summarize_walk_forward, walk_forward_backtest

    ohlcv = ohlcv_map.get("SPY")
    if ohlcv is None or len(ohlcv) < 400:
        return
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  WALK-FORWARD — SPY, 12m train / 3m test, retrain each window     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    results = walk_forward_backtest(
        ohlcv, lambda: MACrossStrategy(fast=20, slow=100),
        symbol="SPY", asset_class="ETF", train_months=12, test_months=3,
        step_months=1, start_date=str(ohlcv.index[0].date()),
        end_date=str(ohlcv.index[-1].date()))
    table = summarize_walk_forward(results)
    if not table.empty:
        print(table.to_string(index=False))
        print(f"\nMean window Sharpe: {table['sharpe'].mean():.2f}   "
              f"windows with Sharpe>0: {(table['sharpe'] > 0).mean():.0%}")


def run_regimes(db, ohlcv_map) -> None:
    from backtesting.engine import BacktestEngine
    from backtesting.regime_tester import regime_sharpe_table, run_regime_tests
    from backtesting.strategies import MomentumStrategy

    ohlcv = ohlcv_map.get("SPY")
    if ohlcv is None:
        return
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  REGIME TESTS — SPY momentum strategy across 7 historical regimes ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    strat = MomentumStrategy(lookback=20, threshold=0.03)
    strat.fit(ohlcv)
    results = run_regime_tests("SPY", "ETF", ohlcv, strat.generate_signals(),
                               engine=BacktestEngine())
    table = regime_sharpe_table(results)
    if not table.empty:
        print(table.to_string(index=False))


def run_monte_carlo(db, ohlcv_map, n_runs: int = 200) -> None:
    from backtesting.engine import BacktestEngine
    from backtesting.monte_carlo import mc_summary, monte_carlo_shuffle
    from backtesting.strategies import MomentumStrategy

    ohlcv = ohlcv_map.get("SPY")
    if ohlcv is None:
        return
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  MONTE CARLO — {n_runs} shuffled-signal runs on SPY (luck baseline)        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    strat = MomentumStrategy(lookback=20, threshold=0.03)
    strat.fit(ohlcv)
    signals = strat.generate_signals()
    engine = BacktestEngine()
    actual = engine.run("SPY", "ETF", ohlcv, signals).report.sharpe_ratio
    t0 = time.time()
    mc = monte_carlo_shuffle("SPY", "ETF", ohlcv, signals, n_runs=n_runs,
                             engine=engine, seed=7)
    summary = mc_summary(mc, actual)
    print(f"Actual Sharpe:          {actual:.3f}")
    print(f"Shuffled mean Sharpe:   {summary['sharpe_mean']:.3f}  (p95 {summary['sharpe_p95']:.3f})")
    print(f"Actual > shuffled pct:  {summary['pctile_of_actual']:.1%}  "
          f"({n_runs} runs in {time.time() - t0:.0f}s)")


def persist_summary(db, synthetic: bool) -> None:
    db.write_portfolio_snapshot({
        "time": dt.datetime.utcnow(), "nav_usd": 1_000_000.0,
        "cash_usd": 1_000_000.0, "invested_usd": 0.0, "daily_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0, "realized_pnl_usd": 0.0, "var_95_usd": 0.0,
        "gross_exposure_usd": 0.0, "position_count": 0,
    })
    db.write_daily_performance({
        "time": dt.datetime.utcnow(), "total_return_pct": 0.0, "cagr": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
        "information_ratio": 0.0, "max_drawdown_pct": 0.0,
        "max_drawdown_duration_days": 0, "daily_var_95": 0.0,
        "volatility_annualized": 0.0, "total_trades": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "expectancy_per_trade_usd": 0.0,
        "alpha_vs_sp500": 0.0, "cost_drag_pct": 0.0,
    })
    db.write_circuit_breaker("SMOKE_TEST", "INFO",
                             "smoke test completed" + (" (SYNTHETIC DATA)" if synthetic else ""))


def run_smoke_test(n_mc_runs: int = 200) -> None:
    t0 = time.time()
    db = get_storage()
    log.info("Storage backend: %s", db.backend)

    synthetic, ohlcv_map = backfill_all(db)
    fetch_context_feeds(db)
    build_features(db, ohlcv_map)
    run_screener(db)
    run_backtests(db, ohlcv_map)
    run_walk_forward(db, ohlcv_map)
    run_regimes(db, ohlcv_map)
    run_monte_carlo(db, ohlcv_map, n_runs=n_mc_runs)
    persist_summary(db, synthetic)

    print(f"\n✅ Smoke test complete in {time.time() - t0:.0f}s "
          f"({'SYNTHETIC data — network unavailable' if synthetic else 'live data'})")
    print(f"   Feature vectors: {len(db.query_feature_vectors()):,} rows · "
          f"symbols: {len(ohlcv_map)}")


if __name__ == "__main__":
    setup_logging()
    run_smoke_test()
