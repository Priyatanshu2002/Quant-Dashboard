"""
Research Test #1: MA Crossover Strategy Backtest
=================================================
Project Agonistes -- Research Lab
Tests the simplest classical strategy (EMA 20/100 crossover) on real 5-year data
across multiple asset classes to validate the data pipeline and establish a baseline.

Output: research/results/01_ma_crossover_results.json
        research/reports/01_ma_crossover_report.txt
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import os
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(r"c:\Users\Priyatanshu Ghosh\Documents\Python Practice\CFA Practice")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import numpy as np

from core.db import get_storage
from backtesting.engine import BacktestEngine
from backtesting.strategies import MACrossStrategy, RsiMeanReversionStrategy, MomentumStrategy
from backtesting.cost_model import CostModel

# -- Configuration ----------------------------------------------------------
# Diverse test universe: US equities, Indian, ETFs, crypto, FX
TEST_SYMBOLS = {
    # US Equities (large-cap)
    "AAPL":         "EQUITY_US",
    "MSFT":         "EQUITY_US",
    "NVDA":         "EQUITY_US",
    "AMZN":         "EQUITY_US",
    "GOOGL":        "EQUITY_US",
    "META":         "EQUITY_US",
    "TSLA":         "EQUITY_US",
    "JPM":          "EQUITY_US",
    # Indian Equities
    "RELIANCE.NS":  "EQUITY_IN",
    "HDFCBANK.NS":  "EQUITY_IN",
    "INFY.NS":      "EQUITY_IN",
    "TCS.NS":       "EQUITY_IN",
    # ETFs
    "SPY":          "ETF",
    "QQQ":          "ETF",
    "GLD":          "ETF",
    "TLT":          "ETF",
    # Crypto
    "BTC-USD":      "CRYPTO",
    "ETH-USD":      "CRYPTO",
    # FX / Rates
    "EURUSD=X":     "FOREX",
    "GC=F":         "ETF",     # Gold futures
}

INITIAL_CASH = 1_000_000.0

# -- Strategy Variants ------------------------------------------------------
STRATEGY_CONFIGS = {
    "ma_cross_20_100":       {"cls": MACrossStrategy, "params": {"fast": 20, "slow": 100, "allow_short": False}},
    "ma_cross_20_100_short": {"cls": MACrossStrategy, "params": {"fast": 20, "slow": 100, "allow_short": True}},
    "ma_cross_10_50":        {"cls": MACrossStrategy, "params": {"fast": 10, "slow": 50, "allow_short": False}},
    "ma_cross_50_200":       {"cls": MACrossStrategy, "params": {"fast": 50, "slow": 200, "allow_short": False}},
}


def run_test():
    print("=" * 80)
    print("  PROJECT AGONISTES - RESEARCH TEST #1: MA CROSSOVER STRATEGY")
    print("=" * 80)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Initial capital: ${INITIAL_CASH:,.0f}")
    print(f"  Symbols: {len(TEST_SYMBOLS)}")
    print(f"  Strategy variants: {len(STRATEGY_CONFIGS)}")
    print("=" * 80)
    print()

    # Load data
    db = get_storage()
    cost_model = CostModel()
    engine = BacktestEngine(cost_model=cost_model, initial_cash=INITIAL_CASH)

    all_results = []
    detailed_results = {}

    for strat_name, strat_cfg in STRATEGY_CONFIGS.items():
        print(f"\n{'-' * 60}")
        print(f"  Strategy: {strat_name}")
        print(f"{'-' * 60}")

        strat_results = []

        for symbol, asset_class in TEST_SYMBOLS.items():
            ohlcv = db.query_ohlcv(symbol)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 250:
                print(f"  SKIP {symbol}: insufficient data ({0 if ohlcv is None else len(ohlcv)} bars)")
                continue

            # Ensure index is DatetimeIndex
            if not isinstance(ohlcv.index, pd.DatetimeIndex):
                if "time" in ohlcv.columns:
                    ohlcv = ohlcv.set_index("time")
                ohlcv.index = pd.to_datetime(ohlcv.index)

            # Ensure required columns
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(ohlcv.columns)):
                print(f"  SKIP {symbol}: missing columns {required - set(ohlcv.columns)}")
                continue

            # Run strategy
            try:
                strategy = strat_cfg["cls"](**strat_cfg["params"])
                strategy.fit(ohlcv)
                signals = strategy.generate_signals()

                # Get benchmark (SPY for US, ^NSEI for India, BTC-USD for crypto)
                benchmark = None
                if asset_class in ("EQUITY_US", "ETF"):
                    bm_data = db.query_ohlcv("SPY")
                    if bm_data is not None and not bm_data.empty:
                        if not isinstance(bm_data.index, pd.DatetimeIndex):
                            if "time" in bm_data.columns:
                                bm_data = bm_data.set_index("time")
                            bm_data.index = pd.to_datetime(bm_data.index)
                        benchmark = bm_data["close"]

                result = engine.run(
                    symbol=symbol,
                    asset_class=asset_class,
                    ohlcv=ohlcv,
                    signals=signals,
                    benchmark=benchmark,
                    strategy_name=strat_name,
                )

                report = result.report
                row = {
                    "strategy": strat_name,
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "period": f"{report.period_start} to {report.period_end}",
                    "bars": len(ohlcv),
                    "total_return_pct": round(report.total_return_pct, 2),
                    "cagr_pct": round(report.cagr, 2),
                    "sharpe": round(report.sharpe_ratio, 3),
                    "sortino": round(report.sortino_ratio, 3),
                    "calmar": round(report.calmar_ratio, 3),
                    "max_drawdown_pct": round(report.max_drawdown_pct, 2),
                    "max_dd_duration_days": report.max_drawdown_duration_days,
                    "volatility_ann_pct": round(report.volatility_annualized, 2),
                    "var_95_pct": round(report.daily_var_95, 3),
                    "total_trades": report.total_trades,
                    "win_rate": round(report.win_rate, 3),
                    "avg_win_pct": round(report.avg_win_pct, 2),
                    "avg_loss_pct": round(report.avg_loss_pct, 2),
                    "profit_factor": round(report.profit_factor, 3),
                    "expectancy_usd": round(report.expectancy_per_trade_usd, 2),
                    "avg_hold_hours": round(report.avg_holding_period_hours, 1),
                    "alpha_vs_spy_pct": round(report.alpha_vs_sp500, 2),
                    "info_ratio": round(report.information_ratio, 3),
                    "commission_usd": round(report.total_commission_paid_usd, 2),
                    "cost_drag_pct": round(report.cost_drag_pct, 3),
                }
                strat_results.append(row)
                all_results.append(row)

                # Print summary
                emoji = "+" if report.total_return_pct > 0 else "-"
                print(f"  {emoji} {symbol:16s} | Return: {report.total_return_pct:+8.2f}% | "
                      f"Sharpe: {report.sharpe_ratio:+6.3f} | MaxDD: {report.max_drawdown_pct:7.2f}% | "
                      f"Trades: {report.total_trades:4d} | WR: {report.win_rate:5.1%}")

            except Exception as e:
                print(f"  ERROR {symbol}: {e}")
                import traceback
                traceback.print_exc()

        if strat_results:
            detailed_results[strat_name] = strat_results

    # -- Aggregate & Save -------------------------------------------------
    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "research" / "results"
    reports_dir = PROJECT_ROOT / "research" / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Build summary table
    if all_results:
        df = pd.DataFrame(all_results)

        # Per-strategy aggregate
        print("\n  Per-Strategy Summary:")
        print(f"  {'Strategy':<25s} {'Avg Return':>10s} {'Avg Sharpe':>10s} {'Avg MaxDD':>10s} "
              f"{'Win Rate':>10s} {'Symbols':>8s}")
        print("  " + "-" * 75)
        for strat_name in STRATEGY_CONFIGS:
            sdf = df[df["strategy"] == strat_name]
            if sdf.empty:
                continue
            print(f"  {strat_name:<25s} "
                  f"{sdf['total_return_pct'].mean():+9.2f}% "
                  f"{sdf['sharpe'].mean():+9.3f} "
                  f"{sdf['max_drawdown_pct'].mean():9.2f}% "
                  f"{sdf['win_rate'].mean():9.1%} "
                  f"{len(sdf):8d}")

        # Best/worst
        print(f"\n  Best performer:  {df.loc[df['sharpe'].idxmax(), 'symbol']} "
              f"({df.loc[df['sharpe'].idxmax(), 'strategy']}) "
              f"Sharpe={df['sharpe'].max():.3f}, "
              f"Return={df.loc[df['sharpe'].idxmax(), 'total_return_pct']:.2f}%")
        print(f"  Worst performer: {df.loc[df['sharpe'].idxmin(), 'symbol']} "
              f"({df.loc[df['sharpe'].idxmin(), 'strategy']}) "
              f"Sharpe={df['sharpe'].min():.3f}, "
              f"Return={df.loc[df['sharpe'].idxmin(), 'total_return_pct']:.2f}%")

        # Per-asset-class
        print("\n  Per-Asset-Class Summary:")
        print(f"  {'Asset Class':<15s} {'Avg Return':>10s} {'Avg Sharpe':>10s} {'Avg MaxDD':>10s}")
        print("  " + "-" * 50)
        for ac in df["asset_class"].unique():
            adf = df[df["asset_class"] == ac]
            print(f"  {ac:<15s} {adf['total_return_pct'].mean():+9.2f}% "
                  f"{adf['sharpe'].mean():+9.3f} "
                  f"{adf['max_drawdown_pct'].mean():9.2f}%")

        # Save JSON
        output = {
            "test_name": "01_ma_crossover",
            "run_time": datetime.now().isoformat(),
            "initial_cash": INITIAL_CASH,
            "n_symbols": len(TEST_SYMBOLS),
            "n_strategies": len(STRATEGY_CONFIGS),
            "total_backtests": len(all_results),
            "results": all_results,
            "per_strategy_avg": {
                strat: {
                    "avg_return_pct": round(df[df["strategy"] == strat]["total_return_pct"].mean(), 2),
                    "avg_sharpe": round(df[df["strategy"] == strat]["sharpe"].mean(), 3),
                    "avg_max_dd_pct": round(df[df["strategy"] == strat]["max_drawdown_pct"].mean(), 2),
                    "avg_win_rate": round(df[df["strategy"] == strat]["win_rate"].mean(), 3),
                    "n_tested": len(df[df["strategy"] == strat]),
                }
                for strat in STRATEGY_CONFIGS if len(df[df["strategy"] == strat]) > 0
            },
        }

        json_path = results_dir / "01_ma_crossover_results.json"
        json_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"\n  Results saved to: {json_path}")

        # Save CSV
        csv_path = results_dir / "01_ma_crossover_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved to: {csv_path}")

    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
