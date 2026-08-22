"""Build a slim SQLite DB containing only market_data OHLCV for the research
lab universe, so it can be shipped to the RunPod pod (the full 758MB DB is too
large for the PTY-only SSH transfer, and the notebook only needs OHLCV).

Usage: .venv/Scripts/python scripts/build_slim_universe_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "agonistes_dev.db"
OUT = ROOT / "data" / "agonistes_slim.db"

UNIVERSE = [
    "AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "UNH", "XOM",
    "HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS",
    "SPY", "QQQ", "IWM", "EEM", "GLD", "TLT", "^NSEI",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "DX-Y.NYB", "^TNX", "GC=F",
]


def main() -> None:
    if not SRC.exists():
        print(f"source DB not found: {SRC}")
        return
    if OUT.exists():
        OUT.unlink()
    src = sqlite3.connect(str(SRC))
    dst = sqlite3.connect(str(OUT))
    cur = dst.cursor()
    cur.execute("CREATE TABLE market_data ("
                "time TEXT, symbol TEXT, asset_class TEXT, source TEXT, "
                "interval TEXT, open REAL, high REAL, low REAL, close REAL, "
                "volume REAL, raw TEXT, dollar_volume REAL)")
    ph = ",".join("?" * len(UNIVERSE))
    rows = src.execute(
        "SELECT time, symbol, asset_class, source, interval, open, high, low, "
        "close, volume, raw, dollar_volume FROM market_data "
        f"WHERE symbol IN ({ph})", UNIVERSE).fetchall()
    cur.executemany("INSERT INTO market_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    dst.commit()
    n = cur.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    syms = cur.execute("SELECT COUNT(DISTINCT symbol) FROM market_data").fetchone()[0]
    print(f"slim DB: {n} rows, {syms} symbols -> {OUT} "
          f"({OUT.stat().st_size/1e6:.1f} MB)")
    src.close(); dst.close()


if __name__ == "__main__":
    sys.exit(main())
