"""Load the desktop's 1-min Databento futures into a daily continuous panel.

Path: Desktop/Research Notebooks and Data/Downloaded Data from Databento/
Symbols: ES, NQ, ZN, CL, DX, 6E, GC (continuous futures, already back-adjusted
  single contracts via the '.v.0' continuous symbol).

Resamples 1-min -> daily (last close; OHLC + volume aggregated), builds a wide
close-price DataFrame (columns = symbols) and returns it. This is the data
Momentum Transformer (arXiv:2112.08534) uses: daily continuous futures.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_DIR = Path(r"C:\Users\Priyatanshu Ghosh\Desktop\Research Notebooks and Data"
                   r"\Downloaded Data from Databento")
SYMBOLS = ["es", "nq", "zn", "cl", "dx", "6e", "gc"]


def load_daily_futures(data_dir: Path | str = DEFAULT_DIR,
                       symbols: list[str] | None = None) -> pd.DataFrame:
    """Return wide daily close-price DataFrame (columns = upper symbols)."""
    symbols = symbols or SYMBOLS
    closes: dict[str, pd.Series] = {}
    for s in symbols:
        f = Path(data_dir) / f"{s}_1min_ohlcv.csv"
        if not f.exists():
            print(f"  ! missing {f.name}")
            continue
        df = pd.read_csv(f, usecols=["ts_event", "close"])
        df["ts_event"] = pd.to_datetime(df["ts_event"])
        closes[s.upper()] = (df.set_index("ts_event")["close"]
                             .resample("1D").last().dropna())
    prices = pd.DataFrame(closes).sort_index()
    return prices


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    p = load_daily_futures()
    print(p.shape)
    print(p.index.min().date(), "->", p.index.max().date())
    print(p.tail(3))
