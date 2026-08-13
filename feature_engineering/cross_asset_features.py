"""Cross-asset features — BTC dominance trend, sector rotation, correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.db import Storage, get_storage
from core.logging import get_logger

log = get_logger(__name__)


def compute_cross_asset_features(symbol: str, ohlcv: pd.DataFrame,
                                 db: Storage | None = None) -> dict:
    """Relative strength vs SPY, BTC correlation, dominance trend."""
    db = db or get_storage()
    f: dict = {}

    # BTC dominance trend (macro snapshot series)
    try:
        macro = db.query_latest_macro()
        if macro and macro.get("btc_dominance") is not None:
            f["btc_dominance_trend"] = macro["btc_dominance"]
    except Exception:  # noqa: BLE001
        pass

    # Correlation to BTC and SPY (rolling 60-bar) when those series exist
    for bench, label in (("BTC-USD", "btc"), ("SPY", "spy")):
        try:
            b = db.query_ohlcv(bench, start=ohlcv.index[0], end=ohlcv.index[-1])
            if len(b) > 30:
                both = pd.concat([ohlcv["close"].rename(symbol),
                                  b["close"].rename(label)], axis=1).dropna()
                f[f"corr_{label}_60"] = both[symbol].pct_change().rolling(60).corr(
                    both[label].pct_change()).iloc[-1]
        except Exception:  # noqa: BLE001
            continue

    # Sector rotation: relative strength vs SPY over 20 bars
    try:
        spy = db.query_ohlcv("SPY", start=ohlcv.index[0], end=ohlcv.index[-1])
        if len(spy) > 20 and len(ohlcv) > 20:
            rel = (ohlcv["close"] / ohlcv["close"].shift(20)) / \
                  (spy["close"] / spy["close"].shift(20))
            f["rel_strength_vs_spy_20"] = (rel.iloc[-1] - 1) * 100
    except Exception:  # noqa: BLE001
        pass

    return {k: (None if pd.isna(v) else float(v)) for k, v in f.items()}
