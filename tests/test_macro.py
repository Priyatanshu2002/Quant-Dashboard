"""Tests for the macro layer (yfinance-based, keyless)."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from core.db import SQLiteStorage
from data_ingestion.macro_feeds.yfinance_macro import fetch_yfinance_macro
from data_ingestion.macro_feeds.treasury_fetcher import fetch_vix_stooq


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


def _fake_yf_download(tickers, **kwargs):
    """Yahoo-style MultiIndex-column DataFrame: (ticker, OHLCV) columns."""
    idx = pd.date_range("2026-08-01", periods=5, freq="B")
    values = {
        "^VIX": [16.5, 16.0, 15.8, 15.4, 15.29],
        "^TNX": [4.60, 4.65, 4.70, 4.72, 4.68],
        "^2YY": [4.10, 4.12, 4.15, 4.13, 4.10],
        "DX-Y.NYB": [100.5, 100.2, 100.1, 99.9, 99.86],
        "GC=F": [2400, 2420, 2450, 2490, 2525.5],
        "BTC-USD": [60000, 61000, 60500, 61500, 62000],
    }
    cols = pd.MultiIndex.from_product([list(tickers), ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(np.zeros((5, len(cols) * 5)) if False else
                      np.random.default_rng(0).normal(100, 5, (5, len(cols))),
                      index=idx, columns=cols)
    for t in tickers:
        for i, v in enumerate(values.get(t, [100] * 5)):
            df.loc[df.index[i], (t, "Close")] = v
            df.loc[df.index[i], (t, "Open")] = v * 0.99
            df.loc[df.index[i], (t, "High")] = v * 1.01
            df.loc[df.index[i], (t, "Low")] = v * 0.98
            df.loc[df.index[i], (t, "Volume")] = 1e6
    return df


def test_yfinance_macro_snapshot(db):
    with mock.patch("data_ingestion.macro_feeds.yfinance_macro.yf.download",
                    side_effect=_fake_yf_download) as m:
        macro = fetch_yfinance_macro(db)
    m.assert_called_once()
    assert macro["vix"] == pytest.approx(15.29, abs=1e-6)
    assert macro["us_10y_yield"] == pytest.approx(0.0468, abs=1e-6)
    assert macro["us_2y_yield"] == pytest.approx(0.0410, abs=1e-6)
    assert macro["yield_curve_spread"] == pytest.approx(0.0058, abs=1e-6)
    assert macro["dxy"] == pytest.approx(99.86, abs=1e-6)
    # gold 5d change: (2525.5 / 2400 - 1) * 100
    assert macro["gold_pct_change_5d"] == pytest.approx((2525.5 / 2400 - 1) * 100, rel=1e-6)

    # persisted → feature layer can read it
    latest = db.query_latest_macro()
    assert latest["vix"] == pytest.approx(15.29, abs=1e-6)
    assert latest["yield_curve_spread"] is not None


def test_yfinance_macro_missing_2y_no_spread(db):
    def _no_2y(tickers, **kwargs):
        df = _fake_yf_download(tickers, **kwargs)
        df = df.drop(columns=[("^2YY", c) for c in ("Open", "High", "Low", "Close", "Volume")])
        return df

    with mock.patch("data_ingestion.macro_feeds.yfinance_macro.yf.download",
                    side_effect=_no_2y):
        macro = fetch_yfinance_macro(db)
    assert macro["vix"] is not None
    assert "yield_curve_spread" not in macro  # degrades gracefully


def test_yfinance_macro_failure_returns_none(db):
    with mock.patch("data_ingestion.macro_feeds.yfinance_macro.yf.download",
                    side_effect=RuntimeError("network down")):
        assert fetch_yfinance_macro(db) is None


def test_stooq_vix_parse():
    csv = "Date,Close\n2026-08-08,16.1\n2026-08-11,15.6\n2026-08-12,15.29\n"
    resp = mock.MagicMock()
    resp.text = csv
    resp.raise_for_status = mock.MagicMock()
    with mock.patch("data_ingestion.macro_feeds.treasury_fetcher.requests.get",
                    return_value=resp):
        assert fetch_vix_stooq() == pytest.approx(15.29)
