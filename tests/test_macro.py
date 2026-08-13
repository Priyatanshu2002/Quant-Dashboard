"""Tests for the macro layer (yfinance-based, keyless)."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from core.db import SQLiteStorage
from data_ingestion.macro_feeds.yfinance_macro import fetch_yfinance_macro
from data_ingestion.macro_feeds.treasury_fetcher import (
    fetch_treasury_curve, fetch_vix_stooq)


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


# ── Full nominal curve + TIPS real yields + breakeven ────────────────────
_NOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
<entry><content type="application/xml"><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-12T00:00:00</d:NEW_DATE>
<d:BC_1MONTH>3.72</d:BC_1MONTH><d:BC_2MONTH>3.66</d:BC_2MONTH>
<d:BC_3MONTH>3.65</d:BC_3MONTH><d:BC_6MONTH>3.58</d:BC_6MONTH>
<d:BC_1YEAR>3.47</d:BC_1YEAR><d:BC_2YEAR>3.47</d:BC_2YEAR>
<d:BC_3YEAR>3.55</d:BC_3YEAR><d:BC_5YEAR>3.74</d:BC_5YEAR>
<d:BC_7YEAR>3.95</d:BC_7YEAR><d:BC_10YEAR>4.19</d:BC_10YEAR>
<d:BC_20YEAR>4.81</d:BC_20YEAR><d:BC_30YEAR>4.86</d:BC_30YEAR>
</m:properties></content></entry>
</feed>"""

_REAL_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
<entry><content type="application/xml"><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-12T00:00:00</d:NEW_DATE>
<d:TC_5YEAR>1.46</d:TC_5YEAR><d:TC_7YEAR>1.69</d:TC_7YEAR>
<d:TC_10YEAR>1.94</d:TC_10YEAR><d:TC_20YEAR>2.39</d:TC_20YEAR>
<d:TC_30YEAR>2.63</d:TC_30YEAR>
</m:properties></content></entry>
</feed>"""


def _fake_treasury_get(url, timeout=30):
    resp = mock.MagicMock()
    resp.raise_for_status = mock.MagicMock()
    if "daily_treasury_real_yield_curve" in url:
        resp.content = _REAL_XML.encode()
    else:
        resp.content = _NOM_XML.encode()
    return resp


def test_treasury_full_curve_and_breakeven(db):
    with mock.patch("data_ingestion.macro_feeds.treasury_fetcher.requests.get",
                    side_effect=_fake_treasury_get):
        macro = fetch_treasury_curve(db)
    assert macro is not None
    # full curve present (≥10 maturities)
    nom = [k for k in ("us_1m_yield", "us_2m_yield", "us_3m_yield", "us_6m_yield",
                       "us_1y_yield", "us_2y_yield", "us_3y_yield", "us_5y_yield",
                       "us_7y_yield", "us_10y_yield", "us_20y_yield", "us_30y_yield")]
    assert sum(1 for k in nom if k in macro) >= 10
    assert macro["us_10y_yield"] == pytest.approx(0.0419, abs=1e-6)
    assert macro["us_30y_yield"] == pytest.approx(0.0486, abs=1e-6)
    # real yields + breakeven
    assert macro["us_10y_real_yield"] == pytest.approx(0.0194, abs=1e-6)
    assert macro["breakeven_inflation"] == pytest.approx(0.0419 - 0.0194, abs=1e-6)
    assert macro["yield_curve_spread"] == pytest.approx(0.0419 - 0.0347, abs=1e-6)
    # persisted → feature layer can read the new fields
    latest = db.query_latest_macro()
    assert latest["us_30y_yield"] == pytest.approx(0.0486, abs=1e-6)
    assert latest["breakeven_inflation"] == pytest.approx(0.0419 - 0.0194, abs=1e-6)


def test_treasury_graceful_when_real_feed_missing(db):
    def _nom_only(url, timeout=30):
        resp = mock.MagicMock()
        resp.raise_for_status = mock.MagicMock()
        resp.content = _NOM_XML.encode()
        return resp

    with mock.patch("data_ingestion.macro_feeds.treasury_fetcher.requests.get",
                    side_effect=_nom_only):
        macro = fetch_treasury_curve(db)
    assert macro is not None
    assert "us_10y_yield" in macro
    assert "breakeven_inflation" not in macro  # degrades gracefully


def test_treasury_none_on_total_failure(db):
    def _boom(url, timeout=30):
        raise RuntimeError("network down")

    with mock.patch("data_ingestion.macro_feeds.treasury_fetcher.requests.get",
                    side_effect=_boom):
        assert fetch_treasury_curve(db) is None
