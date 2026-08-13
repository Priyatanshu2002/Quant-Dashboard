"""Tests for the FRED + BLS series maps: transforms, key-gating, and the
expanded series that "light up" once an API key is configured."""
from __future__ import annotations

from unittest import mock

import pytest

from core.db import SQLiteStorage
from data_ingestion.macro_feeds.bls_fetcher import SERIES as BLS_SERIES
from data_ingestion.macro_feeds.fred_fetcher import (
    SERIES, _bps_to_decimal, _pct, fetch_fred_latest)


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


# ── transforms ───────────────────────────────────────────────────────────
def test_pct_transform():
    assert _pct("4.68") == pytest.approx(0.0468)
    assert _pct("0.25") == pytest.approx(0.0025)


def test_bps_to_decimal_transform():
    assert _bps_to_decimal("256") == pytest.approx(0.0256)
    assert _bps_to_decimal("117.9") == pytest.approx(0.01179)


# ── expanded series maps ─────────────────────────────────────────────────
def test_fred_series_map_has_key_categories():
    fields = {v[0] for v in SERIES.values()}
    # inflation
    assert {"cpi_all_urban", "core_cpi", "ppi_final", "pce_all"} <= fields
    # labour / activity
    assert {"unemployment_rate", "nonfarm_payrolls", "ism_pmi"} <= fields
    # full curve + real + breakeven
    assert {"us_1y_yield", "us_30y_yield", "us_10y_real_yield",
            "t10y_breakeven_ie"} <= fields
    # credit spreads + money + commodity
    assert {"hy_credit_spread", "ig_credit_spread", "m2_supply",
            "wti_price", "brent_price"} <= fields
    assert len(SERIES) >= 30


def test_fred_yield_series_use_pct_transform():
    assert SERIES["DGS10"] == ("us_10y_yield", _pct)
    assert SERIES["DFII10"] == ("us_10y_real_yield", _pct)
    assert SERIES["DFF"] == ("fed_funds_rate", _pct)


def test_bls_series_map_expanded():
    assert BLS_SERIES["CUUR0000SA0"] == "cpi_all_urban"
    assert "ppi_final" in BLS_SERIES.values()
    assert "avg_hourly_earnings" in BLS_SERIES.values()
    assert len(BLS_SERIES) >= 6


# ── key-gated behaviour ──────────────────────────────────────────────────
def test_fred_no_key_returns_none(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.macro_feeds.fred_fetcher.FRED_API_KEY", "")
    assert fetch_fred_latest(db) is None


def test_fred_with_key_fetches_and_persists(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.macro_feeds.fred_fetcher.FRED_API_KEY", "k")
    calls = {"n": 0}

    def _fake_get(url, params, timeout=30):
        calls["n"] += 1
        resp = mock.MagicMock()
        resp.raise_for_status = mock.MagicMock()
        sid = params["series_id"]
        # realistic values per series
        valmap = {"VIXCLS": "15.3", "DGS10": "4.68", "DGS2": "4.10",
                  "DFF": "4.33", "CPIAUCSL": "324.5", "UNRATE": "3.9",
                  "DCOILWTICO": "78.5", "BAMLH0A0HYM2": "356"}
        val = valmap.get(sid, "100")
        resp.json.return_value = {"observations": [{"value": val}]}
        return resp

    with mock.patch("data_ingestion.macro_feeds.fred_fetcher.requests.get",
                    side_effect=_fake_get):
        macro = fetch_fred_latest(db)
    assert macro is not None
    assert calls["n"] == len(SERIES)
    assert macro["vix"] == pytest.approx(15.3)
    assert macro["us_10y_yield"] == pytest.approx(0.0468)   # /100
    assert macro["fed_funds_rate"] == pytest.approx(0.0433)  # /100
    assert macro["hy_credit_spread"] == pytest.approx(0.0356)  # bps→decimal
    assert macro["cpi_all_urban"] == pytest.approx(324.5)      # index unchanged
    # persisted to storage
    latest = db.query_latest_macro()
    assert latest["us_10y_yield"] == pytest.approx(0.0468)
