"""Tests for the equity price feed: asset-class mapping, corporate actions,
dollar-volume persistence, and the Storage.write_corporate_actions round-trip.
"""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from core.db import SQLiteStorage
from data_ingestion.price_feeds.equity_ws import (
    asset_class_for, backfill_equities, fetch_corporate_actions)


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


# ── asset-class mapping (C3 Task 1) ──────────────────────────────────────
@pytest.mark.parametrize("symbol,expected", [
    ("SPY", "ETF"), ("QQQ", "ETF"), ("VTI", "ETF"),
    ("^GSPC", "INDEX"), ("^NSEI", "INDEX"), ("^VIX", "INDEX"),
    ("AAPL", "EQUITY_US"), ("MSFT", "EQUITY_US"),
    ("RELIANCE.NS", "EQUITY_IN"), ("TCS.NS", "EQUITY_IN"),
    ("RELIANCE.BO", "EQUITY_IN"),
    (" spy ", "ETF"),  # case-insensitive + whitespace
])
def test_asset_class_for(symbol, expected):
    assert asset_class_for(symbol) == expected


# ── corporate-actions fetch + persist ────────────────────────────────────
def _fake_ticker():
    t = mock.MagicMock()
    div_idx = pd.to_datetime(["2026-05-01", "2026-02-01"])
    t.dividends = pd.Series([0.25, 0.25], index=div_idx)
    t.splits = pd.Series([4.0], index=pd.to_datetime(["2024-08-28"]))
    t.actions = pd.DataFrame({
        "Dividends": [0.25], "Splits": [0.0],
    }, index=pd.to_datetime(["2026-05-01"]))
    return t


def test_fetch_corporate_actions_flat():
    with mock.patch("data_ingestion.price_feeds.equity_ws.yf.Ticker",
                    return_value=_fake_ticker()):
        acts = fetch_corporate_actions("AAPL")
    types = {a["action_type"] for a in acts}
    assert "DIVIDEND" in types and "SPLIT" in types
    divs = [a for a in acts if a["action_type"] == "DIVIDEND"]
    # dividends from .dividends (+ .actions non-zero), splits from .splits
    assert any(a["amount"] == 0.25 for a in divs)
    assert any(a["action_type"] == "SPLIT" and a["amount"] == 4.0 for a in acts)


def test_write_and_query_corporate_actions(db):
    db.write_corporate_actions("AAPL", [
        {"action_date": "2026-05-01", "action_type": "DIVIDEND", "amount": 0.25},
        {"action_date": "2024-08-28", "action_type": "SPLIT", "amount": 4.0},
    ])
    rows = db.query_corporate_actions("AAPL")
    assert len(rows) == 2
    assert rows[0]["action_type"] == "SPLIT"  # date-ordered
    assert rows[1]["action_type"] == "DIVIDEND"
    # idempotent upsert — same key overwrites, no duplicates
    db.write_corporate_actions("AAPL", [
        {"action_date": "2026-05-01", "action_type": "DIVIDEND", "amount": 0.30},
    ])
    rows = db.query_corporate_actions("AAPL")
    assert len(rows) == 2
    assert any(r["action_type"] == "DIVIDEND" and r["amount"] == 0.30 for r in rows)


# ── backfill: dollar volume persisted + asset-class on each write ────────
def _fake_download(tickers, **kwargs):
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    cols = pd.MultiIndex.from_product([list(tickers), ["Open", "High", "Low",
                                                       "Close", "Volume"]])
    df = pd.DataFrame(100.0, index=idx, columns=cols)
    for t in tickers:
        df[(t, "Close")] = [100.0, 102.0, 105.0]
        df[(t, "Volume")] = [1000, 1000, 1000]
    return df if len(tickers) > 1 else df[tickers[0]]


def test_backfill_equities_persists_dollar_volume_and_actions(db):
    with mock.patch("data_ingestion.price_feeds.equity_ws.yf.download",
                    side_effect=_fake_download), \
         mock.patch("data_ingestion.price_feeds.equity_ws.fetch_corporate_actions",
                    return_value=[{"action_date": "2026-05-01",
                                   "action_type": "DIVIDEND", "amount": 0.25}]):
        results = backfill_equities(["AAPL"], period="2y", storage=db)
    df = results["AAPL"]
    assert "dollar_volume" in df.columns
    assert df["dollar_volume"].iloc[-1] == pytest.approx(105.0 * 1000)
    # persisted through write_ohlcv
    stored = db.query_ohlcv("AAPL")
    assert "dollar_volume" in stored.columns
    assert stored["dollar_volume"].iloc[-1] == pytest.approx(105.0 * 1000)
    # corporate actions persisted
    acts = db.query_corporate_actions("AAPL")
    assert acts and acts[0]["action_type"] == "DIVIDEND"
