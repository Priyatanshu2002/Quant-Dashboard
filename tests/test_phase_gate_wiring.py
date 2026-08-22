"""Tests for the Phase-gate wiring: gating outcomes/drift, on-chain storage,
Prometheus /metrics, and the RL environment's real-data loader."""
from __future__ import annotations

import pytest

from core.db import SQLiteStorage


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


# ── gating outcome + drift support ─────────────────────────────────────
def test_query_recent_gating_returns_logged_rows(db):
    db.write_gating_log({"time": "2026-01-01 00:00:00", "cycle_id": "c1",
                         "symbol": "AAPL", "dominant_side": "BULL",
                         "decision": "TRADE"})
    db.write_gating_log({"time": "2026-01-02 00:00:00", "cycle_id": "c2",
                         "symbol": "MSFT", "dominant_side": "BEAR",
                         "decision": "TRADE"})
    rows = db.query_recent_gating(n=10)
    assert len(rows) == 2
    assert rows[0]["cycle_id"] == "c2"  # newest first


def test_query_gating_outcomes_hit_and_miss(db):
    # Closed trade with positive PnL on a BULL call → HIT
    db.write_gating_log({"time": "2026-01-01 00:00:00", "cycle_id": "c1",
                         "symbol": "AAPL", "dominant_side": "BULL",
                         "decision": "TRADE"})
    db.write_trade({"time": "2026-01-01 00:00:00", "trade_id": "t1", "cycle_id": "c1",
                    "symbol": "AAPL", "entry_time": "2026-01-01",
                    "exit_time": "2026-01-10", "pnl_pct": 0.05})
    # Closed trade with negative PnL on a BEAR call → HIT (short made money)
    db.write_gating_log({"time": "2026-01-02 00:00:00", "cycle_id": "c2",
                         "symbol": "MSFT", "dominant_side": "BEAR",
                         "decision": "TRADE"})
    db.write_trade({"time": "2026-01-02 00:00:00", "trade_id": "t2", "cycle_id": "c2",
                    "symbol": "MSFT", "entry_time": "2026-01-02",
                    "exit_time": "2026-01-09", "pnl_pct": -0.03})
    # Open trade (no exit) must be excluded
    db.write_gating_log({"time": "2026-01-03 00:00:00", "cycle_id": "c3",
                         "symbol": "NVDA", "dominant_side": "BULL",
                         "decision": "TRADE"})
    db.write_trade({"time": "2026-01-03 00:00:00", "trade_id": "t3", "cycle_id": "c3",
                    "symbol": "NVDA", "entry_time": "2026-01-03",
                    "exit_time": "", "pnl_pct": None})

    outcomes = db.query_gating_outcomes(n=10)
    assert len(outcomes) == 2
    by_cycle = {o["cycle_id"]: o for o in outcomes}
    assert by_cycle["c1"]["outcome"] == "HIT"
    assert by_cycle["c2"]["outcome"] == "HIT"


def test_gating_outcome_miss(db):
    db.write_gating_log({"time": "2026-01-01 00:00:00", "cycle_id": "c1",
                         "symbol": "AAPL", "dominant_side": "BULL",
                         "decision": "TRADE"})
    db.write_trade({"time": "2026-01-01 00:00:00", "trade_id": "t1", "cycle_id": "c1",
                    "symbol": "AAPL", "exit_time": "2026-01-10", "pnl_pct": -0.02})
    outcomes = db.query_gating_outcomes(n=10)
    assert outcomes[0]["outcome"] == "MISS"


# ── on-chain snapshot storage ──────────────────────────────────────────
def test_write_and_query_onchain_snapshot(db):
    db.write_onchain_snapshot("dex_volume_by_pair_7d",
                              [{"token_pair": "BTC-USDC", "volume_usd": 1e9}])
    snaps = db.query_onchain_snapshot("dex_volume_by_pair_7d")
    assert len(snaps) == 1
    assert snaps[0]["rows"][0]["token_pair"] == "BTC-USDC"
    assert "ONCHAIN_DEX_VOLUME_BY_PAIR_7D" in db.symbols()


# ── Prometheus metrics render ──────────────────────────────────────────
def test_route_metrics_renders_prometheus_text(db):
    from core.api_server import _route_metrics
    # prime a couple of counters
    from core.api_server import record_screener, record_event
    record_screener(100, 3)
    record_event("macro_snapshot", 5)
    out = _route_metrics(db)
    assert "agonistes_screener_scored_total 100" in out
    assert "agonistes_http_requests_total" in out
    assert "agonistes_gating_total" in out
    assert "agonistes_ohlcv_bars_total" in out


# ── RL environment real-data loader ────────────────────────────────────
def test_load_real_frame_with_store(tmp_path, monkeypatch):
    import pandas as pd
    db = SQLiteStorage(tmp_path / "rl.db")
    # Seed a small feature vector row with a label.
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=10, freq="D"),
        "symbol": "AAPL", "asset_class": "EQUITY_US", "timeframe": "SWING",
        "rsi_14": list(range(10)),
        "macd_histogram": [0.0] * 10,
        "future_return_1d": [0.01] * 10,
    }).set_index("time")
    db.write_feature_vectors(df, "AAPL", "EQUITY_US", "SWING")

    # Route the RL loader's internal store lookup to the temp DB.
    monkeypatch.setattr("feature_engineering.feature_store.get_storage",
                        lambda: db)

    from rl_agent.environment import load_real_frame
    X, R = load_real_frame(symbols=["AAPL"], timeframe="SWING")
    assert X is not None
    assert X.ndim == 2 and X.shape[0] == 10
    assert R.shape == (10,)
    # returns use the 1d label directly (≈0.01)
    assert abs(float(R[0]) - 0.01) < 1e-6
