"""Tests for the LLM analyst (plan §8.1): JSON parsing, fallback verdicts,
cache freshness, and the API route handlers for financials + sentiment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.api_server import _route_financials, _route_sentiment
from core.db import SQLiteStorage
from data_ingestion.sentiment_feeds.llm_analyst import (
    _label_for, _parse_json, analyze_fundamentals, analyze_news_sentiment)


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


# ── JSON tolerance ──────────────────────────────────────────────────────
def test_parse_json_plain():
    assert _parse_json('{"score": 0.5}') == {"score": 0.5}


def test_parse_json_fenced():
    raw = '```json\n{"score": -0.25, "label": "bearish"}\n```'
    assert _parse_json(raw) == {"score": -0.25, "label": "bearish"}


def test_parse_json_with_prefix_text():
    raw = 'Here is the analysis:\n{"a": 1, "b": [1, 2]}\nHope this helps.'
    assert _parse_json(raw) == {"a": 1, "b": [1, 2]}


def test_parse_json_garbage():
    assert _parse_json("no json here at all") is None
    assert _parse_json('{"unclosed": [1, 2') is None


def test_label_for():
    assert _label_for(0.5) == "bullish"
    assert _label_for(-0.5) == "bearish"
    assert _label_for(0.1) == "neutral"


# ── news verdict fallback + caching ─────────────────────────────────────
def test_news_fallback_empty(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: False)
    verdict = analyze_news_sentiment("NOPE", storage=db, db=db, force=True)
    assert verdict["model"] == "fallback-lexicon"
    assert verdict["label"] == "neutral"
    assert verdict["analyzed_events"] == 0


def test_news_verdict_aggregates_events(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: False)
    db.write_sentiment_event("X", "NEWS", 0.6, headline="strong beat", source_weight=0.9)
    db.write_sentiment_event("X", "NEWS", -0.6, headline="guidance cut", source_weight=0.9)
    verdict = analyze_news_sentiment("X", storage=db, db=db, force=True)
    assert verdict["score"] == pytest.approx(0.0, abs=1e-6)
    assert verdict["analyzed_events"] == 2
    assert "positive" in verdict["summary"]


def test_news_verdict_cached_within_freshness(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: False)
    db.upsert_llm_analysis("X", "NEWS", {"score": 0.9, "label": "bullish",
                                         "summary": "cached", "key_points": [],
                                         "risks": [], "analyzed_events": 0}, "m")
    # force=False → cache hit, no new events needed
    verdict = analyze_news_sentiment("X", storage=db, db=db)
    assert verdict["cached"] is True
    assert verdict["score"] == 0.9
    # force=True → recompute from (empty) events
    verdict2 = analyze_news_sentiment("X", storage=db, db=db, force=True)
    assert verdict2["cached"] is False


def test_news_verdict_llm_path(db, monkeypatch):
    """LLM path: raw JSON from the model is parsed and stored."""
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: True)
    async def fake_llm(system, user, temperature=0.3, max_tokens=2048):
        return ('{"score": 0.8, "label": "bullish", "summary": "Great quarter", '
                '"key_points": ["beat"], "risks": ["valuation"]}')
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.call_openrouter_raw",
                        fake_llm)
    db.write_sentiment_event("X", "NEWS", 0.5, headline="something", source_weight=0.9)
    verdict = analyze_news_sentiment("X", storage=db, db=db, force=True)
    assert verdict["model"] == "deepseek/deepseek-v4-flash-0731"
    assert verdict["label"] == "bullish"
    assert verdict["score"] == 0.8
    stored = db.query_latest_llm_analysis("X", kind="NEWS")
    assert stored["verdict"]["summary"] == "Great quarter"


def test_news_verdict_llm_failure_falls_back(db, monkeypatch):
    """503/timeout from the provider → lexicon fallback, still persisted."""
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: True)
    async def boom(system, user, temperature=0.3):
        raise RuntimeError("upstream capacity limits")
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.call_openrouter_raw",
                        boom)
    db.write_sentiment_event("X", "NEWS", 0.6, headline="great news", source_weight=0.9)
    verdict = analyze_news_sentiment("X", storage=db, db=db, force=True)
    assert verdict["model"] == "fallback-lexicon"
    assert verdict["score"] > 0
    assert db.query_latest_llm_analysis("X", kind="NEWS") is not None


# ── fundamental verdict ─────────────────────────────────────────────────
def test_fundamental_fallback_no_data(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: False)
    verdict = analyze_fundamentals("NOPE", storage=db, db=db, force=True)
    assert verdict["rating"] == "HOLD"
    assert verdict["model"] == "fallback-rule-based"


def test_fundamental_fallback_uses_dcf_margin(db, monkeypatch):
    monkeypatch.setattr("data_ingestion.sentiment_feeds.llm_analyst.llm_available",
                        lambda: False)
    db.upsert_fundamental_snapshot({
        "symbol": "X", "asset_class": "EQUITY_US",
        "free_cash_flow": 1000, "revenue_yoy_growth": 0.1,
        "market_cap": 50000, "current_price": 100, "net_debt": 0,
        "forward_pe": 12.0, "roic": 0.18, "dcf_intrinsic_value": 130.0,
        "dcf_margin_of_safety": 0.3,
    })
    verdict = analyze_fundamentals("X", storage=db, db=db, force=True)
    assert verdict["score"] > 0
    assert verdict["fair_value_est"] == 130.0
    assert "margin of safety" in verdict["thesis"].lower()


# ── API route handlers ──────────────────────────────────────────────────
def test_route_financials_bundle(db):
    db.upsert_company_profile("AAPL", {"company_name": "Apple Inc."})
    db.write_financial_statements("AAPL", "income", [
        {"period": "2026-06-30", "total_revenue": 1e6, "net_income": 2e5},
        {"period": "2026-03-31", "total_revenue": 9e5, "net_income": 1e5},
    ])
    db.write_financial_statements("AAPL", "balance", [
        {"period": "2026-06-30", "total_assets": 5e6},
    ])
    db.upsert_fundamental_snapshot({
        "symbol": "AAPL", "asset_class": "EQUITY_US",
        "free_cash_flow": 1000, "revenue_yoy_growth": 0.1,
        "market_cap": 50000, "current_price": 100, "net_debt": 0,
    })
    out = _route_financials(db, {"symbol": ["AAPL"]})
    assert out["profile"]["company_name"] == "Apple Inc."
    assert out["dcf"]["intrinsic_value_per_share"] is not None
    assert [r["period"] for r in out["statements"]["income"]] == \
        ["2026-03-31", "2026-06-30"]  # chronological
    assert out["statements"]["balance"][0]["total_assets"] == 5e6
    assert out["earnings"]["results"] == []
    assert set(out["llm_analyses"]) == {"news", "fundamental"}
    assert out["price_change_pct"] is None  # no market data in tmp db


def test_route_sentiment_aggregate(db):
    db.write_sentiment_event("AAPL", "NEWS", 0.5, headline="h1", source_weight=0.9)
    db.write_sentiment_event("AAPL", "NEWS", -0.5, headline="h2", source_weight=0.9)
    db.write_sentiment_event("AAPL", "STOCKTWITS", 0.6, headline="h3", source_weight=0.7)
    out = _route_sentiment(db, {"symbol": ["AAPL"], "hours": ["24"]})
    assert out["aggregate"]["volume"] == 3
    expected = (0.5 * 0.9 - 0.5 * 0.9 + 0.6 * 0.7) / (0.9 * 2 + 0.7)
    assert out["aggregate"]["score"] == pytest.approx(expected)
    assert set(out["per_source"]) == {"NEWS", "STOCKTWITS"}
    assert out["per_source"]["NEWS"]["score"] == pytest.approx(0.0)
    assert out["llm"] is None
    assert len(out["events"]) == 3


def test_route_sentiment_empty(db):
    out = _route_sentiment(db, {"symbol": ["ZZZ"], "hours": ["72"]})
    assert out["aggregate"]["volume"] == 0
    assert out["aggregate"]["score"] == 0.0
    assert out["per_source"] == {}
    assert out["series"] == []
