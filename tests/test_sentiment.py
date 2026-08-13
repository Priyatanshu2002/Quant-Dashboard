"""Tests for the news sentiment layer (multi-source aggregator + features)."""
from unittest import mock

import pytest

from core.db import SQLiteStorage
from data_ingestion.sentiment_feeds.news_aggregator import (
    fetch_news_events, fetch_stocktwits, fetch_yahoo_news)
from feature_engineering.sentiment_features import compute_sentiment_features


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


def _mock_get(payload, status=200):
    resp = mock.MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.content = b"<rss/>"
    resp.raise_for_status = mock.MagicMock()
    return resp


def test_yahoo_news_persists_scored(db):
    payload = {"news": [
        {"title": "Apple beats earnings record", "link": "http://x/1"},
        {"title": "lawsuit filed against company", "link": "http://x/2"},
        {"title": "plain headline here", "link": "http://x/3"},
    ]}
    with mock.patch("data_ingestion.sentiment_feeds.news_aggregator.requests.get",
                    return_value=_mock_get(payload)) as m:
        events = fetch_yahoo_news("AAPL", db)
    assert len(events) == 3
    m.assert_called_once()
    stored = db.query_sentiment_events("AAPL", hours=24)
    assert len(stored) == 3
    assert all(-1.0 <= e["score"] <= 1.0 for e in stored)
    scores = {e["headline"]: e["score"] for e in stored}
    assert scores["Apple beats earnings record"] > 0
    assert scores["lawsuit filed against company"] < 0
    # source-weighted score blends only the positive headline meaningfully
    feats = compute_sentiment_features("AAPL", db)
    assert feats["sentiment_volume"] == 3
    assert "news_sentiment" in feats


def test_stocktwits_native_sentiment(db):
    payload = {"messages": [
        {"body": "looking great", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "this is terrible", "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "neutral chatter", "entities": {}},
    ]}
    with mock.patch("data_ingestion.sentiment_feeds.news_aggregator.requests.get",
                    return_value=_mock_get(payload)):
        events = fetch_stocktwits("AAPL", db)
    assert len(events) == 3
    stored = db.query_sentiment_events("AAPL", hours=24)
    by_body = {e["headline"]: e["score"] for e in stored}
    assert by_body["looking great"] == pytest.approx(0.6)
    assert by_body["this is terrible"] == pytest.approx(-0.6)
    assert by_body["neutral chatter"] == 0.0  # lexicon: no signal words


def test_fetch_news_events_multisource(db):
    payload = {"news": [{"title": "record profits", "link": ""}]}
    twits = {"messages": [{"body": "bullish day", "entities": {"sentiment": {"basic": "Bullish"}}}]}
    with mock.patch("data_ingestion.sentiment_feeds.news_aggregator.requests.get",
                    side_effect=[_mock_get(payload), _mock_get(twits)]):
        events = fetch_news_events("MSFT", storage=db, sources=["yahoo", "stocktwits"])
    assert len(events) == 2
    sources = {e["source"] for e in db.query_sentiment_events("MSFT", hours=24)}
    assert sources == {"NEWS", "STOCKTWITS"}


def test_sentiment_features_weighted_and_momentum(db):
    db.write_sentiment_event("X", "NEWS", 0.5, source_weight=0.9)
    db.write_sentiment_event("X", "NEWS", 0.5, source_weight=0.9)
    db.write_sentiment_event("X", "STOCKTWITS", -0.6, source_weight=0.7)
    feats = compute_sentiment_features("X", db)
    assert feats["sentiment_volume"] == 3
    expected = (0.5 * 0.9 * 2 + (-0.6) * 0.7) / (0.9 * 2 + 0.7)
    assert feats["sentiment_score"] == pytest.approx(expected)
    assert feats["news_sentiment"] == pytest.approx(0.5)
    assert feats["stocktwits_sentiment"] == pytest.approx(-0.6)
    assert feats["sentiment_extreme"] == 0.0


def test_sentiment_features_empty():
    assert compute_sentiment_features("NOPE")["sentiment_volume"] == 0
