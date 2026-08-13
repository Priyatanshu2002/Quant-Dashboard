"""Tests for plan R4 — SQLite FTS5 full-text search over the corpus
(filings/news/analyses)."""
from __future__ import annotations

import pytest

from core.db import SQLiteStorage


@pytest.fixture
def db(tmp_path):
    return SQLiteStorage(tmp_path / "test.db")


def test_index_and_search_corpus_fts(db):
    db.index_corpus_doc("AAPL", "FILING", "Apple annual report 10-K",
                        "The company reported record revenue from iPhone sales.",
                        url="https://sec.gov/1", ts="2026-06-30")
    db.index_corpus_doc("MSFT", "NEWS", "Microsoft cloud growth",
                        "Azure compute demand accelerated last quarter.",
                        url="https://x.com/2", ts="2026-07-01")
    res = db.search_corpus("record revenue")
    assert len(res) >= 1
    hit = res[0]
    assert hit["symbol"] == "AAPL"
    assert hit["search_method"] == "fts5"
    assert hit["kind"] == "FILING"


def test_search_ranks_relevant_doc_higher(db):
    db.index_corpus_doc("X", "NEWS", "Apple beats earnings again", "strong quarter",
                        ts="2026-07-01")
    db.index_corpus_doc("Y", "NEWS", "Apple supplier delays", "weak guidance",
                        ts="2026-07-01")
    res = db.search_corpus("Apple beats")
    assert len(res) >= 1
    # only the first doc matches both terms
    assert all(r["symbol"] == "X" for r in res)


def test_search_symbol_filter_via_query(db):
    db.index_corpus_doc("AAPL", "NEWS", "iPhone demand strong", "unit sales up",
                        ts="2026-07-01")
    db.index_corpus_doc("MSFT", "NEWS", "Azure demand strong", "cloud up",
                        ts="2026-07-01")
    res = db.search_corpus("symbol: MSFT AND demand")
    assert res and all(r["symbol"] == "MSFT" for r in res)


def test_sentiment_events_auto_indexed(db):
    db.write_sentiment_event("AAPL", "NEWS", 0.6, headline="Apple beats on revenue",
                             source_weight=0.9)
    res = db.search_corpus("Apple beats")
    assert res and res[0]["symbol"] == "AAPL"
    assert res[0]["kind"].startswith("SENTIMENT_")


def test_reindex_corpus_rebuilds(db):
    db.index_corpus_doc("AAPL", "FILING", "10-K", "apple quarterly revenue",
                        ts="2026-06-30")
    n = db.reindex_corpus()
    assert n == 1
    assert db.search_corpus("quarterly revenue")


def test_search_empty_query_returns_empty(db):
    assert db.search_corpus("") == []
    assert db.search_corpus("   ") == []
