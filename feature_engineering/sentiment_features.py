"""Sentiment features (~10, plan §3.4) — from stored sentiment events."""
from __future__ import annotations

from core.db import Storage, get_storage


def compute_sentiment_features(symbol: str, db: Storage | None = None) -> dict:
    db = db or get_storage()
    recent = db.query_sentiment_events(symbol, hours=24)
    if not recent:
        return {"sentiment_score": 0.0, "sentiment_volume": 0, "sentiment_momentum": 0.0}

    scores = [e["score"] for e in recent]
    weights = [e.get("source_weight", 1.0) for e in recent]

    f: dict = {}
    f["sentiment_score"] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    f["sentiment_volume"] = len(recent)
    f["sentiment_positive_pct"] = sum(1 for s in scores if s > 0.2) / len(scores)
    f["sentiment_negative_pct"] = sum(1 for s in scores if s < -0.2) / len(scores)
    f["sentiment_momentum"] = f["sentiment_score"] - db.query_sentiment_avg(symbol, hours=72)

    for source, key in [("GDELT", "gdelt_sentiment"), ("REDDIT", "reddit_sentiment"),
                        ("NEWS", "news_sentiment"), ("GOOGLE", "google_sentiment"),
                        ("YAHOO_RSS", "yahoo_rss_sentiment"),
                        ("STOCKTWITS", "stocktwits_sentiment")]:
        vals = [e["score"] for e in recent if e["source"] == source]
        f[key] = sum(vals) / len(vals) if vals else 0.0

    f["sentiment_extreme"] = float(abs(f["sentiment_score"]) > 0.7)
    return f
