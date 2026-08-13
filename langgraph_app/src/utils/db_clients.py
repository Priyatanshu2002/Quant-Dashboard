"""DB helpers for the debate nodes — read feature/sentiment/macro snapshots."""
from __future__ import annotations

from core.db import Storage, get_storage
from core.logging import get_logger
from feature_engineering.fundamental_features import compute_fundamental_features
from feature_engineering.macro_features import compute_macro_features
from feature_engineering.sentiment_features import compute_sentiment_features

log = get_logger(__name__)


def latest_snapshot_for(symbol: str, db: Storage | None = None) -> dict:
    """Everything the AnalystPack needs from storage (fund + sent + macro)."""
    db = db or get_storage()
    return {
        "fundamental": compute_fundamental_features(symbol, "EQUITY_US", db),
        "sentiment": compute_sentiment_features(symbol, db),
        "macro": compute_macro_features(db),
        "latest_reflection": _latest_reflection(db),
    }


def _latest_reflection(db: Storage) -> str | None:
    import sqlite3
    if db.backend != "sqlite":
        return None
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT lesson_text FROM reflection_prompts WHERE applied=0 "
                "ORDER BY time DESC LIMIT 1").fetchone()
        return row["lesson_text"] if row else None
    except Exception:  # noqa: BLE001
        return None


def log_gating(db: Storage, row: dict) -> None:
    try:
        db.write_gating_log(row)
    except Exception as e:  # noqa: BLE001
        log.warning("gating log write failed: %s", e)
