"""Calendar features (plan §3.5) — earnings/expiry proximity, day-of-week effects."""
from __future__ import annotations

from datetime import date

from core.db import Storage, get_storage


def compute_calendar_features(symbol: str, db: Storage | None = None,
                              today: date | None = None) -> dict:
    db = db or get_storage()
    today = today or date.today()
    f: dict = {}

    next_earnings = db.get_next_earnings_date(symbol)
    if next_earnings:
        f["days_to_earnings"] = (next_earnings - today).days
        f["earnings_week"] = float(f["days_to_earnings"] <= 5)
        f["earnings_month"] = float(f["days_to_earnings"] <= 30)

    next_expiry = db.get_next_fo_expiry(symbol)
    if next_expiry:
        f["days_to_expiry"] = (next_expiry - today).days
        f["expiry_week"] = float((next_expiry - today).days <= 5)

    f["day_of_week"] = today.weekday()
    f["month_end_effect"] = float(today.day >= 25)
    f["quarter_end_effect"] = float(today.month in [3, 6, 9, 12] and today.day >= 20)
    return f
