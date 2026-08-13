"""NSE/BSE filing event watcher.

⚠️ CAUTION (plan §12): before connecting any LIVE broker API, verify
compliance with SEBI regulations on automated trading systems. This module
only watches public announcements for event triggers — it places no orders.

BSE exposes a corporate-announcements feed at api.bseindia.com (headers
required, frequently rate-limited); NSE publishes via nsearchives.nseindia.com.
Both are treated as best-effort: failures degrade to a log line.
"""
from __future__ import annotations

import time
from typing import Iterable

import requests

from core.db import Storage, get_storage
from core.events import EVENT_FUNDAMENTAL, bus
from core.logging import get_logger

log = get_logger(__name__)

BSE_ANNOUNCEMENTS = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
NSE_RESULTS_URL = "https://www.nseindia.com/api/corporates-announcements"


class NseBseWatcher:
    def __init__(self, symbols: Iterable[str], storage: Storage | None = None,
                 poll_seconds: int = 1800):
        self.symbols = list(symbols)
        self.storage = storage or get_storage()
        self.poll_seconds = poll_seconds
        self._seen: set[str] = set()

    def check_bse(self) -> list[dict]:
        """Best-effort BSE announcements check (requires BSE session headers)."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.bseindia.com/",
            "Accept": "application/json",
        }
        try:
            resp = requests.get(BSE_ANNOUNCEMENTS, headers=headers, timeout=15)
            if resp.status_code != 200:
                return []
            rows = resp.json().get("Table", [])
            return [r for r in rows if str(r.get("SC_CODE", "")).isdigit()]
        except Exception as e:  # noqa: BLE001
            log.debug("BSE check failed (expected): %s", e)
            return []

    def check_nse(self) -> list[dict]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        }
        try:
            sess = requests.Session()
            sess.get("https://www.nseindia.com", headers=headers, timeout=15)
            resp = sess.get(NSE_RESULTS_URL, headers=headers, timeout=15)
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception as e:  # noqa: BLE001
            log.debug("NSE check failed (expected): %s", e)
            return []

    def run_forever(self) -> None:
        while True:
            for item in self.check_bse() + self.check_nse():
                key = str(item.get("ATTACHMENTNAME") or item.get("sm_name") or item)
                if key in self._seen:
                    continue
                self._seen.add(key)
                log.info("New NSE/BSE announcement: %s", key)
                import asyncio
                asyncio.run(bus.publish(EVENT_FUNDAMENTAL, {
                    "source": "NSEBSE", "raw": item,
                }))
            time.sleep(self.poll_seconds)


if __name__ == "__main__":
    NseBseWatcher(["RELIANCE.NS"]).run_forever()
