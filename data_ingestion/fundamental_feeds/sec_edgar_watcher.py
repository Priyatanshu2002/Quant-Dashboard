"""SEC EDGAR filing watcher — polls for new 10-K / 10-Q / 8-K filings.

Uses the public EDGAR full-text/browse RSS (no key). SEC requires a
User-Agent with contact info; see https://www.sec.gov/os/accessing-edgar-data.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import requests

from core.db import Storage, get_storage
from core.events import EVENT_FUNDAMENTAL, bus
from core.logging import get_logger

log = get_logger(__name__)

EDGAR_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {
    "User-Agent": "Project Agonistes research@agonistes.local",
    "Accept-Encoding": "gzip, deflate",
}

FILING_TYPES = ["10-K", "10-Q", "8-K"]

# 8-K Item → material-event class (C4 Task 2). Item numbers appear in the
# filing title/URL e.g. "8-K ... Item 2.02". If no item is found we fall
# back to the filing type's default class.
_ITEM_MATERIAL = {
    "1.01": "AGREEMENT", "1.02": "AGREEMENT", "2.01": "ACQUISITION",
    "2.02": "EARNINGS", "2.03": "DEBT", "2.04": "IMPAIRMENT",
    "2.05": "RESTRUCTURING", "2.06": "ASSET_SALE", "3.01": "DELISTING",
    "4.01": "AUDITOR_CHANGE", "4.02": "AUDITOR_OPINION", "5.02": "MANAGEMENT_CHANGE",
    "5.03": "BYLAWS", "5.07": "SHAREHOLDER_VOTE", "7.01": "DISCLOSURE",
    "8.01": "OTHER_EVENT",
}
_DEFAULT_MATERIAL = {"10-K": "ANNUAL_REPORT", "10-Q": "QUARTERLY_REPORT",
                     "8-K": "8K_EVENT"}


def material_event(filing_type: str, title: str = "") -> str:
    """Map a filing (type + optional title) to a material-event class.

    10-K → ANNUAL_REPORT, 10-Q → QUARTERLY_REPORT; 8-K uses the item number
    extracted from the title (e.g. 'Item 2.02' → EARNINGS), else 8K_EVENT.
    """
    t = (filing_type or "").upper()
    if t == "8-K":
        for item, cls in _ITEM_MATERIAL.items():
            if f"item {item}" in title.lower():
                return cls
        return _DEFAULT_MATERIAL["8-K"]
    return _DEFAULT_MATERIAL.get(t, "OTHER")


def load_cik_map() -> dict[str, str]:
    """ticker → CIK (zero-padded 10 digits)."""
    resp = requests.get(EDGAR_TICKERS, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {entry["ticker"]: str(entry["cik_str"]).zfill(10)
            for entry in data.values()}


def recent_filings(cik: str, filing_type: str = "10-K", count: int = 10) -> list[dict]:
    """Most recent filings of a type for a company, from the browse RSS feed."""
    resp = requests.get(
        EDGAR_BROWSE,
        params={"action": "getcompany", "CIK": cik, "type": filing_type,
                "dateb": "", "owner": "include", "count": count, "output": "atom"},
        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns)
        link = entry.find("a:link", ns)
        date = entry.findtext("a:updated", default="", namespaces=ns)
        out.append({"title": title, "url": link.get("href") if link is not None else "",
                    "date": date})
    return out


class SecEdgarWatcher:
    def __init__(self, tickers: Iterable[str], storage: Storage | None = None,
                 poll_seconds: int = 3600):
        self.tickers = list(tickers)
        self.storage = storage or get_storage()
        self.poll_seconds = poll_seconds
        self.cik_map: dict[str, str] = {}
        self._seen: set[tuple[str, str]] = set()

    def refresh_cik_map(self) -> None:
        try:
            self.cik_map = load_cik_map()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not refresh CIK map: %s", e)

    def check_once(self) -> list[dict]:
        """Check for filings newer than the last seen; return the new ones."""
        new_filings = []
        for ticker in self.tickers:
            cik = self.cik_map.get(ticker.upper())
            if not cik:
                continue
            for ftype in FILING_TYPES:
                try:
                    for f in recent_filings(cik, ftype):
                        key = (f["url"], ftype)
                        if key in self._seen:
                            continue
                        self._seen.add(key)
                        f["ticker"] = ticker.upper()
                        f["filing_type"] = ftype
                        f["cik"] = cik
                        f["material_event"] = material_event(ftype, f.get("title", ""))
                        new_filings.append(f)
                except Exception as e:  # noqa: BLE001
                    log.debug("EDGAR poll %s %s failed: %s", ticker, ftype, e)
            time.sleep(0.2)  # be polite to EDGAR
        for f in new_filings:
            log.info("New SEC filing: %s %s %s", f["ticker"], f["filing_type"], f["title"])
        return new_filings

    def run_forever(self) -> None:
        self.refresh_cik_map()
        while True:
            for filing in self.check_once():
                import asyncio
                asyncio.run(bus.publish(EVENT_FUNDAMENTAL, filing))
            time.sleep(self.poll_seconds)


if __name__ == "__main__":
    SecEdgarWatcher(["AAPL", "MSFT"]).run_forever()
