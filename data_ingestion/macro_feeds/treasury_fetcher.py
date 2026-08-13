"""US Treasury fetcher (no key) — full nominal yield curve + TIPS real yields.

Source: US Treasury data warehouse XML API (the legacy `/daily-treasury-rates.csv`
CSV path now 404s; the OData-style XML feeds are the live endpoint):
  nominal: pages/xml?data=daily_treasury_yield_curve
  real:    pages/xml?data=daily_treasury_real_yield_curve

Captures the full par yield curve (1M–30Y) plus the 10Y real (TIPS) yield and
derives breakeven inflation = 10Y nominal − 10Y real. Persists a MacroSnapshot
and publishes EVENT_MACRO.
"""
from __future__ import annotations

import datetime as dt
import io
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from core.db import Storage, get_storage
from core.events import EVENT_MACRO, bus
from core.logging import get_logger

log = get_logger(__name__)

XML_BASE = ("https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml")

# Nominal par yield curve columns: Treasury tag → our field (fraction).
_NOMINAL = {
    "BC_1MONTH": "us_1m_yield", "BC_2MONTH": "us_2m_yield",
    "BC_3MONTH": "us_3m_yield", "BC_6MONTH": "us_6m_yield",
    "BC_1YEAR": "us_1y_yield", "BC_2YEAR": "us_2y_yield",
    "BC_3YEAR": "us_3y_yield", "BC_5YEAR": "us_5y_yield",
    "BC_7YEAR": "us_7y_yield", "BC_10YEAR": "us_10y_yield",
    "BC_20YEAR": "us_20y_yield", "BC_30YEAR": "us_30y_yield",
}

# TIPS real yield curve: tag → field (fraction).
_REAL = {
    "TC_5YEAR": "us_5y_real_yield", "TC_7YEAR": "us_7y_real_yield",
    "TC_10YEAR": "us_10y_real_yield", "TC_20YEAR": "us_20y_real_yield",
    "TC_30YEAR": "us_30y_real_yield",
}


def _fetch_xml_feed(data: str, years_back: int) -> pd.DataFrame | None:
    """Pull one Treasury OData feed, newest-first, filtered to today."""
    today = dt.date.today()
    frames = []
    for year in range(today.year, today.year - years_back - 1, -1):
        url = f"{XML_BASE}?data={data}&field_tdr_date_value={year}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            ns = {
                "a": "http://www.w3.org/2005/Atom",
                "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
                "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
            }
            rows = []
            for content in root.findall(".//a:content", ns):
                props = content.find("m:properties", ns)
                if props is None:
                    continue
                row: dict[str, float] = {}
                for child in props:
                    tag = child.tag.split("}")[-1]
                    if tag == "NEW_DATE":
                        continue
                    try:
                        row[tag] = float(child.text)
                    except (TypeError, ValueError):
                        continue
                if row:
                    date_node = props.find("d:NEW_DATE", ns)
                    if date_node is not None:
                        row["_date"] = str(pd.Timestamp(date_node.text).date())
                    rows.append(row)
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception as e:  # noqa: BLE001
            log.debug("Treasury %s feed year %d failed: %s", data, year, e)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["_date"] <= today.isoformat()].sort_values("_date")
    return df


def fetch_treasury_curve(storage: Storage | None = None,
                         years_back: int = 2) -> dict | None:
    """Fetch full nominal curve + real yields, persist a MacroSnapshot."""
    storage = storage or get_storage()
    nom = _fetch_xml_feed("daily_treasury_yield_curve", years_back)
    real = _fetch_xml_feed("daily_treasury_real_yield_curve", years_back)

    if nom is None or nom.empty:
        return None
    latest = nom.iloc[-1]
    macro: dict[str, float] = {}
    for tag, field in _NOMINAL.items():
        v = latest.get(tag)
        if pd.notna(v):
            macro[field] = float(v) / 100

    if real is not None and not real.empty:
        rrow = real.iloc[-1]
        for tag, field in _REAL.items():
            v = rrow.get(tag)
            if pd.notna(v):
                macro[field] = float(v) / 100

    if "us_10y_yield" in macro and "us_2y_yield" in macro:
        macro["yield_curve_spread"] = macro["us_10y_yield"] - macro["us_2y_yield"]
    # 10Y nominal − 10Y real = breakeven inflation (key macro signal).
    if "us_10y_yield" in macro and "us_10y_real_yield" in macro:
        macro["breakeven_inflation"] = macro["us_10y_yield"] - macro["us_10y_real_yield"]

    if not macro:
        return None
    full = {"ts": dt.datetime.utcnow(), **macro}
    storage.write_macro_snapshot(full)
    import asyncio
    asyncio.run(bus.publish(EVENT_MACRO, full))
    log.info("Treasury curve: %s", {k: round(v, 4) for k, v in macro.items()})
    return full


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_treasury_curve(), indent=2, default=str))


def fetch_vix_stooq() -> float | None:
    """Latest VIX close from Stooq (free, no key). Returns None on failure."""
    try:
        resp = requests.get("https://stooq.com/q/d/l/?s=^vix&i=d", timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if len(lines) < 2:
            return None
        return float(lines[-1].split(",")[-1])
    except Exception as e:  # noqa: BLE001
        log.debug("Stooq VIX fetch failed: %s", e)
        return None
