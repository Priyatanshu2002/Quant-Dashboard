"""SEC EDGAR XBRL 3-statement parser — company facts JSON → snapshot.

Pulls the full XBRL company-facts feed for a CIK and extracts the latest
annual (10-K) income statement, balance sheet and cash-flow items, computes
YoY growth and margins, and returns a FundamentalSnapshot-ready dict.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests

from core.logging import get_logger

log = get_logger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
HEADERS = {"User-Agent": "Project Agonistes research@agonistes.local"}

# us-gaap concept → (our field, scale). Annual = 10-K frames only.
_US_GAAP_FIELDS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("revenue", 1),
    "RevenueFromContractWithCustomerIncludingAssessedTax": ("revenue", 1),
    "Revenues": ("revenue", 1),
    "SalesRevenueNet": ("revenue", 1),
    "GrossProfit": ("gross_profit", 1),
    "OperatingIncomeLoss": ("ebitda", 1),  # fallback proxy, refined below
    "NetIncomeLoss": ("net_income", 1),
    "EarningsPerShareDiluted": ("eps_actual", 1),
    "Assets": ("total_assets", 1),
    "Liabilities": ("total_debt", 1),  # proxy; refined with long-term debt below
    "CashAndCashEquivalentsAtCarryingValue": ("cash_and_equivalents", 1),
    "StockholdersEquity": ("shareholders_equity", 1),
    "LongTermDebtNoncurrent": ("long_term_debt", 1),
    "NetCashProvidedByUsedInOperatingActivities": ("operating_cash_flow", 1),
    "PaymentsToAcquirePropertyPlantAndEquipment": ("capex", -1),  # negative → flip sign
    "CommonStockSharesOutstanding": ("shares_outstanding", 1),
}


def _us_gaap(facts: dict[str, Any]) -> dict:
    """The company-facts payload nests everything under facts.us-gaap."""
    return (facts.get("facts") or {}).get("us-gaap") or facts.get("us-gaap") or {}


def _latest_annual(facts: dict[str, Any], concept: str) -> dict | None:
    """Most recent full-year (10-K, ~12-month duration) value for a concept."""
    unit_data = _us_gaap(facts).get(concept, {}).get("units", {})
    best: dict | None = None
    best_key = ("", "")
    for units in unit_data.values():
        for f in units:
            if f.get("form") not in ("10-K", "10-K/A"):
                continue
            start, end = f.get("start"), f.get("end")
            if start and end:
                try:
                    duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
                except Exception:  # noqa: BLE001
                    duration = 0
                if duration < 300 or duration > 420:   # skip partial periods
                    continue
            key = (str(end or ""), str(start or ""))
            if key > best_key:
                best_key = key
                best = f
        if best:
            return best
    return None


def _annual_series(facts: dict[str, Any], concept: str) -> dict[str, float]:
    """All annual (10-K) values keyed by fiscal year end."""
    out: dict[str, float] = {}
    unit_data = _us_gaap(facts).get(concept, {}).get("units", {})
    for units in unit_data.values():
        for f in units:
            if f.get("form") not in ("10-K", "10-K/A"):
                continue
            end = f.get("end")
            if end:
                out[end] = float(f.get("val"))
    return out


# Statement groupings of the concept map → rows for deep multi-year history.
_INCOME_CONCEPTS = ["revenue", "gross_profit", "ebitda", "net_income", "eps_actual"]
_BALANCE_CONCEPTS = ["total_assets", "total_liabilities", "total_debt", "cash_and_equivalents",
                     "shareholders_equity", "long_term_debt", "shares_outstanding"]
_CASHFLOW_CONCEPTS = ["operating_cash_flow", "capex"]


def _concept_values(facts: dict[str, Any], concept: str, forms: tuple[str, ...],
                    min_dur: int, max_dur: int) -> dict[str, float]:
    """Every value for a concept whose filing duration falls in [min_dur,max_dur],
    keyed by period end. Keeps the most recent value per end (dedup across units)."""
    out: dict[str, float] = {}
    unit_data = _us_gaap(facts).get(concept, {}).get("units", {})
    for units in unit_data.values():
        for f in units:
            if f.get("form") not in forms:
                continue
            start, end = f.get("start"), f.get("end")
            if start and end:
                try:
                    dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
                except Exception:  # noqa: BLE001
                    dur = 0
                if not (min_dur <= dur <= max_dur):
                    continue
            if end:
                out[str(end)] = float(f.get("val"))
    return out


def parse_company_facts_series(cik: str, symbol: str) -> dict:
    """Full multi-year statement series from SEC XBRL company-facts.

    Returns {"income": [...], "balance": [...], "cashflow": [...]} where each
    list contains one normalized row per period end with the `period` date and
    `period_type` "ANNUAL" or "QUARTERLY" — every 10-K / 10-Q on record, not just
    the latest. This is the deep-history source (vs yfinance's ~5 quarters).
    """
    resp = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    facts = resp.json()

    # our concept field -> list of us-gaap concept labels
    reverse: dict[str, list[str]] = {}
    for concept, (field, _scale) in _US_GAAP_FIELDS.items():
        reverse.setdefault(field, []).append(concept)

    def _rows(fields: list[str], forms: tuple[str, ...], min_dur: int, max_dur: int,
              ptype: str) -> list[dict]:
        series: dict[str, dict] = {}
        for field in fields:
            for concept in reverse.get(field, []):
                vals = _concept_values(facts, concept, forms, min_dur, max_dur)
                for end, val in vals.items():
                    series.setdefault(end, {"period": end, "period_type": ptype})[field] = val
        # convert period to ISO date string, newest first
        rows = []
        for end in sorted(series, reverse=True):
            r = series[end]
            r["period"] = str(pd.Timestamp(end).date())
            rows.append(r)
        return rows

    return {
        "income": _rows(_INCOME_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL")
                + _rows(_INCOME_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY"),
        "balance": _rows(_BALANCE_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL")
                 + _rows(_BALANCE_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY"),
        "cashflow": _rows(_CASHFLOW_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL")
                  + _rows(_CASHFLOW_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY"),
    }



def parse_company_facts(cik: str, symbol: str,
                        asset_class: str = "EQUITY_US") -> dict:
    """Download + parse company facts → fundamental snapshot dict (or {} on failure)."""
    resp = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    facts = resp.json()

    snap: dict[str, Any] = {
        "symbol": symbol, "asset_class": asset_class,
        "period_type": "ANNUAL", "raw_data": {"cik": cik},
    }

    # Latest annual values — FIRST concept with data wins per field, so the
    # modern revenue concept is never overridden by a stale legacy concept
    # (e.g. "Revenues" last tagged FY2017 for MSFT).
    for concept, (field, scale) in _US_GAAP_FIELDS.items():
        entry = _latest_annual(facts, concept)
        if not entry:
            continue
        val = float(entry["val"]) * scale
        if field not in snap:
            snap[field] = val
        if "end" in entry and "time" not in snap:
            snap["time"] = dt.datetime.fromisoformat(entry["end"])
            snap["fiscal_year"] = int(entry["end"][:4])

    # Refinements
    if "long_term_debt" in snap:
        snap["total_debt"] = snap.pop("long_term_debt")
    if snap.get("total_debt") is not None and snap.get("cash_and_equivalents") is not None:
        snap["net_debt"] = snap["total_debt"] - snap["cash_and_equivalents"]
    if snap.get("shareholders_equity") and snap.get("total_debt") is not None:
        snap["debt_to_equity"] = snap["total_debt"] / snap["shareholders_equity"]
    if snap.get("revenue"):
        snap["ebitda"] = snap.get("ebitda") or snap.get("gross_profit")
    if snap.get("revenue") and snap.get("gross_profit") is not None:
        snap["gross_margin"] = snap["gross_profit"] / snap["revenue"]
    if snap.get("revenue") and snap.get("ebitda") is not None:
        snap["ebitda_margin"] = snap["ebitda"] / snap["revenue"]
    if snap.get("operating_cash_flow") is not None and snap.get("capex") is not None:
        snap["free_cash_flow"] = snap["operating_cash_flow"] - abs(snap["capex"])

    # YoY growth on the revenue series
    series = _annual_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    if not series:
        series = _annual_series(facts, "Revenues")
    ends = sorted(series.keys())
    if len(ends) >= 2 and series.get(ends[-2]):
        snap["revenue_yoy_growth"] = series[ends[-1]] / series[ends[-2]] - 1

    log.info("Parsed %s (%s): revenue=%s fcf=%s", symbol, cik,
             snap.get("revenue"), snap.get("free_cash_flow"))
    return snap
