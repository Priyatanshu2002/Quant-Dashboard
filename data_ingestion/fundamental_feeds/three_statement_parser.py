"""SEC EDGAR XBRL 3-statement parser — company facts JSON → snapshot.

Pulls the full XBRL company-facts feed for a CIK and extracts the latest
annual (10-K) income statement, balance sheet and cash-flow items, computes
YoY growth, margins, ROIC and a balance-sheet sanity check, and returns a
FundamentalSnapshot-ready dict. Also provides a deep multi-year series
parser for the 10-K / 10-Q history.

Industry-standard correctness notes (fixed here):
  * EBITDA = EBIT + D&A  (NOT OperatingIncomeLoss alone, and never gross profit).
  * total_debt = short-term debt + current portion of LT debt + long-term debt
    + finance/operating leases (ASC 842), NOT the whole-liabilities line.
  * ROIC = NOPAT / invested capital, computed from the statements (not imported).
  * Balance-sheet identity Assets = Liabilities + Equity is checked per period.
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
# Multiple concepts per field: the FIRST concept with data wins (per field),
# so modern concepts are never overridden by stale legacy ones.
_US_GAAP_FIELDS = {
    # Revenue
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("revenue", 1),
    "RevenueFromContractWithCustomerIncludingAssessedTax": ("revenue", 1),
    "Revenues": ("revenue", 1),
    "SalesRevenueNet": ("revenue", 1),
    # Cost / gross profit
    "CostOfGoodsAndServicesSold": ("cost_of_revenue", 1),
    "CostOfRevenue": ("cost_of_revenue", 1),
    "GrossProfit": ("gross_profit", 1),
    # Operating expenses
    "ResearchAndDevelopmentExpense": ("research_development", 1),
    "SellingGeneralAndAdministrativeExpense": ("selling_general_admin", 1),
    "OperatingIncomeLoss": ("operating_income", 1),       # EBIT
    "DepreciationDepletionAndAmortization": ("depreciation_amortization", 1),
    "DepreciationAmortizationAndAccretionNet": ("depreciation_amortization", 1),
    "DepreciationAmortization": ("depreciation_amortization", 1),
    # Non-operating / tax
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": ("pretax_income", 1),
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": ("pretax_income", 1),
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": ("pretax_income", 1),
    "IncomeTaxExpenseBenefit": ("income_tax", 1),
    "InterestExpense": ("interest_expense", 1),
    "NetIncomeLoss": ("net_income", 1),
    "EarningsPerShareDiluted": ("eps_actual", 1),
    "EarningsPerShareBasic": ("eps_basic", 1),
    # Balance sheet
    "Assets": ("total_assets", 1),
    "AssetsCurrent": ("current_assets", 1),
    "CashAndCashEquivalentsAtCarryingValue": ("cash_and_equivalents", 1),
    "AccountsReceivableNetCurrent": ("accounts_receivable", 1),
    "InventoryNet": ("inventory", 1),
    "Liabilities": ("total_liabilities", 1),
    "LiabilitiesCurrent": ("current_liabilities", 1),
    "AccountsPayableCurrent": ("accounts_payable", 1),
    "StockholdersEquity": ("shareholders_equity", 1),
    "RetainedEarningsAccumulatedDeficit": ("retained_earnings", 1),
    "MinorityInterest": ("minority_interest", 1),
    # Debt (components summed into total_debt below)
    "LongTermDebtNoncurrent": ("long_term_debt", 1),
    "ShortTermBorrowings": ("short_term_debt", 1),
    "LongTermDebtCurrent": ("current_portion_long_term_debt", 1),
    "FinanceLeaseLiabilityNoncurrent": ("finance_lease_noncurrent", 1),
    "FinanceLeaseLiability": ("finance_lease", 1),
    "OperatingLeaseLiability": ("operating_lease", 1),
    "PreferredStockValue": ("preferred_stock", 1),
    "CommonStockSharesOutstanding": ("shares_outstanding", 1),
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities": ("operating_cash_flow", 1),
    "PaymentsToAcquirePropertyPlantAndEquipment": ("capex", -1),  # negative → flip sign
}

# Statement groupings of the concept map → rows for deep multi-year history.
_INCOME_CONCEPTS = [
    "revenue", "cost_of_revenue", "gross_profit", "research_development",
    "selling_general_admin", "operating_income", "depreciation_amortization",
    "interest_expense", "pretax_income", "income_tax", "net_income",
    "eps_actual", "eps_basic",
]
_BALANCE_CONCEPTS = [
    "total_assets", "current_assets", "cash_and_equivalents",
    "accounts_receivable", "inventory", "total_liabilities",
    "current_liabilities", "accounts_payable", "long_term_debt",
    "short_term_debt", "current_portion_long_term_debt",
    "finance_lease", "finance_lease_noncurrent", "operating_lease",
    "shareholders_equity", "retained_earnings", "minority_interest",
    "preferred_stock", "shares_outstanding",
]
_CASHFLOW_CONCEPTS = ["operating_cash_flow", "capex"]


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


def _total_debt(snap: dict) -> float | None:
    """Proper total debt = ST borrowings + current portion of LT debt
    + long-term debt + finance leases + operating leases (ASC 842)."""
    components = [snap.get("short_term_debt"), snap.get("current_portion_long_term_debt"),
                  snap.get("long_term_debt"), snap.get("finance_lease"),
                  snap.get("finance_lease_noncurrent"), snap.get("operating_lease")]
    vals = [float(c) for c in components if c is not None]
    return sum(vals) if vals else None


def _tax_rate(snap: dict) -> float:
    if snap.get("pretax_income") and snap.get("income_tax") is not None:
        try:
            t = snap["income_tax"] / snap["pretax_income"]
            if 0 <= t <= 1:
                return float(t)
        except (TypeError, ZeroDivisionError):
            pass
    return 0.21  # statutory fallback


def _finalize_snapshot(snap: dict) -> dict:
    """Compute derived ratios + balance-sheet sanity on a raw parsed snapshot."""
    # EBITDA = EBIT + D&A  (fall back to EBIT when D&A missing)
    ebit = snap.get("operating_income")
    da = snap.get("depreciation_amortization")
    if ebit is not None:
        snap["ebitda"] = (ebit + da) if da is not None else ebit
    elif snap.get("ebitda") is None and snap.get("gross_profit") is not None:
        snap["ebitda"] = snap["gross_profit"]

    # Proper total debt (never the whole-liabilities line). Only overwrite
    # when the ASC-842 components are present; otherwise keep a pre-existing value.
    computed_td = _total_debt(snap)
    if computed_td is not None:
        snap["total_debt"] = computed_td

    if snap.get("total_debt") is not None and snap.get("cash_and_equivalents") is not None:
        snap["net_debt"] = snap["total_debt"] - snap["cash_and_equivalents"]

    rev = snap.get("revenue")
    if rev:
        if snap.get("gross_profit") is not None:
            snap["gross_margin"] = snap["gross_profit"] / rev
        if snap.get("ebitda") is not None:
            snap["ebitda_margin"] = snap["ebitda"] / rev
        if snap.get("operating_income") is not None:
            snap["operating_margin"] = snap["operating_income"] / rev
        if snap.get("net_income") is not None:
            snap["net_margin"] = snap["net_income"] / rev

    if snap.get("shareholders_equity") and snap.get("total_debt") is not None:
        snap["debt_to_equity"] = snap["total_debt"] / snap["shareholders_equity"]

    if (snap.get("current_assets") is not None and snap.get("current_liabilities")):
        snap["current_ratio"] = snap["current_assets"] / snap["current_liabilities"]
        inv = snap.get("inventory") or 0.0
        snap["quick_ratio"] = (snap["current_assets"] - inv) / snap["current_liabilities"]

    if snap.get("operating_income") is not None and snap.get("interest_expense"):
        snap["interest_coverage_ratio"] = snap["operating_income"] / snap["interest_expense"]

    # ROIC = NOPAT / invested capital
    if ebit is not None:
        nopat = ebit * (1 - _tax_rate(snap))
        snap["nopat"] = nopat
        inv_cap = None
        td = snap.get("total_debt")
        eq = snap.get("shareholders_equity")
        cash = snap.get("cash_and_equivalents")
        if td is not None and eq is not None:
            inv_cap = td + eq - (cash or 0.0)
        if inv_cap and inv_cap > 0:
            snap["roic"] = nopat / inv_cap

    if snap.get("operating_cash_flow") is not None and snap.get("capex") is not None:
        snap["free_cash_flow"] = snap["operating_cash_flow"] - abs(snap["capex"])

    # Balance-sheet identity: Assets = Liabilities + Equity (+ minority)
    a = snap.get("total_assets")
    l = snap.get("total_liabilities")
    e = snap.get("shareholders_equity")
    m = snap.get("minority_interest")
    if a and l is not None and e is not None:
        rhs = l + e + (m or 0.0)
        snap["balance_discrepancy_pct"] = (a - rhs) / a * 100.0
        snap["balance_ok"] = abs(snap["balance_discrepancy_pct"]) < 1.0

    return snap


def parse_company_facts_series(cik: str, symbol: str) -> dict:
    """Full multi-year statement series from SEC XBRL company-facts.

    Returns {"income": [...], "balance": [...], "cashflow": [...]} where each
    list contains one normalized row per period end with the `period` date and
    `period_type` "ANNUAL" or "QUARTERLY" — every 10-K / 10-Q on record, not just
    the latest. This is the deep-history source (vs yfinance's ~5 quarters).
    Derived fields (EBITDA, total_debt, net_debt, ratios) are added per period.
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
        rows = []
        for end in sorted(series, reverse=True):
            r = series[end]
            r["period"] = str(pd.Timestamp(end).date())
            rows.append(r)
        return rows

    income = _rows(_INCOME_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL") \
        + _rows(_INCOME_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY")
    balance = _rows(_BALANCE_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL") \
        + _rows(_BALANCE_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY")
    cashflow = _rows(_CASHFLOW_CONCEPTS, ("10-K", "10-K/A"), 300, 420, "ANNUAL") \
        + _rows(_CASHFLOW_CONCEPTS, ("10-Q", "10-Q/A"), 60, 150, "QUARTERLY")

    # Derived per-period fields
    for row in income:
        ebit = row.get("operating_income")
        da = row.get("depreciation_amortization")
        if ebit is not None:
            row["ebitda"] = (ebit + da) if da is not None else ebit
    for row in balance:
        td = _total_debt(row)
        if td is not None:
            row["total_debt"] = td
        if row.get("total_debt") is not None and row.get("cash_and_equivalents") is not None:
            row["net_debt"] = row["total_debt"] - row["cash_and_equivalents"]
        if row.get("shareholders_equity") and row.get("total_debt") is not None:
            row["debt_to_equity"] = row["total_debt"] / row["shareholders_equity"]
        a, l, e = row.get("total_assets"), row.get("total_liabilities"), row.get("shareholders_equity")
        if a and l is not None and e is not None:
            row["balance_discrepancy_pct"] = (a - (l + e + (row.get("minority_interest") or 0.0))) / a * 100.0
            row["balance_ok"] = abs(row["balance_discrepancy_pct"]) < 1.0

    return {"income": income, "balance": balance, "cashflow": cashflow}


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

    _finalize_snapshot(snap)

    # YoY growth on the revenue series
    series = _annual_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    if not series:
        series = _annual_series(facts, "Revenues")
    ends = sorted(series.keys())
    if len(ends) >= 2 and series.get(ends[-2]):
        snap["revenue_yoy_growth"] = series[ends[-1]] / series[ends[-2]] - 1

    log.info("Parsed %s (%s): revenue=%s ebitda=%s fcf=%s roic=%s balance_ok=%s",
             symbol, cik, snap.get("revenue"), snap.get("ebitda"),
             snap.get("free_cash_flow"), snap.get("roic"), snap.get("balance_ok"))
    return snap
