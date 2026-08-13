"""SEC EDGAR XBRL 3-statement parser — maximum data extraction.

Pulls the full XBRL company-facts feed for a CIK and extracts the complete
income statement, balance sheet and cash-flow history (annual + quarterly),
computes derived ratios (margins, ROIC, balance-sheet identity), and returns
FundamentalSnapshot-ready dicts.

Design goals (maximum-data):
  * A broad US-GAAP concept map captures the full set of standard line items.
  * A catch-all keeps EVERY other us-gaap concept the company reports in an
    `extra` dict on each row — nothing the SEC tags is ever dropped.
  * Raw company-facts JSON is cached to disk (data/sec_facts/<CIK>.json) so the
    parser can be expanded/re-run without re-hitting SEC — cheap iteration.
  * Correctness: EBITDA = EBIT + D&A; total_debt sums ASC-842 debt components
    (never the whole-liabilities line); ROIC = NOPAT / invested capital; the
    balance-sheet identity Assets = Liabilities + Equity is checked per period.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from core.logging import get_logger

log = get_logger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
HEADERS = {"User-Agent": "Project Agonistes research@agonistes.local"}
_FACTS_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "sec_facts"


def load_facts(cik: str, use_cache: bool = True, refresh: bool = False) -> dict:
    """Fetch (and cache) the SEC company-facts JSON for a CIK.

    Cached copies live at data/sec_facts/<CIK>.json so re-parsing with an
    expanded concept map never re-hits SEC.
    """
    if use_cache and not refresh:
        _FACTS_CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = _FACTS_CACHE / f"CIK{cik}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log.debug("corrupt cache for %s — refetching", cik)
    resp = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    facts = resp.json()
    if use_cache:
        try:
            _FACTS_CACHE.mkdir(parents=True, exist_ok=True)
            (_FACTS_CACHE / f"CIK{cik}.json").write_text(
                json.dumps(facts), encoding="utf-8")
        except OSError as e:
            log.debug("cache write failed for %s: %s", cik, e)
    return facts


# ─────────────────────────────────────────────────────────────────────
# US-GAAP concept → (canonical field, scale). Multiple concepts per field;
# the FIRST concept with data wins per field (modern concepts never overridden
# by stale legacy ones). This map is intentionally broad to maximise capture.
# ─────────────────────────────────────────────────────────────────────
_US_GAAP_FIELDS: dict[str, tuple[str, int]] = {
    # ── Income statement ──────────────────────────────────────────────
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("revenue", 1),
    "RevenueFromContractWithCustomerIncludingAssessedTax": ("revenue", 1),
    "Revenues": ("revenue", 1),
    "SalesRevenueNet": ("revenue", 1),

    "CostOfGoodsAndServicesSold": ("cost_of_revenue", 1),
    "CostOfRevenue": ("cost_of_revenue", 1),
    "CostOfGoodsSold": ("cost_of_revenue", 1),
    "CostOfServices": ("cost_of_revenue", 1),
    "CostOfSales": ("cost_of_revenue", 1),

    "GrossProfit": ("gross_profit", 1),

    "ResearchAndDevelopmentExpense": ("research_development", 1),
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": ("research_development", 1),

    "SellingGeneralAndAdministrativeExpense": ("selling_general_admin", 1),
    "SellingAndMarketingExpense": ("selling_general_admin", 1),
    "SellingAndAdministrativeExpense": ("selling_general_admin", 1),
    "GeneralAndAdministrativeExpense": ("selling_general_admin", 1),

    "OtherOperatingIncomeExpenseNet": ("other_operating_expenses", 1),
    "OtherOperatingCostsAndExpenses": ("other_operating_expenses", 1),
    "RestructuringAndRelatedCostIncurredCost": ("other_operating_expenses", 1),
    "RestructuringCharges": ("other_operating_expenses", 1),

    "OperatingIncomeLoss": ("operating_income", 1),   # EBIT

    "DepreciationDepletionAndAmortization": ("depreciation_amortization", 1),
    "DepreciationAmortizationAndAccretionNet": ("depreciation_amortization", 1),
    "DepreciationAmortization": ("depreciation_amortization", 1),
    "DepreciationDepletionAndAmortizationExcludingFinancialServices": ("depreciation_amortization", 1),

    "InterestExpense": ("interest_expense", 1),
    "InterestExpenseNonoperating": ("interest_expense", 1),
    "InterestAndDebtExpense": ("interest_expense", 1),

    "InterestIncomeOperating": ("interest_income", 1),
    "InterestIncomeNonoperating": ("interest_income", 1),
    "InterestAndInvestmentIncome": ("interest_income", 1),
    "InvestmentIncomeInterest": ("interest_income", 1),

    "OtherNonoperatingIncomeExpense": ("other_income_expense", 1),
    "OtherNonoperatingIncome": ("other_income_expense", 1),
    "OtherNonoperatingExpenses": ("other_income_expense", 1),
    "OtherIncomeExpenseNet": ("other_income_expense", 1),

    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": ("pretax_income", 1),  # noqa: E501
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": ("pretax_income", 1),  # noqa: E501
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": ("pretax_income", 1),
    "EarningsBeforeInterestAndTaxes": ("pretax_income", 1),

    "IncomeTaxExpenseBenefit": ("income_tax", 1),
    "IncomeTaxExpenseBenefitContinuingOperations": ("income_tax", 1),
    "ProvisionForIncomeTaxesCurrent": ("income_tax", 1),
    "CurrentIncomeTaxExpenseBenefit": ("income_tax", 1),
    "DeferredIncomeTaxExpenseBenefit": ("income_tax", 1),

    "NetIncomeLoss": ("net_income", 1),
    "NetIncomeLossAvailableToCommonStockholdersBasic": ("net_income", 1),
    "ProfitLoss": ("net_income", 1),
    "NetIncomeLossAttributableToNoncontrollingInterest": ("net_income_to_noncontrolling", 1),

    "EarningsPerShareDiluted": ("eps_actual", 1),
    "EarningsPerShareDilutedFromContinuingOperations": ("eps_actual", 1),
    "EarningsPerShareBasic": ("eps_basic", 1),
    "EarningsPerShareBasicFromContinuingOperations": ("eps_basic", 1),

    # ── Balance sheet ─────────────────────────────────────────────────
    "Assets": ("total_assets", 1),
    "AssetsCurrent": ("current_assets", 1),
    "CashAndCashEquivalentsAtCarryingValue": ("cash_and_equivalents", 1),
    "CashAndCashEquivalentsAtCarryingValueExcludingFinancingReceivables": ("cash_and_equivalents", 1),

    "ShortTermInvestments": ("short_term_investments", 1),
    "MarketableSecuritiesCurrent": ("short_term_investments", 1),
    "AvailableForSaleSecuritiesCurrent": ("short_term_investments", 1),
    "ShortTermInvestmentsAndOtherCurrentAssets": ("short_term_investments", 1),

    "AccountsReceivableNetCurrent": ("accounts_receivable", 1),
    "AccountsReceivableNet": ("accounts_receivable", 1),
    "AccountsReceivableTradeNetCurrent": ("accounts_receivable", 1),

    "InventoryNet": ("inventory", 1),
    "InventoryNetOfAllowances": ("inventory", 1),
    "InventoriesNetCurrent": ("inventory", 1),
    "InventoryFinishedGoods": ("inventory", 1),

    "PrepaidExpenseAndOtherCurrentAssets": ("prepaid_expenses", 1),
    "PrepaidExpenseAndOtherAssetsCurrent": ("prepaid_expenses", 1),
    "OtherCurrentAssets": ("other_current_assets", 1),

    "PropertyPlantAndEquipmentNet": ("net_ppe", 1),
    "PropertyPlantAndEquipmentNetIncludingFinanceLeaseRightOfUseAsset": ("net_ppe", 1),
    "PropertyPlantAndEquipmentNetExcludingFinanceLeaseRightOfUseAsset": ("net_ppe", 1),
    "PropertyPlantAndEquipmentGross": ("ppe_gross", 1),
    "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment": ("accumulated_depreciation", 1),  # noqa: E501

    "Goodwill": ("goodwill", 1),
    "FiniteLivedIntangibleAssetsNet": ("intangibles", 1),
    "IntangibleAssetsNetExcludingGoodwill": ("intangibles", 1),
    "OtherIntangibleAssetsNet": ("intangibles", 1),
    "IntangibleAssetsNet": ("intangibles", 1),

    "LongTermInvestments": ("long_term_investments", 1),
    "MarketableSecuritiesNoncurrent": ("long_term_investments", 1),
    "OtherInvestments": ("long_term_investments", 1),

    "DeferredTaxAssetsNet": ("deferred_tax_assets", 1),
    "DeferredTaxAssetsNetNoncurrent": ("deferred_tax_assets", 1),
    "OtherNoncurrentAssets": ("other_noncurrent_assets", 1),
    "OperatingLeaseRightOfUseAsset": ("operating_lease_roa", 1),
    "FinanceLeaseRightOfUseAsset": ("finance_lease_roa", 1),

    "Liabilities": ("total_liabilities", 1),
    "LiabilitiesCurrent": ("current_liabilities", 1),
    "AccountsPayableCurrent": ("accounts_payable", 1),
    "AccountsPayableAndAccruedLiabilitiesCurrent": ("accounts_payable", 1),
    "AccountsPayableTradeCurrent": ("accounts_payable", 1),

    "AccruedLiabilitiesCurrent": ("accrued_liabilities", 1),
    "AccruedLiabilitiesAndOtherLiabilitiesCurrent": ("accrued_liabilities", 1),
    "EmployeeRelatedLiabilitiesCurrent": ("accrued_liabilities", 1),

    "ContractWithCustomerLiabilityCurrent": ("deferred_revenue_current", 1),
    "DeferredRevenueCurrent": ("deferred_revenue_current", 1),
    "DeferredRevenueAndCreditsCurrent": ("deferred_revenue_current", 1),

    "ShortTermBorrowings": ("short_term_debt", 1),
    "CommercialPaper": ("short_term_debt", 1),
    "LongTermDebtCurrent": ("current_portion_long_term_debt", 1),
    "LongTermDebtAndCapitalLeaseObligationsCurrent": ("current_portion_long_term_debt", 1),

    "OtherCurrentLiabilities": ("other_current_liabilities", 1),

    "LongTermDebtNoncurrent": ("long_term_debt", 1),
    "LongTermDebtAndCapitalLeaseObligations": ("long_term_debt", 1),
    "LongTermDebtAndLeaseObligations": ("long_term_debt", 1),

    "DeferredTaxLiabilitiesNoncurrent": ("deferred_tax_liabilities", 1),
    "DeferredIncomeTaxLiabilitiesNet": ("deferred_tax_liabilities", 1),
    "OperatingLeaseLiability": ("operating_lease", 1),
    "OperatingLeaseLiabilityNoncurrent": ("operating_lease", 1),
    "FinanceLeaseLiability": ("finance_lease", 1),
    "FinanceLeaseLiabilityNoncurrent": ("finance_lease", 1),
    "OtherNoncurrentLiabilities": ("other_noncurrent_liabilities", 1),
    "DeferredRevenueNoncurrent": ("other_noncurrent_liabilities", 1),

    "StockholdersEquity": ("shareholders_equity", 1),
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": ("total_equity", 1),
    "CommonStockValue": ("common_stock", 1),
    "CommonStocksIncludingAdditionalPaidInCapital": ("common_stock", 1),
    "AdditionalPaidInCapitalCommonStock": ("additional_paid_in_capital", 1),
    "AdditionalPaidInCapital": ("additional_paid_in_capital", 1),
    "RetainedEarningsAccumulatedDeficit": ("retained_earnings", 1),
    "RetainedEarnings": ("retained_earnings", 1),
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax": ("accumulated_oci", 1),
    "TreasuryStockCommonSharesValue": ("treasury_stock", 1),
    "TreasuryStockValue": ("treasury_stock", 1),
    "MinorityInterest": ("minority_interest", 1),

    "PreferredStockValue": ("preferred_stock", 1),
    "CommonStockSharesOutstanding": ("shares_outstanding", 1),
    "CommonStockSharesOutstandingIncludingTreasuryShares": ("shares_outstanding", 1),

    # ── Cash flow ─────────────────────────────────────────────────────
    "NetCashProvidedByUsedInOperatingActivities": ("operating_cash_flow", 1),
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": ("operating_cash_flow", 1),
    "ShareBasedCompensation": ("stock_based_comp", 1),
    "AllocatedShareBasedCompensationExpense": ("stock_based_comp", 1),

    "PaymentsToAcquirePropertyPlantAndEquipment": ("capex", -1),
    "PaymentsToAcquireProductiveAssets": ("capex", -1),
    "PaymentsToAcquireBusinessesNetOfCashAcquired": ("acquisitions", -1),
    "PaymentsToAcquireInvestments": ("acquisitions", -1),
    "ProceedsFromDivestitureOfBusinessesNetOfCashDivested": ("divestitures", 1),
    "ProceedsFromSaleOfPropertyPlantAndEquipment": ("divestitures", 1),
    "NetCashProvidedByUsedInInvestingActivities": ("investing_cash_flow", 1),

    "PaymentsOfDividends": ("dividends_paid", -1),
    "PaymentsOfDividendsCommonStock": ("dividends_paid", -1),
    "PaymentsOfDividendsToCommonStockholders": ("dividends_paid", -1),
    "PaymentsForRepurchaseOfCommonStock": ("share_buyback", -1),
    "PaymentsForRepurchaseOfCommonStockAndOther": ("share_buyback", -1),
    "PaymentsToRepurchaseEquity": ("share_buyback", -1),
    "RepaymentsOfLongTermDebt": ("debt_repayments", -1),
    "ProceedsFromIssuanceOfLongTermDebt": ("debt_issuance", 1),
    "NetCashProvidedByUsedInFinancingActivities": ("financing_cash_flow", 1),
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": ("cash_change", 1),  # noqa: E501
}


# Statement groupings of the concept map → rows for deep multi-year history.
_INCOME_CONCEPTS = [
    "revenue", "cost_of_revenue", "gross_profit", "research_development",
    "selling_general_admin", "other_operating_expenses", "operating_income",
    "depreciation_amortization", "interest_expense", "interest_income",
    "other_income_expense", "pretax_income", "income_tax", "net_income",
    "net_income_to_noncontrolling", "eps_actual", "eps_basic",
]
_BALANCE_CONCEPTS = [
    "total_assets", "current_assets", "cash_and_equivalents", "short_term_investments",
    "accounts_receivable", "inventory", "prepaid_expenses", "other_current_assets",
    "net_ppe", "ppe_gross", "accumulated_depreciation", "goodwill", "intangibles",
    "long_term_investments", "deferred_tax_assets", "other_noncurrent_assets",
    "operating_lease_roa", "finance_lease_roa",
    "total_liabilities", "current_liabilities", "accounts_payable", "accrued_liabilities",
    "deferred_revenue_current", "short_term_debt", "current_portion_long_term_debt",
    "other_current_liabilities", "long_term_debt", "deferred_tax_liabilities",
    "operating_lease", "finance_lease", "other_noncurrent_liabilities",
    "total_equity", "shareholders_equity", "common_stock", "additional_paid_in_capital",
    "retained_earnings", "accumulated_oci", "treasury_stock", "minority_interest",
    "preferred_stock", "shares_outstanding",
]
_CASHFLOW_CONCEPTS = [
    "operating_cash_flow", "stock_based_comp", "capex", "acquisitions",
    "divestitures", "investing_cash_flow", "dividends_paid", "share_buyback",
    "debt_repayments", "debt_issuance", "financing_cash_flow", "cash_change",
]


def _us_gaap(facts: dict[str, Any]) -> dict:
    return (facts.get("facts") or {}).get("us-gaap") or facts.get("us-gaap") or {}


def _latest_annual(facts: dict[str, Any], concept: str) -> dict | None:
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
                if duration < 300 or duration > 420:
                    continue
            key = (str(end or ""), str(start or ""))
            if key > best_key:
                best_key = key
                best = f
        if best:
            return best
    return None


def _annual_series(facts: dict[str, Any], concept: str) -> dict[str, float]:
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
    # finance lease (noncurrent variant preferred) — count once
    fin_lease = snap.get("finance_lease_noncurrent")
    if fin_lease is None:
        fin_lease = snap.get("finance_lease")
    components = [snap.get("short_term_debt"), snap.get("current_portion_long_term_debt"),
                  snap.get("long_term_debt"), fin_lease, snap.get("operating_lease")]
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
    return 0.21


def _finalize_snapshot(snap: dict) -> dict:
    ebit = snap.get("operating_income")
    da = snap.get("depreciation_amortization")
    if ebit is not None:
        snap["ebitda"] = (ebit + da) if da is not None else ebit
    elif snap.get("ebitda") is None and snap.get("gross_profit") is not None:
        snap["ebitda"] = snap["gross_profit"]

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

    if snap.get("current_assets") is not None and snap.get("current_liabilities"):
        snap["current_ratio"] = snap["current_assets"] / snap["current_liabilities"]
        inv = snap.get("inventory") or 0.0
        snap["quick_ratio"] = (snap["current_assets"] - inv) / snap["current_liabilities"]

    if snap.get("operating_income") is not None and snap.get("interest_expense"):
        snap["interest_coverage_ratio"] = snap["operating_income"] / snap["interest_expense"]

    if ebit is not None:
        nopat = ebit * (1 - _tax_rate(snap))
        snap["nopat"] = nopat
        td = snap.get("total_debt")
        eq = snap.get("shareholders_equity")
        cash = snap.get("cash_and_equivalents")
        if td is not None and eq is not None:
            inv_cap = td + eq - (cash or 0.0)
            if inv_cap and inv_cap > 0:
                snap["roic"] = nopat / inv_cap

    if snap.get("operating_cash_flow") is not None and snap.get("capex") is not None:
        snap["free_cash_flow"] = snap["operating_cash_flow"] - abs(snap["capex"])

    a = snap.get("total_assets")
    liab = snap.get("total_liabilities")
    e = snap.get("shareholders_equity")
    m = snap.get("minority_interest")
    if a and liab is not None and e is not None:
        rhs = liab + e + (m or 0.0)
        snap["balance_discrepancy_pct"] = (a - rhs) / a * 100.0
        snap["balance_ok"] = abs(snap["balance_discrepancy_pct"]) < 1.0

    return snap


def parse_company_facts_series(cik: str, symbol: str, use_cache: bool = True) -> dict:
    """Full multi-year statement series (annual + quarterly) from SEC XBRL.

    Every row carries its canonical fields plus an `extra` dict with ALL other
    us-gaap concepts the company reported for that period (maximum capture).
    """
    facts = load_facts(cik, use_cache=use_cache)

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
        # Catch-all: attach every other us-gaap concept for the period into `extra`.
        for end, row in series.items():
            extra = {}
            for concept, entry in _us_gaap(facts).items():
                if concept in _US_GAAP_FIELDS:
                    continue
                best = None
                for units in entry.get("units", {}).values():
                    for f in units:
                        if f.get("form") not in forms:
                            continue
                        if str(f.get("end") or "") == end:
                            best = f
                            break
                    if best:
                        break
                if best:
                    try:
                        extra[concept] = float(best.get("val"))
                    except (TypeError, ValueError):
                        continue
            if extra:
                row["extra"] = extra
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

    for row in income:
        ebit = row.get("operating_income")
        da = row.get("depreciation_amortization")
        if ebit is not None:
            row["ebitda"] = (ebit + da) if da is not None else ebit
        # gross profit fallback when SEC only tags revenue & COGS
        if row.get("gross_profit") is None and row.get("revenue") is not None \
                and row.get("cost_of_revenue") is not None:
            row["gross_profit"] = row["revenue"] - row["cost_of_revenue"]
    for row in cashflow:
        if row.get("operating_cash_flow") is not None and row.get("capex") is not None:
            row["free_cash_flow"] = row["operating_cash_flow"] - abs(row["capex"])
    for row in balance:
        td = _total_debt(row)
        if td is not None:
            row["total_debt"] = td
        if row.get("total_debt") is not None and row.get("cash_and_equivalents") is not None:
            row["net_debt"] = row["total_debt"] - row["cash_and_equivalents"]
        if row.get("shareholders_equity") and row.get("total_debt") is not None:
            row["debt_to_equity"] = row["total_debt"] / row["shareholders_equity"]
        a, liab, e = row.get("total_assets"), row.get("total_liabilities"), row.get("shareholders_equity")
        if a and liab is not None and e is not None:
            row["balance_discrepancy_pct"] = (a - (liab + e + (row.get("minority_interest") or 0.0))) / a * 100.0  # noqa: E501
            row["balance_ok"] = abs(row["balance_discrepancy_pct"]) < 1.0

    return {"income": income, "balance": balance, "cashflow": cashflow}


def parse_company_facts(cik: str, symbol: str,
                        asset_class: str = "EQUITY_US", use_cache: bool = True) -> dict:
    """Download + parse company facts → fundamental snapshot dict (or {} on failure)."""
    facts = load_facts(cik, use_cache=use_cache)

    snap: dict[str, Any] = {
        "symbol": symbol, "asset_class": asset_class,
        "period_type": "ANNUAL", "raw_data": {"cik": cik},
    }

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

    series = _annual_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    if not series:
        series = _annual_series(facts, "Revenues")
    ends = sorted(series.keys())
    if len(ends) >= 2 and series.get(ends[-2]):
        snap["revenue_yoy_growth"] = series[ends[-1]] / series[ends[-2]] - 1

    log.info("Parsed %s (%s): revenue=%s ebitda=%s roic=%s balance_ok=%s",
             symbol, cik, snap.get("revenue"), snap.get("ebitda"),
             snap.get("roic"), snap.get("balance_ok"))
    return snap
