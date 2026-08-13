"""Quarterly 3-statement fetcher (plan §8) — yfinance quarterly statements.

Fetches 8 quarters of income statement, balance sheet and cash flow for a
ticker, derives per-period ratios (margins, YoY growth, D/E, current ratio,
FCF conversion), and persists them to the `financial_statements` store
(plan §8.4: 3-Statement + DCF dashboard with 8-quarter trends).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from core.db import Storage, get_storage
from core.logging import get_logger

log = get_logger(__name__)

# yfinance line-item label → our canonical key. Keys are tried in order,
# because yfinance labels shift slightly between API revisions.
_INCOME_MAP = [
    ("total_revenue", ("Total Revenue", "Operating Revenue", "Revenue")),
    ("cost_of_revenue", ("Cost Of Revenue", "Cost of Revenue")),
    ("gross_profit", ("Gross Profit",)),
    ("research_development", ("Research And Development", "Research And Development Expenses")),
    ("selling_general_admin", ("Selling General And Administration", "Selling General And Administrative")),
    ("total_operating_expenses", ("Total Operating Expenses",)),
    ("operating_income", ("Operating Income",)),
    ("ebitda", ("EBITDA",)),
    ("interest_expense", ("Interest Expense", "Interest Expense Non Operating")),
    ("interest_income", ("Interest Income", "Interest Income Non Operating", "Interest Income Operating")),
    ("other_income_expense", ("Other Income Expense", "Other Non Operating Income Expenses")),
    ("pretax_income", ("Pretax Income", "Pretax Income Before Non-Recurring Items")),
    ("income_tax", ("Tax Provision", "Income Tax Expense Benefit")),
    ("net_income", ("Net Income", "Net Income Common Stockholders")),
    ("eps_diluted", ("Diluted EPS",)),
    ("eps_basic", ("Basic EPS",)),
    ("shares_outstanding", ("Diluted Average Shares", "Basic Average Shares")),
]

_BALANCE_MAP = [
    ("total_assets", ("Total Assets",)),
    ("current_assets", ("Total Current Assets",)),
    ("cash_and_equivalents", ("Cash And Cash Equivalents",
                              "Cash Cash Equivalents And Short Term Investments")),
    ("short_term_investments", ("Short Term Investments", "Short Term Investments And Other")),
    ("accounts_receivable", ("Accounts Receivable",
                             "Accounts Receivable Trade Net")),
    ("inventory", ("Inventory",)),
    ("net_ppe", ("Net PPE", "Net Property Plant And Equipment")),
    ("goodwill", ("Goodwill",)),
    ("intangibles", ("Goodwill And Other Intangible Assets", "Other Intangible Assets", "Intangible Assets")),
    ("deferred_tax_assets", ("Deferred Tax Assets",)),
    ("total_noncurrent_assets", ("Total Non Current Assets", "Non Current Assets Total")),
    ("total_liabilities", ("Total Liabilities Net Minority Interest", "Total Liabilities")),
    ("current_liabilities", ("Total Current Liabilities",)),
    ("accounts_payable", ("Accounts Payable", "Payables And Accrued Expenses")),
    ("long_term_debt", ("Long Term Debt",)),
    ("total_debt", ("Total Debt",)),
    ("total_noncurrent_liabilities",
     ("Total Non Current Liabilities Net Minority Interest", "Non Current Liabilities Total")),
    ("shareholders_equity", ("Stockholders Equity", "Total Equity Gross Minority Interest")),
    ("retained_earnings", ("Retained Earnings",)),
    ("additional_paid_in_capital", ("Additional Paid In Capital", "Common Stock Additional Paid In Capital")),
    ("accumulated_oci", ("Accumulated Other Comprehensive Income",)),
    ("treasury_stock", ("Treasury Stock",)),
    ("common_stock", ("Common Stock Equity", "Total Common Shares Outstanding")),
]

_CASHFLOW_MAP = [
    ("net_income", ("Net Income", "Net Income Common Stockholders")),
    ("operating_cash_flow", ("Operating Cash Flow",)),
    ("depreciation", ("Depreciation Amortization Depletion", "Depreciation And Amortization")),
    ("stock_based_comp", ("Stock Based Compensation",)),
    ("change_in_working_capital", ("Change In Working Capital", "Changes In Working Capital")),
    ("capex", ("Capital Expenditure", "Investments In Property Plant And Equipment")),
    ("free_cash_flow", ("Free Cash Flow",)),
    ("acquisitions", ("Acquisition Of Businesses", "Cash Acquisitions")),
    ("divestitures", ("Sale Of Business", "Proceeds From Divestiture Of Businesses")),
    ("debt_repayments", ("Repayment Of Debt",)),
    ("debt_issuance", ("Issuance Of Debt", "Proceeds From Issuance Of Debt")),
    ("investing_cash_flow", ("Investing Cash Flow", "Cash Flow From Continuing Investing Activities")),
    ("financing_cash_flow", ("Financing Cash Flow", "Cash Flow From Continuing Financing Activities")),
    ("dividends_paid", ("Common Stock Dividend Paid", "Dividends Paid")),
    ("share_buyback", ("Repurchase Of Capital Stock", "Common Stock Repurchased")),
]

STATEMENTS = {"income": _INCOME_MAP, "balance": _BALANCE_MAP, "cashflow": _CASHFLOW_MAP}


def _extract(frame: pd.DataFrame, mapping: list[tuple[str, tuple[str, ...]]]) -> dict:
    """Map yfinance rows → {key: value} for the latest period columns."""
    if frame is None or frame.empty:
        return {}
    out: dict = {}
    for key, labels in mapping:
        for label in labels:
            if label in frame.index:
                val = frame.loc[label]
                val = val.dropna()
                if not val.empty:
                    out[key] = float(val.iloc[0])
                    break
    return out


def _period_rows(ticker: str, statement: str, mapping: list,
                 periods: int = 8, annual: bool = False) -> list[dict]:
    """One normalized row per period end (most recent first → chronological).

    `annual` selects the annual statements; otherwise quarterly.
    """
    t = yf.Ticker(ticker)
    if statement == "income":
        frame = t.income_stmt if annual else t.quarterly_income_stmt
    elif statement == "balance":
        frame = t.balance_sheet if annual else t.quarterly_balance_sheet
    else:
        frame = t.cashflow if annual else t.quarterly_cashflow
    if frame is None or frame.empty:
        return []
    ptype = "ANNUAL" if annual else "QUARTERLY"
    rows = []
    for period_end in list(frame.columns)[:periods]:
        col = frame[period_end]
        data = {}
        for key, labels in mapping:
            for label in labels:
                if label in frame.index:
                    val = col.get(label)
                    if pd.notna(val):
                        try:
                            data[key] = float(val)
                        except (TypeError, ValueError):
                            pass
                        break
        if data:
            rows.append({"period": str(pd.Timestamp(period_end).date()),
                         "period_type": ptype, **data})
    return rows


def derive_ratios(income: list[dict], balance: list[dict],
                  cashflow: list[dict]) -> list[dict]:
    """Merge the three statements into per-period ratio rows (oldest→newest)."""
    merged: dict[str, dict] = {}
    for rows in (income, balance, cashflow):
        for r in rows:
            merged.setdefault(r["period"], {}).update(r)

    # YoY growth needs the prior-year quarter — sort, then pair by period offset.
    periods = sorted(merged)
    by_key: dict[str, dict] = {}
    for i, p in enumerate(periods):
        cur = merged[p]
        prior = merged[periods[i - 4]] if i >= 4 else None
        rev, rev_prior = cur.get("total_revenue"), prior.get("total_revenue") if prior else None
        eps, eps_prior = cur.get("eps_diluted"), prior.get("eps_diluted") if prior else None
        ocf = cur.get("operating_cash_flow")

        row: dict = {"period": p, "period_type": "QUARTERLY"}
        row["revenue_yoy_growth"] = (rev / rev_prior - 1) if rev and rev_prior else None
        row["eps_yoy_growth"] = (eps / eps_prior - 1) if eps and eps_prior else None
        row["gross_margin"] = (cur.get("gross_profit") / rev) if cur.get("gross_profit") and rev else None
        row["ebitda_margin"] = (cur.get("ebitda") / rev) if cur.get("ebitda") and rev else None
        row["net_margin"] = (cur.get("net_income") / rev) if cur.get("net_income") and rev else None
        de = (cur.get("total_debt") / cur.get("shareholders_equity")
              if cur.get("total_debt") is not None and cur.get("shareholders_equity") else None)
        row["debt_to_equity"] = de
        row["current_ratio"] = (cur.get("current_assets") / cur.get("current_liabilities")
                                if cur.get("current_assets") is not None
                                and cur.get("current_liabilities") else None)
        row["interest_coverage_ratio"] = (
            cur.get("operating_income") / cur.get("interest_expense")
            if cur.get("operating_income") is not None and cur.get("interest_expense") else None)
        row["net_debt"] = (cur.get("total_debt") - cur.get("cash_and_equivalents")
                           if cur.get("total_debt") is not None
                           and cur.get("cash_and_equivalents") is not None else None)
        row["free_cash_flow"] = cur.get("free_cash_flow")
        row["fcf_conversion"] = (cur.get("free_cash_flow") / ocf) if ocf else None
        row["roic"] = (cur.get("net_income") / cur.get("total_assets"))
        by_key[p] = row
    return [by_key[p] for p in periods]


def fetch_quarterly_statements(ticker: str, quarters: int = 8,
                               storage: Storage | None = None) -> dict:
    """Fetch + persist quarterly statements; return {income, balance, cashflow, ratios}."""
    storage = storage or get_storage()
    symbol = ticker.upper()
    income = _period_rows(ticker, "income", _INCOME_MAP, quarters, annual=False)
    balance = _period_rows(ticker, "balance", _BALANCE_MAP, quarters, annual=False)
    cashflow = _period_rows(ticker, "cashflow", _CASHFLOW_MAP, quarters, annual=False)

    if income:
        storage.write_financial_statements(symbol, "income", income)
    if balance:
        storage.write_financial_statements(symbol, "balance", balance)
    if cashflow:
        storage.write_financial_statements(symbol, "cashflow", cashflow)

    ratios = derive_ratios(income, balance, cashflow)
    log.info("3-statement fetch %s: income=%d balance=%d cashflow=%d",
             symbol, len(income), len(balance), len(cashflow))
    return {
        "income": income, "balance": balance,
        "cashflow": cashflow, "ratios": ratios,
    }


def fetch_annual_statements(ticker: str, years: int = 10,
                            storage: Storage | None = None) -> dict:
    """Fetch + persist ANNUAL statements (yfinance income_stmt/balance_sheet/cashflow)."""
    storage = storage or get_storage()
    symbol = ticker.upper()
    income = _period_rows(ticker, "income", _INCOME_MAP, years, annual=True)
    balance = _period_rows(ticker, "balance", _BALANCE_MAP, years, annual=True)
    cashflow = _period_rows(ticker, "cashflow", _CASHFLOW_MAP, years, annual=True)
    if income:
        storage.write_financial_statements(symbol, "income", income)
    if balance:
        storage.write_financial_statements(symbol, "balance", balance)
    if cashflow:
        storage.write_financial_statements(symbol, "cashflow", cashflow)
    log.info("Annual 3-statement fetch %s: income=%d balance=%d cashflow=%d",
             symbol, len(income), len(balance), len(cashflow))
    return {"income": income, "balance": balance, "cashflow": cashflow}


def fetch_company_profile(ticker: str, storage: Storage | None = None) -> dict | None:
    """Best-effort company meta from yfinance .info (maximised capture)."""
    storage = storage or get_storage()
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:  # noqa: BLE001
        log.debug("profile info failed for %s: %s", ticker, e)
        return None
    if not info:
        return None

    def _num(k):
        v = info.get(k)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    meta = {
        "company_name": info.get("longName") or info.get("shortName"),
        "description": info.get("longBusinessSummary"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "market_cap": _num("marketCap"),
        "shares_outstanding": _num("sharesOutstanding") or _num("sharesOutstandingRaw"),
        "shares_float": _num("floatShares"),
        "beta": _num("beta"),
        "trailing_pe": _num("trailingPE"),
        "forward_pe": _num("forwardPE"),
        "peg_ratio": _num("pegRatio"),
        "price_to_book": _num("priceToBook"),
        "price_to_sales": _num("priceToSalesTrailing12Months"),
        "ev_to_ebitda": _num("enterpriseToEbitda"),
        "ev_to_revenue": _num("enterpriseToRevenue"),
        "dividend_yield": _num("dividendYield") or _num("trailingAnnualDividendYield"),
        "dividend_rate": _num("dividendRate"),
        "payout_ratio": _num("payoutRatio"),
        "institutional_ownership": _num("heldPercentInstitutions"),
        "insider_ownership": _num("heldPercentInsiders"),
        "short_percent_of_float": _num("shortPercentOfFloat"),
        "recommendation_key": info.get("recommendationKey"),
        "target_mean_price": _num("targetMeanPrice"),
        "target_high_price": _num("targetHighPrice"),
        "target_low_price": _num("targetLowPrice"),
        "number_of_analysts": _num("numberOfAnalystOpinions"),
        "revenue_growth": _num("revenueGrowth"),
        "earnings_growth": _num("earningsGrowth"),
        "return_on_equity": _num("returnOnEquity"),
        "return_on_assets": _num("returnOnAssets"),
        "profit_margins": _num("profitMargins"),
        "gross_margins": _num("grossMargins"),
        "operating_margins": _num("operatingMargins"),
        "free_cash_flow": _num("freeCashflow"),
        "operating_cash_flow": _num("operatingCashflow"),
        "total_debt": _num("totalDebt"),
        "total_cash": _num("totalCash"),
        "current_ratio": _num("currentRatio"),
        "debt_to_equity": _num("debtToEquity"),
        "book_value": _num("bookValue"),
        "fifty_two_week_high": _num("fiftyTwoWeekHigh"),
        "fifty_two_week_low": _num("fiftyTwoWeekLow"),
        "average_volume": _num("averageVolume"),
        "market": info.get("market"),
        "symbol": ticker.upper(),
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    if meta:
        storage.upsert_company_profile(ticker.upper(), meta)
    return meta


if __name__ == "__main__":
    import json
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        res = fetch_quarterly_statements(t)
        print(f"{t}: income={len(res['income'])} balance={len(res['balance'])} "
              f"cashflow={len(res['cashflow'])} ratios={len(res['ratios'])}")
        for r in res["ratios"][-3:]:
            print(" ", json.dumps(r, default=str)[:200])
