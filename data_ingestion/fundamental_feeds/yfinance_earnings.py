"""Earnings calendar + surprise fetcher (yfinance)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf

from core.db import Storage, get_storage
from core.logging import get_logger

log = get_logger(__name__)


def fetch_earnings_dates(ticker: str, limit: int = 8,
                         storage: Storage | None = None) -> pd.DataFrame:
    """Pull reported/estimated EPS per earnings date; persist the calendar."""
    storage = storage or get_storage()
    t = yf.Ticker(ticker)
    try:
        dates = t.get_earnings_dates(limit=limit)
    except Exception as e:  # noqa: BLE001
        log.warning("No earnings dates for %s: %s", ticker, e)
        return pd.DataFrame()
    if dates is None or dates.empty:
        return pd.DataFrame()

    storage.write_earnings_dates(ticker.upper(), dates.index)
    out = dates.copy()
    out["symbol"] = ticker.upper()
    out["earnings_date"] = out.index
    if "Reported EPS" in out.columns:
        out["eps_actual"] = pd.to_numeric(out["Reported EPS"], errors="coerce")
    if "EPS Estimate" in out.columns:
        out["eps_estimate"] = pd.to_numeric(out["EPS Estimate"], errors="coerce")
    return out


def earnings_surprise(ticker: str) -> dict | None:
    """Latest quarter surprise metrics, or None when unavailable."""
    t = yf.Ticker(ticker)
    try:
        dates = t.get_earnings_dates(limit=4)
    except Exception:  # noqa: BLE001
        return None
    if dates is None or dates.empty:
        return None
    last = dates.iloc[-1]
    try:
        actual = float(last.get("Reported EPS") or float("nan"))
        estimate = float(last.get("EPS Estimate") or float("nan"))
    except (TypeError, ValueError):
        return None
    if pd.isna(actual) or pd.isna(estimate) or estimate == 0:
        return None
    return {
        "eps_actual": actual,
        "eps_estimate": estimate,
        "eps_surprise_pct": actual / estimate - 1,
        "earnings_date": str(dates.index[-1].date()),
    }


def refresh_info_snapshot(ticker: str, storage: Storage | None = None) -> dict | None:
    """Best-effort fundamentals snapshot from yfinance .info (used by screener)."""
    storage = storage or get_storage()
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception as e:  # noqa: BLE001
        log.debug("info failed for %s: %s", ticker, e)
        return None
    if not info:
        return None
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    snap = {
        "symbol": ticker.upper(), "asset_class": "EQUITY_US",
        "period_type": "SNAPSHOT",
        "time": dt.datetime.utcnow(),
        "revenue": info.get("totalRevenue"),
        "gross_profit": info.get("grossProfits"),
        "ebitda": info.get("ebitda"),
        "net_income": info.get("netIncomeToCommon"),
        "eps_actual": info.get("trailingEps"),
        "eps_yoy_growth": info.get("earningsGrowth"),
        "revenue_yoy_growth": info.get("revenueGrowth"),
        "total_debt": info.get("totalDebt"),
        "cash_and_equivalents": info.get("totalCash"),
        "shareholders_equity": info.get("totalStockholderEquity"),
        "current_ratio": info.get("currentRatio"),
        "free_cash_flow": info.get("freeCashflow"),
        "roic": info.get("returnOnCapital"),
        "gross_margin": info.get("grossMargins"),
        "ebitda_margin": info.get("ebitdaMargins"),
        "fcf_yield": (info.get("freeCashflow") / info.get("marketCap"))
                     if info.get("freeCashflow") is not None and info.get("marketCap") else None,
        "market_cap": info.get("marketCap"),
        "current_price": price,
        "forward_pe": info.get("forwardPE"),
        "forward_eps": info.get("forwardEps"),
        "peg_ratio": info.get("pegRatio"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "debt_to_equity": info.get("debtToEquity"),
        "insider_buy_value": info.get("insiderBuyShares") or info.get("insiderPurchases"),
        "insider_sell_value": info.get("insiderSellShares") or info.get("insiderTransactions"),
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "recommendation": info.get("recommendationKey"),
        "number_of_analysts": info.get("numberOfAnalystOpinions"),
        "revenue_estimate": (info.get("revenueEstimate") or {}).get("avg")
        if isinstance(info.get("revenueEstimate"), dict) else None,
    }
    snap = {k: v for k, v in snap.items() if v is not None}
    if snap.get("total_debt") is not None and snap.get("cash_and_equivalents") is not None:
        snap["net_debt"] = snap["total_debt"] - snap["cash_and_equivalents"]
    if snap.get("market_cap") and snap.get("current_price"):
        try:
            shares = snap["market_cap"] / snap["current_price"]
            if snap.get("free_cash_flow") is not None and shares:
                snap["fcf_yield"] = snap["free_cash_flow"] / snap["market_cap"]
        except Exception:  # noqa: BLE001
            pass
    storage.upsert_fundamental_snapshot(snap)
    return snap


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        print(t, refresh_info_snapshot(t))
