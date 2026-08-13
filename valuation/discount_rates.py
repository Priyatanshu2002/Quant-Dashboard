"""Pro-grade discount-rate engine — ERP, country risk, cost of debt, WACC.

Grounded in the Damodaran valuation datasets (ERPMarch26, ctrypremJuly25):
  * Mature-market equity risk premium (July-2025) = 4.21%.
  * Per-country ERP = mature ERP + (country default spread × relative equity
    volatility 1.5046). e.g. US 4.62%, India 7.46%, China 5.25%, Brazil 7.91%.
  * Rating → default spread table (Moody's rating → bps).
  * Synthetic credit rating from interest coverage (EBIT/Interest) → default
    spread → pre-tax cost of debt = rf + (2/3·country spread) + firm spread.
  * WACC = Ke·(E/(D+E)) + Kd·(1−t)·(D/(D+E)) at market-value weights.

Everything degrades gracefully: missing inputs fall back to documented defaults
rather than raising. All assumptions are exposed in the output.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Market data (July-2025 Damodaran) ──────────────────────────────────
MATURE_ERP = 0.0421            # mature-market premium, July 2025
REL_EQ_VOL = 1.5046            # relative equity volatility multiplier (std-equity/std-bond)
DEFAULT_RISK_FREE_RATE = 0.0395  # US 10Y ~3.95% (ERPMarch26)
DEFAULT_BETA = 1.0
DEFAULT_WACC = 0.10

# ERP presets (ERPMarch26.xlsx / Damodaran presets)
ERP_PRESETS = {
    "mature": MATURE_ERP,
    "us_implied": 0.0437,        # S&P implied ERP (US Treasury rf)
    "us_implied_adj_rf": 0.0460, # implied ERP w/ default-risk-adjusted rf
    "historical_us": 0.0544,
    "global": 0.0320,
    "avg_implied_decade": 0.0519,
    "longterm_avg": 0.0425,      # 1960-current avg
    "high": 0.0645,
    "low": 0.0205,
}

# Country → default rating + ERP. ERP = MATURE + (default_spread × REL_EQ_VOL).
# Default spreads from ctrypremJuly25 (bps) → country ERP computed.
_COUNTRY = {
    # name: (moody_rating)
    "US": "Aa1", "UNITED STATES": "Aa1", "USA": "Aa1",
    "INDIA": "Baa3", "CHINA": "A1", "BRAZIL": "Ba1",
    "GERMANY": "Aaa", "AUSTRALIA": "Aaa", "SWITZERLAND": "Aaa",
    "NORDICS": "Aaa", "CANADA": "Aaa", "SINGAPORE": "Aaa",
    "UNITED KINGDOM": "Aa3", "UK": "Aa3", "FRANCE": "Aa3",
    "JAPAN": "A1", "MEXICO": "Baa2", "SOUTH AFRICA": "Ba2",
    "KOREA": "Aa2", "TAIWAN": "Aa3", "NETHERLANDS": "Aaa",
    "ITALY": "Baa3", "SPAIN": "Baa1", "RUSSIA": "Caa1",
}

# Moody's rating → default spread in basis points (ctrypremJuly25).
RATING_SPREAD_BPS = {
    "Aaa": 0, "Aa1": 27, "Aa2": 48.5, "Aa3": 58.9,
    "A1": 69.3, "A2": 83.2, "A3": 117.9,
    "Baa1": 157.2, "Baa2": 187.2, "Baa3": 216.1,
    "Ba1": 246.1, "Ba2": 295.8, "Ba3": 353.6,
    "B1": 442.6, "B2": 540.8, "B3": 639.1,
    "Caa1": 737.3, "Caa2": 885.2, "Caa3": 983.4,
    "Ca": 1179.9, "C": 1750.0,
}

# Synthetic credit rating from interest coverage (EBIT/Interest) for large
# firms — Damodaran synthetic-rating table → rating + firm default spread (bps).
_SYNTHETIC_RATING = [
    (8.50, "Aaa/AAA", 75), (6.50, "Aa2/AA", 100), (5.50, "A1/A+", 125),
    (4.25, "A2/A", 150), (3.00, "A3/A-", 175), (2.50, "Baa2/BBB", 225),
    (2.25, "Ba1/BB+", 275), (2.00, "Ba2/BB", 325), (1.75, "B1/B+", 425),
    (1.50, "B2/B", 525), (1.25, "B3/B-", 625), (0.80, "Caa/Ca", 950),
    (0.65, "Ca2/CC", 1200), (0.20, "C", 1600), (0.0, "D", 2000),
]


def default_spread_for_rating(rating: str) -> float:
    """Moody's rating → default spread (decimal). Unknown → A3 (117.9bps)."""
    r = (rating or "").upper().split("/")[0].strip()
    bps = RATING_SPREAD_BPS.get(r, 117.9)
    return bps / 10_000.0


def synthetic_rating(interest_coverage: float) -> tuple[str, float]:
    """Interest coverage → (rating, firm default spread as decimal)."""
    if interest_coverage is None or interest_coverage <= 0:
        return "D", 0.20
    for threshold, rating, bps in _SYNTHETIC_RATING:
        if interest_coverage >= threshold:
            return rating, bps / 10_000.0
    return "D", 0.20


def country_default_spread(country: str | None) -> float:
    """Country → default spread (decimal). Mature/Aaa countries → 0."""
    if not country:
        return 0.0
    rating = _COUNTRY.get(country.upper().replace(" ", "").strip(), "Aaa")
    return default_spread_for_rating(rating)


def country_erp(country: str | None, mature_erp: float = MATURE_ERP) -> float:
    """Country equity risk premium = mature ERP + (spread × rel equity vol)."""
    return mature_erp + country_default_spread(country) * REL_EQ_VOL


def risk_free_for_currency(currency: str | None, base_rf: float | None,
                           country: str | None = None) -> float:
    """US 10Y by default; for non-USD sovereigns, rf = local bond rate −
    country default spread (approximation using base rf)."""
    rf = base_rf if base_rf is not None else DEFAULT_RISK_FREE_RATE
    return rf


@dataclass
class DiscountRates:
    risk_free: float
    erp: float
    beta: float
    cost_of_equity: float
    cost_of_debt: float
    cost_of_debt_after_tax: float
    tax_rate: float
    equity_value: float | None
    debt_value: float | None
    w_e: float | None
    w_d: float | None
    wacc: float
    country: str | None
    synthetic_rating: str | None
    erp_source: str
    cost_of_debt_source: str
    coverage: str  # full | partial | default


def estimate_discount_rates(
    snap: dict,
    beta: float | None = None,
    rf: float | None = None,
    erp: float | None = None,
    country: str | None = None,
    rating: str | None = None,
    mature_erp: float = MATURE_ERP,
) -> DiscountRates:
    """Estimate full discount-rate stack (Ke, Kd, WACC) for a snapshot.

    Inputs drawn from the snapshot where present: market_cap, total_debt,
    interest_expense, interest_coverage_ratio, pretax_income, income_tax.
    Country/rating/erp overrides come from the caller (profile data).
    """
    rf = rf if rf is not None else DEFAULT_RISK_FREE_RATE
    beta = beta if beta is not None else DEFAULT_BETA

    # ERP: country-aware when we know the country, else caller-supplied or mature
    erp_source = "mature"
    if erp is None:
        erp = country_erp(country, mature_erp) if country else mature_erp
        erp_source = f"country({country})" if country else "mature"
    cost_of_equity = rf + beta * erp

    # Tax rate from the statements
    pretax, tax = snap.get("pretax_income"), snap.get("income_tax")
    tax_rate = 0.21
    if pretax and tax is not None:
        try:
            t = tax / pretax
            if 0 <= t <= 1:
                tax_rate = float(t)
        except (TypeError, ZeroDivisionError):
            pass

    # Cost of debt: rating > synthetic-from-IC > effective-rate > rf+spread
    td = snap.get("total_debt")
    ic = snap.get("interest_coverage_ratio")
    if rating:
        firm_spread = default_spread_for_rating(rating)
        rating_out, cost_debt_source = rating, "rating"
    elif ic and ic > 0:
        rating_out, firm_spread = synthetic_rating(ic)
        cost_debt_source = f"synthetic({rating_out})"
    else:
        ie = snap.get("interest_expense")
        if ie and td and td > 0:
            rating_out, cost_debt_source = None, "effective-interest"
            firm_spread = ie / td  # pre-tax effective rate used directly below
        else:
            rating_out, cost_debt_source = None, "rf-plus-spread"
            firm_spread = 0.015

    if cost_debt_source == "effective-interest":
        cost_of_debt = firm_spread
    else:
        country_spread = country_default_spread(country)
        cost_of_debt = rf + (2.0 / 3.0) * country_spread + firm_spread
    cost_of_debt_after_tax = cost_of_debt * (1 - tax_rate)

    # Market-value weights
    mcap = snap.get("market_cap")
    equity_value = float(mcap) if mcap else None
    debt_value = float(td) if td is not None else None
    if equity_value is not None and debt_value is not None and (equity_value + debt_value) > 0:
        w_e = equity_value / (equity_value + debt_value)
        w_d = debt_value / (equity_value + debt_value)
        wacc = w_e * cost_of_equity + w_d * cost_of_debt_after_tax
        coverage = "full"
    else:
        w_e = w_d = None
        wacc = DEFAULT_WACC
        coverage = "default"

    return DiscountRates(
        risk_free=rf, erp=erp, beta=beta, cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt, cost_of_debt_after_tax=cost_of_debt_after_tax,
        tax_rate=tax_rate, equity_value=equity_value, debt_value=debt_value,
        w_e=w_e, w_d=w_d, wacc=wacc, country=country,
        synthetic_rating=rating_out, erp_source=erp_source,
        cost_of_debt_source=cost_debt_source, coverage=coverage,
    )
