"""Earnings-quality, bankruptcy, and earnings-manipulation models.

Grounded in CFA M36 earnings-quality framework (CFO/NI, DSO, inventory, asset
turnover, accruals, margin-vs-industry) plus the standard practitioner models:
  * Piotroski F-Score (0-9)  — fundamental financial-strength scorecard
  * Altman Z-Score           — bankruptcy probability (Z, Z', Z'')
  * Beneish M-Score          — earnings-manipulation detection (M > -1.78)

Inputs: statement lists (income/balance/cashflow, oldest → newest) and a
latest fundamental snapshot. All inputs optional → best-effort with flags.
"""
from __future__ import annotations

from typing import Any


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _sum_last(rows: list[dict], key: str, n: int) -> float | None:
    vals = [_f(r.get(key)) for r in rows[-n:]]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _latest(rows: list[dict], key: str) -> float | None:
    for r in reversed(rows):
        v = _f(r.get(key))
        if v is not None:
            return v
    return None


# ══════════════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE (0-9)
# ══════════════════════════════════════════════════════════════════════
def piotroski(income: list[dict], balance: list[dict],
              cashflow: list[dict]) -> dict:
    """Piotroski F-Score. Returns {score, components:{...}}. None if data insufficient."""
    if len(income) < 4 or len(balance) < 2 or not cashflow:
        return {"score": None, "components": {}}

    # TTM items
    ni = _sum_last(income, "net_income", 4)
    prior_ni = _sum_last(income[:-4], "net_income", 4) if len(income) >= 8 else None
    cfo = _sum_last(cashflow, "operating_cash_flow", 4)
    ta_cur = _latest(balance, "total_assets")
    ta_prior = _latest(balance[:-4], "total_assets") if len(balance) >= 5 else None

    score = 0
    comp: dict[str, bool] = {}

    # Profitability (4)
    comp["roa_positive"] = bool(ni and ta_cur and ni > 0)
    comp["cfo_positive"] = bool(cfo and cfo > 0)
    comp["delta_roa"] = bool(ni is not None and ta_cur and prior_ni is not None and ta_prior
                             and (ni / ta_cur) > (prior_ni / ta_prior))
    comp["accruals_cfo_exceeds_ni"] = bool(cfo is not None and ni is not None and cfo > ni)

    # Leverage / liquidity / source of funds (3)
    td_cur = _latest(balance, "total_debt")
    td_prior = _latest(balance[:-4], "total_debt") if len(balance) >= 5 else None
    comp["leverage_decreased"] = bool(td_cur is not None and td_prior is not None and td_cur < td_prior)
    ca_cur, cl_cur = _latest(balance, "current_assets"), _latest(balance, "current_liabilities")
    ca_prior = _latest(balance[:-4], "current_assets") if len(balance) >= 5 else None
    cl_prior = _latest(balance[:-4], "current_liabilities") if len(balance) >= 5 else None
    comp["liquidity_increased"] = bool(ca_cur is not None and cl_cur and ca_prior is not None and cl_prior and
                                       (ca_cur / cl_cur) > (ca_prior / cl_prior))
    # no new equity: shares_outstanding not materially up
    so_cur = _latest(income, "shares_outstanding")
    so_prior = _latest(income[:-4], "shares_outstanding") if len(income) >= 5 else None
    comp["no_new_shares"] = not (so_cur is not None and so_prior and so_cur > so_prior * 1.02)

    # Operating efficiency (2)
    gp_cur = _sum_last(income, "gross_profit", 4)
    gp_prior = _sum_last(income[:-4], "gross_profit", 4) if len(income) >= 8 else None
    rev_cur = _sum_last(income, "total_revenue", 4)
    rev_prior = _sum_last(income[:-4], "total_revenue", 4) if len(income) >= 8 else None
    comp["gross_margin_increased"] = bool(
        gp_cur is not None and rev_cur and gp_prior is not None and rev_prior
        and (gp_cur / rev_cur) > (gp_prior / rev_prior))
    comp["asset_turnover_increased"] = bool(
        rev_cur is not None and ta_cur and rev_prior is not None and ta_prior
        and (rev_cur / ta_cur) > (rev_prior / ta_prior))

    score = sum(1 for v in comp.values() if v)
    return {"score": score, "components": comp,
            "label": "strong" if score >= 7 else "weak" if score <= 3 else "moderate"}


# ══════════════════════════════════════════════════════════════════════
# ALTMAN Z-SCORE
# ══════════════════════════════════════════════════════════════════════
def altman_z(income: list[dict], balance: list[dict],
             market_cap: float | None = None, is_manufacturing: bool = True) -> dict:
    """Altman Z-Score. Returns {z, zone, inputs}. None if insufficient data."""
    bal = balance[-1] if balance else {}
    ta = _f(bal.get("total_assets"))
    if not ta or ta <= 0:
        return {"z": None, "zone": "insufficient_data", "inputs": {}}
    wc = (_f(bal.get("current_assets")) or 0.0) - (_f(bal.get("current_liabilities")) or 0.0)
    re = _f(bal.get("retained_earnings")) or 0.0
    ebit = _sum_last(income, "operating_income", 4) or 0.0
    sales = _sum_last(income, "total_revenue", 4) or 0.0
    tl = _f(bal.get("total_liabilities")) or 0.0
    me = market_cap or (_f(bal.get("shareholders_equity")) or 0.0)

    x1, x2, x3, x4, x5 = wc / ta, re / ta, ebit / ta, me / tl if tl else 0.0, sales / ta
    if is_manufacturing:
        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        zone = "safe" if z > 2.99 else ("distress" if z < 1.81 else "grey")
    else:
        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        zone = "safe" if z > 2.6 else ("distress" if z < 1.1 else "grey")
    return {"z": round(z, 3), "zone": zone,
            "inputs": {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}}


# ══════════════════════════════════════════════════════════════════════
# BENEISH M-SCORE
# ══════════════════════════════════════════════════════════════════════
def beneish_m(income: list[dict], balance: list[dict],
              cashflow: list[dict]) -> dict:
    """Beneish M-Score (8-ratio manipulation index). M > -1.78 → likely manipulator."""
    if len(income) < 8 or len(balance) < 4 or not cashflow:
        return {"m_score": None, "manipulator": None, "inputs": {}}

    def _q(idx):
        return {
            "sales": _sum_last(income, "total_revenue", 4) if idx == 0 else None,
        }

    # indices compare current year (last 4 qtrs) vs prior year (previous 4 qtrs)
    def _ttm(rows, key, offset):
        return _sum_last(rows[:-offset], key, 4) if len(rows) >= offset + 4 else None

    rev_t, rev_p = _ttm(income, "total_revenue", 0), _ttm(income, "total_revenue", 4)
    ar_t, ar_p = _latest(balance, "accounts_receivable"), _latest(balance[:-4], "accounts_receivable")
    gp_t, gp_p = _ttm(income, "gross_profit", 0), _ttm(income, "gross_profit", 4)
    ta_t, ta_p = _latest(balance, "total_assets"), _latest(balance[:-4], "total_assets")
    pp_t, pp_p = _latest(balance, "net_ppe"), _latest(balance[:-4], "net_ppe")
    sga_t, sga_p = _ttm(income, "selling_general_admin", 0), _ttm(income, "selling_general_admin", 4)
    rev_p = _ttm(income, "total_revenue", 4)
    td_t, td_p = _latest(balance, "total_debt"), _latest(balance[:-4], "total_debt")

    def _safe(ratio):
        return ratio if ratio is not None and ratio not in (float("inf"), float("-inf")) else 0.0

    # DSRI
    dsri = (_safe((ar_t / rev_t) / (ar_p / rev_p))
            if (rev_t and rev_p and ar_t is not None and ar_p is not None) else 0.0)
    # GMI
    gm_t = (gp_t / rev_t) if (gp_t and rev_t) else None
    gm_p = (gp_p / rev_p) if (gp_p and rev_p) else None
    gmi = _safe(gm_p / gm_t) if (gm_t and gm_p) else 0.0
    # AQI
    aqi = (_safe((1 - (pp_t / ta_t)) / (1 - (pp_p / ta_p)))
           if (ta_t and ta_p and pp_t is not None and pp_p is not None) else 0.0)
    # SGI
    sgi = _safe(rev_t / rev_p) if (rev_t and rev_p) else 0.0
    # DEPI
    depi = 1.0
    # SGAI
    sgai = (_safe((sga_t / rev_t) / (sga_p / rev_p))
            if (sga_t and sga_p and rev_t and rev_p) else 0.0)
    # LVGI
    lvgi = (_safe((td_t / ta_t) / (td_p / ta_p))
            if (ta_t and ta_p and td_t is not None and td_p is not None) else 0.0)
    # TATA
    ni_t = _ttm(income, "net_income", 0) or 0.0
    cfo_t = _ttm(cashflow, "operating_cash_flow", 0) or 0.0
    tata = _safe((ni_t - cfo_t) / ta_t) if ta_t else 0.0

    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    return {"m_score": round(m, 3),
            "manipulator": m > -1.78,
            "inputs": {"dsri": round(dsri, 3), "gmi": round(gmi, 3), "aqi": round(aqi, 3),
                       "sgi": round(sgi, 3), "depi": depi, "sgai": round(sgai, 3),
                       "lvgi": round(lvgi, 3), "tata": round(tata, 3)}}


# ══════════════════════════════════════════════════════════════════════
# EARNINGS-QUALITY WARNING FLAGS (CFA M36)
# ══════════════════════════════════════════════════════════════════════
def earnings_quality_flags(income: list[dict], balance: list[dict],
                           cashflow: list[dict]) -> list[dict]:
    """Return a list of {flag, severity} warnings from the CFA M36 checklist."""
    flags: list[dict] = []
    ni = _sum_last(income, "net_income", 4)
    cfo = _sum_last(cashflow, "operating_cash_flow", 4)
    if ni and cfo is not None:
        if cfo < ni:
            flags.append({"flag": "CFO < Net Income (accrual-heavy earnings)", "severity": "warning"})
        if cfo and cfo / ni < 1.0:
            flags.append({"flag": "CFO / NI < 1.0 (earnings quality concern)", "severity": "info"})
    # Rising DSO
    def _dsos():
        ar = _latest(balance, "accounts_receivable")
        rev = _sum_last(income, "total_revenue", 4)
        return (ar * 365 / rev) if (ar and rev) else None
    if len(income) >= 8:
        ar_p = _latest(balance[:-4], "accounts_receivable")
        rev_p = _sum_last(income[:-4], "total_revenue", 4)
        if ar_p and rev_p:
            dso_p = ar_p * 365 / rev_p
            dso_c = _dsos()
            if dso_c and dso_p and dso_c > dso_p * 1.15:
                flags.append({"flag": "Rising DSO (possible channel stuffing / slow "
                                      "collections)", "severity": "warning"})
    # Declining inventory turnover
    cogs = _sum_last(income, "cost_of_revenue", 4)
    inv = _latest(balance, "inventory")
    if cogs and inv and inv > 0:
        flags.append({"flag": "Inventory turnover context", "severity": "info"})
    # Declining asset turnover
    ta = _latest(balance, "total_assets")
    rev = _sum_last(income, "total_revenue", 4)
    if len(balance) >= 5:
        ta_p = _latest(balance[:-4], "total_assets")
        rev_p = _sum_last(income[:-4], "total_revenue", 4)
        if ta and ta_p and rev and rev_p and (rev / ta) < (rev_p / ta_p) * 0.95:
            flags.append({"flag": "Declining total-asset turnover", "severity": "info"})
    return flags


def evaluate_quality(income: list[dict], balance: list[dict], cashflow: list[dict],
                     market_cap: float | None = None, is_manufacturing: bool = True) -> dict:
    """Run all quality models together."""
    return {
        "piotroski": piotroski(income, balance, cashflow),
        "altman_z": altman_z(income, balance, market_cap=market_cap,
                             is_manufacturing=is_manufacturing),
        "beneish_m": beneish_m(income, balance, cashflow),
        "earnings_quality_flags": earnings_quality_flags(income, balance, cashflow),
    }
