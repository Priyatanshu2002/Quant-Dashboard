"""Portfolio backtest + metric suite (Oxford protocol, arXiv:2603.01820).

Implements Appendix D metrics:
  * portfolio return = cross-sectional mean of weight * next-day return (eq. 8)
  * CAGR, annualized return, Sharpe, HAC (Newey-West) t-stats, hit rate,
    turnover (annualized), turnover xGMV, information ratio vs passive,
    correlation vs passive, max drawdown, Calmar, worst-3m Sharpe,
    min annual Sharpe, CVaR 5%, breakeven transaction cost per asset (App. E)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(weights: pd.DataFrame, panel: pd.DataFrame,
                      max_gross: float = 1.5, cost_bps: float = 0.0) -> pd.DataFrame:
    """Merge weights with next-day returns; daily portfolio return per day.

    weights: (time, symbol, weight); panel: long frame with ret_1 (r_t) and time/symbol.
    Returns DataFrame indexed by time with columns: portfolio_ret, gross_exposure,
    per-symbol weights (wide), and turnover (for cost drag).

    Robustness fixes vs the prior buggy version:
      * weights are gross-capped (sum |w| <= max_gross) so no single model can
        take absurd leverage (previously weight = pos*sigma_tgt*(1/sigma) reached
        ~±600%, inflating Sharpe/CAGR).
      * an optional proportional transaction cost (cost_bps) is applied from the
        daily weight turnover.
    """
    weights = weights.copy()
    weights["time"] = pd.to_datetime(weights["time"])
    panel = panel.copy()
    panel["time"] = pd.to_datetime(panel["time"])
    merged = weights.merge(
        panel[["time", "symbol", "ret_1"]], on=["time", "symbol"], how="left")

    # gross-cap weights per day
    daily_w = merged.groupby("time")["weight"]
    gross = daily_w.transform(lambda s: s.abs().sum())
    scale = np.minimum(1.0, max_gross / gross.replace(0.0, np.nan))
    merged["weight"] = merged["weight"] * scale.fillna(0.0)

    merged["contribution"] = merged["weight"] * merged["ret_1"]
    daily = merged.groupby("time").agg(
        portfolio_ret=("contribution", "mean"),
        gross_exposure=("weight", lambda s: s.abs().sum()),
        n_assets=("weight", "count"))
    daily.index = pd.DatetimeIndex(daily.index)

    # transaction cost drag from turnover (per symbol weight change, sum abs)
    wide = merged.pivot_table(index="time", columns="symbol", values="weight").fillna(0.0)
    wide.index = pd.DatetimeIndex(wide.index)
    turnover = wide.diff().abs().sum(axis=1).reindex(daily.index).fillna(0.0)
    if cost_bps > 0:
        daily["portfolio_ret"] = daily["portfolio_ret"] - turnover * (cost_bps / 1e4)
    daily["turnover"] = turnover
    return daily.join(wide)


def _hac_tstat(returns: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-statistic of the mean."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return 0.0
    n = len(r)
    mu = r.mean()
    e = r - mu
    var = e @ e / n
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(1, min(lags, n - 2))
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        var += 2 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(max(var, 1e-12) / n)
    return float(mu / se)


def annualize(r: pd.Series) -> tuple[float, float, float]:
    """Annualized return, volatility, Sharpe from a daily return series."""
    r = r.dropna()
    if len(r) < 5:
        return 0.0, 0.0, 0.0
    mu = r.mean() * TRADING_DAYS
    sigma = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = mu / sigma if sigma > 0 else 0.0
    return float(mu), float(sigma), float(sharpe)


def full_metrics(weights: pd.DataFrame, panel: pd.DataFrame,
                 passive_ret: pd.Series | None = None,
                 cost_bps: float = 0.0) -> dict:
    """All Appendix-D metrics for one strategy's OOS weights.

    cost_bps: proportional transaction cost (bps per unit turnover) applied to
    the daily returns before computing all metrics. Default 0 keeps behaviour
    cost-free; the benchmark passes a realistic value so the leaderboard shows
    NET-of-cost performance (gross Sharpe/CAGR were misleadingly high).
    """
    pr = portfolio_returns(weights, panel, cost_bps=cost_bps)
    rets = pr["portfolio_ret"].dropna()
    ann_ret, ann_vol, sharpe = annualize(rets)
    n_days = len(rets)
    years = n_days / TRADING_DAYS
    equity = (1 + rets).cumprod()
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0

    # drawdowns
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # worst 3-month Sharpe: rolling 63d
    r63 = rets.rolling(63).apply(lambda s: s.mean() / (s.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS)
                                 if s.std(ddof=1) > 0 else 0.0, raw=True)
    worst_3m = float(r63.min()) if len(r63) else 0.0
    yearly = rets.groupby(rets.index.year).apply(
        lambda s: s.mean() / (s.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS) if len(s) > 20 else 0.0)
    min_ann = float(yearly.min()) if len(yearly) else 0.0

    # CVaR 5% (expected shortfall)
    cvar = float(rets[rets <= rets.quantile(0.05)].mean()) if len(rets) > 20 else 0.0

    # turnover: mean abs change in weights, annualized; xGMV = turnover / 2 (per unit gross)
    w_cols = [c for c in pr.columns if c not in ("portfolio_ret", "n_assets")]
    w = pr[w_cols].fillna(0.0)
    turnover = float(w.diff().abs().sum(axis=1).mean() * TRADING_DAYS)
    xgmv = float(w.diff().abs().sum(axis=1).mean() * TRADING_DAYS / 2)

    hit = float((rets > 0).mean())

    # passive-relative
    if passive_ret is not None:
        pr_al = rets.reindex(passive_ret.index).dropna()
        pa_al = passive_ret.reindex(pr_al.index).dropna()
        excess = pr_al - pa_al
        ir = float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))
        t_hac_passive = _hac_tstat(excess.values)
        corr = float(np.corrcoef(pr_al, pa_al)[0, 1]) if len(pr_al) > 5 else 0.0
    else:
        ir = t_hac_passive = corr = 0.0

    return {
        "cagr": round(cagr, 4), "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4), "sharpe": round(sharpe, 3),
        "t_hac": round(_hac_tstat(rets.values), 2), "hit_rate": round(hit, 3),
        "turnover": round(turnover, 1), "xgmv": round(xgmv, 1),
        "max_dd": round(max_dd, 4), "calmar": round(calmar, 3),
        "worst_3m_sharpe": round(worst_3m, 2), "min_ann_sharpe": round(min_ann, 2),
        "cvar_5": round(cvar, 5), "info_ratio": round(ir, 3),
        "t_hac_vs_passive": round(t_hac_passive, 2), "corr_vs_passive": round(corr, 3),
        "days": n_days,
    }


def breakeven_costs(weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per-asset breakeven transaction cost c* (bps): max friction before PnL = 0.

    c* = sum(w_k * r_{k,t+1}) / sum(|w_k,t - w_k,t-1|)  (Appendix E)
    """
    weights = weights.copy()
    weights["time"] = pd.to_datetime(weights["time"])
    panel = panel.copy()
    panel["time"] = pd.to_datetime(panel["time"])
    merged = weights.merge(panel[["time", "symbol", "ret_1"]],
                           on=["time", "symbol"], how="left")
    merged = merged.sort_values(["symbol", "time"])
    gross = merged.groupby("symbol").apply(
        lambda g: float((g["weight"] * g["ret_1"]).sum()), include_groups=False)
    merged["dweight"] = merged.groupby("symbol")["weight"].diff().abs()
    turnover = merged.groupby("symbol")["dweight"].sum()
    out = pd.DataFrame({"gross_ann": gross / (merged["symbol"].value_counts() / TRADING_DAYS)})
    out["turnover_ann"] = turnover / (merged["symbol"].value_counts() / TRADING_DAYS)
    out["breakeven_bps"] = (gross / turnover.replace(0, np.nan) * 1e4).round(2)
    out = out.sort_values("breakeven_bps", ascending=False)
    return out.reset_index().rename(columns={"index": "symbol"})


def passive_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only daily returns (passive benchmark)."""
    pr = panel.pivot_table(index="time", columns="symbol", values="ret_1")
    return pr.mean(axis=1).dropna()


def equity_curve_from_weights(weights: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    pr = portfolio_returns(weights, panel)
    return (1 + pr["portfolio_ret"].dropna()).cumprod()
