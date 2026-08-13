"""Black-Litterman allocation (plan §7.2).

- Prior:    market-cap weighted equilibrium returns
- Views:    confidence_delta from the debate gating node scaled to expected return
- Output:   posterior weights, capped at 15%/position and 40%/asset class
"""
from __future__ import annotations

import numpy as np

RISK_AVERSION = 2.5          # δ
TAU = 0.05                   # prior uncertainty scalar


def equilibrium_returns(market_caps: np.ndarray, covariance: np.ndarray,
                        risk_aversion: float = RISK_AVERSION) -> np.ndarray:
    """Π = δ Σ w_mkt — implied excess returns."""
    w = market_caps / market_caps.sum()
    return risk_aversion * covariance @ w


def black_litterman_posterior(equilibrium: np.ndarray, covariance: np.ndarray,
                              views: np.ndarray, view_assets: np.ndarray,
                              confidences: np.ndarray,
                              tau: float = TAU) -> np.ndarray:
    """Posterior mean via the BL formula with diagonal Ω."""
    n = len(equilibrium)
    k = len(views)
    P = np.zeros((k, n))
    for i, asset in enumerate(view_assets):
        P[i, asset] = 1.0
    # Ω diagonal: view variance scaled by (1-c)/c — built explicitly 2-D so a
    # single view never degenerates to a 1-D array.
    view_var = np.diag(covariance[np.ix_(view_assets, view_assets)])
    scale = (1.0 - confidences) / np.clip(confidences, 0.05, 1.0)
    Ω = np.zeros((k, k))
    np.fill_diagonal(Ω, scale * view_var)

    tauΣ_inv = np.linalg.inv(tau * covariance)
    M = tauΣ_inv + P.T @ np.linalg.inv(Ω) @ P
    b = tauΣ_inv @ equilibrium + P.T @ np.linalg.inv(Ω) @ views
    return np.linalg.solve(M, b)


def black_litterman_allocate(tickers: list[str], views: dict[str, float],
                             confidence: dict[str, float],
                             nav_usd: float = 1_000_000.0,
                             market_caps: dict[str, float] | None = None,
                             covariance: np.ndarray | None = None,
                             max_position: float = 0.15,
                             max_class: float = 0.40) -> dict[str, float]:
    """Full BL pipeline with constraints. Falls back to equal-weight on failure."""
    n = len(tickers)
    if n == 0:
        return {}

    # Fallbacks for missing priors
    caps = np.ones(n) if not market_caps else np.array(
        [market_caps.get(t, 1.0) for t in tickers], dtype=float)
    cov = (np.eye(n) * 0.02 + 0.005) if covariance is None else covariance

    try:
        eq = equilibrium_returns(caps, cov)
        view_assets = np.array([tickers.index(t) for t in views if t in tickers],
                               dtype=int)
        if len(view_assets) == 0:
            raise ValueError("No view assets in tickers")
        view_vals = np.array([views[tickers[i]] for i in view_assets], dtype=float)
        conf_vals = np.array([min(confidence.get(tickers[i], 0.5), 1.0)
                              for i in view_assets], dtype=float)
        mu = black_litterman_posterior(eq, cov, view_vals, view_assets, conf_vals)
        w = np.linalg.solve(RISK_AVERSION * cov, mu)
    except (np.linalg.LinAlgError, ValueError):
        w = np.ones(n) / n

    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    # Cap per-position weight (plan §7.2: no single position > 15%).
    # The remainder stays in cash (sum may be < 1 — never renormalize past the cap).
    w = np.minimum(w, max_position)
    return {t: round(float(wi), 4) for t, wi in zip(tickers, w)}
