"""Classical ML baselines: linear, tree, kernel, probabilistic, and classification models.

Two feature modes:
  * ``last_step`` (default for linear/kernel models) — only the most recent
    timestep (F features). Avoids the curse of dimensionality on L*F inputs.
  * ``full_window`` (default for tree models) — flattened (L*F) window, which
    trees handle well due to axis-aligned splits.

All models share the same benchmark interface:
  fit on walk-forward train windows → emit OOS position signals in [-1, 1]
  (tanh of the vol-scaled return prediction) → portfolio layer converts to
  vol-targeted weights.

Current model roster
--------------------
Linear:
  ridge, lasso, elasticnet, bayesian_ridge
Tree / boosting:
  random_forest, extra_trees, gbm (sklearn), xgboost, lightgbm, catboost
Kernel:
  svr_rbf
Instance-based:
  knn
Classification-as-signal:
  logistic, lda
Hybrid / ensembles:
  hmm_lgbm  — HMM regime probs prepended → LightGBM
  strategy_xgb — momentum signals prepended → XGBoost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_builder.trainer import WindowedDataset, walk_forward_windows


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_matrix(
    panel: pd.DataFrame,
    feature_cols: list[str],
    symbols: list[str],
    lookback: int,
    last_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Build design matrix + targets.

    Parameters
    ----------
    last_step:
        If True, use only the final timestep → shape (N, F).
        If False, flatten the full window → shape (N, L*F).
    """
    ds = WindowedDataset(panel, feature_cols, lookback, symbols)
    n = len(ds)
    if last_step:
        X = ds.x[:, -1, :]          # (N, F) — most recent bar only
    else:
        X = ds.x.reshape(n, -1)     # (N, L*F) — full flattened window
    # volatility-scaled next-day return target (paper eq. 23), clipped
    target = np.clip(ds.r / np.clip(ds.v, 1e-6, None), -20, 20)
    return X, target, ds.v, ds.r, [(ds.times[i], ds.syms[i]) for i in range(n)]


def _positions_from_pred(pred: np.ndarray) -> np.ndarray:
    """Continuous regression prediction → position in [-1, 1]."""
    return np.tanh(pred)


def _clf_signal(proba: np.ndarray) -> np.ndarray:
    """Classification probability → signed signal: p(up) - p(down).

    Assumes class order [down, up] or [down, neutral, up] based on n_classes.
    """
    if proba.ndim == 1 or proba.shape[1] == 1:
        return 2 * proba.ravel() - 1                      # binary: p(up) → [-1,1]
    if proba.shape[1] == 2:
        return proba[:, 1] - proba[:, 0]                  # p(up) - p(down)
    return proba[:, -1] - proba[:, 0]                      # 3-class: up - down


def _target_class(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Convert vol-scaled return to class label.

    Binary:  0 = down (y < 0),  1 = up (y >= 0)
    Ternary: 0 = down, 1 = neutral (|y| < 0.5σ), 2 = up
    """
    if n_classes == 2:
        return (y >= 0).astype(int)
    labels = np.ones(len(y), dtype=int)     # neutral
    labels[y < -0.5] = 0                    # down
    labels[y > 0.5] = 2                     # up
    return labels


def _run_classical(
    name: str,
    make_fit_predict,
    panel: pd.DataFrame,
    feature_cols: list[str],
    symbols: list[str],
    lookback: int = 64,
    train_months: int = 36,
    test_months: int = 6,
    sigma_tgt: float = 0.10,
    last_step: bool = False,
) -> pd.DataFrame:
    """Generic walk-forward driver.

    make_fit_predict(Xtr, ytr, Xva) -> pred array in [-1, 1] space.
    """
    windows = walk_forward_windows(panel, train_months, test_months)
    frames = []
    for tr, va in windows:
        Xtr, ytr, _, _, _ = _sample_matrix(
            tr, feature_cols, symbols, lookback, last_step=last_step)
        Xva, _, vva, _, rows = _sample_matrix(
            va, feature_cols, symbols, lookback, last_step=last_step)
        pred = make_fit_predict(Xtr, ytr, Xva)
        pos = _positions_from_pred(np.asarray(pred, dtype=float))
        for i, (t, sym) in enumerate(rows):
            frames.append({"time": t, "symbol": sym,
                           "weight": float(pos[i] * sigma_tgt * vva[i])})
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Linear models  (last_step=True — only current bar, F features)
# ---------------------------------------------------------------------------

def ridge_lasso(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, alpha: float = 1.0, penalty: str = "l2", **kw,
) -> pd.DataFrame:
    """Ridge (L2) or Lasso (L1) regression on last-step features."""
    from sklearn.linear_model import Lasso, Ridge
    model = Ridge(alpha=alpha) if penalty == "l2" else Lasso(alpha=alpha, max_iter=5000)
    return _run_classical(
        "ridge" if penalty == "l2" else "lasso",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


def elasticnet_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, alpha: float = 0.5, l1_ratio: float = 0.5, **kw,
) -> pd.DataFrame:
    """ElasticNet — balanced L1+L2 penalty, last-step features."""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    return _run_classical(
        "elasticnet",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


def bayesian_ridge_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """Bayesian Ridge — automatic regularisation, last-step features."""
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge()
    return _run_classical(
        "bayesian_ridge",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


# ---------------------------------------------------------------------------
# Tree / boosting models  (full_window — trees handle high-dim well)
# ---------------------------------------------------------------------------

def random_forest_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """Random Forest regressor — full flattened window."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42)
    return _run_classical(
        "random_forest",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


def extra_trees_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """Extra Trees — faster than RF, higher variance, full window."""
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42)
    return _run_classical(
        "extra_trees",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


def gbm_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """sklearn Gradient Boosting — slower but no external dep, full window."""
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    return _run_classical(
        "gbm",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


def xgboost_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """XGBoost regressor — full window."""
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4)
    return _run_classical(
        "xgboost",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


def lightgbm_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """LightGBM standalone regressor — full window, faster than XGBoost."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        n_jobs=4, verbose=-1)
    return _run_classical(
        "lightgbm",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


def catboost_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, **kw,
) -> pd.DataFrame:
    """CatBoost regressor — handles NaNs natively, full window."""
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.05,
        subsample=0.8, verbose=0, random_seed=42)
    return _run_classical(
        "catboost",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=False, **kw)


# ---------------------------------------------------------------------------
# Kernel model  (last_step=True — SVR chokes on 4800-dim input)
# ---------------------------------------------------------------------------

def svr_rbf_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, C: float = 1.0, epsilon: float = 0.1, **kw,
) -> pd.DataFrame:
    """SVR with RBF kernel — last-step features, StandardScaler pre-processing."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon)),
    ])
    return _run_classical(
        "svr_rbf",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


# ---------------------------------------------------------------------------
# Instance-based  (last_step=True — k-NN on 4800-dim is meaningless)
# ---------------------------------------------------------------------------

def knn_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, n_neighbors: int = 10, **kw,
) -> pd.DataFrame:
    """k-Nearest Neighbours regressor — last-step features, L2 distance."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)),
    ])
    return _run_classical(
        "knn",
        lambda X, y, Xv: model.fit(X, y).predict(Xv),
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


# ---------------------------------------------------------------------------
# Classification-as-signal  (predict direction class → p(up)-p(down))
# ---------------------------------------------------------------------------

def logistic_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, C: float = 1.0, n_classes: int = 2, **kw,
) -> pd.DataFrame:
    """Logistic Regression — predicts up/down class, signal = p(up)-p(down)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=500, n_jobs=-1)),
    ])

    def fit_predict(Xtr, ytr, Xva):
        labels = _target_class(ytr, n_classes)
        clf.fit(Xtr, labels)
        return _clf_signal(clf.predict_proba(Xva))

    return _run_classical(
        "logistic", fit_predict,
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


def lda_ml(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, n_classes: int = 2, **kw,
) -> pd.DataFrame:
    """Linear Discriminant Analysis — fast, closed-form, last-step features."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()

    def fit_predict(Xtr, ytr, Xva):
        labels = _target_class(ytr, n_classes)
        # LDA needs ≥2 samples per class; skip degenerate windows gracefully
        if len(np.unique(labels)) < 2:
            return np.zeros(len(Xva))
        clf.fit(Xtr, labels)
        return _clf_signal(clf.predict_proba(Xva))

    return _run_classical(
        "lda", fit_predict,
        panel, feature_cols, symbols, lookback, last_step=True, **kw)


# ---------------------------------------------------------------------------
# Hybrid / ensemble models (existing + HMM+LightGBM)
# ---------------------------------------------------------------------------

def hmm_lightgbm(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, n_states: int = 3, train_months: int = 36,
    test_months: int = 6, sigma_tgt: float = 0.10,
) -> pd.DataFrame:
    """HMM regime probabilities (per asset, fit on train window) prepended → LightGBM."""
    import lightgbm as lgb
    from hmmlearn.hmm import GaussianHMM

    def fit_regime(tr_panel: pd.DataFrame) -> dict[str, GaussianHMM]:
        models = {}
        for sym in symbols:
            g = tr_panel[tr_panel["symbol"] == sym].sort_values("time")
            rets = g["ret_1"].dropna().to_numpy().reshape(-1, 1)
            if len(rets) < 60:
                continue
            hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                              n_iter=100, random_state=0)
            hmm.fit(rets)
            models[sym] = hmm
        return models

    def regime_matrix(panel_df: pd.DataFrame, models: dict) -> np.ndarray:
        rows = []
        for sym, g in panel_df.groupby("symbol", sort=False):
            g = g.sort_values("time")
            m = models.get(sym)
            n = len(g)
            full = np.zeros((n, n_states))
            if m is not None:
                rets = g["ret_1"].to_numpy().reshape(-1, 1)
                valid = ~np.isnan(rets[:, 0])
                if valid.any():
                    full[valid] = m.predict_proba(rets[valid])
            full_clean = pd.DataFrame(full).ffill().bfill().to_numpy()
            for t in range(lookback, len(g) - 1):
                rows.append(full_clean[t])
        return np.array(rows) if rows else np.zeros((0, n_states))

    frames = []
    for tr, va in walk_forward_windows(panel, train_months, test_months):
        Xtr, ytr, _, _, _ = _sample_matrix(
            tr, feature_cols, symbols, lookback, last_step=False)
        Xva, _, vva, _, rows = _sample_matrix(
            va, feature_cols, symbols, lookback, last_step=False)
        hmm_models = fit_regime(tr)
        Xtr2 = np.hstack([Xtr, regime_matrix(tr, hmm_models)])
        Xva2 = np.hstack([Xva, regime_matrix(va, hmm_models)])
        m = lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            n_jobs=4, verbose=-1)
        m.fit(Xtr2, ytr)
        pos = _positions_from_pred(m.predict(Xva2))
        for i, (t, sym) in enumerate(rows):
            frames.append({"time": t, "symbol": sym,
                           "weight": float(pos[i] * sigma_tgt * vva[i])})
    return pd.DataFrame(frames)


def strategy_xgboost(
    panel: pd.DataFrame, feature_cols: list[str], symbols: list[str],
    lookback: int = 64, train_months: int = 36,
    test_months: int = 6, sigma_tgt: float = 0.10,
) -> pd.DataFrame:
    """Meta-ensemble: momentum strategy signals prepended to features → XGBoost."""
    import xgboost as xgb

    def base_signals(panel_df: pd.DataFrame) -> np.ndarray:
        sig = []
        for sym, g in panel_df.groupby("symbol", sort=False):
            g = g.sort_values("time")
            s = np.column_stack([
                g["ret_norm_21"].to_numpy(), g["ret_norm_63"].to_numpy(),
                g["ret_norm_126"].to_numpy(), g["macd_signal"].to_numpy()])
            s_clean = pd.DataFrame(s).ffill().bfill().to_numpy()
            for t in range(lookback, len(g) - 1):
                sig.append(s_clean[t])
        return np.array(sig) if sig else np.zeros((0, 4))

    frames = []
    for tr, va in walk_forward_windows(panel, train_months, test_months):
        Xtr, ytr, _, _, _ = _sample_matrix(
            tr, feature_cols, symbols, lookback, last_step=False)
        Xva, _, vva, _, rows = _sample_matrix(
            va, feature_cols, symbols, lookback, last_step=False)
        Xtr2 = np.hstack([Xtr, base_signals(tr)])
        Xva2 = np.hstack([Xva, base_signals(va)])
        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4)
        m.fit(Xtr2, ytr)
        pos = _positions_from_pred(m.predict(Xva2))
        for i, (t, sym) in enumerate(rows):
            frames.append({"time": t, "symbol": sym,
                           "weight": float(pos[i] * sigma_tgt * vva[i])})
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Registry (imported by run.py)
# ---------------------------------------------------------------------------

#: All callable classical models keyed by benchmark name
CLASSICAL_REGISTRY: dict[str, callable] = {
    # Linear
    "ridge":          lambda p, f, s, **kw: ridge_lasso(p, f, s, penalty="l2", **kw),
    "lasso":          lambda p, f, s, **kw: ridge_lasso(p, f, s, penalty="l1", **kw),
    "elasticnet":     elasticnet_ml,
    "bayesian_ridge": bayesian_ridge_ml,
    # Tree / boosting
    "random_forest":  random_forest_ml,
    "extra_trees":    extra_trees_ml,
    "gbm":            gbm_ml,
    "xgboost":        xgboost_ml,
    "lightgbm":       lightgbm_ml,
    "catboost":       catboost_ml,
    # Kernel
    "svr_rbf":        svr_rbf_ml,
    # Instance-based
    "knn":            knn_ml,
    # Classification-as-signal
    "logistic":       logistic_ml,
    "lda":            lda_ml,
    # Hybrid
    "hmm_lgbm":       hmm_lightgbm,
    "strategy_xgb":   strategy_xgboost,
}
