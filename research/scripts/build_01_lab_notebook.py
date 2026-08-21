"""Build research/notebooks/01_empirical_research_lab.ipynb — the 6-stage
empirical research lab for Project Agonistes.

The notebook is GPU-aware and mode-configurable via environment variables so
the SAME artifact runs a fast smoke pass locally (CPU) or the full-strength
benchmark on a RunPod GPU pod:

    RUN_MODE          : "smoke" | "full"         (default full)
    AG_MODELS         : comma-separated model roster override
    AG_TRAIN_MONTHS   : int (smoke 24 / full 36)
    AG_TEST_MONTHS    : int (smoke 4  / full 6)
    AG_EPOCHS         : int (smoke 15 / full 60)
    AG_LOOKBACK       : int (smoke 32 / full 64)
    AG_BENCH_DIR      : artifact output dir (default data/benchmark)

Usage:
    python research/scripts/build_01_lab_notebook.py

Emits .ipynb (JSON) via nbformat with cells pre-created; execution is done
with nbconvert/papermill on the target host so outputs are recorded per-cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "notebooks" / "01_empirical_research_lab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# Shared preamble / config (injected verbatim into the notebook)
# ---------------------------------------------------------------------------

PRELUDE = r'''
import os, sys, json, time
from pathlib import Path

# Anchor to the project root (strategy_builder/ + core/ + data/) regardless of
# the kernel's cwd (Jupyter sets kernel cwd to the notebook's directory).
_cwd = Path(os.getcwd())
PROJECT_ROOT = _cwd
for _p in [PROJECT_ROOT, *_cwd.parents]:
    if (_p / "strategy_builder").is_dir() and (_p / "core").is_dir():
        PROJECT_ROOT = _p
        break
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"[lab] project root: {PROJECT_ROOT}")

import numpy as np
import pandas as pd

RUN_MODE = os.environ.get("RUN_MODE", "full").strip().lower()
print(f"[lab] RUN_MODE = {RUN_MODE}")
import torch
print(f"[lab] device  = cuda available -> {torch.cuda.is_available()}")
'''

CONFIG = r'''
# ---------------------------------------------------------------------------
# Config: read from data/benchmark/lab_config.json (survives kernel env
# inheritance) with env-var fallback. One knob for smoke vs full-strength.
# ---------------------------------------------------------------------------
import json
from pathlib import Path
from strategy_builder.trainer import default_device

# Resolve project root regardless of kernel cwd (Jupyter sets it to the
# notebook's directory). Independent of the PRELUDE cell.
_p_root = Path(os.getcwd())
for _p in [_p_root, *_p_root.parents]:
    if (_p / "strategy_builder").is_dir() and (_p / "core").is_dir():
        _p_root = _p
        break

_cfg = {}
_cfg_path = Path(os.environ.get("AG_CONFIG", str(_p_root / "data/benchmark/lab_config.json")))
if _cfg_path.exists():
    try:
        _cfg = json.loads(_cfg_path.read_text())
    except Exception as _e:  # noqa: BLE001
        print(f"[lab] warning: could not read {_cfg_path}: {_e}")
RUN_MODE = _cfg.get("run_mode", os.environ.get("RUN_MODE", "full")).strip().lower()

def _env_int(name, default):
    try:
        return int(_cfg.get(name, os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default

def _env_models(name, default):
    raw = _cfg.get(name, os.environ.get(name, "")).strip()
    return [m.strip() for m in raw.split(",") if m.strip()] or default

DEVICE = default_device()

# Universe: balanced 29-asset cross-asset panel (US/India/ETF/FX/rates/commo)
UNIVERSE = [
    # US large caps
    "AAPL","AMZN","GOOGL","JPM","META","MSFT","NVDA","TSLA","UNH","XOM",
    # Indian bluechips
    "HDFCBANK.NS","INFY.NS","RELIANCE.NS","SBIN.NS","TCS.NS",
    # Liquid ETFs / indices
    "SPY","QQQ","IWM","EEM","GLD","TLT","^NSEI",
    # FX / rates / commodities
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDINR=X","DX-Y.NYB","^TNX","GC=F",
]

# Walk-forward geometry
TRAIN_MONTHS = 24 if RUN_MODE == "smoke" else 36
TEST_MONTHS  = 4  if RUN_MODE == "smoke" else 6
LOOKBACK     = 32 if RUN_MODE == "smoke" else 64
HIDDEN       = 16 if RUN_MODE == "smoke" else 32
EPOCHS       = 15 if RUN_MODE == "smoke" else 60
SEEDS        = 2  if RUN_MODE == "smoke" else 3
TOP_SEEDS    = 1  if RUN_MODE == "smoke" else 2
BATCH_SIZE   = 256
LR           = 1e-3
SIGMA_TGT    = 0.10
PATIENCE     = 10

# Model rosters (each overridable independently via config/env)
NEURAL_MODELS     = _env_models("neural", ["vlstm", "tft", "nlinear", "lstm", "patchtst"])
CLASSICAL_MODELS  = _env_models("classical", ["ridge", "elasticnet", "xgboost", "lightgbm", "catboost", "knn"])
VOLATILITY_MODELS = _env_models("volatility", ["garch11", "egarch", "gjr_garch", "har_rv"])
BENCH_DIR = Path(_cfg.get("bench_dir", os.environ.get("AG_BENCH_DIR", str(_p_root / "data/benchmark"))))
BENCH_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_MODE:", RUN_MODE, "| Train months:", TRAIN_MONTHS,
      "| Test months:", TEST_MONTHS, "| Lookback:", LOOKBACK,
      "| Epochs:", EPOCHS)
'''

# ---------------------------------------------------------------------------
# Stage cells
# ---------------------------------------------------------------------------

S1_MD = md("""# Stage 1 — Data Panel Extraction & Verification

Connect to the local SQLite `data/agonistes_dev.db`, pull a balanced 29-asset
cross-asset universe, and verify: first/last rows, date range, null counts,
summary stats, and that there are **no missing dates** and **no forward-looking
leakage** (the next-day target must never be available at time t in the panel).""")

S1_CODE = code(r'''
from core.db import get_storage
db = get_storage()

def load_universe(universe, min_bars=400):
    closes = {}
    for sym in universe:
        ohlcv = db.query_ohlcv(sym)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < min_bars:
            print(f"  SKIP {sym}: insufficient data "
                  f"({0 if ohlcv is None else len(ohlcv)} bars)")
            continue
        closes[sym] = ohlcv["close"]
    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
    return prices

prices = load_universe(UNIVERSE)
print(f"Loaded {prices.shape[1]} assets x {prices.shape[0]} bars")
print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
prices.head()
''')

S1_CODE2 = code(r'''
print("--- LAST 5 ROWS ---")
prices.tail()

# --- Null counts ---
nulls = prices.isna().sum()
print("--- Nulls per asset ---")
print(nulls[nulls > 0] if (nulls > 0).any() else "No nulls in the close panel.")

# --- Missing business days check (forward-looking safety) ---
mask = prices.notna().sum(axis=1)
all_present = prices[mask >= len(prices.columns) * 0.8]
idx = pd.to_datetime(all_present.index)
expected = pd.bdate_range(idx.min(), idx.max())
missing = expected.difference(idx)
print(f"Business days expected: {len(expected)} | present: {len(idx)} | missing: {len(missing)}")
if len(missing):
    print("Missing business days:", list(missing)[:10])
''')

S1_CODE3 = code(r'''
# --- Summary statistics for a few representative assets ---
summary = pd.DataFrame({
    "mean_ret":  prices.pct_change().mean(),
    "std_ret":   prices.pct_change().std(),
    "annual_vol":prices.pct_change().std() * np.sqrt(252),
    "min_close": prices.min(),
    "max_close": prices.max(),
}).round(4)
summary.head(10)
''')

S2_MD = md("""# Stage 2 — Feature Engineering & Target Construction

Compute standardized feature representations from the Oxford DL-for-finance
protocol:

- Volatility-normalized log returns at 1d / 5d / 21d / 63d / 126d:
  $$r^{\\text{norm}}_{t,h} = \\frac{r_{t,h}}{\\sigma_t \\sqrt{h}}$$
- Parkinson & Garman-Klass volatility estimators.
- Volatility-scaled next-day target:  $y_t = r_{t+1} / \\sigma_t$

Print tensor dimensions `(N, Lookback, F)` and render a correlation heatmap.""")

S2_CODE = code(r'''
from strategy_builder.features import build_features, build_target, FEATURE_COLS

# Build long-format panel with features + target, exactly like the benchmark.
from strategy_builder.features import build_universe_frame
panel = build_universe_frame(prices)
symbols = sorted(panel["symbol"].unique())
print(f"Panel: {len(panel):,} rows, {len(symbols)} symbols, "
      f"{panel['time'].min().date()} -> {panel['time'].max().date()}")
print("Feature columns:", FEATURE_COLS)
print("Target column  : target (vol-scaled next-day return, clipped)")
panel.head(3)
''')

S2_CODE2 = code(r'''
# --- Tensor geometry (N, Lookback, F) for a single asset ---
from strategy_builder.trainer import WindowedDataset
sample_sym = panel["symbol"].iloc[0]
sub = panel[panel["symbol"] == sample_sym].sort_values("time")
ds = WindowedDataset(sub, FEATURE_COLS, lookback=LOOKBACK, symbols=[sample_sym])
print(f"Tensor shape: (N={ds.x.shape[0]}, Lookback={ds.x.shape[1]}, F={ds.x.shape[2]})")
print(f"Sample x[0]: {ds.x[0].shape}, next-day raw return r[0] = {ds.r[0]:.5f}")
''')

S2_CODE3 = code(r'''
# --- Feature correlation heatmap (representative asset) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

feats = sub[FEATURE_COLS].dropna().corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(feats.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
ax.set_xticklabels(feats.columns, rotation=90); ax.set_yticklabels(feats.columns)
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title(f"Feature correlation — {sample_sym}")
plt.tight_layout(); plt.show()
''')

S3_MD = md("""# Stage 3 — Walk-Forward Validation Engine (with Purge & Embargo)

Expanding-window walk-forward with a **strict purge gap** and **embargo**
between train and validation to eliminate lookahead bias: the validation
samples whose labels overlap the training horizon are purged, and an embargo
of `embargo_days` is dropped from the boundary.

> Without purge/embargo, overlapping labels leak information from the
> validation window back into training — inflating OOS metrics.""")
# (purge/embargo semantics are enforced in the splitter below)

S3_CODE = code(r'''
from datetime import timedelta

def walk_forward_purged(panel, train_months, test_months,
                        embargo_days=None, min_train=500, min_test=100):
    """Expanding-window splitter with purge + embargo.

    Returns list of (train, test) DataFrames. Embargo (in business days) drops
    a safety gap between the last training label and the first test bar.
    """
    start, end = panel["time"].min(), panel["time"].max()
    if embargo_days is None:
        embargo_days = max(5, int(test_months * 21 * 0.1))
    windows = []
    cursor = start + pd.DateOffset(months=train_months)
    step = pd.DateOffset(months=test_months)
    while cursor + step <= end:
        tr_end = cursor
        te_end = cursor + step
        tr = panel[(panel["time"] >= start) & (panel["time"] < tr_end)].copy()
        te = panel[(panel["time"] >= te_end - pd.Timedelta(days=embargo_days))
                   & (panel["time"] < te_end)].copy()
        # purge: drop test rows that overlap the last label window of train
        te = te[te["time"] >= tr_end + pd.Timedelta(days=embargo_days)]
        if len(tr) > min_train and len(te) > min_test:
            windows.append((tr, te))
        cursor += step
    return windows

WINDOWS = walk_forward_purged(panel, TRAIN_MONTHS, TEST_MONTHS,
                              embargo_days=max(5, int(TEST_MONTHS * 21 * 0.1)))
print(f"{len(WINDOWS)} walk-forward windows (train {TRAIN_MONTHS}m / "
      f"test {TEST_MONTHS}m, embargo purged)")

rows = []
for i, (tr, te) in enumerate(WINDOWS):
    rows.append({
        "window": i,
        "train_start": tr["time"].min().date(),
        "train_end":   tr["time"].max().date(),
        "train_rows":  len(tr),
        "test_start":  te["time"].min().date(),
        "test_end":    te["time"].max().date(),
        "test_rows":   len(te),
        "test_assets": te["symbol"].nunique(),
    })
wf_table = pd.DataFrame(rows)
display(wf_table) if "display" in dir() else print(wf_table.to_string(index=False))
''')

S4_MD = md("""# Stage 4 — Model Training & Diagnostic Inspection

For each model family, fit on each walk-forward train slice and predict on the
OOS test slice:

1. **Neural sequence models** — NLinear, AR(1)x, DLinear, LSTM, PatchTST
   (PyTorch), trained on the pooled-Sharpe objective, **on the GPU pod**.
2. **Volatility regime models** — GARCH(1,1), EGARCH, GJR-GARCH, HAR-RV;
   inspect conditional variance.
3. **Instance & tree models** — k-NN, Lasso/ElasticNet, XGBoost, LightGBM,
   CatBoost.

Inspect raw predicted signals `s_t ∈ [-1,1]` and target position weights
`w_t = s_t · (σ_tgt / σ_t)`.

> In **smoke** mode this cell is trivially fast; in **full** mode on the pod it
> is the heavy GPU workload.""")

S4_CODE = code(r'''
from strategy_builder.trainer import run_benchmark_model

def run_neural(name):
    t0 = time.time()
    res = run_benchmark_model(
        name, panel, FEATURE_COLS, symbols,
        lookback=LOOKBACK, hidden=HIDDEN, seeds=SEEDS, top_seeds=TOP_SEEDS,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        sigma_tgt=SIGMA_TGT, use_ticker_emb=True, verbose=True, device=DEVICE)
    res["model"] = name
    res["seconds"] = round(time.time() - t0, 1)
    return res

NEURAL_RESULTS = {}
for name in NEURAL_MODELS:
    print(f"\n===== NEURAL: {name} =====")
    NEURAL_RESULTS[name] = run_neural(name)
    print(f"{name}: {len(NEURAL_RESULTS[name]['weights']):,} weight rows, "
          f"{NEURAL_RESULTS[name]['windows']} windows, "
          f"{NEURAL_RESULTS[name]['seconds']}s")
    # save per-model OOS weights
    NEURAL_RESULTS[name]["weights"].to_csv(BENCH_DIR / f"weights_{name}.csv", index=False)
''')

S4_CODE2 = code(r'''
# --- Inspect signals & weights for the first neural model ---
first_name = list(NEURAL_RESULTS.keys())[0]
w = NEURAL_RESULTS[first_name]["weights"]
print(f"Model: {first_name} | OOS weight rows: {len(w):,}")

# reconstruct position signal s_t in [-1,1] from weight = s * sigma_tgt / sigma
merged = w.merge(panel[["time", "symbol", "sigma"]], on=["time", "symbol"], how="left")
merged["signal"] = merged["weight"] / (SIGMA_TGT / merged["sigma"].clip(lower=1e-6))
print("\nPosition signal s_t:")
print(merged["signal"].describe().round(3))
print("\nTarget weight w_t:")
print(merged["weight"].describe().round(5))
print("\nSample of raw weights (time | symbol | weight):")
print(w.head(8).to_string(index=False))
''')

S4_CODE3 = code(r'''
# --- Volatility regime models: inspect conditional variance ---
import matplotlib
matplotlib.use("Agg")
from strategy_builder.volatility_models import (garch11_variance, egarch_variance,
                                                gjr_garch_variance, har_rv_forecast)

vol_sample = "SPY"
g = panel[panel["symbol"] == vol_sample].sort_values("time")
rets = g["ret_1"].fillna(0.0).to_numpy()

def series(cond_var):
    return pd.Series(cond_var, index=g["time"].values).dropna()

vols = {
    "GARCH(1,1)": series(garch11_variance(rets)),
    "EGARCH":     series(egarch_variance(rets)),
    "GJR-GARCH":  series(gjr_garch_variance(rets)),
    "EWMA(0.94)": series(np.fromiter(
        __import__("strategy_builder.volatility_models", fromlist=["ewma_variance"])
        .ewma_variance(rets), dtype=float)),
}
for name_, s in vols.items():
    print(f"{name_:<12s} mean σ² = {s.mean():.2e}  last σ = {np.sqrt(s.iloc[-1]):.4f}")

fig, ax = plt.subplots(figsize=(11, 4))
for name_, s in vols.items():
    ax.plot(s.index, np.sqrt(s).to_numpy(), label=name_, linewidth=1)
ax.set_title(f"Conditional volatility σ_t — {vol_sample}")
ax.set_ylabel("σ_t"); ax.legend(ncol=4); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
''')

S4_CODE4 = code(r'''
# --- Classical / tree / instance models ---
from strategy_builder.classical import CLASSICAL_REGISTRY

CLASSICAL_RESULTS = {}
for name in CLASSICAL_MODELS:
    if name not in CLASSICAL_REGISTRY:
        print(f"SKIP {name}: not in registry")
        continue
    t0 = time.time()
    try:
        weights = CLASSICAL_REGISTRY[name](
            panel, FEATURE_COLS, symbols,
            lookback=LOOKBACK, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS)
        CLASSICAL_RESULTS[name] = {"model": name, "weights": weights,
                                   "seconds": round(time.time() - t0, 1)}
        weights.to_csv(BENCH_DIR / f"weights_{name}.csv", index=False)
        print(f"{name}: {len(weights):,} weight rows, "
              f"{CLASSICAL_RESULTS[name]['seconds']}s")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        CLASSICAL_RESULTS[name] = {"model": name, "error": str(e),
                                   "weights": pd.DataFrame(columns=["time","symbol","weight"])}
''')

S5_MD = md("""# Stage 5 — Rigorous Financial Metric & Statistical Verification

For every model's combined OOS portfolio return series compute:

- Annualized **Sharpe** & **Sortino** ratio
- **CAGR %** & **Max Drawdown %**
- **Calmar** & **Win Rate (Hit %)**
- **Newey-West HAC** t-statistic (H₀: zero alpha)
- **Breakeven Transaction Friction** $c^*$ (bps)

Render an interactive cumulative equity curve (models vs Buy-and-Hold).""")

S5_CODE = code(r'''
from strategy_builder.backtest import (full_metrics, passive_benchmark,
                                       breakeven_costs, equity_curve_from_weights)

passive = passive_benchmark(panel)
print("Passive (equal-weight long-only) daily returns:", len(passive))

ALL_WEIGHTS = {}
for d in list(NEURAL_RESULTS.values()) + list(CLASSICAL_RESULTS.values()):
    if d.get("weights") is not None and not d["weights"].empty:
        ALL_WEIGHTS[d["model"]] = d["weights"]

summary = []
for name, w in ALL_WEIGHTS.items():
    try:
        m = full_metrics(w, panel, passive)
        m["model"] = name
        summary.append(m)
        print(f"{name:<12s} sharpe={m['sharpe']:+.3f} cagr={m['cagr']*100:+.1f}% "
              f"maxDD={m['max_dd']*100:.1f}% calmar={m['calmar']:.2f} "
              f"hit={m['hit_rate']*100:.1f}% t_hac={m['t_hac']:+.2f} "
              f"c*={m.get('breakeven_bps', 0):.1f}bps")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: METRIC FAIL {e}")

summary.sort(key=lambda s: s.get("sharpe", -99), reverse=True)
leaderboard = pd.DataFrame(summary)
leaderboard_path = BENCH_DIR / "leaderboard.json"
leaderboard_path.write_text(
    json.dumps({"mode": RUN_MODE, "generated": pd.Timestamp.now().isoformat(),
                "summary": summary}, indent=2, default=str),
    encoding="utf-8")
print("\nLeaderboard saved ->", leaderboard_path)
print("\n===== LEADERBOARD (sorted by OOS Sharpe) =====")
cols = ["model","sharpe","cagr","max_dd","calmar","hit_rate","t_hac","turnover"]
print(leaderboard[cols].to_string(index=False))
''')

S5_CODE2 = code(r'''
# --- Cumulative equity curves vs Buy-and-Hold passive ---
import matplotlib
matplotlib.use("Agg")
fig, ax = plt.subplots(figsize=(12, 6))
bench = (1 + passive).cumprod()
ax.plot(bench.index, bench.to_numpy(), color="black", linewidth=2,
        label="Buy & Hold (passive)")

for name, w in ALL_WEIGHTS.items():
    try:
        ec = equity_curve_from_weights(w, panel)
        ax.plot(ec.index, ec.to_numpy(), linewidth=1, label=name)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: curve fail {e}")
ax.set_yscale("log")
ax.set_title("Cumulative equity — all models vs Buy & Hold")
ax.set_ylabel("Equity (log)"); ax.legend(ncol=4, fontsize=8)
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
''')

S5_CODE3 = code(r'''
# --- Breakeven transaction friction c* (bps) for top models ---
be = {}
top = summary[: min(4, len(summary))]
for m in top:
    name = m["model"]
    try:
        be[name] = breakeven_costs(ALL_WEIGHTS[name], panel).head(8).to_dict("records")
    except Exception as e:  # noqa: BLE001
        be[name] = {"error": str(e)}
(BENCH_DIR / "breakeven.json").write_text(json.dumps(be, indent=2, default=str),
                                          encoding="utf-8")
for name, recs in be.items():
    print(f"\n{name} — top-8 breakeven costs (bps):")
    if isinstance(recs, list):
        for r in recs:
            print(f"  {r.get('symbol','?'):<12s} c* = {r.get('breakeven_bps',0):>8.1f} bps "
                  f"(turnover/yr {r.get('turnover_ann',0):.1f})")
    else:
        print(" ", recs)
''')

S6_MD = md("""# Stage 6 — Modular Extensibility (Paper Replication Template)

A standardized `@register_model` decorator so any newly published arXiv/NeurIPS
trading paper or custom alpha formula can be added in a single cell and
**automatically** evaluated against the master leaderboard.

New models only need to implement the standard interface:
`fn(panel, feature_cols, symbols, **kw) -> DataFrame[time, symbol, weight]`.""")

S6_CODE = code(r'''
# --- Model registry with @register_model ---
import functools

MODEL_REGISTRY = {}

def register_model(name=None, family="custom", description=""):
    """Decorator: register a model factory for the master leaderboard.

    The wrapped callable must return DataFrame[time, symbol, weight].
    """
    def deco(fn):
        key = name or fn.__name__
        MODEL_REGISTRY[key] = {
            "fn": fn, "family": family, "description": description,
        }
        return fn
    return deco

@register_model("momentum_cross", family="alpha",
                description="21d vs 63d EWMA crossover, vol-scaled")
def momentum_cross(panel, feature_cols, symbols, lookback=64,
                   train_months=36, test_months=6, sigma_tgt=0.10):
    rows = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("time")
        fast = g["close"] if "close" in g else np.exp(np.log1p(g["ret_1"]).cumsum())
        e21 = fast.ewm(span=21, adjust=False).mean()
        e63 = fast.ewm(span=63, adjust=False).mean()
        sig = np.sign(e21 - e63)
        for i, (_, r) in enumerate(g.iterrows()):
            vs = 1.0 / max(float(r.get("sigma", 1e-4)), 1e-6)
            rows.append({"time": r["time"], "symbol": sym,
                         "weight": float(sig.iloc[i] * sigma_tgt * vs)})
    return pd.DataFrame(rows)

print("Registered models:", sorted(MODEL_REGISTRY))
''')

S6_CODE2 = code(r'''
# --- Evaluate every registered custom model on the master leaderboard ---
for key, spec in MODEL_REGISTRY.items():
    t0 = time.time()
    try:
        w = spec["fn"](panel, FEATURE_COLS, symbols, lookback=LOOKBACK,
                       train_months=TRAIN_MONTHS, test_months=TEST_MONTHS)
        w.to_csv(BENCH_DIR / f"weights_{key}.csv", index=False)
        m = full_metrics(w, panel, passive)
        m["model"] = key; m["family"] = spec["family"]
        summary.append(m)
        print(f"[{spec['family']:>5s}] {key:<20s} sharpe={m['sharpe']:+.3f} "
              f"cagr={m['cagr']*100:+.1f}% maxDD={m['max_dd']*100:.1f}% "
              f"t_hac={m['t_hac']:+.2f} ({round(time.time()-t0,1)}s)")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"[{key}] FAILED: {e}")

# Rewrite leaderboard to include custom models
summary.sort(key=lambda s: s.get("sharpe", -99), reverse=True)
leaderboard = pd.DataFrame(summary)
(BENCH_DIR / "leaderboard.json").write_text(
    json.dumps({"mode": RUN_MODE, "generated": pd.Timestamp.now().isoformat(),
                "summary": summary}, indent=2, default=str), encoding="utf-8")
print("\nFinal leaderboard:")
cols = ["model","family","sharpe","cagr","max_dd","calmar","hit_rate","t_hac"]
print(leaderboard[[c for c in cols if c in leaderboard.columns]].to_string(index=False))
''')

FOOTER = md("""## End of 6-stage empirical research lab

- **Stage 1** — verified clean, leakage-free 29-asset cross-asset panel.
- **Stage 2** — standardized vol-normalized features + target; tensor `(N, L, F)`.
- **Stage 3** — purge & embargo walk-forward splits.
- **Stage 4** — neural / volatility / tree models trained (GPU on pod).
- **Stage 5** — full metric suite + equity curves + breakeven costs.
- **Stage 6** — `@register_model` paper-replication template.

Artifacts: `data/benchmark/weights_*.csv`, `data/benchmark/leaderboard.json`,
`data/benchmark/breakeven.json`.

To add a new paper model, define it in a cell with `@register_model`, then
re-run Stage 6 — it is scored automatically on the master leaderboard.""")


def build() -> Path:
    cells = [
        md("# Project Agonistes — Empirical Research Lab (01)"),
        md("Cell-by-cell, zero-black-box research protocol across 6 stages. "
           "Run **smoke** (`RUN_MODE=smoke`) for a fast end-to-end sanity pass, "
           "or **full** (`RUN_MODE=full`) for the complete benchmark on the GPU pod."),
        code(PRELUDE),
        code(CONFIG),
        S1_MD, S1_CODE, S1_CODE2, S1_CODE3,
        S2_MD, S2_CODE, S2_CODE2, S2_CODE3,
        S3_MD, S3_CODE,
        S4_MD, S4_CODE, S4_CODE2, S4_CODE3, S4_CODE4,
        S5_MD, S5_CODE, S5_CODE2, S5_CODE3,
        S6_MD, S6_CODE, S6_CODE2,
        FOOTER,
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote notebook: {OUT}")
    print(f"Cells: {len(cells)}")
    return OUT


if __name__ == "__main__":
    build()
