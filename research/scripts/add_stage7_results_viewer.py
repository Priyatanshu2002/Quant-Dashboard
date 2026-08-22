"""Append Stage 7 — Research Results Viewer to the empirical research lab notebook.

Loads the committed research artifacts (research/results/*.csv + *.json) and
renders them as tables + plots inside the notebook, so the user can observe all
research results directly in Jupyter.

Idempotent: if a Stage 7 section already exists, it is replaced in place.
Existing cells (Stage 1-6 + recorded outputs) are left untouched.

Usage:
    python research/scripts/add_stage7_results_viewer.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "research" / "notebooks" / "01_empirical_research_lab.ipynb"


def md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(source)


def code(source: str) -> dict:
    return nbf.v4.new_code_cell(source)


MARKER = "# ==== STAGE 7 : RESEARCH RESULTS VIEWER ===="

STAGE7 = [
    md("# Stage 7 — Research Results Viewer\n\n"
       "Loads the committed research artifacts from `research/results/` and renders "
       "them as tables + charts so every result can be observed directly here."),

    code(f"""
{MARKER}
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("research/results")
print("Artifacts in research/results:")
for p in sorted(RESULTS.glob("*")):
    print(f"  {{p.name:<45}} {{p.stat().st_size:>9,}} B")
"""),

    md("### 7.1 MA-Crossover strategy backtests\n\n"
       "Per-strategy × per-symbol results: total return, CAGR, Sharpe, Sortino, "
       "Calmar, max drawdown, win rate, profit factor, alpha vs SPY."),

    code(f"""
{MARKER}
csv_path = RESULTS / "01_ma_crossover_results.csv"
if csv_path.exists():
    ma = pd.read_csv(csv_path)
    print(f"{{len(ma)}} backtests across {{ma['strategy'].nunique()}} strategies "
          f"x {{ma['symbol'].nunique()}} symbols\\n")
    show = ["strategy", "symbol", "total_return_pct", "cagr_pct", "sharpe",
            "sortino", "calmar", "max_drawdown_pct", "win_rate", "profit_factor",
            "alpha_vs_spy_pct", "info_ratio"]
    display(ma[[c for c in show if c in ma.columns]].head(25))
else:
    print("01_ma_crossover_results.csv not found")
"""),

    code(f"""
{MARKER}
json_path = RESULTS / "01_ma_crossover_results.json"
if json_path.exists():
    d = json.load(open(json_path, encoding="utf-8"))
    avg = d.get("per_strategy_avg", {{}})
    rows = [{{"strategy": k, **v}} for k, v in avg.items()]
    avg_df = pd.DataFrame(rows)
    print("Per-strategy averages (across symbols):")
    display(avg_df.sort_values("avg_sharpe", ascending=False))
else:
    print("01_ma_crossover_results.json not found")
"""),

    code(f"""
{MARKER}
# Sharpe / return / drawdown by strategy (average across symbols)
if "avg_df" in dir():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = avg_df["strategy"]
    for ax, col, title in [
        (axes[0], "avg_sharpe", "Avg Sharpe"),
        (axes[1], "avg_return_pct", "Avg Return %"),
        (axes[2], "avg_max_dd_pct", "Avg Max Drawdown %"),
    ]:
        ax.bar(x, avg_df[col], color="#4f8cff")
        ax.set_title(title); ax.set_xlabel("Strategy"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); plt.show()
"""),

    md("### 7.2 Cross-sectional model weights (strategy_builder benchmark)\n\n"
       "The committed `weights_*.csv` are each model's per-symbol allocation over "
       "time from the benchmark. Plot the top exposures and turnover."),

    code(f"""
{MARKER}
import glob as _glob
weight_files = sorted(_glob.glob(str(RESULTS / "weights_*.csv")))
print(f"Found {{len(weight_files)}} weight files: {{[Path(f).name for f in weight_files]}}")
weight_frames = {{}}
for f in weight_files:
    name = Path(f).stem.replace("weights_", "")
    weight_frames[name] = pd.read_csv(f, parse_dates=["time"])
    print(f"  {{name:<12}} rows={{len(weight_frames[name]):,}}  cols={{list(weight_frames[name].columns)}}")
"""),

    code(f"""
{MARKER}
# Turnover + top exposures per model (avg |dw| per rebalance)
if weight_frames:
    rows = []
    for name, w in weight_frames.items():
        w = w.sort_values("time")
        # daily turnover = sum of |weight change| / 2
        piv = w.pivot_table(index="time", columns="symbol", values="weight").fillna(0)
        if len(piv) > 1:
            turnover = (piv.diff().abs().sum(axis=1) / 2).mean()
        else:
            turnover = float("nan")
        rows.append({{"model": name, "rows": len(w), "avg_daily_turnover": turnover}})
    wt = pd.DataFrame(rows)
    print("Daily turnover per model (avg across the OOS window):")
    display(wt)
"""),

    code(f"""
{MARKER}
# Top-N exposures over time for the best available model (first with data)
if weight_frames:
    name = next(iter(weight_frames))
    w = weight_frames[name].sort_values("time")
    piv = w.pivot_table(index="time", columns="symbol", values="weight").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 4))
    # Signed weights (long/short) can't be stacked — use plain lines.
    piv.plot(ax=ax, alpha=0.6, legend=False)
    ax.set_title(f"Exposures over time — {{name}}")
    ax.set_ylabel("weight"); ax.set_xlabel("time")
    # legend: top 6 symbols by mean weight
    top = piv.mean().sort_values(ascending=False).head(6).index
    ax.legend(top, loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout(); plt.show()
"""),
]


def _src(c) -> str:
    return c.source if isinstance(c.source, str) else "".join(c.source)


def main() -> None:
    nb = nbf.read(NB, as_version=4)
    # Idempotent: drop any existing Stage 7 section (starts at the first cell
    # whose source marks the stage — the whole tail is the section).
    cut = next((i for i, c in enumerate(nb.cells)
                if "# Stage 7" in _src(c) or MARKER in _src(c)), None)
    if cut is not None:
        nb.cells = nb.cells[:cut]
    nb.cells.extend(STAGE7)
    for c in nb.cells:
        c.setdefault("id", "")
    nbf.write(nb, NB)
    print(f"Stage 7 appended to {NB} ({len(nb.cells)} cells total)")


if __name__ == "__main__":
    main()
