#!/usr/bin/env python3
"""Reference chart: Oxford benchmark (arXiv:2603.01820) published OOS Sharpe by model.

Purely the paper's reported numbers (Table 1/2, 2010-2025, gross, vol-targeted 10%)
— NOT results from our local run (which awaits a GPU machine).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("data/charts")
OUT.mkdir(parents=True, exist_ok=True)

# (model, family, sharpe_2010_2025, cagr, max_dd)
DATA = [
    ("VLSTM (VSN+LSTM)",        "hybrid",      2.40, 0.263, -0.229),
    ("LPatchTST (LSTM+PatchTST)", "hybrid",    2.31, 0.255, -0.174),
    ("TFT",                     "tft",         2.20, 0.240, -0.232),
    ("xLSTM",                   "recurrent",   1.79, 0.194, -0.141),
    ("PsLSTM",                  "recurrent",   1.74, 0.187, -0.131),
    ("VxLSTM (VSN+xLSTM)",      "recurrent",   1.69, 0.194, -0.118),
    ("LSTM",                    "recurrent",   1.48, 0.135, -0.342),
    ("VSN+Mamba2",              "ssm",         1.10, 0.097, -0.163),
    ("Mamba2",                  "ssm",         0.78, 0.059, -0.263),
    ("PatchTST",                "transformer", 0.76, 0.085, -0.176),
    ("AR1x",                    "linear",      0.77, 0.081, -0.167),
    ("NLinear",                 "linear",      0.66, 0.066, -0.180),
    ("DLinear",                 "linear",      0.64, 0.075, -0.180),
    ("iTransformer",            "transformer", 0.35, 0.031, -0.264),
    ("Passive (equal-weight)",  "passive",     0.48, 0.044, -0.308),
]

FAMILY_COLORS = {
    "hybrid": "#2e7d32", "recurrent": "#1565c0", "tft": "#6a1b9a",
    "ssm": "#ef6c00", "transformer": "#00838f", "linear": "#757575",
    "passive": "#b71c1c",
}

rows = sorted(DATA, key=lambda r: r[2])
fig, ax = plt.subplots(figsize=(11, 7.5))
y = np.arange(len(rows))
bars = ax.barh(y, [r[2] for r in rows],
               color=[FAMILY_COLORS[r[1]] for r in rows], height=0.62)
for yi, r in zip(y, rows):
    ax.text(r[2] + 0.03, yi, f"{r[2]:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.text(-0.07, yi, f"{r[3]*100:.0f}%  ·  DD {r[4]*100:.0f}%",
            va="center", ha="right", fontsize=7.5, color="#555")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlabel("Out-of-sample Sharpe ratio (annualized, gross, vol-targeted 10%)", fontsize=10)
ax.set_title("Oxford DL-for-Finance benchmark — published OOS Sharpe 2010–2025\n"
             "Saly-Kaufmann, Wood, Calliess & Zohren, arXiv:2603.01820 (Table 1/2)\n"
             "Right labels: CAGR % · max drawdown %. Reference only — Agonistes run pending GPU.",
             fontsize=10.5)
ax.set_xlim(-0.75, 2.9)
ax.axvline(1.0, ls="--", lw=0.8, color="#999")
ax.text(1.02, len(rows) - 0.4, "Sharpe = 1.0 (institutional quality line)", fontsize=8, color="#777")
legend = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
ax.legend(legend, [k.upper() for k in FAMILY_COLORS], loc="lower right", fontsize=8, ncol=2)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
fig.savefig(OUT / "paper_benchmark_sharpe_reference.png", dpi=150)
print(f"saved {OUT / 'paper_benchmark_sharpe_reference.png'}")
