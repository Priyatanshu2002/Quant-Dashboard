"""Regenerate data/benchmark/leaderboard_full.json (canonical model benchmark).

The full OOS walk-forward leaderboard (12 models) is served to the dashboard
via GET /api/benchmark and rendered on the React /benchmark page. Because
data/ is gitignored, this script embeds the results so the canonical file can
be recreated on any machine without re-running the (RunPod) benchmark.

Usage:
    python scripts/rebuild_benchmark_leaderboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark" / "leaderboard_full.json"

# model, sharpe, cagr, max_dd, calmar, hit_rate, t_hac, info_ratio
# (cagr/max_dd are fractions; None = weight output near-flat in OOS, so ~0)
ROWS = [
    ("nlinear",          3.852,  3.12,  -0.114, 27.3,  0.638,  2.81,  2.97),
    ("knn",              1.57,   None,  None,   None,  0.538,  1.03,  0.90),
    ("momentum_cross",   1.53,   0.978, -0.413, 2.37,  0.557,  3.14,  0.93),
    ("elasticnet",       1.51,   None,  None,   None,  0.546,  0.62,  0.74),
    ("tft",              0.48,   0.089, -0.164, 0.54,  0.538,  0.33,  0.13),
    ("patchtst",         -0.17,  -0.058, -0.104, -0.55, 0.554,  -0.15, -0.35),
    ("xgboost",          -1.99,  None,  None,   None,  0.438,  -1.92, -1.40),
    ("catboost",         -2.17,  None,  None,   None,  0.515,  -1.72, -1.37),
    ("lightgbm",         -2.50,  None,  None,   None,  0.415,  -2.36, -1.74),
    ("vlstm",            -4.41,  -0.44, -0.26,  -1.69, 0.431,  -2.93, -2.63),
    ("lstm",             -4.93,  -0.17, -0.095, -1.79, 0.408,  -3.52, -2.97),
    ("ridge",            -5.95,  None,  None,   None,  0.377,  -3.42, -0.69),
]


def main() -> None:
    summary = [
        {
            "model": m, "sharpe": s, "cagr": c, "max_dd": dd, "calmar": k,
            "hit_rate": h, "t_hac": t, "info_ratio": i,
        }
        for m, s, c, dd, k, h, t, i in ROWS
    ]
    doc = {
        "mode": "full_walk_forward",
        "generated": "2026-08-16T04:44:00.000000",
        "description": (
            "Full OOS walk-forward benchmark (36m train / 6m test), ranked by Sharpe. "
            "Run on RunPod RTX 4090 (torch 2.9.1+cu130) · 29-asset universe · 60 epochs "
            "· 3 seeds. null cagr/max_dd/calmar = model weight output was near-flat in "
            "OOS (tiny/noise positions)."
        ),
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary)} models -> {OUT}")


if __name__ == "__main__":
    main()
