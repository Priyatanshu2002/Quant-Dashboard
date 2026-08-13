"""SHAP / attention feature importance — drives feature pruning (plan §3 note).

For TFT, the interpretable multi-head attention gives per-step feature
importance directly (VariableSelectionNetwork). This module extracts it and
ranks features; a SHAP fallback is available for tabular baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.logging import get_logger

log = get_logger(__name__)


def attention_importance(model, dataloader) -> pd.DataFrame:
    """Aggregate TFT encoder attention across batches → feature importance."""
    import torch

    model.eval()
    importance: dict[str, float] = {}
    with torch.no_grad():
        for batch in dataloader:
            enc, _ = model.encode(batch)
            # encoder_attention shape: (batch, heads, time)
            attn = model._get_attention(batch, enc)
            if attn is None:
                continue
            weights = attn["encoder_attention"].mean(dim=(0, 1))  # (time,)
            vals = enc["encoder_lengths"].float()
            norm = vals.sum().clamp(min=1)
            for i, feat in enumerate(batch.encoder_cont):
                importance[feat] = importance.get(feat, 0.0) + (
                    (attn["encoder_attention"] * batch.encoder_cont[feat].unsqueeze(1).abs())
                    .mean().item() / max(norm.item(), 1e-8))
    if not importance:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(sorted(importance.items(), key=lambda kv: -kv[1]),
                      columns=["feature", "importance"])
    df["importance_rank"] = range(1, len(df) + 1)
    return df


def shap_feature_importance(X: pd.DataFrame, y: pd.Series, model=None,
                            n_samples: int = 200) -> pd.DataFrame:
    """Tabular SHAP fallback (e.g., for a gradient-boosted baseline)."""
    import shap

    X = X.dropna().sample(min(n_samples, len(X)), random_state=42)
    y = y.loc[X.index]
    if model is None:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    import numpy as np
    mean_abs = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"feature": X.columns, "importance": mean_abs})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def prune_low_importance(importance: pd.DataFrame, threshold_rank: int = 30) -> list[str]:
    """Keep the top-N features; returns the pruned-away list."""
    dropped = importance.iloc[threshold_rank:]["feature"].tolist()
    log.info("SHAP pruning: dropping %d low-importance features: %s",
             len(dropped), dropped[:10])
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="data/checkpoints")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    log.info("Run after training: python -m transformer_model.shap_analysis "
             "--checkpoints data/checkpoints")


if __name__ == "__main__":
    main()
