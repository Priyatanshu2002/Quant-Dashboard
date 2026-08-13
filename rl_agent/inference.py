"""RL inference — turn a gated trade into an execution order (used by Node H)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def predict_order(model_path: str | Path = "data/rl_agent/ppo_agonistes.zip",
                  obs: np.ndarray | None = None) -> dict:
    """Run the trained policy on an observation → execution action dict."""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    if obs is None:
        obs = np.zeros(128, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return {
        "position_size_pct": float(np.clip(action[0], 0, 1)),
        "order_type": ["LIMIT", "MARKET", "TWAP"][int(round(action[1]))],
        "timing_delay_bars": int(round(action[2])),
    }
