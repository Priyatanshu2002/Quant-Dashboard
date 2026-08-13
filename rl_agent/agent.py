"""PPO agent wrapper (Stable-Baselines3) — plan §6.2 config."""
from __future__ import annotations

from pathlib import Path

import yaml

from core.logging import get_logger

log = get_logger(__name__)

CONFIG_PATH = Path(__file__).parent / "configs" / "ppo_config.yaml"


def load_ppo_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(env, config: dict | None = None):
    """Create a PPO model from the plan §6.2 configuration."""
    from stable_baselines3 import PPO

    config = config or load_ppo_config()
    return PPO(
        policy=config.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=config.get("learning_rate", 3.0e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        policy_kwargs=config.get("policy_kwargs", {"net_arch": [256, 256, 128]}),
        verbose=1,
        seed=42,
    )
