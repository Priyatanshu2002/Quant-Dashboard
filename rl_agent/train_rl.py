"""PPO training loop with eval callback (plan §6.2)."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.logging import get_logger, setup_logging

log = get_logger(__name__)


def train(env, model, total_timesteps: int = 1_000_000,
          eval_freq: int = 10_000, n_eval_episodes: int = 20,
          reward_threshold: float = 1.5, save_dir: Path = Path("data/rl_agent")):
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    save_dir.mkdir(parents=True, exist_ok=True)
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_cb = EvalCallback(
        env, best_model_save_path=str(save_dir), log_path=str(save_dir),
        eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        deterministic=True, callback_on_new_best=stop_cb)

    log.info("Training PPO for %d timesteps (target reward %.2f)",
             total_timesteps, reward_threshold)
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(save_dir / "ppo_agonistes"))
    log.info("Saved final model to %s", save_dir / "ppo_agonistes.zip")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Train the PPO execution agent")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "ppo_config.yaml"))
    args = ap.parse_args()

    from rl_agent.agent import build_agent, load_ppo_config
    from rl_agent.environment import get_environment

    config = load_ppo_config(Path(args.config))
    env = get_environment({"commission_rate": 0.001, "max_drawdown_limit": 0.15})
    model = build_agent(env, config)
    train(env, model, total_timesteps=args.timesteps,
          eval_freq=config.get("eval_freq", 10_000),
          n_eval_episodes=config.get("n_eval_episodes", 20),
          reward_threshold=config.get("reward_threshold", 1.5))


if __name__ == "__main__":
    main()
