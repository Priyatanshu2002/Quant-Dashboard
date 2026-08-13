"""Custom Gymnasium trading environment (plan §6.1).

The RL agent decides HOW and WHEN to execute a trade that the Gating Node
has already approved. It does not decide WHAT to trade.

Action: [position_size_pct (0-1), order_type (0=LIMIT/1=MARKET/2=TWAP), timing_delay (0-2 bars)]
Observation: 128-dim feature vector including portfolio state.
"""
from __future__ import annotations

import numpy as np

from rl_agent.reward import compute_reward, estimate_slippage

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]


def _make_env_class():
    if gym is None:
        raise RuntimeError("gymnasium not installed: pip install -e '.[rl]'")

    class TradingEnvironment(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, config: dict | None = None):
            super().__init__()
            self.config = config or {
                "commission_rate": 0.001,
                "max_drawdown_limit": 0.15,
                "portfolio_volatility": 0.012,
            }
            # Action: [size (0-1), order type (0-2), timing delay (0-2 bars)]
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0, 0], dtype=np.float32),
                high=np.array([1.0, 2, 2], dtype=np.float32),
                dtype=np.float32,
            )
            # Observation: feature vector (128-dim) including portfolio state
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
            self._step = 0
            self._nav = 1.0
            self._peak_nav = 1.0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            self._nav = 1.0
            self._peak_nav = 1.0
            return self._obs(), {}

        def step(self, action):
            size = float(np.clip(action[0], 0.0, 1.0))
            order_type = int(round(float(action[1])))
            delay = int(round(float(action[2])))

            # Market simulation: the approved trade has drift μ=0.0008, σ=0.01/bar.
            raw_return = float(self.np_random.normal(0.0008, 0.01)) * (1 + delay * 0.3)
            slippage = estimate_slippage(size, ["LIMIT", "MARKET", "TWAP"][order_type])
            self._nav *= (1 + raw_return * size)
            self._peak_nav = max(self._peak_nav, self._nav)
            drawdown = 1.0 - self._nav / self._peak_nav

            reward = compute_reward(
                raw_return, size, self.config["portfolio_volatility"], drawdown,
                slippage, self.config["commission_rate"],
                self.config["max_drawdown_limit"])

            self._step += 1
            terminated = self._step >= 250
            return self._obs(), reward, terminated, False, {"nav": self._nav}

        def _obs(self) -> np.ndarray:
            obs = np.zeros(128, dtype=np.float32)
            obs[0] = self._nav
            obs[1] = 1.0 - self._nav / self._peak_nav
            obs[2] = self._step / 250.0
            obs[3:] = self.np_random.normal(0, 0.1, 125).astype(np.float32)
            return obs

        def render(self):
            return None

    return TradingEnvironment


TradingEnvironment = _make_env_class() if GYM_AVAILABLE else None  # type: ignore[assignment,misc]


def get_environment(config: dict | None = None):
    """Instantiate the environment (raises a clear error if gymnasium is absent)."""
    if TradingEnvironment is None:
        raise RuntimeError("gymnasium not installed: pip install -e '.[rl]'")
    return TradingEnvironment(config)
