"""Custom Gymnasium trading environment (plan §6.1).

The RL agent decides HOW and WHEN to execute a trade that the Gating Node
has already approved. It does not decide WHAT to trade.

Action: [position_size_pct (0-1), order_type (0=LIMIT/1=MARKET/2=TWAP), timing_delay (0-2 bars)]
Observation: 128-dim feature vector including portfolio state.

Data: uses REAL labeled feature vectors from the feature store when available
(observations = feature rows, return = realized forward return from the label),
with a synthetic fallback only when no real data exists — so PPO trains on
actual market features, not noise.
"""
from __future__ import annotations

import numpy as np

from core.logging import get_logger
from rl_agent.reward import compute_reward, estimate_slippage

log = get_logger(__name__)

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]

OBS_DIM = 128


def load_real_frame(symbols: list[str] | None = None, timeframe: str = "SWING",
                    max_rows: int = 20_000):
    """Load labeled feature vectors from the store into a plain array.

    Returns (features: np.ndarray float32 [N, OBS_DIM], returns: np.ndarray
    float32 [N]) where `returns` is the realized forward return per row. Returns
    (None, None) when the store has no usable rows, so the environment can fall
    back to synthetic data.
    """
    try:
        from feature_engineering.feature_store import load_training_frame
        df = load_training_frame(symbols=symbols, timeframe=timeframe)
    except Exception:  # noqa: BLE001
        return None, None
    if df is None or df.empty:
        return None, None

    # Forward-return label columns we can use as the realized per-step return.
    ret_cols = ["future_return_1d", "future_return_5d", "future_return_20d"]
    label = next((c for c in ret_cols if c in df.columns), None)
    if label is None:
        return None, None

    # Numeric feature columns = everything except identifiers/labels/timestamps.
    skip = {"symbol", "asset_class", "sector", "exchange", "time", "timeframe",
            *ret_cols}
    feat_cols = [c for c in df.columns if c not in skip]
    feat_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)
                 and df[c].notna().any()]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        dtype=np.float32)
    # Realized return: scale the 1d label if present, else use 5d/5 as a per-bar
    # proxy so rewards are on a comparable per-step scale.
    if label == "future_return_1d":
        r = df[label].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        divisor = 5 if label == "future_return_5d" else 20
        r = df[label].fillna(0.0).to_numpy(dtype=np.float32) / divisor

    if len(X) < 2 or X.shape[1] == 0:
        return None, None
    return X, r


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
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

            # Load real data (graceful fallback to synthetic).
            symbols = self.config.get("symbols")
            timeframe = self.config.get("timeframe", "SWING")
            self._X, self._R = load_real_frame(symbols=symbols, timeframe=timeframe)
            self._using_real = self._X is not None
            self._i = 0
            self._n = self._X.shape[0] if self._using_real else 250
            self._episode_len = self._n - 1
            self._step = 0
            self._nav = 1.0
            self._peak_nav = 1.0
            if not self._using_real:
                log.warning("RL env: no real feature data for %s — using synthetic fallback",
                            symbols)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            self._nav = 1.0
            self._peak_nav = 1.0
            if self._using_real:
                self._i = int(self.np_random.integers(0, max(1, self._episode_len)))
            return self._obs(), {}

        def step(self, action):
            size = float(np.clip(action[0], 0.0, 1.0))
            order_type = int(round(float(action[1])))
            delay = int(round(float(action[2])))

            if self._using_real:
                # Realized per-bar return at this observation, modulated by the
                # chosen timing delay (delayed entries skip forward bars).
                j = min(self._i + delay, self._n - 1)
                raw_return = float(self._R[j]) if j < len(self._R) else 0.0
                self._i = j + 1
            else:
                # Synthetic fallback: drift μ=0.0008, σ=0.01/bar.
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
            terminated = self._step >= self._episode_len
            return self._obs(), reward, terminated, False, {"nav": self._nav}

        def _obs(self) -> np.ndarray:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[0] = self._nav
            obs[1] = 1.0 - self._nav / self._peak_nav
            obs[2] = self._step / float(max(self._episode_len, 1))
            if self._using_real and self._i < self._n:
                feats = self._X[self._i]
                obs[3:3 + min(feats.shape[0], OBS_DIM - 3)] = feats[:OBS_DIM - 3]
            else:
                obs[3:] = self.np_random.normal(0, 0.1, OBS_DIM - 3).astype(np.float32)
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
