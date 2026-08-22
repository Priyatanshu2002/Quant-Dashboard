"""Node H — RL Execution Agent: order type, size and timing (plan §6).

Uses the trained PPO policy when available; otherwise a rule-based policy
(MARKET order, no delay, size from the portfolio decision).
"""
from __future__ import annotations

from pathlib import Path

from langgraph_app.src.state import DebateState
from core.logging import get_logger

log = get_logger(__name__)


def node_h_execution(state: DebateState) -> DebateState:
    decision = state["portfolio_decision"]
    order = {
        "symbol": decision["symbol"],
        "side": "BUY" if decision["direction"] == "LONG" else "SELL",
        "order_type": "MARKET",
        "position_size_pct": decision["position_size_pct"],
        "notional_usd": decision["notional_usd"],
        "timing_delay_bars": 0,
    }

    model_path = Path("data/rl_agent/ppo_agonistes.zip")
    if model_path.exists():
        try:
            import numpy as np
            from stable_baselines3 import PPO
            from rl_agent.environment import TradingEnvironment

            env = TradingEnvironment({
                "commission_rate": 0.001, "max_drawdown_limit": 0.15,
                "symbols": [decision["symbol"]],
            })
            model = PPO.load(str(model_path))
            obs, _ = env.reset()
            # If real feature data was loaded, leave the real obs; otherwise fall
            # back to a stable zero-centred baseline (not pure noise).
            if not env._using_real:
                obs[:] = np.zeros(128, dtype=np.float32)
                obs[0] = 1.0
            action, _ = model.predict(obs, deterministic=True)
            size, order_type, delay = float(action[0]), int(round(action[1])), int(round(action[2]))
            order.update({
                "position_size_pct": round(min(max(size, 0.0), 0.15), 4),
                "order_type": ["LIMIT", "MARKET", "TWAP"][order_type],
                "timing_delay_bars": delay,
            })
            log.info("Node H: PPO policy applied: %s", order)
        except Exception as e:  # noqa: BLE001
            log.warning("RL inference failed (%s) — rule-based order", e)
    else:
        log.info("Node H: no PPO checkpoint (data/rl_agent) — rule-based MARKET order")

    return {"execution_order": order}
