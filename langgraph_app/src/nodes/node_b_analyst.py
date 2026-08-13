"""Node B — Analyst: attach the TFT signal + produce a neutral summary."""
from __future__ import annotations

from pathlib import Path

from langgraph_app.src.state import DebateState
from core.logging import get_logger

log = get_logger(__name__)


def node_b_analyst(state: DebateState) -> DebateState:
    pack = state["analyst_pack"]
    tft_signal = None

    # Attach a real TFT signal when a trained checkpoint exists.
    ckpt_dir = Path("data/checkpoints")
    if ckpt_dir.exists() and any(ckpt_dir.glob("*.ckpt")):
        try:
            from transformer_model.inference import infer_signal
            from core.db import get_storage
            import yaml

            config = yaml.safe_load(
                open("transformer_model/configs/tft_swing.yaml", encoding="utf-8"))
            ohlcv = get_storage().query_ohlcv(pack.symbol)
            if not ohlcv.empty:
                tft_signal = infer_signal(pack.symbol, pack.asset_class, ohlcv,
                                          ckpt_dir, config)
        except Exception as e:  # noqa: BLE001
            log.warning("TFT inference unavailable for %s: %s", pack.symbol, e)

    if tft_signal is not None:
        pack.tft_direction = tft_signal.direction
        pack.tft_conviction = tft_signal.conviction
        pack.return_1d_p50 = tft_signal.return_1d_p50
        pack.return_5d_p50 = tft_signal.return_5d_p50
        pack.tft_top_features = tft_signal.top_features

    summary = (
        f"{pack.symbol}: screener {pack.composite_score:.0f}/100, "
        f"TFT {pack.tft_direction} (conv {pack.tft_conviction:.2f}), "
        f"RSI {pack.rsi_14}, macro regime {pack.macro_regime}"
    )
    log.info("Node B: %s", summary)
    return {"tft_signal": tft_signal, "analyst_summary": summary}
