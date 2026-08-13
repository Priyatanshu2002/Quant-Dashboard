"""TFT signal model (plan §4) — Temporal Fusion Transformer via pytorch-forecasting.

Heavy ML dependencies are imported lazily; this module also defines the
TFTSignal output dataclass used by the LangGraph debate layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TFTSignal:
    symbol: str
    asset_class: str
    timeframe: str                      # INTRADAY / SWING / LONGTERM

    # Predictions per horizon
    return_1d_p10: float | None = None  # Downside 10th percentile
    return_1d_p50: float | None = None  # Median expected return
    return_1d_p90: float | None = None  # Upside 90th percentile
    return_5d_p50: float | None = None
    return_20d_p50: float | None = None

    # Derived signals
    direction: str = "NEUTRAL"          # LONG / SHORT / NEUTRAL
    conviction: float = 0.0             # 1 - prediction_interval_width
    volatility_forecast: float = 0.0

    # What drove this prediction (from TFT attention weights)
    top_features: list[tuple[str, float]] = field(default_factory=list)


def signal_from_quantiles(symbol: str, asset_class: str, timeframe: str,
                          p10: float, p50: float, p90: float,
                          long_threshold: float = 0.002,
                          short_threshold: float = -0.002,
                          top_features: list[tuple[str, float]] | None = None) -> TFTSignal:
    """Derive a TFTSignal from predicted return quantiles (1d horizon)."""
    width = abs(p90 - p10)
    conviction = float(max(0.0, min(1.0, 1.0 - width / 0.06)))  # wide interval → low conviction

    if p50 >= long_threshold and p10 > -p50 * 0.5:
        direction = "LONG"
    elif p50 <= short_threshold and p90 < -p50 * 0.5:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return TFTSignal(
        symbol=symbol, asset_class=asset_class, timeframe=timeframe,
        return_1d_p10=p10, return_1d_p50=p50, return_1d_p90=p90,
        direction=direction, conviction=conviction,
        volatility_forecast=width / 2, top_features=top_features or [],
    )


class AgonistesTFT:
    """TFT construction per plan §4.2 (imports pytorch-forecasting lazily)."""

    # Static features (don't change over time for a given asset)
    STATIC_CATEGORICALS = ["asset_class", "sector", "exchange"]
    STATIC_REALS = ["market_cap_log", "avg_daily_volume_log"]

    # Time-varying known-future (we know them in advance — calendar)
    TIME_VARYING_KNOWN = [
        "days_to_earnings", "day_of_week", "month_end_effect",
        "quarter_end_effect", "days_to_expiry",
    ]

    # Time-varying unknown (observed up to present only)
    TIME_VARYING_UNKNOWN = [
        "rsi_14", "rsi_7", "macd_histogram", "bb_pct_b", "atr_pct",
        "volume_z_score", "vwap_pct", "price_vs_ema200_pct", "realized_vol_20",
        "adx_14", "cci_20", "stoch_k", "return_1bar", "return_5bar", "return_20bar",
        "eps_surprise_pct", "revenue_yoy_growth", "dcf_margin_of_safety",
        "fcf_yield", "roic", "debt_to_equity", "earnings_call_sentiment",
        "margin_trend", "revenue_accel", "forward_pe", "peg_ratio",
        "sentiment_score", "sentiment_momentum", "reddit_sentiment",
        "gdelt_sentiment", "sentiment_extreme",
        "vix", "yield_curve_spread", "btc_dominance", "dxy",
    ]

    TARGETS = ["future_return_1d", "future_return_5d", "future_return_20d"]

    def build_model(self, training_dataset, hidden_size: int = 128,
                    attention_head_size: int = 4, dropout: float = 0.1,
                    hidden_continuous_size: int = 64, learning_rate: float = 1e-3):
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        return TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
        )
