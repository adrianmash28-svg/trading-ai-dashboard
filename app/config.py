from dataclasses import dataclass, field

from strategy_research_worker import (
    APPROVED_EMA_LONG,
    APPROVED_EMA_SHORT,
    APPROVED_MOMENTUM_WEIGHTS,
    APPROVED_REL_VOL,
    APPROVED_RR_WEIGHTS,
    APPROVED_RSI_LONG,
    APPROVED_RSI_SHORT,
    APPROVED_SCORE_THRESHOLDS,
    APPROVED_STOP_MULTIPLIERS,
    APPROVED_STRUCTURE_WEIGHTS,
    APPROVED_TP1_MULTIPLIERS,
    APPROVED_TP2_MULTIPLIERS,
    APPROVED_TREND_WEIGHTS,
    APPROVED_VOLUME_WEIGHTS,
)


@dataclass(slots=True)
class ResearchConfig:
    research_interval_seconds: int = 30
    challengers_per_cycle: int = 3
    minimum_trade_count: int = 20
    max_allowed_drawdown: float = 1500.0
    minimum_profit_factor: float = 1.05
    minimum_win_rate: float = 45.0
    minimum_out_of_sample_score: float = 0.05
    maximum_win_rate_drift: float = 18.0
    maximum_profit_factor_drift: float = 0.75
    paper_trading_min_cycles: int = 2
    paper_trading_min_signals: int = 6
    promotion_min_signal_count: int = 6
    promotion_margin_over_champion: float = 50.0
    promotion_win_rate_margin: float = 0.0
    promotion_profit_factor_margin: float = 0.02
    max_log_history_length: int = 200
    max_recent_challengers: int = 50
    max_rejected_challengers: int = 50
    max_promotion_history: int = 50
    backtest_symbols: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "IWM", "DIA"]
    )
    mutation_ranges: dict[str, list[float | int]] = field(
        default_factory=lambda: {
            "score_threshold": list(APPROVED_SCORE_THRESHOLDS),
            "rsi_long_min": list(APPROVED_RSI_LONG),
            "rsi_short_max": list(APPROVED_RSI_SHORT),
            "rel_vol_min": list(APPROVED_REL_VOL),
            "ema_short_len": list(APPROVED_EMA_SHORT),
            "ema_long_len": list(APPROVED_EMA_LONG),
            "stop_multiplier": list(APPROVED_STOP_MULTIPLIERS),
            "tp1_multiplier": list(APPROVED_TP1_MULTIPLIERS),
            "tp2_multiplier": list(APPROVED_TP2_MULTIPLIERS),
            "trend_weight": list(APPROVED_TREND_WEIGHTS),
            "momentum_weight": list(APPROVED_MOMENTUM_WEIGHTS),
            "volume_weight": list(APPROVED_VOLUME_WEIGHTS),
            "structure_weight": list(APPROVED_STRUCTURE_WEIGHTS),
            "rr_weight": list(APPROVED_RR_WEIGHTS),
            "max_position_size": [0.5, 0.75, 1.0],
            "cooldown_bars": [1, 3, 5, 8],
            "lookback_window": [10, 20, 30],
            "signal_weighting": [0.8, 1.0, 1.2],
        }
    )


research_config = ResearchConfig()
