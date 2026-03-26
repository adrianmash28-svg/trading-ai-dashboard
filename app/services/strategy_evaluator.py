from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import ResearchConfig
from strategy_research_worker import run_strategy_backtest, summarize_backtest_results


@dataclass(slots=True)
class EvaluationResult:
    results_summary: dict[str, Any]
    backtest_metrics: dict[str, Any]
    rejection_reasons: list[str]


def _empty_summary() -> dict[str, Any]:
    return {
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "average_trade": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "profit_factor": 0.0,
        "sharpe_or_simple_risk_adjusted_score": 0.0,
        "learning_score": 0.0,
    }


def _enhance_summary(summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    enhanced = {**_empty_summary(), **(summary or {})}
    if trades is None or trades.empty:
        return enhanced
    pnl = pd.to_numeric(trades.get("pnl", 0.0), errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    enhanced["average_trade"] = round(float(pnl.mean()), 2)
    enhanced["profit_factor"] = round(float(wins.sum() / abs(losses.sum() or 1.0)), 2) if not wins.empty else 0.0
    std = float(pnl.std(ddof=0) or 0.0)
    enhanced["sharpe_or_simple_risk_adjusted_score"] = round(float((pnl.mean() / std) if std else pnl.mean()), 4)
    return enhanced


def _select_split_column(trades: pd.DataFrame) -> str | None:
    for name in ("close_time", "timestamp"):
        if name in trades.columns:
            return name
    return None


def _split_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    split_column = _select_split_column(trades)
    if not split_column:
        midpoint = max(len(trades) * 7 // 10, 1)
        return trades.iloc[:midpoint].copy(), trades.iloc[midpoint:].copy()
    ordered = trades.copy()
    ordered[split_column] = pd.to_datetime(ordered[split_column], errors="coerce")
    ordered = ordered.sort_values(split_column)
    midpoint = max(len(ordered) * 7 // 10, 1)
    return ordered.iloc[:midpoint].copy(), ordered.iloc[midpoint:].copy()


def _calculate_summary(trades: pd.DataFrame) -> dict[str, Any]:
    return _enhance_summary(summarize_backtest_results(trades), trades)


def _metric_unstable(in_sample: dict[str, Any], out_of_sample: dict[str, Any], config: ResearchConfig) -> bool:
    return (
        abs(float(in_sample.get("win_rate", 0.0)) - float(out_of_sample.get("win_rate", 0.0))) > config.maximum_win_rate_drift
        or abs(float(in_sample.get("profit_factor", 0.0)) - float(out_of_sample.get("profit_factor", 0.0))) > config.maximum_profit_factor_drift
    )


def _build_rejection_reasons(overall: dict[str, Any], in_sample: dict[str, Any], out_of_sample: dict[str, Any], config: ResearchConfig) -> list[str]:
    reasons: list[str] = []
    if int(overall.get("num_trades", 0)) < config.minimum_trade_count:
        reasons.append("Too few total trades")
    if int(out_of_sample.get("num_trades", 0)) < max(config.minimum_trade_count // 2, 5):
        reasons.append("Too few out-of-sample trades")
    if abs(float(overall.get("max_drawdown", 0.0))) > config.max_allowed_drawdown:
        reasons.append("Max drawdown cap exceeded")
    if float(overall.get("profit_factor", 0.0)) < config.minimum_profit_factor:
        reasons.append("Profit factor below minimum threshold")
    if float(overall.get("win_rate", 0.0)) < config.minimum_win_rate:
        reasons.append("Win rate below minimum threshold")
    if float(out_of_sample.get("sharpe_or_simple_risk_adjusted_score", 0.0)) < config.minimum_out_of_sample_score:
        reasons.append("Out-of-sample score below minimum threshold")
    if float(in_sample.get("total_pnl", 0.0)) > 0 and float(out_of_sample.get("total_pnl", 0.0)) <= 0:
        reasons.append("Overfit profile detected")
    if _metric_unstable(in_sample, out_of_sample, config):
        reasons.append("Metrics unstable between in-sample and out-of-sample")
    return reasons


def evaluate_strategy(parameters: dict[str, Any], config: ResearchConfig) -> EvaluationResult:
    trades = run_strategy_backtest(config.backtest_symbols, parameters)
    in_sample_trades, out_sample_trades = _split_trades(trades)
    overall_summary = _calculate_summary(trades)
    in_sample_summary = _calculate_summary(in_sample_trades)
    out_sample_summary = _calculate_summary(out_sample_trades)
    rejection_reasons = _build_rejection_reasons(overall_summary, in_sample_summary, out_sample_summary, config)
    backtest_metrics = {
        "overall": overall_summary,
        "in_sample": in_sample_summary,
        "out_of_sample": out_sample_summary,
        "safety_passed": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }
    return EvaluationResult(
        results_summary=overall_summary,
        backtest_metrics=backtest_metrics,
        rejection_reasons=rejection_reasons,
    )


def build_paper_metrics_from_backtest(
    challenger_backtest: dict[str, Any],
    champion_backtest: dict[str, Any],
    cycles_observed: int,
    config: ResearchConfig,
) -> dict[str, Any]:
    out_sample = {**_empty_summary(), **(challenger_backtest.get("out_of_sample") or {})}
    champion_out_sample = {**_empty_summary(), **(champion_backtest.get("out_of_sample") or {})}
    signal_count = int(out_sample.get("num_trades", 0))
    validation_passed = (
        cycles_observed >= config.paper_trading_min_cycles
        and signal_count >= config.paper_trading_min_signals
        and float(out_sample.get("total_pnl", 0.0)) >= float(champion_out_sample.get("total_pnl", 0.0))
        and abs(float(out_sample.get("max_drawdown", 0.0))) <= config.max_allowed_drawdown
    )
    status = "passed" if validation_passed else ("active" if cycles_observed < config.paper_trading_min_cycles else "rejected")
    return {
        "status": status,
        "paper_pnl": round(float(out_sample.get("total_pnl", 0.0)), 2),
        "paper_win_rate": round(float(out_sample.get("win_rate", 0.0)), 2),
        "paper_drawdown": round(float(out_sample.get("max_drawdown", 0.0)), 2),
        "signal_count": signal_count,
        "cycles_observed": cycles_observed,
        "comparison_vs_champion": {
            "paper_pnl_delta": round(float(out_sample.get("total_pnl", 0.0)) - float(champion_out_sample.get("total_pnl", 0.0)), 2),
            "paper_win_rate_delta": round(float(out_sample.get("win_rate", 0.0)) - float(champion_out_sample.get("win_rate", 0.0)), 2),
            "paper_drawdown_delta": round(float(out_sample.get("max_drawdown", 0.0)) - float(champion_out_sample.get("max_drawdown", 0.0)), 2),
        },
        "validation_passed": validation_passed,
    }
