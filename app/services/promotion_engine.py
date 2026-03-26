from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.config import ResearchConfig


def should_promote_challenger(champion: dict[str, Any], challenger: dict[str, Any], config: ResearchConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    champion_backtest = champion.get("backtest_metrics", {}) or {}
    challenger_backtest = challenger.get("backtest_metrics", {}) or {}
    challenger_paper = challenger.get("paper_metrics", {}) or {}
    champion_overall = champion_backtest.get("overall", {}) or champion.get("results_summary", {}) or {}
    challenger_overall = challenger_backtest.get("overall", {}) or challenger.get("results_summary", {}) or {}

    if not challenger_backtest.get("safety_passed"):
        reasons.append("Backtest safety filters not passed")
    if float((challenger_backtest.get("out_of_sample") or {}).get("total_pnl", 0.0)) <= 0:
        reasons.append("Out-of-sample validation not passed")
    if not challenger_paper.get("validation_passed"):
        reasons.append("Paper trading validation not passed")
    if int(challenger_paper.get("signal_count", 0)) < config.promotion_min_signal_count:
        reasons.append("Paper test signal count too low")
    if abs(float(challenger_paper.get("paper_drawdown", 0.0))) > config.max_allowed_drawdown:
        reasons.append("Paper drawdown exceeded limit")
    if float(challenger_overall.get("total_pnl", 0.0)) < float(champion_overall.get("total_pnl", 0.0)) + config.promotion_margin_over_champion:
        reasons.append("P&L improvement margin not met")
    if float(challenger_overall.get("win_rate", 0.0)) < float(champion_overall.get("win_rate", 0.0)) + config.promotion_win_rate_margin:
        reasons.append("Win rate improvement margin not met")
    if float(challenger_overall.get("profit_factor", 0.0)) < float(champion_overall.get("profit_factor", 0.0)) + config.promotion_profit_factor_margin:
        reasons.append("Profit factor improvement margin not met")
    return (not reasons, reasons)


def build_promotion_history_entry(previous_champion: dict[str, Any], new_champion: dict[str, Any], promoted_at: str) -> dict[str, Any]:
    return {
        "time": promoted_at,
        "from_id": previous_champion.get("id", ""),
        "from_version": previous_champion.get("version"),
        "to_id": new_champion.get("id", ""),
        "to_version": new_champion.get("version"),
        "summary": {
            "previous_total_pnl": (previous_champion.get("results_summary") or {}).get("total_pnl", 0.0),
            "new_total_pnl": (new_champion.get("results_summary") or {}).get("total_pnl", 0.0),
            "paper_pnl": (new_champion.get("paper_metrics") or {}).get("paper_pnl", 0.0),
        },
    }


def promote_challenger(registry: dict[str, Any], challenger: dict[str, Any], promoted_at: str) -> dict[str, Any]:
    previous_champion = deepcopy(registry.get("champion"))
    challenger_copy = deepcopy(challenger)
    challenger_copy["status"] = "champion"
    challenger_copy["testing_status"] = "active"
    challenger_copy["lifecycle_stage"] = "live"
    challenger_copy["latest_result_status"] = "Promoted"
    challenger_copy["promotion_status"] = "Promoted to champion"
    challenger_copy["paper_probation_passed"] = True
    challenger_copy["promotion_date"] = promoted_at
    challenger_copy["last_updated_at"] = promoted_at

    if previous_champion:
        previous_champion["status"] = "retired"
        previous_champion["latest_result_status"] = "Replaced"
        previous_champion["promotion_status"] = "Replaced by challenger"
        previous_champion["last_updated_at"] = promoted_at
        registry["previous_champion"] = previous_champion

    registry["champion"] = challenger_copy
    registry["current_champion"] = challenger_copy
    registry["challenger"] = None
    registry["current_challenger"] = None
    registry["last_promotion_at"] = promoted_at
    registry.setdefault("promotion_history", []).append(build_promotion_history_entry(previous_champion or {}, challenger_copy, promoted_at))
    return registry
