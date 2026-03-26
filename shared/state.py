import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
STRATEGY_REGISTRY_FILE = BASE_DIR / "strategy_registry.json"
ALGO_UPDATE_STATE_FILE = BASE_DIR / "algo_update_state.json"
DEFAULT_MARKET_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "IWM", "DIA"]
MAX_ACTIVITY_HISTORY = 200
MAX_ENTITY_HISTORY = 100


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_strategy_parameters() -> dict[str, Any]:
    return {
        "score_threshold": 65,
        "rsi_long_min": 53,
        "rsi_short_max": 47,
        "rel_vol_min": 1.4,
        "ema_short_len": 20,
        "ema_long_len": 50,
        "stop_multiplier": 1.0,
        "tp1_multiplier": 1.7,
        "tp2_multiplier": 2.6,
        "trend_weight": 34,
        "momentum_weight": 18,
        "volume_weight": 16,
        "structure_weight": 24,
        "rr_weight": 24,
        "max_position_size": 1.0,
        "cooldown_bars": 3,
        "lookback_window": 20,
        "signal_weighting": 1.0,
    }


def _empty_backtest_metrics() -> dict[str, Any]:
    return {
        "overall": {},
        "in_sample": {},
        "out_of_sample": {},
        "safety_passed": False,
        "rejection_reasons": [],
        "evaluated_at": "",
    }


def _empty_paper_metrics() -> dict[str, Any]:
    return {
        "status": "not_started",
        "start_time": "",
        "end_time": "",
        "paper_pnl": 0.0,
        "paper_win_rate": 0.0,
        "paper_drawdown": 0.0,
        "signal_count": 0,
        "cycles_observed": 0,
        "comparison_vs_champion": {},
        "validation_passed": False,
    }


def default_strategy_record(
    strategy_id: str,
    version: int,
    status: str,
    parameters: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    created = created_at or now_str()
    return {
        "id": strategy_id,
        "version": version,
        "parent_id": "",
        "source": "seed",
        "parameters": deepcopy(parameters or default_strategy_parameters()),
        "created_at": created,
        "last_updated_at": created,
        "last_tested_at": "",
        "status": status,
        "testing_status": "active" if status == "champion" else "generated",
        "lifecycle_stage": "live" if status == "champion" else "generated",
        "latest_result_status": "Awaiting evaluation" if status != "champion" else "Live",
        "promotion_status": "Live champion" if status == "champion" else "",
        "paper_probation_passed": status == "champion",
        "results_summary": {},
        "backtest_metrics": _empty_backtest_metrics(),
        "paper_metrics": _empty_paper_metrics(),
        "rejection_reason": "",
        "rejection_reasons": [],
        "promotion_date": "",
    }


def default_summary_metrics() -> dict[str, Any]:
    return {
        "cycles_completed": 0,
        "cycles_failed": 0,
        "active_challenger_count": 0,
        "rejected_challenger_count": 0,
        "active_paper_test_count": 0,
        "promotion_count": 0,
        "total_experiments": 0,
        "experiments_tested_today": 0,
        "last_cycle_duration_seconds": 0.0,
        "last_cycle_result": "",
    }


def default_strategy_registry() -> dict[str, Any]:
    champion = default_strategy_record("champion-v1", 1, "champion")
    return {
        "worker_status": "offline",
        "research_worker_status": "offline",
        "last_heartbeat": "",
        "research_worker_last_seen": "",
        "current_cycle_number": 0,
        "last_cycle_started_at": "",
        "last_cycle_completed_at": "",
        "last_activity_time": "",
        "last_research_run": "",
        "last_experiment_started_at": "",
        "last_experiment_finished_at": "",
        "last_challenger_result": "",
        "last_rejection_reason": "",
        "last_promotion_at": "",
        "champion": champion,
        "current_champion": champion,
        "previous_champion": None,
        "challenger": None,
        "current_challenger": None,
        "recent_challengers": [],
        "rejected_challengers": [],
        "active_paper_tests": [],
        "paper_test_history": [],
        "promotion_history": [],
        "experiments": [],
        "experiment_index": 0,
        "research_activity": [],
        "latest_worker_logs": [],
        "summary_metrics": default_summary_metrics(),
    }


def default_algo_update_state() -> dict[str, Any]:
    return {
        "last_sent_at": "",
        "last_schedule_slot": "",
        "last_strategy_signature": "",
        "last_strategy_signature_hash": "",
        "last_message": "",
        "last_pnl": 0.0,
        "last_win_rate": 0.0,
        "last_trade_count": 0,
        "last_experiments_tested": 0,
        "last_promoted_since_update": False,
        "last_rejected_since_update": False,
    }


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(default)
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return data if isinstance(data, dict) else deepcopy(default)
    except Exception:
        return deepcopy(default)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.flush()
        os.fsync(file_obj.fileno())
        temp_name = file_obj.name
    os.replace(temp_name, path)


def _normalize_results_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    payload = summary if isinstance(summary, dict) else {}
    return {
        "total_pnl": round(float(payload.get("total_pnl", 0.0) or 0.0), 2),
        "win_rate": round(float(payload.get("win_rate", 0.0) or 0.0), 2),
        "max_drawdown": round(float(payload.get("max_drawdown", 0.0) or 0.0), 2),
        "num_trades": int(payload.get("num_trades", 0) or 0),
        "average_trade": round(float(payload.get("average_trade", 0.0) or 0.0), 2),
        "average_win": round(float(payload.get("average_win", 0.0) or 0.0), 2),
        "average_loss": round(float(payload.get("average_loss", 0.0) or 0.0), 2),
        "profit_factor": round(float(payload.get("profit_factor", 0.0) or 0.0), 2),
        "sharpe_or_simple_risk_adjusted_score": round(
            float(payload.get("sharpe_or_simple_risk_adjusted_score", 0.0) or 0.0),
            4,
        ),
        "learning_score": round(float(payload.get("learning_score", 0.0) or 0.0), 2),
    }


def _normalize_backtest_metrics(backtest_metrics: dict[str, Any] | None) -> dict[str, Any]:
    payload = backtest_metrics if isinstance(backtest_metrics, dict) else {}
    reasons = payload.get("rejection_reasons", [])
    return {
        "overall": _normalize_results_summary(payload.get("overall")),
        "in_sample": _normalize_results_summary(payload.get("in_sample")),
        "out_of_sample": _normalize_results_summary(payload.get("out_of_sample")),
        "safety_passed": bool(payload.get("safety_passed", False)),
        "rejection_reasons": reasons if isinstance(reasons, list) else [],
        "evaluated_at": str(payload.get("evaluated_at", "")),
    }


def _normalize_paper_metrics(paper_metrics: dict[str, Any] | None) -> dict[str, Any]:
    payload = {**_empty_paper_metrics(), **(paper_metrics or {})}
    comparison = payload.get("comparison_vs_champion", {})
    if not isinstance(comparison, dict):
        comparison = {}
    return {
        "status": str(payload.get("status", "not_started") or "not_started"),
        "start_time": str(payload.get("start_time", "") or ""),
        "end_time": str(payload.get("end_time", "") or ""),
        "paper_pnl": round(float(payload.get("paper_pnl", 0.0) or 0.0), 2),
        "paper_win_rate": round(float(payload.get("paper_win_rate", 0.0) or 0.0), 2),
        "paper_drawdown": round(float(payload.get("paper_drawdown", 0.0) or 0.0), 2),
        "signal_count": int(payload.get("signal_count", 0) or 0),
        "cycles_observed": int(payload.get("cycles_observed", 0) or 0),
        "comparison_vs_champion": {
            "paper_pnl_delta": round(float(comparison.get("paper_pnl_delta", 0.0) or 0.0), 2),
            "paper_win_rate_delta": round(float(comparison.get("paper_win_rate_delta", 0.0) or 0.0), 2),
            "paper_drawdown_delta": round(float(comparison.get("paper_drawdown_delta", 0.0) or 0.0), 2),
        },
        "validation_passed": bool(payload.get("validation_passed", False)),
    }


def normalize_strategy_record(strategy: dict[str, Any] | None, fallback_status: str = "challenger") -> dict[str, Any] | None:
    if not isinstance(strategy, dict):
        return None
    version = int(strategy.get("version", 0) or 0)
    strategy_id = str(strategy.get("id", "") or "")
    status = str(strategy.get("status", fallback_status) or fallback_status)
    base = default_strategy_record(strategy_id or f"{status}-v{version or 0}", version or 0, status)
    merged = {**base, **strategy}
    merged["parameters"] = merged.get("parameters") if isinstance(merged.get("parameters"), dict) else deepcopy(default_strategy_parameters())
    merged["results_summary"] = _normalize_results_summary(merged.get("results_summary"))
    merged["backtest_metrics"] = _normalize_backtest_metrics(merged.get("backtest_metrics"))
    merged["paper_metrics"] = _normalize_paper_metrics(merged.get("paper_metrics"))
    rejection_reasons = merged.get("rejection_reasons", [])
    merged["rejection_reasons"] = rejection_reasons if isinstance(rejection_reasons, list) else []
    if not merged.get("rejection_reason") and merged["rejection_reasons"]:
        merged["rejection_reason"] = merged["rejection_reasons"][0]
    return merged


def normalize_strategy_registry(registry: dict[str, Any] | None) -> dict[str, Any]:
    defaults = default_strategy_registry()
    raw = registry if isinstance(registry, dict) else {}
    normalized = deepcopy(defaults)
    normalized.update(raw)

    champion = normalize_strategy_record(raw.get("champion") or raw.get("current_champion"), fallback_status="champion")
    if not champion:
      champion = deepcopy(defaults["champion"])
    challenger = normalize_strategy_record(raw.get("challenger") or raw.get("current_challenger"), fallback_status="challenger")
    previous_champion = normalize_strategy_record(raw.get("previous_champion"), fallback_status="champion")

    normalized["champion"] = champion
    normalized["current_champion"] = champion
    normalized["challenger"] = challenger
    normalized["current_challenger"] = challenger
    normalized["previous_champion"] = previous_champion

    normalized["experiments"] = [
        exp for exp in (normalize_strategy_record(item, fallback_status="challenger") for item in raw.get("experiments", []))
        if exp
    ]
    normalized["recent_challengers"] = [
        exp for exp in (normalize_strategy_record(item, fallback_status="challenger") for item in raw.get("recent_challengers", []))
        if exp
    ][-MAX_ENTITY_HISTORY:]
    normalized["rejected_challengers"] = [
        exp for exp in (normalize_strategy_record(item, fallback_status="rejected") for item in raw.get("rejected_challengers", []))
        if exp
    ][-MAX_ENTITY_HISTORY:]
    normalized["active_paper_tests"] = [
        exp for exp in (normalize_strategy_record(item, fallback_status="paper_testing") for item in raw.get("active_paper_tests", []))
        if exp
    ][-MAX_ENTITY_HISTORY:]
    normalized["paper_test_history"] = [
        exp for exp in (normalize_strategy_record(item, fallback_status="paper_testing") for item in raw.get("paper_test_history", []))
        if exp
    ][-MAX_ENTITY_HISTORY:]
    promotion_history = raw.get("promotion_history", [])
    normalized["promotion_history"] = promotion_history if isinstance(promotion_history, list) else []

    activity = raw.get("research_activity", [])
    logs = raw.get("latest_worker_logs", [])
    normalized["research_activity"] = activity if isinstance(activity, list) else []
    normalized["latest_worker_logs"] = logs if isinstance(logs, list) else list(normalized["research_activity"])
    normalized["research_activity"] = normalized["research_activity"][-MAX_ACTIVITY_HISTORY:]
    normalized["latest_worker_logs"] = normalized["latest_worker_logs"][-MAX_ACTIVITY_HISTORY:]

    summary_metrics = raw.get("summary_metrics", {})
    normalized["summary_metrics"] = {**default_summary_metrics(), **(summary_metrics if isinstance(summary_metrics, dict) else {})}
    normalized["worker_status"] = str(raw.get("worker_status") or raw.get("research_worker_status") or "offline")
    normalized["research_worker_status"] = normalized["worker_status"]
    normalized["last_heartbeat"] = str(raw.get("last_heartbeat") or raw.get("research_worker_last_seen") or "")
    normalized["research_worker_last_seen"] = normalized["last_heartbeat"]
    normalized["current_cycle_number"] = int(raw.get("current_cycle_number", 0) or 0)
    normalized["experiment_index"] = max(int(raw.get("experiment_index", len(normalized["experiments"])) or 0), len(normalized["experiments"]))
    if not normalized["champion"].get("created_at"):
        normalized["champion"]["created_at"] = now_str()
    return normalized


def ensure_strategy_registry() -> dict[str, Any]:
    if not STRATEGY_REGISTRY_FILE.exists():
        registry = default_strategy_registry()
        _save_json(STRATEGY_REGISTRY_FILE, registry)
        return registry
    return normalize_strategy_registry(_load_json(STRATEGY_REGISTRY_FILE, default_strategy_registry()))


def load_strategy_registry() -> dict[str, Any]:
    return ensure_strategy_registry()


def save_strategy_registry(registry: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_strategy_registry(registry)
    _save_json(STRATEGY_REGISTRY_FILE, normalized)
    return normalized


def ensure_algo_update_state() -> dict[str, Any]:
    if not ALGO_UPDATE_STATE_FILE.exists():
        state = default_algo_update_state()
        _save_json(ALGO_UPDATE_STATE_FILE, state)
        return state
    loaded_state = _load_json(ALGO_UPDATE_STATE_FILE, default_algo_update_state())
    return {**default_algo_update_state(), **loaded_state}


def load_algo_update_state() -> dict[str, Any]:
    return ensure_algo_update_state()


def save_algo_update_state(state: dict[str, Any] | None) -> dict[str, Any]:
    normalized = {**default_algo_update_state(), **(state or {})}
    _save_json(ALGO_UPDATE_STATE_FILE, normalized)
    return normalized


def append_research_activity(registry: dict[str, Any], message: str, level: str = "info", event_time: str = "") -> dict[str, Any]:
    event_timestamp = event_time or now_str()
    entry = {
        "time": event_timestamp,
        "message": message,
        "level": level,
    }
    activity = list(registry.get("research_activity", []))
    logs = list(registry.get("latest_worker_logs", []))
    activity.append(entry)
    logs.append(entry)
    registry["research_activity"] = activity[-MAX_ACTIVITY_HISTORY:]
    registry["latest_worker_logs"] = logs[-MAX_ACTIVITY_HISTORY:]
    registry["last_activity_time"] = event_timestamp
    return registry


def append_worker_log(registry: dict[str, Any], message: str, level: str = "info", event_time: str = "") -> dict[str, Any]:
    return append_research_activity(registry, message=message, level=level, event_time=event_time)


def record_heartbeat(registry: dict[str, Any], status: str = "running", event_time: str = "") -> dict[str, Any]:
    heartbeat_time = event_time or now_str()
    registry["worker_status"] = status
    registry["research_worker_status"] = status
    registry["last_heartbeat"] = heartbeat_time
    registry["research_worker_last_seen"] = heartbeat_time
    return registry


def _format_timestamp(value: str) -> str:
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _experiments_tested_today(registry: dict[str, Any]) -> int:
    today = datetime.now().date()
    total = 0
    for exp in registry.get("experiments", []) or []:
        created_at = pd.to_datetime(exp.get("created_at", ""), errors="coerce")
        if pd.isna(created_at):
            continue
        if created_at.date() == today:
            total += 1
    return total


def _normalize_activity_entry(entry: dict[str, Any] | None) -> dict[str, Any]:
    payload = entry if isinstance(entry, dict) else {}
    return {
        "time": _format_timestamp(str(payload.get("time", "") or "")),
        "message": str(payload.get("message", "") or ""),
        "level": str(payload.get("level", "info") or "info"),
    }


def _normalize_strategy_snapshot(strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    record = normalize_strategy_record(strategy)
    if not record:
        return None
    return {
        "id": record.get("id", ""),
        "version": record.get("version"),
        "parent_id": record.get("parent_id", ""),
        "source": record.get("source", ""),
        "status": record.get("status", ""),
        "testing_status": record.get("testing_status", ""),
        "lifecycle_stage": record.get("lifecycle_stage", ""),
        "latest_result_status": record.get("latest_result_status", ""),
        "promotion_status": record.get("promotion_status", ""),
        "paper_probation_passed": bool(record.get("paper_probation_passed", False)),
        "created_at": _format_timestamp(record.get("created_at", "")),
        "last_updated_at": _format_timestamp(record.get("last_updated_at", "")),
        "last_tested_at": _format_timestamp(record.get("last_tested_at", "")),
        "promotion_date": _format_timestamp(record.get("promotion_date", "")),
        "parameters": record.get("parameters", {}),
        "results_summary": _normalize_results_summary(record.get("results_summary")),
        "backtest_metrics": _normalize_backtest_metrics(record.get("backtest_metrics")),
        "paper_metrics": _normalize_paper_metrics(record.get("paper_metrics")),
        "rejection_reason": record.get("rejection_reason", ""),
        "rejection_reasons": record.get("rejection_reasons", []),
    }


def get_strategy_lab_summary_snapshot() -> dict[str, Any]:
    registry = load_strategy_registry()
    algo_state = load_algo_update_state()
    active_paper_tests = [_normalize_strategy_snapshot(item) for item in registry.get("active_paper_tests", [])]
    active_paper_tests = [item for item in active_paper_tests if item]
    recent_challengers = [_normalize_strategy_snapshot(item) for item in registry.get("recent_challengers", [])]
    recent_challengers = [item for item in recent_challengers if item]
    rejected = [_normalize_strategy_snapshot(item) for item in registry.get("rejected_challengers", [])]
    rejected = [item for item in rejected if item]
    promotion_history = list(registry.get("promotion_history", []))
    summary_metrics = {**default_summary_metrics(), **(registry.get("summary_metrics", {}) or {})}
    summary_metrics["experiments_tested_today"] = _experiments_tested_today(registry)
    summary_metrics["total_experiments"] = len(registry.get("experiments", []) or [])
    return {
        "current_cycle_number": registry.get("current_cycle_number", 0),
        "worker_status": registry.get("worker_status", "offline"),
        "research_worker_status": registry.get("research_worker_status", "offline"),
        "last_heartbeat": _format_timestamp(registry.get("last_heartbeat", "")),
        "research_worker_last_seen": _format_timestamp(registry.get("research_worker_last_seen", "")),
        "last_activity_time": _format_timestamp(registry.get("last_activity_time", "")),
        "last_research_run": _format_timestamp(registry.get("last_research_run", "")),
        "last_experiment_started_at": _format_timestamp(registry.get("last_experiment_started_at", "")),
        "last_experiment_finished_at": _format_timestamp(registry.get("last_experiment_finished_at", "")),
        "last_cycle_started_at": _format_timestamp(registry.get("last_cycle_started_at", "")),
        "last_cycle_completed_at": _format_timestamp(registry.get("last_cycle_completed_at", "")),
        "last_challenger_result": registry.get("last_challenger_result", ""),
        "last_rejection_reason": registry.get("last_rejection_reason", ""),
        "last_promotion_time": _format_timestamp(registry.get("last_promotion_at", "")),
        "last_update_sent": _format_timestamp(algo_state.get("last_sent_at", "")),
        "experiments_tested_today": summary_metrics["experiments_tested_today"],
        "total_experiments": summary_metrics["total_experiments"],
        "current_champion": _normalize_strategy_snapshot(registry.get("champion")),
        "current_challenger": _normalize_strategy_snapshot(registry.get("challenger")),
        "active_paper_tests": active_paper_tests,
        "recent_challengers": list(reversed(recent_challengers[-10:])),
        "rejected_challengers": list(reversed(rejected[-10:])),
        "promotion_history": list(reversed(promotion_history[-10:])),
        "summary_metrics": summary_metrics,
    }


def get_strategy_lab_activity_snapshot(limit: int = 25) -> dict[str, Any]:
    registry = load_strategy_registry()
    activity = [_normalize_activity_entry(entry) for entry in registry.get("latest_worker_logs", []) or []]
    activity = [entry for entry in activity if entry["time"] or entry["message"]]
    activity.reverse()
    limit = max(limit, 0)
    return {
        "research_worker_status": registry.get("research_worker_status", "offline"),
        "last_heartbeat": _format_timestamp(registry.get("last_heartbeat", "")),
        "last_activity_time": _format_timestamp(registry.get("last_activity_time", "")),
        "count": min(len(activity), limit),
        "items": activity[:limit],
    }


def get_strategy_lab_experiments_snapshot(limit: int = 50) -> dict[str, Any]:
    registry = load_strategy_registry()
    experiments = [_normalize_strategy_snapshot(exp) for exp in registry.get("experiments", []) or []]
    experiments = [exp for exp in experiments if exp]
    experiments.sort(key=lambda exp: exp.get("created_at", ""), reverse=True)
    limit = max(limit, 0)
    return {
        "total_experiments": len(experiments),
        "items": experiments[:limit],
    }


def get_strategy_lab_paper_tests_snapshot(limit: int = 25) -> dict[str, Any]:
    registry = load_strategy_registry()
    active = [_normalize_strategy_snapshot(exp) for exp in registry.get("active_paper_tests", []) or []]
    history = [_normalize_strategy_snapshot(exp) for exp in registry.get("paper_test_history", []) or []]
    active = [exp for exp in active if exp]
    history = [exp for exp in history if exp]
    active.sort(key=lambda exp: exp.get("created_at", ""), reverse=True)
    history.sort(key=lambda exp: exp.get("last_updated_at", ""), reverse=True)
    limit = max(limit, 0)
    return {
        "active_count": len(active),
        "history_count": len(history),
        "active_items": active[:limit],
        "history_items": history[:limit],
    }


def get_strategy_lab_promotions_snapshot(limit: int = 25) -> dict[str, Any]:
    registry = load_strategy_registry()
    promotions = list(registry.get("promotion_history", []) or [])
    promotions.reverse()
    limit = max(limit, 0)
    return {
        "count": min(len(promotions), limit),
        "items": promotions[:limit],
    }


def get_market_overview_snapshot() -> dict[str, Any]:
    summary = get_strategy_lab_summary_snapshot()
    champion = summary.get("current_champion") or {}
    challenger = summary.get("current_challenger") or {}
    tracked_symbols = [
        {
            "symbol": symbol,
            "group": "Index / ETF" if symbol in {"SPY", "QQQ", "IWM", "DIA"} else "Equity",
        }
        for symbol in DEFAULT_MARKET_SYMBOLS
    ]
    return {
        "tracked_symbols": tracked_symbols,
        "tracked_symbol_count": len(tracked_symbols),
        "market_note": "Initial market surface using the shared tracked universe. Full live charting and symbol detail migration will come later.",
        "research_worker_status": summary.get("research_worker_status", "offline"),
        "current_champion_version": champion.get("version"),
        "current_challenger_version": challenger.get("version") if challenger else None,
        "last_activity_time": summary.get("last_activity_time", ""),
        "experiments_tested_today": summary.get("experiments_tested_today", 0),
        "active_paper_test_count": len(summary.get("active_paper_tests", [])),
    }


def get_api_status_snapshot() -> dict[str, Any]:
    summary = get_strategy_lab_summary_snapshot()
    champion = summary.get("current_champion") or {}
    challenger = summary.get("current_challenger") or {}
    return {
        "app": "Mash Terminal",
        "status": "ready",
        "mode": "skeleton",
        "current_champion_version": champion.get("version"),
        "current_champion_id": champion.get("id", ""),
        "current_challenger_version": challenger.get("version") if challenger else None,
        "last_activity_time": summary.get("last_activity_time", ""),
        "last_research_run": summary.get("last_research_run", ""),
        "last_promotion_time": summary.get("last_promotion_time", ""),
        "last_update_sent": summary.get("last_update_sent", ""),
        "last_challenger_result": summary.get("last_challenger_result", ""),
        "last_rejection_reason": summary.get("last_rejection_reason", ""),
        "research_worker_status": summary.get("research_worker_status", "offline"),
        "research_worker_last_seen": summary.get("research_worker_last_seen", ""),
        "last_heartbeat": summary.get("last_heartbeat", ""),
        "current_cycle_number": summary.get("current_cycle_number", 0),
        "experiments_tested_today": summary.get("experiments_tested_today", 0),
        "total_experiments": summary.get("total_experiments", 0),
        "active_paper_test_count": len(summary.get("active_paper_tests", [])),
        "promotion_count": len(summary.get("promotion_history", [])),
    }
