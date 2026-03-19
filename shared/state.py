import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
STRATEGY_REGISTRY_FILE = BASE_DIR / "strategy_registry.json"
ALGO_UPDATE_STATE_FILE = BASE_DIR / "algo_update_state.json"


def default_strategy_parameters():
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
    }


def default_strategy_registry():
    return {
        "champion": {
            "id": "champion-v1",
            "version": 1,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion",
            "paper_probation_passed": True,
            "created_at": "",
            "last_tested_at": "",
            "testing_status": "active",
            "latest_result_status": "Awaiting test",
        },
        "previous_champion": None,
        "challenger": None,
        "experiments": [],
        "experiment_index": 0,
        "last_activity_time": "",
        "last_experiment_started_at": "",
        "last_experiment_finished_at": "",
        "last_research_run": "",
        "last_challenger_result": "",
        "last_promotion_at": "",
        "last_rejection_reason": "",
        "promotion_history": [],
        "research_worker_status": "offline",
        "research_worker_last_seen": "",
        "research_activity": [],
    }


def default_algo_update_state():
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


def _load_json(path: Path, default: dict):
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def _save_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_strategy_registry(registry: dict | None):
    defaults = default_strategy_registry()
    raw = registry if isinstance(registry, dict) else {}
    champion = raw.get("champion")
    if not isinstance(champion, dict):
        champion = defaults["champion"]
    else:
        champion = {**defaults["champion"], **champion}
        if not isinstance(champion.get("parameters"), dict):
            champion["parameters"] = default_strategy_parameters()
        if not isinstance(champion.get("results_summary"), dict):
            champion["results_summary"] = {}

    normalized = {**defaults, **raw}
    normalized["champion"] = champion
    normalized["previous_champion"] = raw.get("previous_champion") if isinstance(raw.get("previous_champion"), dict) else None
    normalized["challenger"] = raw.get("challenger") if isinstance(raw.get("challenger"), dict) else None
    normalized["experiments"] = raw.get("experiments") if isinstance(raw.get("experiments"), list) else []
    normalized["promotion_history"] = raw.get("promotion_history") if isinstance(raw.get("promotion_history"), list) else []
    normalized["research_activity"] = raw.get("research_activity") if isinstance(raw.get("research_activity"), list) else []
    normalized["experiment_index"] = max(int(raw.get("experiment_index", len(normalized["experiments"])) or 0), len(normalized["experiments"]))
    return normalized


def ensure_strategy_registry():
    registry = normalize_strategy_registry(_load_json(STRATEGY_REGISTRY_FILE, default_strategy_registry()))
    if not registry["champion"].get("created_at"):
        registry["champion"]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_json(STRATEGY_REGISTRY_FILE, registry)
    return registry


def load_strategy_registry():
    return ensure_strategy_registry()


def save_strategy_registry(registry):
    normalized = normalize_strategy_registry(registry)
    _save_json(STRATEGY_REGISTRY_FILE, normalized)
    return normalized


def ensure_algo_update_state():
    state = _load_json(ALGO_UPDATE_STATE_FILE, default_algo_update_state())
    normalized = {**default_algo_update_state(), **state}
    _save_json(ALGO_UPDATE_STATE_FILE, normalized)
    return normalized


def load_algo_update_state():
    return ensure_algo_update_state()


def save_algo_update_state(state):
    normalized = {**default_algo_update_state(), **(state or {})}
    _save_json(ALGO_UPDATE_STATE_FILE, normalized)
    return normalized


def append_research_activity(registry: dict, message: str, level: str = "info", event_time: str = ""):
    event_timestamp = event_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    activity = list(registry.get("research_activity", []))
    activity.append(
        {
            "time": event_timestamp,
            "message": message,
            "level": level,
        }
    )
    registry["research_activity"] = activity[-50:]
    registry["last_activity_time"] = event_timestamp
    return registry


def _format_timestamp(value: str):
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _experiments_tested_today(registry: dict) -> int:
    experiments = registry.get("experiments", []) or []
    today = datetime.now().date()
    total = 0
    for exp in experiments:
        created_at = pd.to_datetime(exp.get("created_at", ""), errors="coerce")
        if pd.isna(created_at):
            continue
        if created_at.date() == today:
            total += 1
    return total


def _normalize_results_summary(summary: dict | None):
    payload = summary if isinstance(summary, dict) else {}
    return {
        "total_pnl": round(float(payload.get("total_pnl", 0.0) or 0.0), 2),
        "win_rate": round(float(payload.get("win_rate", 0.0) or 0.0), 2),
        "max_drawdown": round(float(payload.get("max_drawdown", 0.0) or 0.0), 2),
        "num_trades": int(payload.get("num_trades", 0) or 0),
        "average_win": round(float(payload.get("average_win", 0.0) or 0.0), 2),
        "average_loss": round(float(payload.get("average_loss", 0.0) or 0.0), 2),
        "learning_score": round(float(payload.get("learning_score", 0.0) or 0.0), 2),
    }


def _normalize_strategy_snapshot(strategy: dict | None):
    if not isinstance(strategy, dict):
        return None
    return {
        "id": strategy.get("id", ""),
        "version": strategy.get("version"),
        "status": strategy.get("status", ""),
        "promotion_status": strategy.get("promotion_status", ""),
        "paper_probation_passed": bool(strategy.get("paper_probation_passed", False)),
        "created_at": _format_timestamp(strategy.get("created_at", "")),
        "last_tested_at": _format_timestamp(strategy.get("last_tested_at", "")),
        "testing_status": strategy.get("testing_status", ""),
        "latest_result_status": strategy.get("latest_result_status", ""),
        "promotion_date": _format_timestamp(strategy.get("promotion_date", "")),
        "parameters": strategy.get("parameters", {}) if isinstance(strategy.get("parameters"), dict) else {},
        "results_summary": _normalize_results_summary(strategy.get("results_summary")),
    }


def _normalize_activity_entry(entry: dict | None):
    payload = entry if isinstance(entry, dict) else {}
    return {
        "time": _format_timestamp(payload.get("time", "")),
        "message": payload.get("message", ""),
        "level": payload.get("level", "info"),
    }


def get_strategy_lab_summary_snapshot():
    registry = load_strategy_registry()
    algo_state = load_algo_update_state()
    return {
        "current_champion": _normalize_strategy_snapshot(registry.get("champion")),
        "current_challenger": _normalize_strategy_snapshot(registry.get("challenger")),
        "research_worker_status": registry.get("research_worker_status", "offline"),
        "research_worker_last_seen": _format_timestamp(registry.get("research_worker_last_seen", "")),
        "last_activity_time": _format_timestamp(registry.get("last_activity_time", "")),
        "last_research_run": _format_timestamp(registry.get("last_research_run", "")),
        "last_experiment_started_at": _format_timestamp(registry.get("last_experiment_started_at", "")),
        "last_experiment_finished_at": _format_timestamp(registry.get("last_experiment_finished_at", "")),
        "last_challenger_result": registry.get("last_challenger_result", ""),
        "last_rejection_reason": registry.get("last_rejection_reason", ""),
        "last_promotion_time": _format_timestamp(registry.get("last_promotion_at", "")),
        "last_update_sent": _format_timestamp(algo_state.get("last_sent_at", "")),
        "experiments_tested_today": _experiments_tested_today(registry),
        "total_experiments": len(registry.get("experiments", []) or []),
    }


def get_strategy_lab_activity_snapshot(limit: int = 25):
    registry = load_strategy_registry()
    activity = [_normalize_activity_entry(entry) for entry in registry.get("research_activity", []) or []]
    activity = [entry for entry in activity if entry["time"] or entry["message"]]
    activity.reverse()
    return {
        "research_worker_status": registry.get("research_worker_status", "offline"),
        "last_activity_time": _format_timestamp(registry.get("last_activity_time", "")),
        "count": min(len(activity), max(limit, 0)),
        "items": activity[: max(limit, 0)],
    }


def get_strategy_lab_experiments_snapshot(limit: int = 50):
    registry = load_strategy_registry()
    experiments = [_normalize_strategy_snapshot(exp) for exp in registry.get("experiments", []) or []]
    experiments = [exp for exp in experiments if exp]
    experiments.sort(key=lambda exp: exp.get("created_at", ""), reverse=True)
    return {
        "total_experiments": len(experiments),
        "items": experiments[: max(limit, 0)],
    }


def get_api_status_snapshot():
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
        "experiments_tested_today": summary.get("experiments_tested_today", 0),
        "total_experiments": summary.get("total_experiments", 0),
    }
