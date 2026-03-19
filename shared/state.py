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


def get_api_status_snapshot():
    registry = load_strategy_registry()
    algo_state = load_algo_update_state()
    champion = registry.get("champion", {}) or {}
    challenger = registry.get("challenger") or {}
    return {
        "app": "Mash Terminal",
        "status": "ready",
        "mode": "skeleton",
        "current_champion_version": champion.get("version"),
        "current_champion_id": champion.get("id", ""),
        "current_challenger_version": challenger.get("version") if challenger else None,
        "last_activity_time": _format_timestamp(registry.get("last_activity_time", "")),
        "last_research_run": _format_timestamp(registry.get("last_research_run", "")),
        "last_promotion_time": _format_timestamp(registry.get("last_promotion_at", "")),
        "last_update_sent": _format_timestamp(algo_state.get("last_sent_at", "")),
        "last_challenger_result": registry.get("last_challenger_result", ""),
        "last_rejection_reason": registry.get("last_rejection_reason", ""),
        "research_worker_status": registry.get("research_worker_status", "offline"),
        "research_worker_last_seen": _format_timestamp(registry.get("research_worker_last_seen", "")),
        "experiments_tested_today": _experiments_tested_today(registry),
        "total_experiments": len(registry.get("experiments", []) or []),
    }
