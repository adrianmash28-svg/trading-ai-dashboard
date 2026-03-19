import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
STRATEGY_REGISTRY_FILE = BASE_DIR / "strategy_registry.json"
ALGO_UPDATE_STATE_FILE = BASE_DIR / "algo_update_state.json"


def _load_json(path: Path, default: dict):
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def load_strategy_registry():
    return _load_json(
        STRATEGY_REGISTRY_FILE,
        {
            "champion": {},
            "challenger": None,
            "experiments": [],
            "last_activity_time": "",
            "last_research_run": "",
            "last_promotion_at": "",
            "last_challenger_result": "",
            "last_rejection_reason": "",
            "research_worker_status": "offline",
            "research_worker_last_seen": "",
            "research_activity": [],
        },
    )


def load_algo_update_state():
    return _load_json(
        ALGO_UPDATE_STATE_FILE,
        {
            "last_sent_at": "",
            "last_message": "",
        },
    )


def _format_timestamp(value: str) -> str:
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
