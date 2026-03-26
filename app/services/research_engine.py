from __future__ import annotations

import json
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any

from app.config import ResearchConfig, research_config
from app.services.challenger_generator import generate_challenger_variants
from app.services.promotion_engine import promote_challenger, should_promote_challenger
from app.services.strategy_evaluator import build_paper_metrics_from_backtest, evaluate_strategy
from shared.state import (
    append_worker_log,
    default_strategy_record,
    load_strategy_registry,
    normalize_strategy_record,
    record_heartbeat,
    save_strategy_registry,
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parameter_signature(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True)


def _entity_sort_key(item: dict[str, Any]) -> str:
    return str(item.get("last_updated_at") or item.get("created_at") or "")


class ResearchEngine:
    def __init__(self, config: ResearchConfig | None = None):
        self.config = config or research_config

    def _log(self, registry: dict[str, Any], message: str, level: str = "info") -> None:
        print(message, flush=True)
        append_worker_log(registry, message, level=level)

    def _ensure_champion_evaluated(self, registry: dict[str, Any]) -> dict[str, Any]:
        champion = normalize_strategy_record(registry.get("champion"), fallback_status="champion")
        if not champion:
            champion = default_strategy_record("champion-v1", 1, "champion")
        if champion.get("backtest_metrics", {}).get("evaluated_at"):
            registry["champion"] = champion
            registry["current_champion"] = champion
            return champion

        self._log(registry, f"Backtesting champion {champion['id']}")
        evaluation = evaluate_strategy(champion["parameters"], self.config)
        champion["results_summary"] = evaluation.results_summary
        champion["backtest_metrics"] = {
            **evaluation.backtest_metrics,
            "evaluated_at": _now(),
        }
        champion["latest_result_status"] = "Baseline ready"
        champion["last_tested_at"] = _now()
        champion["last_updated_at"] = champion["last_tested_at"]
        registry["champion"] = champion
        registry["current_champion"] = champion
        return champion

    def _existing_parameter_signatures(self, registry: dict[str, Any]) -> set[str]:
        signatures = set()
        for collection_name in ("experiments", "recent_challengers", "rejected_challengers", "active_paper_tests"):
            for item in registry.get(collection_name, []) or []:
                parameters = item.get("parameters", {})
                if isinstance(parameters, dict) and parameters:
                    signatures.add(_parameter_signature(parameters))
        champion = registry.get("champion") or {}
        if isinstance(champion.get("parameters"), dict):
            signatures.add(_parameter_signature(champion["parameters"]))
        return signatures

    def _next_experiment_version(self, registry: dict[str, Any]) -> int:
        return int(registry.get("experiment_index", len(registry.get("experiments", []))) or 0) + 1

    def _build_challenger(self, champion: dict[str, Any], parameters: dict[str, Any], version: int) -> dict[str, Any]:
        created_at = _now()
        challenger = default_strategy_record(f"challenger-v{version}", version, "challenger", parameters, created_at=created_at)
        challenger["parent_id"] = champion.get("id", "")
        challenger["source"] = champion.get("id", "")
        challenger["testing_status"] = "generated"
        challenger["lifecycle_stage"] = "generated"
        challenger["latest_result_status"] = "Generated"
        return challenger

    def _reject_challenger(self, registry: dict[str, Any], challenger: dict[str, Any], reasons: list[str]) -> None:
        challenger["status"] = "rejected"
        challenger["testing_status"] = "rejected"
        challenger["lifecycle_stage"] = "rejected"
        challenger["latest_result_status"] = "Rejected"
        challenger["promotion_status"] = "Rejected before paper trading"
        challenger["rejection_reasons"] = reasons
        challenger["rejection_reason"] = reasons[0] if reasons else "Rejected"
        challenger["last_updated_at"] = _now()
        registry["last_challenger_result"] = f"Rejected: {challenger['rejection_reason']}"
        registry["last_rejection_reason"] = challenger["rejection_reason"]
        registry.setdefault("rejected_challengers", []).append(deepcopy(challenger))
        self._log(registry, f"Rejected / Promoted: rejected {challenger['id']} ({challenger['rejection_reason']})", level="warning")

    def _queue_for_paper_testing(self, registry: dict[str, Any], challenger: dict[str, Any]) -> None:
        challenger["status"] = "paper_testing"
        challenger["testing_status"] = "paper_testing"
        challenger["lifecycle_stage"] = "paper_testing"
        challenger["latest_result_status"] = "Paper testing"
        challenger["promotion_status"] = "Awaiting paper validation"
        challenger["paper_metrics"]["status"] = "active"
        challenger["paper_metrics"]["start_time"] = _now()
        challenger["paper_metrics"]["cycles_observed"] = 0
        challenger["paper_metrics"]["signal_count"] = 0
        challenger["last_updated_at"] = _now()
        registry["challenger"] = deepcopy(challenger)
        registry["current_challenger"] = deepcopy(challenger)
        registry.setdefault("active_paper_tests", []).append(deepcopy(challenger))
        self._log(registry, f"Challenger sent to paper testing: {challenger['id']}")

    def _evaluate_new_challengers(self, registry: dict[str, Any], champion: dict[str, Any]) -> None:
        signatures = self._existing_parameter_signatures(registry)
        variants = generate_challenger_variants(champion, self.config, signatures, registry.get("current_cycle_number", 0))
        if not variants:
            self._log(registry, "No new challenger variants available this cycle", level="warning")
            return

        best_candidate: dict[str, Any] | None = None
        best_pnl = float("-inf")
        for parameters in variants:
            version = self._next_experiment_version(registry)
            challenger = self._build_challenger(champion, parameters, version)
            registry["experiment_index"] = version
            registry["last_experiment_started_at"] = challenger["created_at"]
            self._log(registry, f"Challenger generated: {challenger['id']}")
            self._log(registry, f"Testing new strategy... {challenger['id']}")
            try:
                evaluation = evaluate_strategy(challenger["parameters"], self.config)
                challenger["results_summary"] = evaluation.results_summary
                challenger["backtest_metrics"] = {
                    **evaluation.backtest_metrics,
                    "evaluated_at": _now(),
                }
                challenger["last_tested_at"] = _now()
                challenger["last_updated_at"] = challenger["last_tested_at"]
                registry["last_experiment_finished_at"] = challenger["last_tested_at"]
                self._log(registry, f"Backtest complete... {challenger['id']}")
                self._log(registry, f"Challenger backtested: {challenger['id']}")
                if evaluation.rejection_reasons:
                    self._reject_challenger(registry, challenger, evaluation.rejection_reasons)
                else:
                    if float(challenger["results_summary"].get("total_pnl", 0.0)) > best_pnl:
                        best_pnl = float(challenger["results_summary"].get("total_pnl", 0.0))
                        best_candidate = deepcopy(challenger)
                registry.setdefault("experiments", []).append(deepcopy(challenger))
                registry.setdefault("recent_challengers", []).append(deepcopy(challenger))
            except Exception as exc:
                challenger["status"] = "rejected"
                challenger["testing_status"] = "error"
                challenger["lifecycle_stage"] = "rejected"
                challenger["latest_result_status"] = "Error"
                challenger["rejection_reason"] = f"Evaluation error: {exc}"
                challenger["rejection_reasons"] = [challenger["rejection_reason"]]
                challenger["last_updated_at"] = _now()
                registry.setdefault("experiments", []).append(deepcopy(challenger))
                registry.setdefault("rejected_challengers", []).append(deepcopy(challenger))
                self._log(registry, f"Worker error / recovery: bad challenger {challenger['id']} ({exc})", level="error")

        if best_candidate:
            self._queue_for_paper_testing(registry, best_candidate)
            registry["last_challenger_result"] = f"Paper testing: {best_candidate['id']}"
            registry["last_rejection_reason"] = ""
        elif not variants:
            registry["last_challenger_result"] = "No challenger generated"

    def _advance_paper_tests(self, registry: dict[str, Any], champion: dict[str, Any]) -> None:
        active_tests = []
        champion_backtest = champion.get("backtest_metrics", {}) or {}
        for candidate in registry.get("active_paper_tests", []) or []:
            challenger = normalize_strategy_record(candidate, fallback_status="paper_testing")
            if not challenger:
                continue
            current_cycles = int((challenger.get("paper_metrics") or {}).get("cycles_observed", 0)) + 1
            paper_metrics = build_paper_metrics_from_backtest(
                challenger.get("backtest_metrics", {}) or {},
                champion_backtest,
                current_cycles,
                self.config,
            )
            challenger["paper_metrics"] = {
                **challenger.get("paper_metrics", {}),
                **paper_metrics,
                "start_time": challenger.get("paper_metrics", {}).get("start_time") or _now(),
                "end_time": _now() if paper_metrics["status"] in {"passed", "rejected"} else "",
            }
            challenger["last_updated_at"] = _now()
            if paper_metrics["status"] == "active":
                self._log(registry, f"Paper testing active for {challenger['id']}")
                active_tests.append(challenger)
                continue

            if paper_metrics["validation_passed"]:
                self._log(registry, f"Paper validation passed for {challenger['id']}")
                promote, reasons = should_promote_challenger(champion, challenger, self.config)
                if promote:
                    promoted_at = _now()
                    promote_challenger(registry, challenger, promoted_at)
                    registry["last_challenger_result"] = f"Promoted / Promoted: {challenger['id']}"
                    registry["last_rejection_reason"] = ""
                    self._log(registry, f"Rejected / Promoted: promoted {challenger['id']}", level="success")
                    champion = registry["champion"]
                else:
                    challenger["status"] = "rejected"
                    challenger["testing_status"] = "rejected"
                    challenger["lifecycle_stage"] = "rejected"
                    challenger["latest_result_status"] = "Rejected after paper test"
                    challenger["rejection_reasons"] = reasons
                    challenger["rejection_reason"] = reasons[0] if reasons else "Promotion rules not met"
                    challenger["promotion_status"] = "Rejected after paper validation"
                    registry.setdefault("rejected_challengers", []).append(deepcopy(challenger))
                    registry["last_challenger_result"] = f"Rejected: {challenger['rejection_reason']}"
                    registry["last_rejection_reason"] = challenger["rejection_reason"]
                    self._log(registry, f"Rejected / Promoted: rejected {challenger['id']} ({challenger['rejection_reason']})", level="warning")
            else:
                challenger["status"] = "rejected"
                challenger["testing_status"] = "rejected"
                challenger["lifecycle_stage"] = "rejected"
                challenger["latest_result_status"] = "Paper test failed"
                challenger["rejection_reason"] = "Paper trading validation failed"
                challenger["rejection_reasons"] = ["Paper trading validation failed"]
                challenger["promotion_status"] = "Rejected after paper validation"
                registry.setdefault("rejected_challengers", []).append(deepcopy(challenger))
                registry["last_challenger_result"] = "Rejected: Paper trading validation failed"
                registry["last_rejection_reason"] = "Paper trading validation failed"
                self._log(registry, f"Rejected / Promoted: rejected {challenger['id']} (paper validation failed)", level="warning")
            registry.setdefault("paper_test_history", []).append(deepcopy(challenger))
        registry["active_paper_tests"] = [deepcopy(item) for item in active_tests]

    def _refresh_summary_metrics(self, registry: dict[str, Any], cycle_result: str, cycle_duration: float) -> None:
        summary_metrics = registry.get("summary_metrics", {}) or {}
        summary_metrics["cycles_completed"] = int(summary_metrics.get("cycles_completed", 0) or 0) + 1
        summary_metrics["active_challenger_count"] = len(registry.get("recent_challengers", []) or [])
        summary_metrics["rejected_challenger_count"] = len(registry.get("rejected_challengers", []) or [])
        summary_metrics["active_paper_test_count"] = len(registry.get("active_paper_tests", []) or [])
        summary_metrics["promotion_count"] = len(registry.get("promotion_history", []) or [])
        summary_metrics["total_experiments"] = len(registry.get("experiments", []) or [])
        summary_metrics["experiments_tested_today"] = sum(
            1 for item in registry.get("experiments", []) or [] if str(item.get("created_at", "")).startswith(datetime.now().strftime("%Y-%m-%d"))
        )
        summary_metrics["last_cycle_duration_seconds"] = round(cycle_duration, 2)
        summary_metrics["last_cycle_result"] = cycle_result
        registry["summary_metrics"] = summary_metrics

        registry["recent_challengers"] = sorted(
            registry.get("recent_challengers", []) or [],
            key=_entity_sort_key,
        )[-self.config.max_recent_challengers :]
        registry["rejected_challengers"] = sorted(
            registry.get("rejected_challengers", []) or [],
            key=_entity_sort_key,
        )[-self.config.max_rejected_challengers :]
        registry["paper_test_history"] = sorted(
            registry.get("paper_test_history", []) or [],
            key=_entity_sort_key,
        )[-self.config.max_recent_challengers :]
        registry["promotion_history"] = (registry.get("promotion_history", []) or [])[-self.config.max_promotion_history :]

    def run_cycle(self) -> dict[str, Any]:
        cycle_started = datetime.now()
        registry = load_strategy_registry()
        registry["current_cycle_number"] = int(registry.get("current_cycle_number", 0) or 0) + 1
        registry["last_cycle_started_at"] = _now()
        registry["last_research_run"] = registry["last_cycle_started_at"]
        record_heartbeat(registry, status="running", event_time=registry["last_cycle_started_at"])
        self._log(registry, f"Cycle started #{registry['current_cycle_number']}")
        champion = self._ensure_champion_evaluated(registry)
        self._advance_paper_tests(registry, champion)
        self._evaluate_new_challengers(registry, registry["champion"])
        registry["last_cycle_completed_at"] = _now()
        record_heartbeat(registry, status="idle", event_time=registry["last_cycle_completed_at"])
        cycle_result = registry.get("last_challenger_result") or "Cycle completed"
        self._refresh_summary_metrics(registry, cycle_result, (datetime.now() - cycle_started).total_seconds())
        self._log(registry, f"Cycle completed #{registry['current_cycle_number']}: {cycle_result}", level="success")
        return save_strategy_registry(registry)

    def run_cycle_safely(self) -> dict[str, Any]:
        registry = load_strategy_registry()
        try:
            return self.run_cycle()
        except Exception:
            error_time = _now()
            record_heartbeat(registry, status="error", event_time=error_time)
            registry["last_cycle_completed_at"] = error_time
            summary_metrics = registry.get("summary_metrics", {}) or {}
            summary_metrics["cycles_failed"] = int(summary_metrics.get("cycles_failed", 0) or 0) + 1
            registry["summary_metrics"] = summary_metrics
            self._log(registry, "Worker error / recovery: cycle failed, continuing", level="error")
            append_worker_log(registry, traceback.format_exc(), level="error", event_time=error_time)
            return save_strategy_registry(registry)
