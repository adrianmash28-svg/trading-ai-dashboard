from __future__ import annotations

import argparse
import time

from app.config import research_config
from app.services.research_engine import ResearchEngine
from shared.state import append_worker_log, load_strategy_registry, record_heartbeat, save_strategy_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Nonstop strategy research worker")
    parser.add_argument("--once", action="store_true", help="Run one research cycle and exit")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=research_config.research_interval_seconds,
        help="Seconds to sleep between research cycles",
    )
    args = parser.parse_args()

    engine = ResearchEngine(research_config)
    registry = load_strategy_registry()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    record_heartbeat(registry, status="starting", event_time=start_time)
    append_worker_log(registry, "worker started", level="info", event_time=start_time)
    save_strategy_registry(registry)
    print("worker started", flush=True)

    if args.once:
        engine.run_cycle_safely()
        return

    sleep_seconds = max(int(args.sleep_seconds or research_config.research_interval_seconds), 1)
    while True:
        cycle_started = time.strftime("%Y-%m-%d %H:%M:%S")
        registry = load_strategy_registry()
        record_heartbeat(registry, status="running", event_time=cycle_started)
        append_worker_log(registry, "heartbeat", level="info", event_time=cycle_started)
        save_strategy_registry(registry)
        engine.run_cycle_safely()
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
