import time
import traceback

from shared.state import append_research_activity, load_strategy_registry, save_strategy_registry
from strategy_research_worker import run_research_iteration


SLEEP_SECONDS = 2


def main():
    print("Research worker started", flush=True)
    registry = load_strategy_registry()
    registry["research_worker_status"] = "running"
    append_research_activity(registry, "Research worker started")
    save_strategy_registry(registry)

    while True:
        try:
            print("Testing new strategy...", flush=True)
            print("Backtest started...", flush=True)
            registry = run_research_iteration()
            print("Backtest complete...", flush=True)

            last_result = str(registry.get("last_challenger_result", "") or "")
            challenger = registry.get("challenger") or {}
            challenger_id = challenger.get("id", "challenger")
            if challenger:
                print(f"Generated challenger {challenger_id}", flush=True)
            if "Promoted" in last_result or "Promotable" in last_result:
                print(f"Promoted: {last_result}", flush=True)
            else:
                print(f"Rejected: {last_result or 'No new challenger'}", flush=True)
        except Exception:
            print("Research worker error, continuing...", flush=True)
            traceback.print_exc()
            registry = load_strategy_registry()
            registry["research_worker_status"] = "error"
            append_research_activity(registry, "Research worker error, continuing", level="error")
            save_strategy_registry(registry)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
