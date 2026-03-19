import time
import traceback

from strategy_research_worker import run_research_iteration


SLEEP_SECONDS = 2


def main():
    while True:
        try:
            print("Testing new strategy...", flush=True)
            registry = run_research_iteration()
            print("Backtest complete...", flush=True)

            last_result = str(registry.get("last_challenger_result", "") or "")
            if "Promoted" in last_result or "Promotable" in last_result:
                print(f"Promoted: {last_result}", flush=True)
            else:
                print(f"Rejected / Promoted: {last_result or 'No new challenger'}", flush=True)
        except Exception:
            print("Research worker error, continuing...", flush=True)
            traceback.print_exc()

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
