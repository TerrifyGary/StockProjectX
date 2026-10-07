from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime

from pymongo import MongoClient

from stock_news.config import Settings
from stock_news.news_window import latest_completed_window, next_run_time

POLL_INTERVAL_SECONDS = 60
FAILED_RUN_RETRY_SECONDS = 15 * 60


def _run_window(start: datetime, end: datetime) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "stock_news.cli",
        "--window-start",
        start.isoformat(),
        "--window-end",
        end.isoformat(),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Scheduled collection failed with exit code {error.returncode}", flush=True)


def main() -> None:
    settings = Settings.from_environment()
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    runs = client[settings.mongo_database]["collection_runs"]
    try:
        last_attempted: dict[datetime, float] = {}
        while True:
            start, end = latest_completed_window(
                settings.window_timezone, settings.window_start_hour,
                window_hours=settings.window_hours,
            )
            run_filter = {"window_start": start, "window_end": end}
            current_brief = client[settings.mongo_database]["current_brief"].find_one(
                {"_id": "current", "window_end": end}, {"_id": 1}
            )
            run_record = runs.find_one(run_filter)
            run_complete = bool(
                run_record
                and int(run_record.get("failed", 0)) == 0
                and current_brief is not None
            )
            if run_complete:
                last_attempted.pop(start, None)
            else:
                now_monotonic = time.monotonic()
                last_attempt = last_attempted.get(start)
                if (
                    last_attempt is None
                    or now_monotonic - last_attempt >= FAILED_RUN_RETRY_SECONDS
                ):
                    print(
                        f"Running collection for missing Taiwan-time window "
                        f"{start.isoformat()} to {end.isoformat()}.",
                        flush=True,
                    )
                    last_attempted[start] = now_monotonic
                    _run_window(start, end)
                    completed_record = runs.find_one(run_filter)
                    completed_brief = client[settings.mongo_database]["current_brief"].find_one(
                        {"_id": "current", "window_end": end}, {"_id": 1}
                    )
                    if (
                        completed_record
                        and int(completed_record.get("failed", 0)) == 0
                        and completed_brief is not None
                    ):
                        last_attempted.pop(start, None)
                    else:
                        print(
                            "The collection run is incomplete or a feed failed; will retry "
                            "this window after the retry interval.",
                            flush=True,
                        )

            run_at = next_run_time(settings.window_timezone, settings.window_start_hour)
            seconds_to_boundary = max(
                (run_at - datetime.now(run_at.tzinfo)).total_seconds(), 1
            )
            time.sleep(min(POLL_INTERVAL_SECONDS, seconds_to_boundary))
    finally:
        client.close()


if __name__ == "__main__":
    main()
