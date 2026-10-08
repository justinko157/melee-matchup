"""Record one week of real start.gg responses as a test fixture.

Usage (from the repo root):
    uv run python -m scripts.record_startgg_fixture 2025-01-06
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import RecordingClient
from pipeline.ingest.startgg import StartGGClient

FIXTURE_DIR = Path("tests/fixtures/startgg")


class NullPublisher:
    def publish(self, topic: str, key: str, value: dict) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("week", help="Monday partition key, e.g. 2025-01-06")
    parser.add_argument("--min-attendees", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv(".env")
    client = RecordingClient(StartGGClient(os.environ.get("STARTGG_API_TOKEN", "")))
    stats = collect_week(
        client,
        NullPublisher(),
        args.week,
        run_id="fixture-recording",
        min_attendees=args.min_attendees,
    )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIXTURE_DIR / f"week_{args.week}.json"
    client.save(out)
    print(f"{stats} api_calls={client.api_calls} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
