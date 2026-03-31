"""CLI entry point for the data collection pipeline."""

import argparse
import logging
from datetime import datetime, timezone

from .api_client import StartGGClient
from .collector import MeleeCollector
from .database import MeleeDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Major Melee tournament slugs for targeted collection
MAJOR_SLUGS = [
    "genesis",
    "the-big-house",
    "super-smash-con",
    "shine",
    "pound",
    "smash-summit",
    "battle-of-bc",
    "get-on-my-level",
    "collision",
    "evo",
    "dreamhack",
    "ceo",
    "low-tier-city",
    "full-bloom",
    "smash-factor",
    "the-function",
    "fete",
    "double-down",
    "lost-tech-city",
    "ludwig-smash-invitational",
]


def date_to_timestamp(date_str: str) -> int:
    """Convert YYYY-MM-DD to Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def main():
    parser = argparse.ArgumentParser(
        description="Collect competitive Melee tournament data from start.gg"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start of date range (YYYY-MM-DD). Default: 2018-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-03-31",
        help="End of date range (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--min-attendees",
        type=int,
        default=50,
        help="Minimum tournament attendees to include. Default: 50",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database. Default: data/raw/melee.db",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-collect events even if they already have data",
    )

    args = parser.parse_args()

    after_ts = date_to_timestamp(args.start_date)
    before_ts = date_to_timestamp(args.end_date)

    logger.info(f"Collecting Melee data from {args.start_date} to {args.end_date}")
    logger.info(f"Minimum attendees: {args.min_attendees}")

    client = StartGGClient()
    db = MeleeDB(args.db_path) if args.db_path else MeleeDB()

    collector = MeleeCollector(client, db)

    try:
        collector.collect_date_range(
            after_date=after_ts,
            before_date=before_ts,
            min_attendees=args.min_attendees,
            skip_existing=not args.no_skip,
        )

        stats = db.get_stats()
        logger.info("Collection complete!")
        logger.info(
            f"  {stats['tournaments']} tournaments, {stats['events']} events, "
            f"{stats['players']} players, {stats['sets']} sets, {stats['games']} games"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
