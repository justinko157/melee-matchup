"""Assert bronze contents after the smoke test's ingestion runs.

Usage: python -m tests.smoke.check_bronze --fixture PATH --week KEY --copies N
"""

import argparse

import trino

from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import FixtureClient


class CountingPublisher:
    def publish(self, topic: str, key: str, value: dict) -> None:
        pass


def scalar_row(cursor, sql: str) -> tuple:
    cursor.execute(sql)
    return tuple(cursor.fetchone())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--week", required=True)
    parser.add_argument("--copies", type=int, required=True)
    args = parser.parse_args()

    expected = collect_week(
        FixtureClient(args.fixture), CountingPublisher(), args.week, run_id="check"
    )
    # An empty fixture would make every count below trivially match.
    assert expected.tournaments > 0, f"fixture has no eligible tournaments for {args.week}"
    assert expected.sets > 0, f"fixture has no sets for {args.week}"
    cursor = trino.dbapi.connect(
        host="trino", port=8080, user="smoke", catalog="lakekeeper", schema="bronze"
    ).cursor()

    for table, unique in (
        ("startgg_tournaments", expected.tournaments),
        ("startgg_sets", expected.sets),
    ):
        rows, distinct = scalar_row(
            cursor, f"SELECT count(*), count(DISTINCT entity_id) FROM {table}"
        )
        assert rows == unique * args.copies, (
            f"{table}: {rows} rows, expected {unique * args.copies}"
        )
        assert distinct == unique, f"{table}: {distinct} distinct ids, expected {unique}"
        print(f"{table}: {rows} rows, {distinct} distinct ids")

    (rejects,) = scalar_row(cursor, "SELECT count(*) FROM startgg_rejects")
    assert rejects == 0, f"{rejects} rejected messages"
    print("startgg_rejects: 0 rows")


if __name__ == "__main__":
    main()
