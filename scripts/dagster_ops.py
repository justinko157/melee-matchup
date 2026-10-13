"""Drive the local Dagster instance over GraphQL: launch jobs and backfills, poll status.

Usage:
  uv run python scripts/dagster_ops.py launch train_matchup_model
  uv run python scripts/dagster_ops.py run-status RUN_ID
  uv run python scripts/dagster_ops.py backfill --start 2018-01-01 --end 2024-12-30
  uv run python scripts/dagster_ops.py backfill-status BACKFILL_ID

Set DAGSTER_GRAPHQL_URL if the webserver isn't at http://localhost:3000/graphql.
Exits non-zero, with the error on stderr, when the call fails or Dagster
returns anything other than the expected success type.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta

LOCATION = "melee"
REPOSITORY = "__repository__"

LAUNCH_RUN = """
mutation($params: ExecutionParams!) {
  launchRun(executionParams: $params) {
    __typename
    ... on LaunchRunSuccess { run { runId } }
    ... on PythonError { message }
    ... on RunConfigValidationInvalid { errors { message } }
    ... on PipelineNotFoundError { message }
  }
}"""
RUN_STATUS = """
query($id: ID!) { runOrError(runId: $id) { __typename ... on Run { status } } }"""
LAUNCH_BACKFILL = """
mutation($params: LaunchBackfillParams!) {
  launchPartitionBackfill(backfillParams: $params) {
    __typename
    ... on LaunchBackfillSuccess { backfillId }
    ... on PythonError { message }
    ... on InvalidSubsetError { message }
  }
}"""
BACKFILL_STATUS = """
query($id: String!) {
  partitionBackfillOrError(backfillId: $id) {
    __typename
    ... on PartitionBackfill { status numPartitions partitionStatusCounts { runStatus count } }
  }
}"""


def weekly_partition_keys(start: date, end: date) -> list[str]:
    """Every Monday from start to end, inclusive (the startgg_raw partition keys)."""
    if start.weekday() != 0 or end.weekday() != 0:
        raise ValueError("Weekly partitions start on a Monday; both bounds must be Mondays")
    keys, day = [], start
    while day <= end:
        keys.append(day.isoformat())
        day += timedelta(days=7)
    return keys


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def graphql(url: str, query: str, variables: dict) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps({"query": query, "variables": variables}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        fail(f"Dagster GraphQL at {url} returned HTTP {error.code}: {error.reason}")
    except urllib.error.URLError as error:
        fail(f"Could not reach Dagster GraphQL at {url}: {error.reason} (set DAGSTER_GRAPHQL_URL?)")
    if body.get("errors"):
        fail(f"GraphQL errors from {url}: {json.dumps(body['errors'])}")
    return body["data"]


def expect(result: dict, typename: str) -> dict:
    """Return the result if it is the success type; otherwise report it and exit non-zero."""
    if result.get("__typename") != typename:
        fail(f"Expected {typename}, got: {json.dumps(result)}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("launch").add_argument("job")
    commands.add_parser("run-status").add_argument("run_id")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--start", type=date.fromisoformat, required=True)
    backfill.add_argument("--end", type=date.fromisoformat, required=True)
    backfill.add_argument("--asset", default="startgg_raw")
    commands.add_parser("backfill-status").add_argument("backfill_id")
    args = parser.parse_args(argv)
    url = os.getenv("DAGSTER_GRAPHQL_URL", "http://localhost:3000/graphql")

    if args.command == "launch":
        selector = {
            "repositoryLocationName": LOCATION,
            "repositoryName": REPOSITORY,
            "jobName": args.job,
        }
        result = graphql(url, LAUNCH_RUN, {"params": {"selector": selector, "runConfigData": {}}})
        print(json.dumps(expect(result["launchRun"], "LaunchRunSuccess")))
    elif args.command == "run-status":
        result = graphql(url, RUN_STATUS, {"id": args.run_id})
        print(json.dumps(expect(result["runOrError"], "Run")))
    elif args.command == "backfill":
        keys = weekly_partition_keys(args.start, args.end)
        params = {
            "assetSelection": [{"path": [args.asset]}],
            "partitionNames": keys,
            "title": f"{args.asset} {keys[0]}..{keys[-1]}",
        }
        result = graphql(url, LAUNCH_BACKFILL, {"params": params})
        print(
            len(keys),
            "partitions:",
            json.dumps(expect(result["launchPartitionBackfill"], "LaunchBackfillSuccess")),
        )
    else:
        result = graphql(url, BACKFILL_STATUS, {"id": args.backfill_id})
        print(json.dumps(expect(result["partitionBackfillOrError"], "PartitionBackfill")))


if __name__ == "__main__":
    main()
