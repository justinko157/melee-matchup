#!/usr/bin/env bash
# End-to-end smoke test: recorded week -> Kafka -> bronze -> Trino, twice.
# Uses a separate Compose project with no host ports and no daemon, so it can
# run next to the dev stack and nothing else launches runs mid-test.
set -euo pipefail
cd "$(dirname "$0")/.."

# Git Bash would rewrite the absolute container paths below into Windows paths.
export MSYS_NO_PATHCONV=1

WEEK=2025-01-06
FIXTURE=/app/tests/fixtures/startgg/week_${WEEK}.json
COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.smoke.yml -p melee-smoke)

cleanup() { "${COMPOSE[@]}" down -v --remove-orphans >/dev/null || true; }
trap cleanup EXIT
cleanup

"${COMPOSE[@]}" up -d --build --wait dagster-code trino

in_code() {
  "${COMPOSE[@]}" exec -T \
    -e STARTGG_FIXTURE_PATH="$FIXTURE" -e STARTGG_API_TOKEN=unused-in-fixture-mode \
    dagster-code "$@"
}

for copies in 1 2; do
  echo "--- ingestion pass $copies"
  in_code dagster asset materialize -m pipeline.dagster_defs --select startgg_raw --partition "$WEEK"
  in_code dagster asset materialize -m pipeline.dagster_defs --select bronze_startgg
  in_code python -m tests.smoke.check_bronze --fixture "$FIXTURE" --week "$WEEK" --copies "$copies"
done

echo "smoke test passed"
