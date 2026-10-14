# Silver and Gold Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn append-only Iceberg bronze into deduplicated silver tables (incremental `MERGE INTO`) and fully recomputed gold tables (PyIceberg). Then move the XGBoost model (MLflow registry with a gated `champion` alias) and the Streamlit app onto gold, and retire SQLite and `src/`.

**Architecture:**
- **Silver.** A Spark batch job (`silver_job.py`) reads only the bronze snapshots appended since a per-table watermark (Iceberg incremental append scan). It parses the JSON payloads into five typed silver tables and MERGEs them with a "newer or equal `source_ingested_at` wins" rule.
- **Gold.** A plain-Python pass reads silver through PyIceberg and runs the existing Elo/feature engine in time order. It replaces three gold tables in one commit each.
- **Orchestration.** Dagster runs silver and gold on `AutomationCondition.eager()`, trains daily into an MLflow server, and runs weekly Iceberg maintenance.
- **App.** The app reads gold through PyIceberg and loads `models:/melee-matchup@champion`.

**Tech Stack:** Python 3.12, uv, PySpark 4.0.4 with Iceberg 1.11.0 (Lakekeeper REST catalog, Silo S3), PyIceberg 0.12.0, Dagster 1.13.25, MLflow 3.16.1 (server image `ghcr.io/mlflow/mlflow:v3.16.1-full`), XGBoost, Streamlit, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-08-silver-gold-design.md`. Read it before starting a task; it is the binding authority. The ingestion foundation spec (`docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md`) describes the bronze layer this builds on.

## Global Constraints

**Tooling and versions**
- Python is pinned to 3.12. Use `uv` for everything (`uv add`, `uv lock`, `uv sync --all-extras`, `uv run`). Never use pip, including inside Dockerfiles (use `uv pip install` or `uv sync`).
- Package pins (exact):
  - `pyiceberg[pyarrow,pyiceberg-core]==0.12.0` in the `pipeline` and `app` extras;
  - `pyiceberg[sql-sqlite]==0.12.0` in the `dev` extra;
  - `mlflow==3.16.1` in the `ml` extra (moved out of `app`).
  - Every existing pin (pyspark 4.0.4, dagster 1.13.25, Iceberg 1.11.0 jars, images) stays as is.
- Images: MLflow is `ghcr.io/mlflow/mlflow:v3.16.1-full`, which includes `psycopg2-binary` and `boto3` via `mlflow[db,extras]`. Every other image keeps its existing pin. Local images are `melee-pipeline:dev` and `melee-app:dev`.

**Names (exact)**
- Silver tables: `silver.tournaments`, `silver.events`, `silver.players`, `silver.sets`, `silver.games`, `silver._watermarks`.
- Gold tables: `gold.set_features`, `gold.player_ratings`, `gold.head_to_head`. Gold table property: `melee.silver_sets_snapshot`.
- Silver lineage columns: `source_ingested_at`, `source_snapshot_id`, `silver_updated_at`.
- Drop reasons: `too_few_slots`, `missing_player`, `not_completed`, `unresolved_winner`.
- Dagster:
  - assets `silver_startgg`, `gold_startgg`, `matchup_model`;
  - job `lakehouse_maintenance`;
  - checks `no_orphan_sets`, `drop_rate_below_10pct`, `winner_is_a_player`, `features_match_silver`, `no_null_features`, `completed_at_non_decreasing`, `model_beats_baseline`.
- Pools, each with limit 1: `startgg_api`, `spark_lakehouse` (renamed from `spark_bronze`, shared by the bronze, silver and maintenance jobs), `gold`, `training`.
- Schedules use `execution_timezone="America/Los_Angeles"`:
  - training daily at 07:00 (`0 7 * * *`);
  - maintenance Sundays at 03:00 (`0 3 * * 0`).
- MLflow:
  - experiment `melee-matchup`;
  - registered model `melee-matchup`;
  - alias `champion`;
  - run and model-version tag `gold_snapshot_id`;
  - artifact `shap_metadata.json`.
- Environment variables, each with a local default:
  - `LAKEKEEPER_URL` (`http://lakekeeper:8181/catalog`);
  - `LAKEHOUSE_WAREHOUSE` (`melee`);
  - `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`. When these are unset, credentials are vended by Lakekeeper;
  - `MLFLOW_TRACKING_URI` (`http://mlflow:5000`);
  - `MLFLOW_PORT` (5000);
  - `APP_PORT` (8501).

**Secrets**
- The start.gg token must never appear in logs, exceptions, metadata, MLflow or Dagster config.
- Never print `.env`.
- Never commit secrets. S3 secrets reach Dagster only through `dg.EnvVar`.

**Line endings**
- Files mounted or copied into Linux containers (`infra/**`, `scripts/*.sh`, Dockerfiles) must be LF in the working tree.
- After `git add`, check with `git ls-files --eol <paths>`: the `w/` column must be `lf`.

**Before every commit**
- Run `uv run ruff format pipeline tests/pipeline tests/smoke scripts app.py` and `uv run ruff check tests/ app.py pipeline/ scripts/`. While `src/` still exists (before Task 14), also run `uv run ruff check src/`.
- Commit messages: plain, no `Co-Authored-By` trailer and no "Generated with Claude Code".

**The running dev stack** (project `melee`) holds the 2025 backfill.
- Never run `make nuke`, `docker compose down -v`, or anything that deletes its volumes.
- On this machine host port 3000 is taken, so export `DAGSTER_WEBSERVER_PORT=3001` before any `make up` and use `http://localhost:3001`.
- `make` may not be on PATH in agent shells. Tasks write `$MAKE`; set it first with `MAKE=/c/Users/justin_lifement/AppData/Local/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/make.exe`.
- In Git Bash, prefix `docker compose exec/run` commands that pass absolute container paths with `MSYS_NO_PATHCONV=1`.

**Memory**
- Docker has 7.3 GB and host memory is tight.
- Run `make smoke` locally at most once (Task 15), in the foreground. CI's `spark-and-smoke` job is the authoritative end-to-end run.
- Prefer unit tests with a PyIceberg `SqlCatalog`, and Spark tests via `make test-spark`.

**Windows host tests.** PyIceberg's default PyArrow FileIO cannot handle `file:///C:/...` warehouse paths. The SQL catalog therefore uses `py-io-impl = pyiceberg.io.fsspec.FsspecFileIO`, which was verified on this host with pyiceberg 0.12.0 plus `pyiceberg-core`.

**Outward actions and long runs**
- Do not modify `src/`, `app.py` or `tests/test_*.py` except in the tasks that move or retire them.
- Do not push, open PRs, or start the 2018–2024 backfill without explicit user confirmation (Tasks 16–17 say where to stop and ask).

## Review Focus

1. **Silver's first run on a stack where bronze tables don't exist yet.** This happens on a fresh stack, or when automation fires early. Expected: silver creates its tables, reads nothing, prints an all-zero summary, and its checks pass. Pinned by `test_missing_bronze_tables_is_a_no_op` (Task 3).
2. **Gold built from silver tables that exist but are empty.** Expected: empty gold tables are written, all three gold checks pass, no exception. Pinned by `test_empty_silver_writes_empty_gold` (Task 6).
3. **Null seeds or null `numAttendees` from start.gg.** Expected: `seed_diff` treats a missing seed as 0, `num_attendees` becomes 0, and no model feature is NaN. Pinned by `test_null_seeds_and_attendees_never_produce_nan` (Task 5).
4. **A training split that leaves the train or test side empty**, for example one fixture week with the default 2025-07-01 split. Expected: a clear `ValueError` naming the empty side and the row counts, not a cryptic scikit-learn error. Pinned by `test_empty_split_raises_a_clear_error` (Task 7).
5. **The app on a stack where no model has been promoted yet**, or where the catalog is unreachable. Expected: a "The data platform is unavailable: …" message naming the component, never a stack trace. Pinned by `test_missing_champion_raises_champion_not_found` (Task 13) and `test_app_reports_an_unreachable_catalog` (Task 13).

---

## File Structure

```
pipeline/
  model_features.py                  CORE_FEATURES / SEED_FEATURES / TARGET (no third-party imports)
  lakehouse.py                       LakehouseSettings, catalog properties, PyIceberg read helpers
  spark_jobs/
    locks.py                         exclusive_lock(): shared flock for bronze and silver jobs
    bronze_job.py                    (modified) lock_checkpoints delegates to locks.py
    silver.py                        pure bronze->silver DataFrame transforms, DDL, MERGE SQL
    watermarks.py                    Snapshot, ReadPlan, plan_read() (no Spark import)
    silver_job.py                    run_silver(spark) + spark-submit entrypoint
    maintenance_job.py               rewrite/expire/remove-orphans per table + entrypoint
  gold/
    __init__.py
    features.py                      EloEngine + chronological feature pass (from src/features.py)
    schema.py                        pandera features schema (tests and validation)
    tables.py                        gold identifiers + pyarrow schemas
    derive.py                        player_ratings(), head_to_head() from the pass's final state
    writer.py                        to_arrow(), replace_table(), write_gold()
    build.py                         load_silver(), build_gold(), gold_checks(), __main__
  ml/
    __init__.py
    names.py                         EXPERIMENT, MODEL_NAME, CHAMPION_ALIAS, GOLD_SNAPSHOT_TAG
    train.py                         split/evaluate/model functions on DataFrames (from src/model.py)
    tuning.py                        Optuna tuning (from src/tuning.py)
    promotion.py                     PromotionDecision, decide_promotion()
    registry.py                      train_and_register(), champion_gold_snapshot()
  serving/
    __init__.py
    gold_reader.py                   load_player_ratings(), load_head_to_head(), lookup_h2h()
    champion.py                      Champion, ChampionNotFound, load_champion()
  dagster_defs/
    pools.py                         pool name constants
    resources.py                     (modified) SparkJobResource gains silver/maintenance; Lakehouse/Mlflow resources
    assets.py                        (modified) bronze pool renamed
    lakehouse_assets.py              silver_startgg, gold_startgg
    training.py                      matchup_model, train job, daily schedule
    maintenance.py                   lakehouse_maintenance job + weekly schedule
    automation.py                    (modified) adds lakehouse_automation sensor
    __init__.py                      (modified) wires everything
scripts/
  dagster_ops.py                     launch jobs/backfills and poll status over Dagster GraphQL
  smoke.sh                           (modified) adds silver, gold, model, check_lakehouse
infra/
  docker-compose.yml                 (modified) mlflow, postgres-databases, app, pools, env
  docker-compose.smoke.yml           (modified) mlflow/app ports reset
  postgres/init-databases.sql        (modified) mlflow database
  pipeline.Dockerfile                (modified) ml extra
Dockerfile                           (modified) app on gold: copies pipeline serving modules only
app.py                               (modified) reads gold + MLflow champion
tests/pipeline/
  conftest.py                        (modified) Iceberg-enabled spark fixture, sql catalog, mlflow fixtures
  lakehouse_data.py                  silver test-table writer for SqlCatalog tests
  ml_data.py                         synthetic gold features for training tests
  test_lakehouse.py  test_silver_ddl.py  test_silver_transform.py  test_watermarks.py
  test_silver_job.py  test_silver_lock.py  test_elo.py  test_gold_features.py
  test_gold_schema.py  test_gold_build.py  test_ml_train.py  test_promotion.py
  test_registry.py  test_dagster_lakehouse.py  test_dagster_training.py
  test_maintenance_sql.py  test_maintenance_job.py  test_dagster_maintenance.py
  test_gold_reader.py  test_champion.py  test_app.py  test_retirement.py
  test_dagster_ops_script.py  test_infra_config.py (modified)
tests/smoke/check_lakehouse.py       silver/gold/MLflow assertions for the smoke test
Removed (Task 14): src/ (all), tests/test_database.py. Moved earlier (Task 5): tests/test_elo.py,
  tests/test_features.py, tests/test_validation.py -> tests/pipeline/.
```

---

### Task 1: Dependencies, shared feature names and the lakehouse catalog helper

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `infra/pipeline.Dockerfile`
- Create: `pipeline/model_features.py`, `pipeline/lakehouse.py`
- Modify: `tests/pipeline/conftest.py`, `tests/pipeline/test_infra_config.py`
- Test: `tests/pipeline/test_lakehouse.py`

**Interfaces:**
- Produces:
  - `pipeline.model_features`: `SEED_FEATURES: list[str]`, `CORE_FEATURES: list[str]`, `TARGET = "p1_won"`.
  - `pipeline.lakehouse.LakehouseSettings`, a frozen dataclass with fields `catalog_type="rest"`, `uri="http://lakekeeper:8181/catalog"`, `warehouse="melee"`, `s3_endpoint=None`, `s3_access_key=None`, `s3_secret_key=None`, `s3_region="local-01"`. It has `.from_env(env=os.environ)`.
  - `catalog_properties(settings) -> dict[str, str]`.
  - `load_lakehouse_catalog(settings=None) -> pyiceberg.catalog.Catalog`.
  - `read_table(catalog, identifier, columns: tuple[str, ...], row_filter=AlwaysTrue(), snapshot_id=None) -> pd.DataFrame`.
  - `current_snapshot_id(catalog, identifier) -> int | None`.
  - Pytest fixtures `sql_settings` (a `LakehouseSettings` for a SQLite catalog in `tmp_path`) and `sql_catalog`.

- [ ] **Step 1: Update the dependencies**

Edit `pyproject.toml`'s optional dependencies so they read:

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "pytest>=8.0",
    "pandera>=0.18",
    "pre-commit>=3.6",
    "pyiceberg[sql-sqlite]==0.12.0",
]
ml = [
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.18",
    "shap>=0.45",
    "optuna>=3.5",
    "mlflow==3.16.1",
]
app = [
    "streamlit>=1.30",
    "pyiceberg[pyarrow,pyiceberg-core]==0.12.0",
]
pipeline = [
    "confluent-kafka>=2.16",
    "dagster>=1.13.25",
    "dagster-postgres>=0.29.25",
    "dagster-webserver>=1.13.25",
    "pyspark==4.0.4",
    "trino>=0.340",
    "pyiceberg[pyarrow,pyiceberg-core]==0.12.0",
]
```

Then run:

```bash
uv lock
uv sync --all-extras
uv run python -c "import pyiceberg, mlflow, pyiceberg_core; print(pyiceberg.__version__, mlflow.__version__)"
```

Expected: the last line prints `0.12.0 3.16.1`.

- [ ] **Step 2: Install the `ml` extra in the pipeline image**

In `infra/pipeline.Dockerfile`, change the sync line to:

```dockerfile
RUN uv sync --frozen --no-install-project --extra pipeline --extra ml --extra dev
```

- [ ] **Step 3: Write the failing tests**

Add to the bottom of `tests/pipeline/conftest.py`:

```python
@pytest.fixture
def sql_settings(tmp_path):
    """A PyIceberg SQL catalog in a temp dir; stands in for Lakekeeper in unit tests."""
    pytest.importorskip("sqlalchemy")
    from pipeline.lakehouse import LakehouseSettings

    return LakehouseSettings(
        catalog_type="sql",
        uri=f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}",
        warehouse=tmp_path.as_uri(),
    )


@pytest.fixture
def sql_catalog(sql_settings):
    from pipeline.lakehouse import load_lakehouse_catalog

    return load_lakehouse_catalog(sql_settings)
```

Create `tests/pipeline/test_lakehouse.py`:

```python
import pyarrow as pa
import pytest

from pipeline.lakehouse import (
    LakehouseSettings,
    catalog_properties,
    current_snapshot_id,
    read_table,
)


def test_rest_catalog_requests_vended_credentials():
    props = catalog_properties(LakehouseSettings())
    assert props == {
        "type": "rest",
        "uri": "http://lakekeeper:8181/catalog",
        "warehouse": "melee",
        "header.X-Iceberg-Access-Delegation": "vended-credentials",
    }


def test_explicit_s3_credentials_disable_vending():
    props = catalog_properties(
        LakehouseSettings(
            s3_endpoint="http://localhost:9000", s3_access_key="key", s3_secret_key="secret"
        )
    )
    assert "header.X-Iceberg-Access-Delegation" not in props
    assert props["s3.endpoint"] == "http://localhost:9000"
    assert props["s3.access-key-id"] == "key"
    assert props["s3.secret-access-key"] == "secret"
    assert props["s3.region"] == "local-01"


def test_explicit_credentials_need_a_secret():
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        catalog_properties(LakehouseSettings(s3_access_key="key"))


def test_settings_from_env_defaults():
    assert LakehouseSettings.from_env({}) == LakehouseSettings()


def test_settings_from_env_overrides():
    settings = LakehouseSettings.from_env(
        {
            "LAKEKEEPER_URL": "http://catalog/x",
            "LAKEHOUSE_WAREHOUSE": "w",
            "S3_ENDPOINT": "http://s3",
            "S3_ACCESS_KEY": "a",
            "S3_SECRET_KEY": "b",
        }
    )
    assert settings.uri == "http://catalog/x"
    assert settings.warehouse == "w"
    assert (settings.s3_endpoint, settings.s3_access_key, settings.s3_secret_key) == (
        "http://s3",
        "a",
        "b",
    )


def test_sql_catalog_uses_fsspec_io(sql_settings):
    props = catalog_properties(sql_settings)
    assert props["type"] == "sql"
    assert props["py-io-impl"] == "pyiceberg.io.fsspec.FsspecFileIO"


def test_sql_catalog_round_trip(sql_catalog):
    sql_catalog.create_namespace_if_not_exists("demo")
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    table = sql_catalog.create_table("demo.t", schema=schema)
    assert current_snapshot_id(sql_catalog, "demo.t") is None
    table.append(pa.table({"id": [1, 2], "name": ["a", "b"]}, schema=schema))
    frame = read_table(sql_catalog, "demo.t", ("id",))
    assert sorted(frame["id"].tolist()) == [1, 2]
    assert list(frame.columns) == ["id"]
    assert current_snapshot_id(sql_catalog, "demo.t") is not None
```

Add to `tests/pipeline/test_infra_config.py`:

```python
def test_pipeline_image_installs_the_ml_extra():
    dockerfile = Path("infra/pipeline.Dockerfile").read_text(encoding="utf-8")
    assert "--extra pipeline --extra ml --extra dev" in dockerfile
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_lakehouse.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.lakehouse'`.

- [ ] **Step 5: Write the implementation**

Create `pipeline/model_features.py`:

```python
"""Model feature names shared by gold, training and the app. No third-party imports."""

SEED_FEATURES = ["seed_diff"]
CORE_FEATURES = [
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "recent_wr_diff",
    "p1_h2h_wr",
    "h2h_total",
    "seed_diff",
    "num_attendees",
]
TARGET = "p1_won"
```

Create `pipeline/lakehouse.py`:

```python
"""PyIceberg access to the lakehouse for gold, training and the app."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.expressions import AlwaysTrue, BooleanExpression

# PyArrow's local FileIO can't parse file:///C:/... warehouse paths on Windows;
# fsspec's can, so the SQL catalog (used by tests) always uses it.
FSSPEC_IO = "pyiceberg.io.fsspec.FsspecFileIO"


@dataclass(frozen=True)
class LakehouseSettings:
    """Where the Iceberg catalog lives. The defaults match the Compose stack."""

    catalog_type: str = "rest"
    uri: str = "http://lakekeeper:8181/catalog"
    warehouse: str = "melee"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "local-01"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "LakehouseSettings":
        defaults = cls()
        return cls(
            catalog_type=env.get("LAKEHOUSE_CATALOG_TYPE", defaults.catalog_type),
            uri=env.get("LAKEKEEPER_URL", defaults.uri),
            warehouse=env.get("LAKEHOUSE_WAREHOUSE", defaults.warehouse),
            s3_endpoint=env.get("S3_ENDPOINT") or None,
            s3_access_key=env.get("S3_ACCESS_KEY") or None,
            s3_secret_key=env.get("S3_SECRET_KEY") or None,
            s3_region=env.get("S3_REGION", defaults.s3_region),
        )


def catalog_properties(settings: LakehouseSettings) -> dict[str, str]:
    """PyIceberg catalog properties. Without explicit S3 keys, Lakekeeper vends them."""
    if settings.catalog_type == "sql":
        return {
            "type": "sql",
            "uri": settings.uri,
            "warehouse": settings.warehouse,
            "py-io-impl": FSSPEC_IO,
        }
    if settings.catalog_type != "rest":
        raise ValueError(f"Unsupported catalog type: {settings.catalog_type}")
    props = {"type": "rest", "uri": settings.uri, "warehouse": settings.warehouse}
    if not settings.s3_access_key:
        props["header.X-Iceberg-Access-Delegation"] = "vended-credentials"
        return props
    if not settings.s3_secret_key:
        raise ValueError("S3_ACCESS_KEY is set but S3_SECRET_KEY is not")
    props.update(
        {
            "s3.access-key-id": settings.s3_access_key,
            "s3.secret-access-key": settings.s3_secret_key,
            "s3.region": settings.s3_region,
        }
    )
    if settings.s3_endpoint:
        props["s3.endpoint"] = settings.s3_endpoint
    return props


def load_lakehouse_catalog(settings: LakehouseSettings | None = None) -> Catalog:
    return load_catalog("melee", **catalog_properties(settings or LakehouseSettings.from_env()))


def read_table(
    catalog: Catalog,
    identifier: str,
    columns: tuple[str, ...],
    row_filter: BooleanExpression = AlwaysTrue(),
    snapshot_id: int | None = None,
) -> pd.DataFrame:
    """Selected columns of an Iceberg table as pandas, optionally pinned to a snapshot."""
    scan = catalog.load_table(identifier).scan(
        row_filter=row_filter, selected_fields=columns, snapshot_id=snapshot_id
    )
    return scan.to_pandas()


def current_snapshot_id(catalog: Catalog, identifier: str) -> int | None:
    snapshot = catalog.load_table(identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_lakehouse.py tests/pipeline/test_infra_config.py -v`
Expected: 7 passed in `test_lakehouse.py`. All of `test_infra_config.py` passes, including the new test.

Run: `uv run pytest`
Expected: everything passes with no warnings (the spark tests are deselected).

- [ ] **Step 7: Rebuild the images on the running dev stack**

This rebuilds `melee-pipeline:dev` with the new extras and restarts changed containers. Volumes are kept.

```bash
export DAGSTER_WEBSERVER_PORT=3001
$MAKE up      # MAKE=/c/Users/justin_lifement/AppData/Local/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/make.exe
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-code python -c "import pyiceberg, mlflow, xgboost; print('ok')"
```

Expected: `make up` exits 0 and the last command prints `ok`. The image build can take several minutes; run it with a long timeout.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock infra/pipeline.Dockerfile pipeline/model_features.py pipeline/lakehouse.py tests/pipeline/conftest.py tests/pipeline/test_lakehouse.py tests/pipeline/test_infra_config.py
git ls-files --eol infra/pipeline.Dockerfile
git commit -m "Add PyIceberg and MLflow pins, shared model feature names and lakehouse catalog helper"
```

---

### Task 2: Silver transforms and DDL (pure Spark functions)

**Files:**
- Create: `pipeline/spark_jobs/silver.py`
- Modify: `tests/pipeline/conftest.py` (Iceberg-enabled Spark session)
- Test: `tests/pipeline/test_silver_ddl.py`, `tests/pipeline/test_silver_transform.py`

**Interfaces:**
- Consumes:
  - `BRONZE_TABLES` from `pipeline.spark_jobs.bronze`: `{"tournament": "bronze.startgg_tournaments", "set": "bronze.startgg_sets"}`.
  - The bronze column names `kafka_partition`, `kafka_offset`, `ingested_at`, `query_variables`, `payload`.
  - `MELEE_VIDEOGAME_ID` from `pipeline.ingest.startgg`.
- Produces, in `pipeline.spark_jobs.silver`:
  - Constants:
    - `SILVER_NAMESPACE = "silver"`, `WATERMARKS_TABLE = "silver._watermarks"`;
    - `MERGE_ORDER = ("tournaments", "events", "players", "sets", "games")`;
    - `KEYS` (entity → key column);
    - `LINEAGE_COLUMNS`, `ORDER_COLUMNS = ("source_ingested_at", "kafka_partition", "kafka_offset")`;
    - `DROP_REASONS`.
  - Table helpers: `table_name(entity) -> str`, `business_columns(entity) -> tuple[str, ...]`, `silver_columns(entity) -> tuple[str, ...]` (business plus lineage), `silver_table_ddl(entity) -> str`, `WATERMARKS_DDL: str`.
  - Transforms:
    - `parse_tournaments(bronze) -> DataFrame`, `tournament_rows(parsed) -> DataFrame`, `event_rows(parsed) -> DataFrame`;
    - `parse_sets(bronze) -> DataFrame`, which adds `drop_reason`;
    - `set_rows(parsed)`, `player_rows(parsed)`, `game_rows(parsed)`;
    - `drop_counts(parsed) -> tuple[int, dict[str, int]]`, returning (bronze rows, drops per reason);
    - `newest_per_key(df, key) -> DataFrame`;
    - `finalize(df, entity, snapshot_id) -> DataFrame`, which outputs exactly `silver_columns(entity)`.
  - `merge_sql(entity, view) -> str`.
  - Every `*_rows` function returns the business columns plus `ORDER_COLUMNS`.

- [ ] **Step 1: Give the Spark test session an Iceberg catalog**

Replace the `spark` fixture in `tests/pipeline/conftest.py` with the version below. In the pipeline container the jar is already in the `/opt/ivy` cache; CI downloads it from Maven Central.

```python
ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """Local SparkSession with an Iceberg Hadoop catalog named `local` as the default.

    Only Spark-marked tests use it; they need Java.
    """
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("iceberg-warehouse")
    session = (
        SparkSession.builder.master("local[1]")
        .appName("pipeline-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.jars.packages", ICEBERG_RUNTIME)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", warehouse.as_uri())
        .config("spark.sql.defaultCatalog", "local")
        .getOrCreate()
    )
    yield session
    session.stop()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/pipeline/test_silver_ddl.py`:

```python
"""Silver DDL column lists must match the transform column lists. No SparkSession needed."""

import re

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.silver import (  # noqa: E402
    MERGE_ORDER,
    WATERMARKS_DDL,
    silver_columns,
    silver_table_ddl,
)


def ddl_columns(ddl: str) -> tuple[str, ...]:
    body = re.search(r"\((.*)\)\s*USING", ddl, re.DOTALL).group(1)
    return tuple(column.split()[0] for column in body.split(","))


@pytest.mark.parametrize("entity", MERGE_ORDER)
def test_silver_ddl_matches_silver_columns(entity):
    ddl = silver_table_ddl(entity)
    assert ddl_columns(ddl) == silver_columns(entity)
    assert "'format-version'='2'" in ddl


def test_watermarks_ddl_columns():
    assert ddl_columns(WATERMARKS_DDL) == ("bronze_table", "snapshot_id", "updated_at")
```

Create `tests/pipeline/test_silver_transform.py`:

```python
import json
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.silver import (  # noqa: E402
    drop_counts,
    event_rows,
    game_rows,
    newest_per_key,
    parse_sets,
    parse_tournaments,
    player_rows,
    set_rows,
    tournament_rows,
)

BRONZE_SCHEMA = (
    "kafka_partition INT, kafka_offset BIGINT, ingested_at TIMESTAMP, "
    "query_variables STRING, payload STRING"
)
T0 = datetime(2025, 1, 6, 12, tzinfo=UTC)
T1 = datetime(2025, 1, 7, 12, tzinfo=UTC)


def bronze(spark, rows):
    """rows: (payload dict, query_variables dict, ingested_at, offset)."""
    return spark.createDataFrame(
        [(0, offset, ts, json.dumps(qv), json.dumps(p)) for p, qv, ts, offset in rows],
        BRONZE_SCHEMA,
    )


def slot(entrant_id, player_id, tag, seed=None, initial_seed=None, score=None):
    participants = [{"player": {"id": player_id, "gamerTag": tag}}] if player_id else []
    return {
        "seed": {"seedNum": seed} if seed is not None else None,
        "standing": {"placement": 1, "stats": {"score": {"value": score}}},
        "entrant": {
            "id": entrant_id,
            "name": tag,
            "initialSeedNum": initial_seed,
            "participants": participants,
        },
    }


def set_node(set_id=100, state=3, winner=11, slots=None, games=None, completed=1736200000):
    return {
        "id": set_id,
        "displayScore": "A 2 - B 1",
        "winnerId": winner,
        "round": 1,
        "fullRoundText": "Winners Round 1",
        "completedAt": completed,
        "totalGames": 3,
        "state": state,
        "games": games,
        "slots": slots
        if slots is not None
        else [slot(11, 1, "Alice", seed=1, score=2.0), slot(12, 2, "Bob", initial_seed=4, score=1.0)],
        "phaseGroup": {"phase": {"name": "Pools"}},
    }


def tournament_node(tid=7, events=None):
    return {
        "id": tid,
        "name": "Genesis",
        "slug": "tournament/genesis",
        "numAttendees": 500,
        "startAt": 1736150400,
        "endAt": 1736323200,
        "city": "San Jose",
        "addrState": "CA",
        "countryCode": "US",
        "isOnline": False,
        "events": events
        if events is not None
        else [
            {"id": 70, "name": "Melee Singles", "slug": "s", "numEntrants": 400, "videogame": {"id": 1}},
            {"id": 71, "name": "Ultimate", "slug": "u", "numEntrants": 900, "videogame": {"id": 1386}},
        ],
    }


def test_tournament_rows_are_typed(spark):
    parsed = parse_tournaments(bronze(spark, [(tournament_node(), {}, T0, 1)]))
    (row,) = tournament_rows(parsed).collect()
    assert row.tournament_id == 7
    assert row.state == "CA"
    assert row.num_attendees == 500
    assert row.start_at == datetime(2025, 1, 6, 8, 0)  # UTC session time zone, naive
    assert row.is_online is False
    assert row.source_ingested_at == datetime(2025, 1, 6, 12, 0)


def test_events_keep_only_melee(spark):
    parsed = parse_tournaments(bronze(spark, [(tournament_node(), {}, T0, 1)]))
    rows = event_rows(parsed).collect()
    assert [(r.event_id, r.tournament_id, r.num_entrants) for r in rows] == [(70, 7, 400)]


def test_completed_set_becomes_a_row(spark):
    parsed = parse_sets(bronze(spark, [(set_node(), {"eventId": 70}, T0, 1)]))
    (row,) = set_rows(parsed).collect()
    assert row.set_id == 100
    assert row.event_id == 70
    assert (row.player1_id, row.player2_id, row.winner_player_id) == (1, 2, 1)
    assert (row.player1_seed, row.player2_seed) == (1, 4)  # seedNum, then initialSeedNum
    assert (row.player1_score, row.player2_score) == (2.0, 1.0)
    assert row.phase_name == "Pools"
    assert row.completed_at == datetime(2025, 1, 6, 21, 46, 40)


def test_unplayable_sets_are_dropped_with_a_reason(spark):
    rows = [
        (set_node(set_id=1, slots=[slot(11, 1, "Alice")]), {"eventId": 70}, T0, 1),
        (set_node(set_id=2, slots=[slot(11, 1, "A"), slot(12, None, "bye")]), {"eventId": 70}, T0, 2),
        (set_node(set_id=3, state=2), {"eventId": 70}, T0, 3),
        (set_node(set_id="preview_9_1"), {"eventId": 70}, T0, 4),
        (set_node(set_id=5, winner=999), {"eventId": 70}, T0, 5),
        (set_node(set_id=6), {"eventId": 70}, T0, 6),
    ]
    parsed = parse_sets(bronze(spark, rows))
    total, dropped = drop_counts(parsed)
    assert total == 6
    assert dropped == {
        "too_few_slots": 1,
        "missing_player": 1,
        "not_completed": 2,
        "unresolved_winner": 1,
    }
    assert [r.set_id for r in set_rows(parsed).collect()] == [6]


def test_games_match_characters_by_entrant(spark):
    games = [
        {
            "id": 900,
            "orderNum": 1,
            "winnerId": 12,
            "stage": {"id": 3, "name": "Battlefield"},
            # Listed with player 2's entrant first, to prove matching is by entrant id.
            "selections": [
                {"entrant": {"id": 12}, "character": {"id": 2, "name": "Fox"}},
                {"entrant": {"id": 11}, "character": {"id": 1, "name": "Marth"}},
            ],
        },
        {"id": None, "orderNum": 2, "winnerId": 11, "stage": None, "selections": None},
    ]
    parsed = parse_sets(bronze(spark, [(set_node(games=games), {"eventId": 70}, T0, 1)]))
    (row,) = game_rows(parsed).collect()
    assert (row.game_id, row.set_id, row.game_number) == (900, 100, 1)
    assert (row.player1_character_name, row.player2_character_name) == ("Marth", "Fox")
    assert row.stage_name == "Battlefield"
    assert row.completed_at == datetime(2025, 1, 6, 21, 46, 40)


def test_dropped_sets_contribute_no_games_or_players(spark):
    games = [{"id": 901, "orderNum": 1, "winnerId": 11, "stage": None, "selections": []}]
    parsed = parse_sets(bronze(spark, [(set_node(state=2, games=games), {"eventId": 70}, T0, 1)]))
    assert game_rows(parsed).count() == 0
    assert player_rows(parsed).count() == 0


def test_players_take_the_newest_tag(spark):
    old = set_node(set_id=1)
    new = set_node(set_id=2, slots=[slot(11, 1, "Alice2"), slot(12, 2, "Bob")])
    parsed = parse_sets(
        bronze(spark, [(old, {"eventId": 70}, T0, 1), (new, {"eventId": 70}, T1, 2)])
    )
    players = {r.player_id: r.gamer_tag for r in newest_per_key(player_rows(parsed), "player_id").collect()}
    assert players == {1: "Alice2", 2: "Bob"}


def test_newest_version_wins_then_offset_breaks_ties(spark):
    v1 = set_node(set_id=1, slots=[slot(11, 1, "A", score=1.0), slot(12, 2, "B", score=2.0)], winner=12)
    v2 = set_node(set_id=1, slots=[slot(11, 1, "A", score=2.0), slot(12, 2, "B", score=0.0)])
    v3 = set_node(set_id=1, slots=[slot(11, 1, "A", score=3.0), slot(12, 2, "B", score=0.0)])
    rows = [
        (v2, {"eventId": 70}, T1, 5),
        (v1, {"eventId": 70}, T0, 9),  # older ingest loses despite the higher offset
        (v3, {"eventId": 70}, T1, 6),  # same ingest time as v2: higher offset wins
    ]
    (row,) = newest_per_key(set_rows(parse_sets(bronze(spark, rows))), "set_id").collect()
    assert row.player1_score == 3.0


def test_event_id_comes_from_query_variables(spark):
    parsed = parse_sets(bronze(spark, [(set_node(), {"eventId": 12345}, T0, 1)]))
    assert set_rows(parsed).first().event_id == 12345
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_silver_ddl.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.spark_jobs.silver'`.

- [ ] **Step 4: Write the implementation**

Create `pipeline/spark_jobs/silver.py`:

```python
"""Pure transforms from bronze rows to silver rows, plus silver DDL and MERGE SQL.

Parsing rules come from the retired src/collector.py:
- a set needs two slots, both with a player, and state 3;
- the winner is resolved from winnerId to the matching entrant's player.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pipeline.ingest.startgg import MELEE_VIDEOGAME_ID

SILVER_NAMESPACE = "silver"
WATERMARKS_TABLE = f"{SILVER_NAMESPACE}._watermarks"
MERGE_ORDER = ("tournaments", "events", "players", "sets", "games")
KEYS = {
    "tournaments": "tournament_id",
    "events": "event_id",
    "players": "player_id",
    "sets": "set_id",
    "games": "game_id",
}
LINEAGE_COLUMNS = ("source_ingested_at", "source_snapshot_id", "silver_updated_at")
ORDER_COLUMNS = ("source_ingested_at", "kafka_partition", "kafka_offset")
DROP_REASONS = ("too_few_slots", "missing_player", "not_completed", "unresolved_winner")

SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "tournaments": (
        ("tournament_id", "BIGINT"),
        ("name", "STRING"),
        ("slug", "STRING"),
        ("num_attendees", "BIGINT"),
        ("start_at", "TIMESTAMP"),
        ("end_at", "TIMESTAMP"),
        ("city", "STRING"),
        ("state", "STRING"),
        ("country_code", "STRING"),
        ("is_online", "BOOLEAN"),
    ),
    "events": (
        ("event_id", "BIGINT"),
        ("tournament_id", "BIGINT"),
        ("name", "STRING"),
        ("slug", "STRING"),
        ("num_entrants", "BIGINT"),
    ),
    "players": (("player_id", "BIGINT"), ("gamer_tag", "STRING")),
    "sets": (
        ("set_id", "BIGINT"),
        ("event_id", "BIGINT"),
        ("round", "BIGINT"),
        ("full_round_text", "STRING"),
        ("phase_name", "STRING"),
        ("completed_at", "TIMESTAMP"),
        ("total_games", "BIGINT"),
        ("display_score", "STRING"),
        ("entrant1_id", "BIGINT"),
        ("entrant2_id", "BIGINT"),
        ("player1_id", "BIGINT"),
        ("player2_id", "BIGINT"),
        ("winner_player_id", "BIGINT"),
        ("player1_score", "DOUBLE"),
        ("player2_score", "DOUBLE"),
        ("player1_seed", "BIGINT"),
        ("player2_seed", "BIGINT"),
    ),
    "games": (
        ("game_id", "BIGINT"),
        ("set_id", "BIGINT"),
        ("game_number", "BIGINT"),
        ("winner_entrant_id", "BIGINT"),
        ("stage_id", "BIGINT"),
        ("stage_name", "STRING"),
        ("player1_character_id", "BIGINT"),
        ("player1_character_name", "STRING"),
        ("player2_character_id", "BIGINT"),
        ("player2_character_name", "STRING"),
        ("completed_at", "TIMESTAMP"),
    ),
}
PARTITIONS = {"sets": "months(completed_at)", "games": "months(completed_at)"}
# v2 + copy-on-write keeps the tables readable by PyIceberg without delete files.
TABLE_PROPERTIES = (
    "TBLPROPERTIES ('format-version'='2', 'write.merge.mode'='copy-on-write', "
    "'write.update.mode'='copy-on-write', 'write.delete.mode'='copy-on-write')"
)


def table_name(entity: str) -> str:
    return f"{SILVER_NAMESPACE}.{entity}"


def business_columns(entity: str) -> tuple[str, ...]:
    return tuple(name for name, _ in SCHEMAS[entity])


def silver_columns(entity: str) -> tuple[str, ...]:
    return business_columns(entity) + LINEAGE_COLUMNS


def silver_table_ddl(entity: str) -> str:
    columns = ", ".join(f"{name} {kind}" for name, kind in SCHEMAS[entity])
    lineage = "source_ingested_at TIMESTAMP, source_snapshot_id BIGINT, silver_updated_at TIMESTAMP"
    partition = f"PARTITIONED BY ({PARTITIONS[entity]})" if entity in PARTITIONS else ""
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name(entity)} ({columns}, {lineage}) "
        f"USING iceberg {partition} {TABLE_PROPERTIES}"
    )


WATERMARKS_DDL = (
    f"CREATE TABLE IF NOT EXISTS {WATERMARKS_TABLE} "
    f"(bronze_table STRING, snapshot_id BIGINT, updated_at TIMESTAMP) "
    f"USING iceberg {TABLE_PROPERTIES}"
)


def _struct(**fields: DataType) -> StructType:
    return StructType([StructField(name, kind) for name, kind in fields.items()])


EVENT_NODE = _struct(
    id=LongType(),
    name=StringType(),
    slug=StringType(),
    numEntrants=LongType(),
    videogame=_struct(id=LongType()),
)
TOURNAMENT_PAYLOAD = _struct(
    id=LongType(),
    name=StringType(),
    slug=StringType(),
    numAttendees=LongType(),
    startAt=LongType(),
    endAt=LongType(),
    city=StringType(),
    addrState=StringType(),
    countryCode=StringType(),
    isOnline=BooleanType(),
    events=ArrayType(EVENT_NODE),
)
ENTRANT = _struct(
    id=LongType(),
    name=StringType(),
    initialSeedNum=LongType(),
    participants=ArrayType(_struct(player=_struct(id=LongType(), gamerTag=StringType()))),
)
SLOT = _struct(
    seed=_struct(seedNum=LongType()),
    standing=_struct(
        placement=LongType(), stats=_struct(score=_struct(value=DoubleType()))
    ),
    entrant=ENTRANT,
)
GAME = _struct(
    id=LongType(),
    orderNum=LongType(),
    winnerId=LongType(),
    stage=_struct(id=LongType(), name=StringType()),
    selections=ArrayType(
        _struct(
            entrant=_struct(id=LongType()),
            character=_struct(id=LongType(), name=StringType()),
        )
    ),
)
# The set id is a string because unplayed sets have ids like "preview_123_1".
SET_PAYLOAD = _struct(
    id=StringType(),
    displayScore=StringType(),
    winnerId=LongType(),
    round=LongType(),
    fullRoundText=StringType(),
    completedAt=LongType(),
    totalGames=LongType(),
    state=LongType(),
    games=ArrayType(GAME),
    slots=ArrayType(SLOT),
    phaseGroup=_struct(phase=_struct(name=StringType())),
)


def _nth(array: Column, n: int) -> Column:
    # try_element_at returns null past the end; plain indexing errors under ANSI mode.
    return F.try_element_at(array, F.lit(n))


def _source(bronze: DataFrame, payload_schema: StructType) -> list[Column]:
    return [
        F.col("ingested_at").alias("source_ingested_at"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.from_json(F.col("payload"), payload_schema).alias("p"),
    ]


def parse_tournaments(bronze: DataFrame) -> DataFrame:
    """One typed row per bronze tournament row (duplicates kept)."""
    return bronze.select(*_source(bronze, TOURNAMENT_PAYLOAD)).where(F.col("p.id").isNotNull())


def tournament_rows(parsed: DataFrame) -> DataFrame:
    return parsed.select(
        F.col("p.id").alias("tournament_id"),
        F.col("p.name").alias("name"),
        F.col("p.slug").alias("slug"),
        F.col("p.numAttendees").alias("num_attendees"),
        F.timestamp_seconds(F.col("p.startAt")).alias("start_at"),
        F.timestamp_seconds(F.col("p.endAt")).alias("end_at"),
        F.col("p.city").alias("city"),
        F.col("p.addrState").alias("state"),
        F.col("p.countryCode").alias("country_code"),
        F.col("p.isOnline").alias("is_online"),
        *ORDER_COLUMNS,
    )


def event_rows(parsed: DataFrame) -> DataFrame:
    """Melee events from each tournament payload's events array."""
    events = parsed.select(
        F.col("p.id").alias("tournament_id"), F.explode(F.col("p.events")).alias("e"), *ORDER_COLUMNS
    )
    return events.where(
        F.col("e.id").isNotNull() & (F.col("e.videogame.id") == F.lit(MELEE_VIDEOGAME_ID))
    ).select(
        F.col("e.id").alias("event_id"),
        "tournament_id",
        F.col("e.name").alias("name"),
        F.col("e.slug").alias("slug"),
        F.col("e.numEntrants").alias("num_entrants"),
        *ORDER_COLUMNS,
    )


def parse_sets(bronze: DataFrame) -> DataFrame:
    """One row per bronze set row with typed columns and a drop_reason (null = keep)."""
    df = bronze.select(
        *_source(bronze, SET_PAYLOAD),
        F.get_json_object(F.col("query_variables"), "$.eventId").try_cast("long").alias("event_id"),
    )
    slots = F.col("p.slots")
    slot1, slot2 = _nth(slots, 1), _nth(slots, 2)
    entrant1, entrant2 = slot1["entrant"], slot2["entrant"]
    player1 = _nth(entrant1["participants"], 1)["player"]
    player2 = _nth(entrant2["participants"], 1)["player"]
    set_id = F.col("p.id").try_cast("long")
    winner = F.col("p.winnerId")
    drop_reason = (
        F.when(F.coalesce(F.size(slots), F.lit(0)) < 2, "too_few_slots")
        .when(player1["id"].isNull() | player2["id"].isNull(), "missing_player")
        .when(set_id.isNull() | ~F.coalesce(F.col("p.state") == 3, F.lit(False)), "not_completed")
        .when(
            ~F.coalesce((winner == entrant1["id"]) | (winner == entrant2["id"]), F.lit(False)),
            "unresolved_winner",
        )
    )
    return df.select(
        *ORDER_COLUMNS,
        set_id.alias("set_id"),
        "event_id",
        F.col("p.round").alias("round"),
        F.col("p.fullRoundText").alias("full_round_text"),
        F.col("p.phaseGroup.phase.name").alias("phase_name"),
        F.timestamp_seconds(F.col("p.completedAt")).alias("completed_at"),
        F.col("p.totalGames").alias("total_games"),
        F.col("p.displayScore").alias("display_score"),
        entrant1["id"].alias("entrant1_id"),
        entrant2["id"].alias("entrant2_id"),
        player1["id"].alias("player1_id"),
        player2["id"].alias("player2_id"),
        F.when(winner == entrant1["id"], player1["id"])
        .when(winner == entrant2["id"], player2["id"])
        .alias("winner_player_id"),
        slot1["standing"]["stats"]["score"]["value"].alias("player1_score"),
        slot2["standing"]["stats"]["score"]["value"].alias("player2_score"),
        F.coalesce(slot1["seed"]["seedNum"], entrant1["initialSeedNum"]).alias("player1_seed"),
        F.coalesce(slot2["seed"]["seedNum"], entrant2["initialSeedNum"]).alias("player2_seed"),
        player1["gamerTag"].alias("player1_tag"),
        player2["gamerTag"].alias("player2_tag"),
        F.col("p.games").alias("games"),
        drop_reason.alias("drop_reason"),
    )


def _kept(parsed: DataFrame) -> DataFrame:
    return parsed.where(F.col("drop_reason").isNull())


def set_rows(parsed: DataFrame) -> DataFrame:
    return _kept(parsed).select(*business_columns("sets"), *ORDER_COLUMNS)


def player_rows(parsed: DataFrame) -> DataFrame:
    kept = _kept(parsed)
    return kept.select(
        F.col("player1_id").alias("player_id"), F.col("player1_tag").alias("gamer_tag"), *ORDER_COLUMNS
    ).unionByName(
        kept.select(
            F.col("player2_id").alias("player_id"),
            F.col("player2_tag").alias("gamer_tag"),
            *ORDER_COLUMNS,
        )
    )


def _character(entrant_column: str) -> Column:
    matches = F.filter(F.col("g.selections"), lambda s: s["entrant"]["id"] == F.col(entrant_column))
    return _nth(matches, 1)["character"]


def game_rows(parsed: DataFrame) -> DataFrame:
    games = _kept(parsed).select(
        "set_id",
        "completed_at",
        "entrant1_id",
        "entrant2_id",
        *ORDER_COLUMNS,
        F.explode(F.col("games")).alias("g"),
    )
    character1, character2 = _character("entrant1_id"), _character("entrant2_id")
    return games.where(F.col("g.id").isNotNull()).select(
        F.col("g.id").alias("game_id"),
        "set_id",
        F.col("g.orderNum").alias("game_number"),
        F.col("g.winnerId").alias("winner_entrant_id"),
        F.col("g.stage.id").alias("stage_id"),
        F.col("g.stage.name").alias("stage_name"),
        character1["id"].alias("player1_character_id"),
        character1["name"].alias("player1_character_name"),
        character2["id"].alias("player2_character_id"),
        character2["name"].alias("player2_character_name"),
        "completed_at",
        *ORDER_COLUMNS,
    )


def drop_counts(parsed: DataFrame) -> tuple[int, dict[str, int]]:
    """(bronze set rows read, dropped rows per reason), in one pass."""
    counts = {row.drop_reason: row["count"] for row in parsed.groupBy("drop_reason").count().collect()}
    total = sum(counts.values())
    return total, {reason: int(counts.get(reason, 0)) for reason in DROP_REASONS}


def newest_per_key(df: DataFrame, key: str) -> DataFrame:
    """Keep the newest version of each key: latest ingest, then highest Kafka position."""
    order = [F.col(column).desc() for column in ORDER_COLUMNS]
    ranked = df.withColumn("_rank", F.row_number().over(Window.partitionBy(key).orderBy(*order)))
    return ranked.where(F.col("_rank") == 1).drop("_rank")


def finalize(df: DataFrame, entity: str, snapshot_id: int) -> DataFrame:
    """Silver column order, with the bronze snapshot the batch was read up to."""
    return df.select(
        *business_columns(entity),
        "source_ingested_at",
        F.lit(snapshot_id).cast("bigint").alias("source_snapshot_id"),
        F.current_timestamp().alias("silver_updated_at"),
    )


def merge_sql(entity: str, view: str) -> str:
    """Upsert one staged batch. Only an equal-or-newer bronze version overwrites a row."""
    key = KEYS[entity]
    return f"""
        MERGE INTO {table_name(entity)} AS t
        USING {view} AS s
        ON t.{key} = s.{key}
        WHEN MATCHED AND s.source_ingested_at >= t.source_ingested_at THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_silver_ddl.py -v`
Expected: 6 passed.

Run: `$MAKE test-spark`
Expected: the 9 new `test_silver_transform.py` tests pass, along with the existing bronze Spark tests. In the summary, `test_silver_transform.py` shows 9 passed.

Fallback: if `try_element_at`, `try_cast` or `F.filter` with an outer column reference is rejected by Spark 4.0.4, use these instead and record the change:
- `F.element_at` inside `F.when(F.size(arr) >= n, ...)`;
- `F.col(...).cast("long")` behind a `F.when(F.col(...).rlike("^[0-9]+$"), ...)`;
- a join on `explode(selections)`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/spark_jobs/silver.py tests/pipeline/conftest.py tests/pipeline/test_silver_ddl.py tests/pipeline/test_silver_transform.py
git commit -m "Add silver transforms and DDL with drop reasons and newest-version dedupe"
```

---

### Task 3: Watermarks, incremental reads and the silver MERGE run

**Files:**
- Create: `pipeline/spark_jobs/watermarks.py`, `pipeline/spark_jobs/silver_job.py` (with `run_silver` only; the entrypoint comes in Task 4)
- Test: `tests/pipeline/test_watermarks.py`, `tests/pipeline/test_silver_job.py`

**Interfaces:**
- Consumes: everything listed under "Produces" for `pipeline.spark_jobs.silver` in Task 2, plus `BRONZE_TABLES` and `bronze_table_ddl` from `pipeline.spark_jobs.bronze`.
- Produces:
  - `pipeline.spark_jobs.watermarks`:
    - `Snapshot(snapshot_id: int, operation: str)`;
    - `ReadPlan(kind: str, end: int | None = None, start: int | None = None)`, where `kind` is one of `"skip"`, `"full"` or `"incremental"`;
    - `INCREMENTAL_SAFE = frozenset({"append", "replace"})`;
    - `plan_read(watermark: int | None, lineage: list[Snapshot]) -> ReadPlan`.
  - `pipeline.spark_jobs.silver_job`:
    - `BRONZE_SOURCES = {"tournaments": "bronze.startgg_tournaments", "sets": "bronze.startgg_sets"}`;
    - `run_silver(spark) -> dict`, which returns the summary `{"tables": {entity: {"inserted": int, "updated": int}}, "dropped": {reason: int}, "bronze_rows_read": int, "bronze_tournament_rows_read": int, "orphan_sets": int, "winner_not_player": int}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_watermarks.py`:

```python
from pipeline.spark_jobs.watermarks import ReadPlan, Snapshot, plan_read

LINEAGE = [Snapshot(1, "append"), Snapshot(2, "append"), Snapshot(3, "append")]


def test_empty_table_is_skipped():
    assert plan_read(None, []) == ReadPlan("skip")


def test_first_run_reads_everything():
    assert plan_read(None, LINEAGE) == ReadPlan("full", end=3)


def test_unchanged_table_is_skipped():
    assert plan_read(3, LINEAGE) == ReadPlan("skip", end=3)


def test_advanced_table_is_read_incrementally():
    assert plan_read(1, LINEAGE) == ReadPlan("incremental", end=3, start=1)


def test_expired_watermark_falls_back_to_full_scan():
    assert plan_read(99, LINEAGE) == ReadPlan("full", end=3)


def test_overwrite_after_the_watermark_forces_a_full_scan():
    lineage = [Snapshot(1, "append"), Snapshot(2, "overwrite"), Snapshot(3, "append")]
    assert plan_read(1, lineage) == ReadPlan("full", end=3)
```

Create `tests/pipeline/test_silver_job.py`:

```python
import json
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze import BRONZE_TABLES, bronze_table_ddl  # noqa: E402
from pipeline.spark_jobs.silver import MERGE_ORDER, WATERMARKS_TABLE, table_name  # noqa: E402
from pipeline.spark_jobs.silver_job import run_silver  # noqa: E402

BRONZE_SETS, BRONZE_TOURNAMENTS = BRONZE_TABLES["set"], BRONZE_TABLES["tournament"]
COLUMNS = (
    "kafka_topic", "kafka_partition", "kafka_offset", "kafka_timestamp", "kafka_key", "entity",
    "entity_id", "ingested_at", "dagster_run_id", "partition_week", "query_name",
    "query_variables", "payload",
)  # fmt: skip
OFFSETS = iter(range(1, 10**9))


@pytest.fixture
def lakehouse(spark):
    """Fresh bronze and silver tables for each test."""
    for entity in MERGE_ORDER:
        spark.sql(f"DROP TABLE IF EXISTS {table_name(entity)}")
    spark.sql(f"DROP TABLE IF EXISTS {WATERMARKS_TABLE}")
    for table in (BRONZE_SETS, BRONZE_TOURNAMENTS):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS bronze")
    return spark


def create_bronze(spark):
    for table in (BRONZE_SETS, BRONZE_TOURNAMENTS):
        spark.sql(bronze_table_ddl(table))


def append(spark, table, entity, payload, query_variables, ingested_at):
    """One bronze append = one Iceberg snapshot."""
    row = (
        "topic", 0, next(OFFSETS), ingested_at, str(payload["id"]), entity, str(payload["id"]),
        ingested_at, "run", "2025-01-06", "Q", json.dumps(query_variables), json.dumps(payload),
    )  # fmt: skip
    schema = spark.table(table).schema
    spark.createDataFrame([row], schema).select(*COLUMNS).writeTo(table).append()


def tournament(tid=7, event_ids=(70,)):
    events = [{"id": e, "name": "Singles", "slug": "s", "numEntrants": 64, "videogame": {"id": 1}} for e in event_ids]
    return {"id": tid, "name": "T", "slug": "t", "numAttendees": 100, "startAt": 1736150400,
            "endAt": 1736200000, "city": "X", "addrState": "CA", "countryCode": "US",
            "isOnline": False, "events": events}  # fmt: skip


def a_set(set_id, score=2.0):
    def slot(eid, pid, tag, value):
        return {"seed": {"seedNum": 1}, "standing": {"placement": 1, "stats": {"score": {"value": value}}},
                "entrant": {"id": eid, "name": tag, "initialSeedNum": 1,
                            "participants": [{"player": {"id": pid, "gamerTag": tag}}]}}  # fmt: skip

    return {"id": set_id, "displayScore": "", "winnerId": 11, "round": 1, "fullRoundText": "R1",
            "completedAt": 1736200000, "totalGames": 3, "state": 3, "games": [],
            "slots": [slot(11, 1, "A", score), slot(12, 2, "B", 0.0)],
            "phaseGroup": {"phase": {"name": "Pools"}}}  # fmt: skip


T0 = datetime(2025, 1, 6, tzinfo=UTC)
T1 = datetime(2025, 1, 7, tzinfo=UTC)


def count(spark, entity):
    return spark.table(table_name(entity)).count()


def test_missing_bronze_tables_is_a_no_op(lakehouse):
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 0
    assert summary["tables"]["sets"] == {"inserted": 0, "updated": 0}
    assert summary["orphan_sets"] == 0
    assert all(count(lakehouse, e) == 0 for e in MERGE_ORDER)


def test_first_run_reads_everything(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T0)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 2
    assert summary["tables"]["sets"] == {"inserted": 2, "updated": 0}
    assert summary["tables"]["players"] == {"inserted": 2, "updated": 0}
    assert (count(lakehouse, "tournaments"), count(lakehouse, "events")) == (1, 1)
    watermarks = {r.bronze_table: r.snapshot_id for r in lakehouse.table(WATERMARKS_TABLE).collect()}
    assert set(watermarks) == {BRONZE_SETS, BRONZE_TOURNAMENTS}


def test_second_run_reads_only_new_snapshots(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 1
    assert summary["bronze_tournament_rows_read"] == 0
    assert summary["tables"]["sets"] == {"inserted": 1, "updated": 0}
    assert count(lakehouse, "sets") == 2


def test_unchanged_bronze_is_a_no_op(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 0
    assert all(v == {"inserted": 0, "updated": 0} for v in summary["tables"].values())


def test_older_version_never_overwrites_newer(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100, score=3.0), {"eventId": 70}, T1)
    run_silver(lakehouse)
    append(lakehouse, BRONZE_SETS, "set", a_set(100, score=1.0), {"eventId": 70}, T0)  # replayed old copy
    summary = run_silver(lakehouse)
    assert summary["tables"]["sets"] == {"inserted": 0, "updated": 0}
    assert lakehouse.table(table_name("sets")).first().player1_score == 3.0


def test_expired_watermark_falls_back_to_full_scan(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T0)
    run_silver(lakehouse)
    lakehouse.sql(f"UPDATE {WATERMARKS_TABLE} SET snapshot_id = -1 WHERE bronze_table = '{BRONZE_SETS}'")
    append(lakehouse, BRONZE_SETS, "set", a_set(102), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 3  # full scan
    assert summary["tables"]["sets"] == {"inserted": 1, "updated": 2}  # re-reads are same-value updates
    assert count(lakehouse, "sets") == 3


def test_compaction_snapshot_does_not_force_a_full_scan(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T1)
    lakehouse.sql(
        f"CALL local.system.rewrite_data_files(table => '{BRONZE_SETS}', "
        f"options => map('min-input-files', '1'))"
    ).collect()
    append(lakehouse, BRONZE_SETS, "set", a_set(102), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 2  # the two appends after the watermark
    assert count(lakehouse, "sets") == 3


def test_orphan_sets_are_counted(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(event_ids=(70,)), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 999}, T0)
    summary = run_silver(lakehouse)
    assert summary["orphan_sets"] == 1
    assert summary["winner_not_player"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_watermarks.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.spark_jobs.watermarks'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/spark_jobs/watermarks.py`:

```python
"""Decide how much of a bronze table the silver job must read. No Spark imports."""

from dataclasses import dataclass

# Compaction (replace) snapshots carry no new rows and are skipped by Iceberg's
# incremental append scan. Anything else after the watermark forces a full read.
INCREMENTAL_SAFE = frozenset({"append", "replace"})


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: int
    operation: str


@dataclass(frozen=True)
class ReadPlan:
    kind: str  # "skip" | "full" | "incremental"
    end: int | None = None
    start: int | None = None


def plan_read(watermark: int | None, lineage: list[Snapshot]) -> ReadPlan:
    """lineage: the table's live current-ancestor snapshots, oldest first."""
    if not lineage:
        return ReadPlan("skip")
    current = lineage[-1].snapshot_id
    if watermark == current:
        return ReadPlan("skip", end=current)
    ids = [snapshot.snapshot_id for snapshot in lineage]
    if watermark is None or watermark not in ids:
        return ReadPlan("full", end=current)
    later = lineage[ids.index(watermark) + 1 :]
    if any(snapshot.operation not in INCREMENTAL_SAFE for snapshot in later):
        return ReadPlan("full", end=current)
    return ReadPlan("incremental", end=current, start=watermark)
```

Create `pipeline/spark_jobs/silver_job.py`:

```python
"""Upsert new bronze rows into the silver tables.

Run with spark-submit. The last line of stdout is a JSON summary.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline.spark_jobs.bronze import BRONZE_TABLES
from pipeline.spark_jobs.silver import (
    DROP_REASONS,
    KEYS,
    MERGE_ORDER,
    SILVER_NAMESPACE,
    WATERMARKS_DDL,
    WATERMARKS_TABLE,
    drop_counts,
    event_rows,
    finalize,
    game_rows,
    merge_sql,
    newest_per_key,
    parse_sets,
    parse_tournaments,
    player_rows,
    set_rows,
    silver_table_ddl,
    table_name,
    tournament_rows,
)
from pipeline.spark_jobs.watermarks import ReadPlan, Snapshot, plan_read

BRONZE_SOURCES = {"tournaments": BRONZE_TABLES["tournament"], "sets": BRONZE_TABLES["set"]}


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {SILVER_NAMESPACE}")
    for entity in MERGE_ORDER:
        spark.sql(silver_table_ddl(entity))
    spark.sql(WATERMARKS_DDL)


def read_watermarks(spark: SparkSession) -> dict[str, int]:
    return {row.bronze_table: row.snapshot_id for row in spark.table(WATERMARKS_TABLE).collect()}


def snapshot_lineage(spark: SparkSession, table: str) -> list[Snapshot]:
    """Live snapshots on the current branch, oldest first ([] if the table doesn't exist)."""
    if not spark.catalog.tableExists(table):
        return []
    rows = spark.sql(
        f"""
        SELECT h.snapshot_id, s.operation
        FROM {table}.history AS h
        JOIN {table}.snapshots AS s ON h.snapshot_id = s.snapshot_id
        WHERE h.is_current_ancestor
        ORDER BY h.made_current_at
        """
    ).collect()
    return [Snapshot(row.snapshot_id, row.operation) for row in rows]


def read_bronze(spark: SparkSession, table: str, plan: ReadPlan) -> DataFrame:
    reader = spark.read.format("iceberg")
    if plan.kind == "incremental":
        reader = reader.option("start-snapshot-id", str(plan.start)).option(
            "end-snapshot-id", str(plan.end)
        )
    else:
        reader = reader.option("snapshot-id", str(plan.end))
    return reader.load(table)


def count_changes(spark: SparkSession, entity: str, staged: DataFrame) -> dict[str, int]:
    """How many staged rows will be inserted, and how many will update an existing row."""
    key = KEYS[entity]
    existing = spark.table(table_name(entity)).select(
        F.col(key), F.col("source_ingested_at").alias("_target_ts")
    )
    row = (
        staged.join(existing, key, "left")
        .agg(
            F.sum(F.when(F.col("_target_ts").isNull(), 1).otherwise(0)).alias("inserted"),
            F.sum(
                F.when(F.col("source_ingested_at") >= F.col("_target_ts"), 1).otherwise(0)
            ).alias("updated"),
        )
        .first()
    )
    return {"inserted": int(row.inserted or 0), "updated": int(row.updated or 0)}


def merge_into(spark: SparkSession, entity: str, staged: DataFrame) -> dict[str, int]:
    staged = staged.persist()
    try:
        changes = count_changes(spark, entity, staged)
        view = f"staged_{entity}"
        staged.createOrReplaceTempView(view)
        spark.sql(merge_sql(entity, view))
        return changes
    finally:
        staged.unpersist()


def write_watermarks(spark: SparkSession, snapshots: dict[str, int]) -> None:
    new = spark.createDataFrame(
        list(snapshots.items()), "bronze_table STRING, snapshot_id BIGINT"
    ).withColumn("updated_at", F.current_timestamp())
    new.createOrReplaceTempView("new_watermarks")
    spark.sql(
        f"""
        MERGE INTO {WATERMARKS_TABLE} AS t USING new_watermarks AS s
        ON t.bronze_table = s.bronze_table
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def integrity_counts(spark: SparkSession) -> dict[str, int]:
    sets, events = table_name("sets"), table_name("events")
    orphans = spark.sql(
        f"SELECT count(*) AS n FROM {sets} AS s LEFT ANTI JOIN {events} AS e "
        f"ON s.event_id = e.event_id"
    ).first()
    bad_winners = spark.sql(
        f"SELECT count(*) AS n FROM {sets} WHERE winner_player_id IS NOT NULL "
        f"AND winner_player_id NOT IN (player1_id, player2_id)"
    ).first()
    return {"orphan_sets": int(orphans.n), "winner_not_player": int(bad_winners.n)}


def run_silver(spark: SparkSession) -> dict:
    """Read new bronze snapshots, MERGE them into silver, then advance the watermarks."""
    ensure_tables(spark)
    watermarks = read_watermarks(spark)
    plans = {
        source: plan_read(watermarks.get(table), snapshot_lineage(spark, table))
        for source, table in BRONZE_SOURCES.items()
    }
    summary = {
        "tables": {entity: {"inserted": 0, "updated": 0} for entity in MERGE_ORDER},
        "dropped": {reason: 0 for reason in DROP_REASONS},
        "bronze_rows_read": 0,
        "bronze_tournament_rows_read": 0,
    }
    staged: dict[str, DataFrame] = {}

    plan = plans["tournaments"]
    if plan.kind != "skip":
        bronze = read_bronze(spark, BRONZE_SOURCES["tournaments"], plan)
        summary["bronze_tournament_rows_read"] = bronze.count()
        parsed = parse_tournaments(bronze)
        staged["tournaments"] = finalize(
            newest_per_key(tournament_rows(parsed), "tournament_id"), "tournaments", plan.end
        )
        staged["events"] = finalize(newest_per_key(event_rows(parsed), "event_id"), "events", plan.end)

    plan = plans["sets"]
    if plan.kind != "skip":
        parsed = parse_sets(read_bronze(spark, BRONZE_SOURCES["sets"], plan))
        summary["bronze_rows_read"], summary["dropped"] = drop_counts(parsed)
        staged["players"] = finalize(newest_per_key(player_rows(parsed), "player_id"), "players", plan.end)
        staged["sets"] = finalize(newest_per_key(set_rows(parsed), "set_id"), "sets", plan.end)
        staged["games"] = finalize(newest_per_key(game_rows(parsed), "game_id"), "games", plan.end)

    for entity in MERGE_ORDER:
        if entity in staged:
            summary["tables"][entity] = merge_into(spark, entity, staged[entity])

    advanced = {BRONZE_SOURCES[s]: p.end for s, p in plans.items() if p.kind != "skip"}
    if advanced:
        write_watermarks(spark, advanced)
    summary.update(integrity_counts(spark))
    return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_watermarks.py -v`
Expected: 6 passed.

Run: `$MAKE test-spark`
Expected: `test_silver_job.py` shows 8 passed, and all other Spark tests still pass.

Fallback if `spark.read.format("iceberg")...load("bronze.startgg_sets")` can't resolve the two-part name against the default catalog: build the reader as `spark.read.options(**opts).table(table)` instead. The options are the same. Record the change.

Fallback for `test_compaction_snapshot_does_not_force_a_full_scan`: if Iceberg rejects an incremental range that contains a `replace` snapshot, do the following and record it:
1. Remove `"replace"` from `INCREMENTAL_SAFE`. The full scan then takes over after compaction, and the MERGE keeps it correct.
2. Change that test's expectation to `bronze_rows_read == 3`.
3. Add a sixth case to `test_watermarks.py` asserting that a `replace` snapshot forces a full scan.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/spark_jobs/watermarks.py pipeline/spark_jobs/silver_job.py tests/pipeline/test_watermarks.py tests/pipeline/test_silver_job.py
git commit -m "Merge new bronze snapshots into silver with watermarks and an equal-or-newer rule"
```

---

### Task 4: Silver job entrypoint, shared lock and a live run on the dev stack

**Files:**
- Create: `pipeline/spark_jobs/locks.py`
- Modify: `pipeline/spark_jobs/bronze_job.py`, `pipeline/spark_jobs/silver_job.py`, `Makefile`
- Test: `tests/pipeline/test_silver_lock.py`

**Interfaces:**
- Consumes: `run_silver` (Task 3).
- Produces:
  - `pipeline.spark_jobs.locks.exclusive_lock(directory: str, job: str) -> IO`. It exits with `another {job} job is running (lock held on {directory}/.lock)`.
  - The entrypoint `spark-submit pipeline/spark_jobs/silver_job.py --lock-dir /checkpoints/silver`. Its last stdout line is the `run_silver` summary as JSON.
  - Make target `silver-once`.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_silver_lock.py`:

```python
import pytest

from pipeline.spark_jobs.locks import exclusive_lock


def test_silver_lock_refuses_a_second_job(tmp_path):
    pytest.importorskip("fcntl")  # the jobs only run in Linux containers
    held = exclusive_lock(str(tmp_path / "silver"), "silver")
    with pytest.raises(SystemExit, match="another silver job is running") as info:
        exclusive_lock(str(tmp_path / "silver"), "silver")
    assert info.value.code != 0
    held.close()
    exclusive_lock(str(tmp_path / "silver"), "silver").close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/pipeline/test_silver_lock.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.spark_jobs.locks'`. On Windows, once the module exists, the test is skipped because `fcntl` is unavailable; it runs in CI and in the container.

- [ ] **Step 3: Write the implementation**

Create `pipeline/spark_jobs/locks.py`:

```python
"""One-at-a-time guard for Spark jobs that share state on the checkpoint volume."""

import os
import sys
from typing import IO


def exclusive_lock(directory: str, job: str) -> IO:
    """Hold an exclusive lock on directory/.lock, or exit if another job holds it.

    Keep the returned file open for as long as the job runs.
    """
    import fcntl  # Linux only; imported here so modules still import on Windows hosts

    os.makedirs(directory, exist_ok=True)
    lock_file = open(os.path.join(directory, ".lock"), "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        sys.exit(f"another {job} job is running (lock held on {directory}/.lock)")
    return lock_file
```

In `pipeline/spark_jobs/bronze_job.py`:
- Remove `import os`; only `lock_checkpoints` used it. Keep `import sys` (`await_all` uses it) and `from typing import IO` (the return annotation).
- Replace the body of `lock_checkpoints` so the function reads:

```python
def lock_checkpoints(checkpoint_root: str) -> IO:
    """Hold an exclusive lock on the checkpoint root, or exit if another job holds it.

    `make bronze-once` and Dagster bronze runs share the checkpoint volume, and two
    streaming queries on one checkpoint corrupt it. Keep the returned file open.
    """
    return exclusive_lock(checkpoint_root, "bronze")
```

Add `from pipeline.spark_jobs.locks import exclusive_lock` and keep `from typing import IO` for the annotation.

Append to `pipeline/spark_jobs/silver_job.py`:

```python
def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from pipeline.spark_jobs.locks import exclusive_lock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-dir", required=True)
    args = parser.parse_args(argv)
    lock = exclusive_lock(args.lock_dir, "silver")

    spark = SparkSession.builder.appName("silver-startgg").getOrCreate()
    summary = run_silver(spark)
    spark.stop()
    lock.close()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
```

In the `Makefile`:
- add `silver-once` to `.PHONY`;
- add this target after `bronze-once`:

```makefile
silver-once:
	$(COMPOSE) run --rm pipeline-tools spark-submit pipeline/spark_jobs/silver_job.py \
		--lock-dir /checkpoints/silver
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_silver_lock.py tests/pipeline/test_bronze_job.py -v`
Expected: on Windows, `test_silver_lock.py` is 1 skipped, and the bronze job tests pass (their lock test is also skipped on Windows).

Run: `$MAKE test-spark`
Expected: all Spark tests pass.

Run, inside the container (where `fcntl` exists):

```bash
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env run --rm --no-deps pipeline-tools pytest -m "" tests/pipeline/test_silver_lock.py tests/pipeline/test_bronze_job.py -q
```

Expected: everything passes, 0 skipped.

- [ ] **Step 5: Run silver against the real 2025 bronze data (dev stack)**

```bash
$MAKE silver-once 2>/dev/null | tail -1
$MAKE silver-once 2>/dev/null | tail -1
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T trino trino --execute "
SELECT 'tournaments', count(*) FROM lakekeeper.silver.tournaments UNION ALL
SELECT 'events', count(*) FROM lakekeeper.silver.events UNION ALL
SELECT 'players', count(*) FROM lakekeeper.silver.players UNION ALL
SELECT 'sets', count(*) FROM lakekeeper.silver.sets UNION ALL
SELECT 'games', count(*) FROM lakekeeper.silver.games"
```

Expected:
- **First run:** the JSON summary shows
  - `bronze_rows_read` equal to the bronze set row count (152424 on the current stack);
  - `tournaments.inserted` 1462;
  - drop counts below 10% of rows read;
  - `orphan_sets` 0 and `winner_not_player` 0.
- **Second run:** `bronze_rows_read` 0 and all-zero table counts.
- **Trino:** returns the five counts. Record them in the task report.

The first run does a full scan of the whole bronze table and may take several minutes.

- [ ] **Step 6: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/spark_jobs/locks.py pipeline/spark_jobs/bronze_job.py pipeline/spark_jobs/silver_job.py Makefile tests/pipeline/test_silver_lock.py
git commit -m "Add the silver job entrypoint with a shared checkpoint-volume lock"
```

---
### Task 5: Move the Elo and feature engine to gold (DataFrame in, no SQLite)

**Files:**
- Create: `pipeline/gold/__init__.py` (empty), `pipeline/gold/features.py`, `pipeline/gold/schema.py`
- Create: `tests/pipeline/test_elo.py`, `tests/pipeline/test_gold_features.py`, `tests/pipeline/test_gold_schema.py`
- Delete: `tests/test_elo.py`, `tests/test_features.py`, `tests/test_validation.py`. They are moved here. `src/` itself is removed in Task 14.

**Interfaces:**
- Consumes: `CORE_FEATURES` from `pipeline.model_features`.
- Produces, in `pipeline.gold.features`:
  - `DEFAULT_ELO = 1500`, `K_FACTOR = 32`, `ELO_SCALE = 400`, `RECENT_WINDOW = 30`.
  - `EloEngine`, unchanged from `src/features.py`.
  - `SET_INPUT_COLUMNS`: the columns a silver-derived sets frame must have, namely `set_id`, `event_id`, `completed_at`, `player1_id`, `player2_id`, `winner_player_id`, `player1_score`, `player2_score`, `player1_seed`, `player2_seed`, `tournament_id`, `tournament_name`, `num_attendees`.
  - `FEATURE_COLUMNS: tuple[str, ...]`, the old `features.parquet` columns in the same order.
  - `FeatureState`, a dataclass with `elo`, `recent_wins`, `h2h`, `wins`, `losses`, `last_played` and `pair_last_played`.
  - `eligible_sets(sets) -> pd.DataFrame`.
  - `build_features(sets) -> tuple[pd.DataFrame, FeatureState]`.
- Also produces `pipeline.gold.schema.features_schema`, a pandera `DataFrameSchema` with ordered columns equal to `FEATURE_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_elo.py` with exactly the content of `tests/test_elo.py`, except for the import line:

```python
from pipeline.gold.features import DEFAULT_ELO, EloEngine
```

The 8 test functions stay unchanged. Then delete `tests/test_elo.py`.

Create `tests/pipeline/test_gold_features.py`:

```python
import random

import numpy as np
import pandas as pd

from pipeline.gold.features import FEATURE_COLUMNS, build_features
from pipeline.model_features import CORE_FEATURES

BASE = {
    "event_id": 10,
    "tournament_id": 1,
    "tournament_name": "T1",
    "num_attendees": 200,
    "player1_score": 2.0,
    "player2_score": 1.0,
    "player1_seed": 1,
    "player2_seed": 4,
}


def sets_frame(rows):
    frame = pd.DataFrame([{**BASE, **row} for row in rows])
    frame["completed_at"] = pd.to_datetime(frame["completed_at"], unit="s", utc=True)
    return frame


THREE_SETS = [
    # Alice (100) beats Bob (200) at T1
    {"set_id": 1, "completed_at": 1000010, "player1_id": 100, "player2_id": 200, "winner_player_id": 100},
    # Bob beats Charlie (300) at T2
    {"set_id": 2, "completed_at": 2000010, "player1_id": 200, "player2_id": 300, "winner_player_id": 200},
    # Alice beats Bob again at T2
    {"set_id": 3, "completed_at": 2000020, "player1_id": 100, "player2_id": 200, "winner_player_id": 100},
]


def test_features_no_leakage():
    df, _ = build_features(sets_frame(THREE_SETS))
    first, third = df.iloc[0], df.iloc[2]
    assert first["p1_elo"] == 1500.0
    assert first["p2_elo"] == 1500.0
    assert first["p1_sets_played"] == 0
    assert first["h2h_total"] == 0
    assert third["h2h_total"] == 1
    assert third["p1_elo"] != 1500.0


def test_features_target_correct():
    df, _ = build_features(sets_frame(THREE_SETS))
    assert df["p1_won"].tolist() == [1, 1, 1]


def test_features_chronological_order():
    df, _ = build_features(sets_frame(list(reversed(THREE_SETS))))
    assert df["completed_at"].is_monotonic_increasing
    assert df["set_id"].tolist() == [1, 2, 3]
    assert tuple(df.columns) == FEATURE_COLUMNS


def test_equal_completed_at_breaks_ties_by_set_id():
    rows = [
        {"set_id": 7, "completed_at": 5000, "player1_id": 1, "player2_id": 2, "winner_player_id": 1},
        {"set_id": 5, "completed_at": 5000, "player1_id": 1, "player2_id": 3, "winner_player_id": 3},
    ]
    df, _ = build_features(sets_frame(rows))
    assert df["set_id"].tolist() == [5, 7]
    assert df.iloc[1]["p1_sets_played"] == 1  # set 5 happened first


def test_each_row_only_counts_earlier_sets():
    rng = random.Random(3)
    rows = []
    for i in range(40):
        p1, p2 = rng.sample([1, 2, 3, 4], 2)
        rows.append({"set_id": i, "completed_at": 1000 + i, "player1_id": p1, "player2_id": p2,
                     "winner_player_id": rng.choice([p1, p2])})  # fmt: skip
    df, _ = build_features(sets_frame(rows))
    played: dict[int, int] = {}
    for row in df.itertuples():
        assert row.p1_sets_played == played.get(row.player1_id, 0)
        assert row.p2_sets_played == played.get(row.player2_id, 0)
        played[row.player1_id] = played.get(row.player1_id, 0) + 1
        played[row.player2_id] = played.get(row.player2_id, 0) + 1


def test_null_seeds_and_attendees_never_produce_nan():
    row = {**THREE_SETS[0], "player1_seed": None, "num_attendees": None}
    df, _ = build_features(sets_frame([row]))
    assert df.iloc[0]["seed_diff"] == -4  # missing seed counts as 0
    assert df.iloc[0]["num_attendees"] == 0
    assert not df[CORE_FEATURES].isna().any().any()


def test_ineligible_sets_are_excluded():
    rows = [
        THREE_SETS[0],
        {**THREE_SETS[1], "winner_player_id": None},
        {**THREE_SETS[2], "completed_at": np.nan},
    ]
    df, _ = build_features(sets_frame(rows))
    assert df["set_id"].tolist() == [1]


def test_state_tracks_records_for_gold_tables():
    _, state = build_features(sets_frame(THREE_SETS))
    assert (state.wins[100], state.losses[100]) == (2, 0)
    assert (state.wins[200], state.losses[200]) == (1, 2)
    assert state.h2h[(100, 200)] == [2, 0]
    assert state.pair_last_played[(100, 200)] == pd.Timestamp(2000020, unit="s", tz="UTC")
    assert state.last_played[300] == pd.Timestamp(2000010, unit="s", tz="UTC")
```

Create `tests/pipeline/test_gold_schema.py`:

```python
import pandas as pd
import pandera.pandas as pandera
import pytest

from pipeline.gold.features import FEATURE_COLUMNS
from pipeline.gold.schema import features_schema


def _make_valid_row(**overrides):
    row = {
        "set_id": 1,
        "event_id": 100,
        "tournament_id": 10,
        "tournament_name": "Genesis 9",
        "completed_at": pd.Timestamp("2023-11-14 22:13:20", tz="UTC"),
        "player1_id": 1,
        "player2_id": 2,
        "p1_elo": 1800.0,
        "p2_elo": 1600.0,
        "elo_diff": 200.0,
        "p1_expected": 0.76,
        "p1_sets_played": 50,
        "p2_sets_played": 30,
        "p1_recent_wr": 0.7,
        "p2_recent_wr": 0.5,
        "recent_wr_diff": 0.2,
        "p1_h2h_wins": 3,
        "p2_h2h_wins": 1,
        "p1_h2h_wr": 0.75,
        "h2h_total": 4,
        "p1_seed": 1,
        "p2_seed": 5,
        "seed_diff": -4,
        "num_attendees": 500,
        "p1_score": 3.0,
        "p2_score": 1.0,
        "p1_won": 1,
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    features_schema.validate(pd.DataFrame([_make_valid_row()]))


def test_negative_elo_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(p1_elo=-100.0)]))


def test_invalid_target_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(p1_won=2)]))


def test_same_players_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(player1_id=1, player2_id=1)]))


def test_h2h_inconsistency_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(h2h_total=10)]))


def test_missing_seed_counts_as_zero_in_seed_diff():
    features_schema.validate(pd.DataFrame([_make_valid_row(p1_seed=None, seed_diff=-5)]))


def test_schema_columns_match_feature_columns():
    assert tuple(features_schema.columns) == FEATURE_COLUMNS
```

Delete `tests/test_features.py` and `tests/test_validation.py`. They depend on `src.database` and `src.validation`; their cases are ported above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_elo.py tests/pipeline/test_gold_features.py tests/pipeline/test_gold_schema.py -v`
Expected: collection errors, `ModuleNotFoundError: No module named 'pipeline.gold'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/gold/__init__.py` as an empty file.

Create `pipeline/gold/features.py`:

```python
"""Chronological feature pass over silver sets (moved from src/features.py).

Every feature for a set uses only sets completed before it. Sets are processed
in (completed_at, set_id) order so ties resolve the same way on every run.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

DEFAULT_ELO = 1500
K_FACTOR = 32
# Scale factor: 400 means a 400-point gap = 10x expected performance
ELO_SCALE = 400
RECENT_WINDOW = 30  # last N sets for rolling win rate

SET_INPUT_COLUMNS = (
    "set_id",
    "event_id",
    "completed_at",
    "player1_id",
    "player2_id",
    "winner_player_id",
    "player1_score",
    "player2_score",
    "player1_seed",
    "player2_seed",
    "tournament_id",
    "tournament_name",
    "num_attendees",
)
FEATURE_COLUMNS = (
    "set_id",
    "event_id",
    "tournament_id",
    "tournament_name",
    "completed_at",
    "player1_id",
    "player2_id",
    "p1_elo",
    "p2_elo",
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "p1_recent_wr",
    "p2_recent_wr",
    "recent_wr_diff",
    "p1_h2h_wins",
    "p2_h2h_wins",
    "p1_h2h_wr",
    "h2h_total",
    "p1_seed",
    "p2_seed",
    "seed_diff",
    "num_attendees",
    "p1_score",
    "p2_score",
    "p1_won",
)


class EloEngine:
    """Tracks Elo ratings for all players over time."""

    def __init__(self, default_rating: float = DEFAULT_ELO, k: float = K_FACTOR):
        self.default_rating = default_rating
        self.k = k
        self.ratings: dict[int, float] = {}
        self.games_played: dict[int, int] = defaultdict(int)

    def get_rating(self, player_id: int) -> float:
        return self.ratings.get(player_id, self.default_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that player A beats player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / ELO_SCALE))

    def update(self, winner_id: int, loser_id: int) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_winner_rating, new_loser_rating).

        Uses a dynamic K-factor: higher for players with fewer games
        to allow faster convergence for newcomers.
        """
        r_w = self.get_rating(winner_id)
        r_l = self.get_rating(loser_id)

        expected_w = self.expected_score(r_w, r_l)

        # Dynamic K: higher for newer players (first 30 games)
        k_w = self.k * max(1.0, 2.0 - self.games_played[winner_id] / 30)
        k_l = self.k * max(1.0, 2.0 - self.games_played[loser_id] / 30)

        self.ratings[winner_id] = r_w + k_w * (1.0 - expected_w)
        self.ratings[loser_id] = r_l + k_l * (0.0 - (1.0 - expected_w))

        self.games_played[winner_id] += 1
        self.games_played[loser_id] += 1

        return self.ratings[winner_id], self.ratings[loser_id]


@dataclass
class FeatureState:
    """Everything the pass knows after the last set; gold's player and h2h tables use it."""

    elo: EloEngine = field(default_factory=EloEngine)
    recent_wins: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))
    # (lower id, higher id) -> [lower id's wins, higher id's wins]
    h2h: dict[tuple[int, int], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    wins: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    losses: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_played: dict[int, pd.Timestamp] = field(default_factory=dict)
    pair_last_played: dict[tuple[int, int], pd.Timestamp] = field(default_factory=dict)


def _int_or_none(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _float_or_none(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def eligible_sets(sets: pd.DataFrame) -> pd.DataFrame:
    """Sets with a winner and a completion time, in (completed_at, set_id) order."""
    mask = sets["winner_player_id"].notna() & sets["completed_at"].notna()
    ordered = sets[mask].sort_values(["completed_at", "set_id"], kind="mergesort")
    return ordered.reset_index(drop=True)


def build_features(sets: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Build the feature matrix (FEATURE_COLUMNS) and return the final state."""
    state = FeatureState()
    rows = []
    for row in eligible_sets(sets).itertuples(index=False):
        p1, p2 = int(row.player1_id), int(row.player2_id)
        winner = int(row.winner_player_id)

        # --- FEATURES (computed BEFORE this set) ---
        p1_elo = state.elo.get_rating(p1)
        p2_elo = state.elo.get_rating(p2)
        p1_games = state.elo.games_played[p1]
        p2_games = state.elo.games_played[p2]

        p1_recent = state.recent_wins[p1][-RECENT_WINDOW:]
        p2_recent = state.recent_wins[p2][-RECENT_WINDOW:]
        p1_recent_wr = sum(p1_recent) / len(p1_recent) if p1_recent else 0.5
        p2_recent_wr = sum(p2_recent) / len(p2_recent) if p2_recent else 0.5

        pair = (min(p1, p2), max(p1, p2))
        record = state.h2h[pair]
        if p1 == pair[0]:
            p1_h2h_wins, p2_h2h_wins = record[0], record[1]
        else:
            p1_h2h_wins, p2_h2h_wins = record[1], record[0]
        h2h_total = p1_h2h_wins + p2_h2h_wins

        p1_seed, p2_seed = _int_or_none(row.player1_seed), _int_or_none(row.player2_seed)
        p1_won = int(winner == p1)

        rows.append(
            {
                "set_id": int(row.set_id),
                "event_id": int(row.event_id),
                "tournament_id": int(row.tournament_id),
                "tournament_name": row.tournament_name,
                "completed_at": row.completed_at,
                "player1_id": p1,
                "player2_id": p2,
                "p1_elo": p1_elo,
                "p2_elo": p2_elo,
                "elo_diff": p1_elo - p2_elo,
                "p1_expected": state.elo.expected_score(p1_elo, p2_elo),
                "p1_sets_played": p1_games,
                "p2_sets_played": p2_games,
                "p1_recent_wr": p1_recent_wr,
                "p2_recent_wr": p2_recent_wr,
                "recent_wr_diff": p1_recent_wr - p2_recent_wr,
                "p1_h2h_wins": p1_h2h_wins,
                "p2_h2h_wins": p2_h2h_wins,
                "p1_h2h_wr": p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5,
                "h2h_total": h2h_total,
                "p1_seed": p1_seed,
                "p2_seed": p2_seed,
                # A missing seed counts as 0, as in the original pipeline.
                "seed_diff": (p1_seed or 0) - (p2_seed or 0),
                "num_attendees": _int_or_none(row.num_attendees) or 0,
                "p1_score": _float_or_none(row.player1_score),
                "p2_score": _float_or_none(row.player2_score),
                "p1_won": p1_won,
            }
        )

        # --- UPDATE STATE (after this set) ---
        loser = p2 if winner == p1 else p1
        state.elo.update(winner, loser)
        state.recent_wins[p1].append(p1_won == 1)
        state.recent_wins[p2].append(p1_won == 0)
        state.h2h[pair][0 if winner == pair[0] else 1] += 1
        state.wins[winner] += 1
        state.losses[loser] += 1
        state.last_played[p1] = state.last_played[p2] = row.completed_at
        state.pair_last_played[pair] = row.completed_at

    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS)), state
```

Create `pipeline/gold/schema.py`:

```python
"""Pandera schema for gold.set_features (moved from src/validation.py)."""

import pandera.pandas as pandera

features_schema = pandera.DataFrameSchema(
    columns={
        "set_id": pandera.Column("int64", nullable=False, unique=True, coerce=True),
        "event_id": pandera.Column("int64", nullable=False, coerce=True),
        "tournament_id": pandera.Column("int64", nullable=False, coerce=True),
        "tournament_name": pandera.Column(str, nullable=False, coerce=True),
        "completed_at": pandera.Column("datetime64[us, UTC]", nullable=False, coerce=True),
        "player1_id": pandera.Column("int64", nullable=False, coerce=True),
        "player2_id": pandera.Column("int64", nullable=False, coerce=True),
        "p1_elo": pandera.Column(float, pandera.Check.gt(0), nullable=False),
        "p2_elo": pandera.Column(float, pandera.Check.gt(0), nullable=False),
        "elo_diff": pandera.Column(float, nullable=False),
        "p1_expected": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "p1_sets_played": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p2_sets_played": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_recent_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "p2_recent_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "recent_wr_diff": pandera.Column(float, pandera.Check.in_range(-1, 1), nullable=False),
        "p1_h2h_wins": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p2_h2h_wins": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_h2h_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "h2h_total": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_seed": pandera.Column("Int64", nullable=True, coerce=True),
        "p2_seed": pandera.Column("Int64", nullable=True, coerce=True),
        "seed_diff": pandera.Column("int64", nullable=False, coerce=True),
        "num_attendees": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_score": pandera.Column(float, nullable=True),
        "p2_score": pandera.Column(float, nullable=True),
        "p1_won": pandera.Column("int64", pandera.Check.isin([0, 1]), coerce=True),
    },
    checks=[
        pandera.Check(
            lambda df: (df["player1_id"] != df["player2_id"]).all(),
            error="player1_id and player2_id must differ",
        ),
        pandera.Check(
            lambda df: (df["h2h_total"] == df["p1_h2h_wins"] + df["p2_h2h_wins"]).all(),
            error="h2h_total must equal p1_h2h_wins + p2_h2h_wins",
        ),
        pandera.Check(
            lambda df: (
                df["seed_diff"] == df["p1_seed"].fillna(0) - df["p2_seed"].fillna(0)
            ).all(),
            error="seed_diff must equal p1_seed - p2_seed, with a missing seed as 0",
        ),
    ],
    ordered=True,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_elo.py tests/pipeline/test_gold_features.py tests/pipeline/test_gold_schema.py -v`
Expected: 8, 8 and 7 passed (23 in total), with no warnings.

If pandera rejects a dtype string under pandas 3, use the pandas dtype object instead and record the change: `pd.DatetimeTZDtype("us", "UTC")` or `pd.Int64Dtype()`.

Run: `uv run pytest`
Expected: all pass. `tests/test_database.py` still passes; it is removed in Task 14.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/gold tests/pipeline/test_elo.py tests/pipeline/test_gold_features.py tests/pipeline/test_gold_schema.py
git rm tests/test_elo.py tests/test_features.py tests/test_validation.py
git commit -m "Move the Elo and feature engine to gold, reading a sets DataFrame"
```

---

### Task 6: Gold tables: derive, write with PyIceberg, build from silver

**Files:**
- Create: `pipeline/gold/tables.py`, `pipeline/gold/derive.py`, `pipeline/gold/writer.py`, `pipeline/gold/build.py`
- Create: `tests/pipeline/lakehouse_data.py`
- Test: `tests/pipeline/test_gold_build.py`
- Modify: `Makefile` (`gold-once` target)

**Interfaces:**
- Consumes:
  - `build_features`, `FeatureState`, `FEATURE_COLUMNS` (Task 5);
  - `read_table`, `current_snapshot_id`, `load_lakehouse_catalog` (Task 1);
  - `CORE_FEATURES`;
  - the `sql_catalog` fixture.
- Produces:
  - `pipeline.gold.tables`:
    - `SET_FEATURES = "gold.set_features"`, `PLAYER_RATINGS = "gold.player_ratings"`, `HEAD_TO_HEAD = "gold.head_to_head"`;
    - `SILVER_SNAPSHOT_PROPERTY = "melee.silver_sets_snapshot"`;
    - `SET_FEATURES_SCHEMA`, `PLAYER_RATINGS_SCHEMA`, `HEAD_TO_HEAD_SCHEMA` (pyarrow);
    - `GOLD_TABLES: dict[str, tuple[pa.Schema, str | None]]`, mapping each identifier to (schema, year-partition column).
  - `pipeline.gold.derive`: `player_ratings(state, players) -> pd.DataFrame`, `head_to_head(state) -> pd.DataFrame`.
  - `pipeline.gold.writer`:
    - `to_arrow(frame, schema) -> pa.Table`;
    - `ensure_table(catalog, identifier) -> Table`;
    - `replace_table(catalog, identifier, data, properties) -> int | None`, which returns the new snapshot id;
    - `write_gold(catalog, frames, silver_sets_snapshot) -> dict[str, int | None]`.
  - `pipeline.gold.build`:
    - `GoldBuild`, a dataclass with `frames`, `eligible_silver_sets`, `silver_sets_snapshot` and `snapshots`;
    - `load_silver(catalog)`;
    - `build_gold(catalog) -> GoldBuild`;
    - `gold_checks(build) -> dict[str, tuple[bool, dict]]`, keyed by `features_match_silver`, `no_null_features` and `completed_at_non_decreasing`.
  - `tests/pipeline/lakehouse_data.py`: `write_silver(catalog, *, sets=(), events=(), tournaments=(), players=())` and `silver_set(...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/lakehouse_data.py`:

```python
"""Write the silver columns gold reads into a PyIceberg catalog (SqlCatalog in tests)."""

from datetime import UTC, datetime

import pyarrow as pa

TS = pa.timestamp("us", tz="UTC")
SILVER_SCHEMAS = {
    "silver.sets": pa.schema(
        [
            ("set_id", pa.int64()),
            ("event_id", pa.int64()),
            ("completed_at", TS),
            ("player1_id", pa.int64()),
            ("player2_id", pa.int64()),
            ("winner_player_id", pa.int64()),
            ("player1_score", pa.float64()),
            ("player2_score", pa.float64()),
            ("player1_seed", pa.int64()),
            ("player2_seed", pa.int64()),
        ]
    ),
    "silver.events": pa.schema([("event_id", pa.int64()), ("tournament_id", pa.int64())]),
    "silver.tournaments": pa.schema(
        [("tournament_id", pa.int64()), ("name", pa.string()), ("num_attendees", pa.int64())]
    ),
    "silver.players": pa.schema([("player_id", pa.int64()), ("gamer_tag", pa.string())]),
}


def silver_set(set_id, p1, p2, winner, ts, event_id=10, seeds=(1, 2)):
    return {
        "set_id": set_id,
        "event_id": event_id,
        "completed_at": datetime.fromtimestamp(ts, UTC) if ts is not None else None,
        "player1_id": p1,
        "player2_id": p2,
        "winner_player_id": winner,
        "player1_score": 2.0,
        "player2_score": 1.0,
        "player1_seed": seeds[0],
        "player2_seed": seeds[1],
    }


def write_silver(catalog, *, sets=(), events=(), tournaments=(), players=()):
    catalog.create_namespace_if_not_exists("silver")
    for identifier, rows in (
        ("silver.sets", sets),
        ("silver.events", events),
        ("silver.tournaments", tournaments),
        ("silver.players", players),
    ):
        schema = SILVER_SCHEMAS[identifier]
        table = catalog.create_table_if_not_exists(identifier, schema=schema)
        if rows:
            table.append(pa.Table.from_pylist(list(rows), schema=schema))
```

Create `tests/pipeline/test_gold_build.py`:

```python
import pandas as pd
from pyiceberg.transforms import YearTransform

from pipeline.gold.build import build_gold, gold_checks
from pipeline.gold.tables import (
    HEAD_TO_HEAD,
    PLAYER_RATINGS,
    SET_FEATURES,
    SILVER_SNAPSHOT_PROPERTY,
)
from pipeline.lakehouse import current_snapshot_id
from tests.pipeline.lakehouse_data import silver_set, write_silver

EVENTS = [{"event_id": 10, "tournament_id": 1}]
TOURNAMENTS = [{"tournament_id": 1, "name": "Genesis", "num_attendees": 300}]
PLAYERS = [{"player_id": 100, "gamer_tag": "Alice"}, {"player_id": 200, "gamer_tag": "Bob"}]
SETS = [
    silver_set(1, 100, 200, 100, 1_000_000),
    silver_set(2, 200, 100, 100, 2_000_000),
    silver_set(3, 100, 300, 300, 3_000_000),  # player 300 has no silver.players row
]


def rows(catalog, identifier):
    return catalog.load_table(identifier).scan().to_pandas()


def test_build_gold_writes_three_tables(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build = build_gold(sql_catalog)
    assert len(rows(sql_catalog, SET_FEATURES)) == 3
    assert len(rows(sql_catalog, PLAYER_RATINGS)) == 3
    assert len(rows(sql_catalog, HEAD_TO_HEAD)) == 2
    silver_snapshot = current_snapshot_id(sql_catalog, "silver.sets")
    assert build.silver_sets_snapshot == silver_snapshot
    properties = sql_catalog.load_table(SET_FEATURES).properties
    assert properties[SILVER_SNAPSHOT_PROPERTY] == str(silver_snapshot)
    assert all(passed for passed, _ in gold_checks(build).values())


def test_rebuild_replaces_instead_of_appending(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    build_gold(sql_catalog)
    assert len(rows(sql_catalog, SET_FEATURES)) == 3
    assert len(rows(sql_catalog, HEAD_TO_HEAD)) == 2


def test_player_ratings_and_head_to_head(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    ratings = rows(sql_catalog, PLAYER_RATINGS).set_index("player_id")
    assert ratings.loc[100, "gamer_tag"] == "Alice"
    assert ratings.loc[300, "gamer_tag"] == "300"  # no tag in silver: fall back to the id
    assert (ratings.loc[100, "wins"], ratings.loc[100, "losses"]) == (2, 1)
    assert ratings.loc[100, "sets_played"] == 3
    assert ratings.loc[100, "last_played_at"] == pd.Timestamp(3_000_000, unit="s", tz="UTC")
    h2h = rows(sql_catalog, HEAD_TO_HEAD).set_index(["player_a_id", "player_b_id"])
    assert tuple(h2h.loc[(100, 200), ["a_wins", "b_wins", "total"]]) == (2, 0, 2)


def test_orphan_set_breaks_features_match_silver(sql_catalog):
    orphan = silver_set(4, 100, 200, 200, 4_000_000, event_id=99)
    write_silver(
        sql_catalog, sets=[*SETS, orphan], events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS
    )
    passed, metadata = gold_checks(build_gold(sql_catalog))["features_match_silver"]
    assert not passed
    assert metadata == {"gold_rows": 3, "eligible_silver_sets": 4}


def test_empty_silver_writes_empty_gold(sql_catalog):
    write_silver(sql_catalog)
    build = build_gold(sql_catalog)
    for identifier in (SET_FEATURES, PLAYER_RATINGS, HEAD_TO_HEAD):
        assert len(rows(sql_catalog, identifier)) == 0
    assert all(passed for passed, _ in gold_checks(build).values())


def test_set_features_is_partitioned_by_year(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    (field,) = sql_catalog.load_table(SET_FEATURES).spec().fields
    assert isinstance(field.transform, YearTransform)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_gold_build.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.gold.build'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/gold/tables.py`:

```python
"""Gold table identifiers and their Arrow schemas."""

import pyarrow as pa

GOLD_NAMESPACE = "gold"
SET_FEATURES = f"{GOLD_NAMESPACE}.set_features"
PLAYER_RATINGS = f"{GOLD_NAMESPACE}.player_ratings"
HEAD_TO_HEAD = f"{GOLD_NAMESPACE}.head_to_head"
SILVER_SNAPSHOT_PROPERTY = "melee.silver_sets_snapshot"

TS = pa.timestamp("us", tz="UTC")
INT, FLOAT = pa.int64(), pa.float64()

SET_FEATURES_SCHEMA = pa.schema(
    [
        ("set_id", INT),
        ("event_id", INT),
        ("tournament_id", INT),
        ("tournament_name", pa.string()),
        ("completed_at", TS),
        ("player1_id", INT),
        ("player2_id", INT),
        ("p1_elo", FLOAT),
        ("p2_elo", FLOAT),
        ("elo_diff", FLOAT),
        ("p1_expected", FLOAT),
        ("p1_sets_played", INT),
        ("p2_sets_played", INT),
        ("p1_recent_wr", FLOAT),
        ("p2_recent_wr", FLOAT),
        ("recent_wr_diff", FLOAT),
        ("p1_h2h_wins", INT),
        ("p2_h2h_wins", INT),
        ("p1_h2h_wr", FLOAT),
        ("h2h_total", INT),
        ("p1_seed", INT),
        ("p2_seed", INT),
        ("seed_diff", INT),
        ("num_attendees", INT),
        ("p1_score", FLOAT),
        ("p2_score", FLOAT),
        ("p1_won", INT),
    ]
)
PLAYER_RATINGS_SCHEMA = pa.schema(
    [
        ("player_id", INT),
        ("gamer_tag", pa.string()),
        ("elo", FLOAT),
        ("sets_played", INT),
        ("recent_wr", FLOAT),
        ("wins", INT),
        ("losses", INT),
        ("last_played_at", TS),
    ]
)
HEAD_TO_HEAD_SCHEMA = pa.schema(
    [
        ("player_a_id", INT),
        ("player_b_id", INT),
        ("a_wins", INT),
        ("b_wins", INT),
        ("total", INT),
        ("last_played_at", TS),
    ]
)
# identifier -> (schema, column partitioned by year or None)
GOLD_TABLES: dict[str, tuple[pa.Schema, str | None]] = {
    SET_FEATURES: (SET_FEATURES_SCHEMA, "completed_at"),
    PLAYER_RATINGS: (PLAYER_RATINGS_SCHEMA, None),
    HEAD_TO_HEAD: (HEAD_TO_HEAD_SCHEMA, None),
}
```

Create `pipeline/gold/derive.py`:

```python
"""Player and head-to-head tables from the feature pass's final state."""

import pandas as pd

from pipeline.gold.features import RECENT_WINDOW, FeatureState
from pipeline.gold.tables import HEAD_TO_HEAD_SCHEMA, PLAYER_RATINGS_SCHEMA


def _tag(tag, player_id: int) -> str:
    return tag if isinstance(tag, str) and tag else str(player_id)


def player_ratings(state: FeatureState, players: pd.DataFrame) -> pd.DataFrame:
    tags = dict(zip(players["player_id"], players["gamer_tag"], strict=True))
    rows = []
    for player_id in sorted(state.elo.ratings):
        recent = state.recent_wins[player_id][-RECENT_WINDOW:]
        rows.append(
            {
                "player_id": player_id,
                "gamer_tag": _tag(tags.get(player_id), player_id),
                "elo": state.elo.get_rating(player_id),
                "sets_played": state.elo.games_played[player_id],
                "recent_wr": sum(recent) / len(recent) if recent else 0.5,
                "wins": state.wins[player_id],
                "losses": state.losses[player_id],
                "last_played_at": state.last_played[player_id],
            }
        )
    return pd.DataFrame(rows, columns=PLAYER_RATINGS_SCHEMA.names)


def head_to_head(state: FeatureState) -> pd.DataFrame:
    rows = [
        {
            "player_a_id": a,
            "player_b_id": b,
            "a_wins": record[0],
            "b_wins": record[1],
            "total": record[0] + record[1],
            "last_played_at": state.pair_last_played[(a, b)],
        }
        for (a, b), record in sorted(state.h2h.items())
    ]
    return pd.DataFrame(rows, columns=HEAD_TO_HEAD_SCHEMA.names)
```

Create `pipeline/gold/writer.py`:

```python
"""Replace gold tables in one Iceberg commit each."""

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table
from pyiceberg.transforms import YearTransform

from pipeline.gold.tables import GOLD_TABLES, SILVER_SNAPSHOT_PROPERTY


def to_arrow(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)


def ensure_table(catalog: Catalog, identifier: str) -> Table:
    schema, year_column = GOLD_TABLES[identifier]
    catalog.create_namespace_if_not_exists(identifier.split(".")[0])
    if catalog.table_exists(identifier):
        return catalog.load_table(identifier)
    table = catalog.create_table(identifier, schema=schema)
    if year_column:
        with table.update_spec() as update:
            update.add_field(year_column, YearTransform(), f"{year_column}_year")
    return catalog.load_table(identifier)


def replace_table(
    catalog: Catalog, identifier: str, data: pa.Table, properties: dict[str, str]
) -> int | None:
    """Swap the table's contents and properties atomically; return the new snapshot id."""
    table = ensure_table(catalog, identifier)
    with table.transaction() as tx:
        if table.current_snapshot() is None:
            tx.append(data)  # overwrite on a never-written table warns about an empty delete
        else:
            tx.overwrite(data)
        tx.set_properties(properties)
    snapshot = catalog.load_table(identifier).current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def write_gold(
    catalog: Catalog, frames: dict[str, pd.DataFrame], silver_sets_snapshot: int | None
) -> dict[str, int | None]:
    snapshot = "" if silver_sets_snapshot is None else str(silver_sets_snapshot)
    properties = {SILVER_SNAPSHOT_PROPERTY: snapshot}
    return {
        identifier: replace_table(
            catalog, identifier, to_arrow(frames[identifier], schema), properties
        )
        for identifier, (schema, _) in GOLD_TABLES.items()
    }
```

Create `pipeline/gold/build.py`:

```python
"""Recompute every gold table from silver. Run directly for a one-off build."""

import json
import logging
from dataclasses import dataclass

import pandas as pd
from pyiceberg.catalog import Catalog

from pipeline.gold.derive import head_to_head, player_ratings
from pipeline.gold.features import build_features
from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES
from pipeline.gold.writer import write_gold
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog, read_table
from pipeline.model_features import CORE_FEATURES

SET_COLUMNS = (
    "set_id",
    "event_id",
    "completed_at",
    "player1_id",
    "player2_id",
    "winner_player_id",
    "player1_score",
    "player2_score",
    "player1_seed",
    "player2_seed",
)


@dataclass
class GoldBuild:
    frames: dict[str, pd.DataFrame]
    eligible_silver_sets: int
    silver_sets_snapshot: int | None
    snapshots: dict[str, int | None]


def load_silver(catalog: Catalog) -> tuple[pd.DataFrame, pd.DataFrame, int | None, int]:
    """(sets joined to their tournament, players, silver.sets snapshot, eligible set count).

    Sets whose event or tournament is missing drop out of the join, as they did in
    the original SQL. The eligible count is taken before the join, so a gap shows
    up in the features_match_silver check.
    """
    snapshot = current_snapshot_id(catalog, "silver.sets")
    sets = read_table(catalog, "silver.sets", SET_COLUMNS, snapshot_id=snapshot)
    events = read_table(catalog, "silver.events", ("event_id", "tournament_id"))
    tournaments = read_table(
        catalog, "silver.tournaments", ("tournament_id", "name", "num_attendees")
    ).rename(columns={"name": "tournament_name"})
    players = read_table(catalog, "silver.players", ("player_id", "gamer_tag"))
    eligible = int((sets["winner_player_id"].notna() & sets["completed_at"].notna()).sum())
    joined = sets.merge(events, on="event_id", how="inner").merge(
        tournaments, on="tournament_id", how="inner"
    )
    return joined, players, snapshot, eligible


def build_gold(catalog: Catalog) -> GoldBuild:
    sets, players, snapshot, eligible = load_silver(catalog)
    features, state = build_features(sets)
    frames = {
        SET_FEATURES: features,
        PLAYER_RATINGS: player_ratings(state, players),
        HEAD_TO_HEAD: head_to_head(state),
    }
    snapshots = write_gold(catalog, frames, snapshot)
    return GoldBuild(frames, eligible, snapshot, snapshots)


def gold_checks(build: GoldBuild) -> dict[str, tuple[bool, dict]]:
    features = build.frames[SET_FEATURES]
    nulls = int(features[CORE_FEATURES].isna().sum().sum())
    ordered = bool(features["completed_at"].is_monotonic_increasing)
    return {
        "features_match_silver": (
            len(features) == build.eligible_silver_sets,
            {"gold_rows": len(features), "eligible_silver_sets": build.eligible_silver_sets},
        ),
        "no_null_features": (nulls == 0, {"null_feature_values": nulls}),
        "completed_at_non_decreasing": (ordered, {}),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build = build_gold(load_lakehouse_catalog())
    summary = {name: len(frame) for name, frame in build.frames.items()}
    summary["checks"] = {name: passed for name, (passed, _) in gold_checks(build).items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
```

In the `Makefile`:
- add `gold-once` to `.PHONY`;
- add this target after `silver-once`:

```makefile
gold-once:
	$(COMPOSE) run --rm pipeline-tools python -m pipeline.gold.build
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_gold_build.py -v`
Expected: 6 passed, with no warnings. If PyIceberg emits a warning that these tests trigger, fix its cause; don't filter it.

- [ ] **Step 5: Build gold on the dev stack**

This is the first live PyIceberg ↔ Lakekeeper round trip.

```bash
$MAKE gold-once 2>/dev/null | tail -1
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T trino trino --execute "
SELECT 'set_features', count(*) FROM lakekeeper.gold.set_features UNION ALL
SELECT 'player_ratings', count(*) FROM lakekeeper.gold.player_ratings UNION ALL
SELECT 'head_to_head', count(*) FROM lakekeeper.gold.head_to_head"
```

Expected:
- The JSON summary shows non-zero row counts and all three checks `true`.
- `gold.set_features` equals silver's eligible sets (compare with `SELECT count(*) FROM lakekeeper.silver.sets WHERE winner_player_id IS NOT NULL AND completed_at IS NOT NULL`).
- Trino reads all three tables.

Fallback if PyIceberg can't read or write with vended credentials (for example a 403, or an endpoint error for `silo`):
- Add `S3_ENDPOINT: http://silo:9000`, `S3_ACCESS_KEY: ${SILO_ROOT_USER:-melee-admin}` and `S3_SECRET_KEY: ${SILO_ROOT_PASSWORD:-melee-admin-password}` to the `pipeline-tools` and `dagster-code` environments in `infra/docker-compose.yml`. Task 13 must then also add them to the `app` service.
- Re-run, and record the change and the original error in the task report.

- [ ] **Step 6: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/gold Makefile tests/pipeline/lakehouse_data.py tests/pipeline/test_gold_build.py
git commit -m "Build gold feature, player and head-to-head tables from silver with PyIceberg"
```

---

### Task 7: Move training to DataFrames (pipeline/ml/train.py, tuning)

**Files:**
- Create: `pipeline/ml/__init__.py` (empty), `pipeline/ml/names.py`, `pipeline/ml/train.py`, `pipeline/ml/tuning.py`
- Create: `tests/pipeline/ml_data.py`
- Test: `tests/pipeline/test_ml_train.py`

**Interfaces:**
- Consumes: `CORE_FEATURES`, `SEED_FEATURES`, `TARGET`, `SET_FEATURES`, `FEATURE_COLUMNS`, `read_table`, `current_snapshot_id`.
- Produces:
  - `pipeline.ml.names`: `EXPERIMENT = "melee-matchup"`, `MODEL_NAME = "melee-matchup"`, `CHAMPION_ALIAS = "champion"`, `GOLD_SNAPSHOT_TAG = "gold_snapshot_id"`, `SHAP_ARTIFACT = "shap_metadata.json"`.
  - `pipeline.ml.train`:
    - `ModelResult`, unchanged fields;
    - `DEFAULT_SPLIT_AT = pd.Timestamp("2025-07-01", tz="UTC")`, `DEFAULT_MIN_SETS = 5`;
    - split functions: `filter_min_sets(df, min_sets)`, `temporal_split(df, split_at)`, `quantile_split(df, q)`, and `split_data(df, *, split_at=None, split_quantile=None, min_sets=5) -> (train, test)`, which raises `ValueError` on an empty side;
    - model functions: `evaluate(...)`, `train_baseline(train, test)`, `train_logistic(train, test)`, `train_xgboost(train, test)`;
    - reporting: `shap_metadata(model, X_sample) -> dict`, `walk_forward_cv(df, min_sets=5, n_folds=4)`, `results_table(results)`;
    - `load_features(catalog) -> tuple[pd.DataFrame, int | None]`.
  - `pipeline.ml.tuning`: `objective(trial, train, test, log_to_mlflow=True) -> float`, `tune(df, n_trials=50, min_sets=5, log_to_mlflow=True)`.
  - `tests/pipeline/ml_data.py`: `synthetic_gold_features(n=800, seed=7) -> pd.DataFrame` with every `FEATURE_COLUMNS` column. In it, `elo_diff` drives the outcome and `seed_diff` is noise.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/ml_data.py`:

```python
"""Synthetic gold.set_features rows where Elo predicts the winner and seeds are noise."""

import numpy as np
import pandas as pd

from pipeline.gold.features import FEATURE_COLUMNS


def synthetic_gold_features(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 250, n)
    expected = 1 / (1 + 10 ** (-elo_diff / 400))
    p1_won = (rng.random(n) < expected).astype("int64")
    p1_seed = rng.integers(1, 64, n)
    p2_seed = rng.integers(1, 64, n)
    frame = pd.DataFrame(
        {
            "set_id": np.arange(n, dtype="int64"),
            "event_id": 10,
            "tournament_id": 1,
            "tournament_name": "Synthetic",
            "completed_at": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "player1_id": rng.integers(1, 50, n),
            "player2_id": rng.integers(51, 100, n),
            "p1_elo": 1500 + elo_diff / 2,
            "p2_elo": 1500 - elo_diff / 2,
            "elo_diff": elo_diff,
            "p1_expected": expected,
            "p1_sets_played": rng.integers(5, 200, n),
            "p2_sets_played": rng.integers(5, 200, n),
            "p1_recent_wr": 0.5,
            "p2_recent_wr": 0.5,
            "recent_wr_diff": 0.0,
            "p1_h2h_wins": 0,
            "p2_h2h_wins": 0,
            "p1_h2h_wr": 0.5,
            "h2h_total": 0,
            "p1_seed": p1_seed,
            "p2_seed": p2_seed,
            "seed_diff": p1_seed - p2_seed,
            "num_attendees": 200,
            "p1_score": 2.0,
            "p2_score": 1.0,
            "p1_won": p1_won,
        }
    )
    return frame[list(FEATURE_COLUMNS)]
```

Create `tests/pipeline/test_ml_train.py`:

```python
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from optuna.trial import FixedTrial  # noqa: E402

from pipeline.ml.train import (  # noqa: E402
    filter_min_sets,
    quantile_split,
    shap_metadata,
    split_data,
    temporal_split,
    train_baseline,
    train_logistic,
    train_xgboost,
)
from pipeline.ml.tuning import objective  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_temporal_split_uses_a_timestamp():
    df = synthetic_gold_features(100)
    cut = df["completed_at"].iloc[60]
    train, test = temporal_split(df, cut)
    assert (len(train), len(test)) == (60, 40)
    assert train["completed_at"].max() < test["completed_at"].min()


def test_quantile_split_keeps_time_order():
    train, test = quantile_split(synthetic_gold_features(101), 0.8)
    assert len(train) == 80
    assert train["completed_at"].max() < test["completed_at"].min()


def test_filter_min_sets_requires_both_players():
    df = synthetic_gold_features(10)
    df.loc[0, "p1_sets_played"] = 2
    df.loc[1, "p2_sets_played"] = 2
    assert len(filter_min_sets(df, 5)) == 8


def test_models_report_metrics_and_xgboost_beats_the_seed_baseline():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    baseline = train_baseline(train, test)
    logistic = train_logistic(train, test)
    xgb = train_xgboost(train, test)
    for result in (baseline, logistic, xgb):
        assert 0 < result.log_loss_val < 1
        assert 0 <= result.accuracy <= 1
    assert xgb.log_loss_val < baseline.log_loss_val


def test_shap_metadata_lists_the_core_features():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    model = train_xgboost(train, test).model
    metadata = shap_metadata(model, test[CORE_FEATURES])
    assert metadata["feature_names"] == CORE_FEATURES
    assert set(metadata["mean_abs_shap"]) == set(CORE_FEATURES)
    assert isinstance(metadata["base_value"], float)


def test_empty_split_raises_a_clear_error():
    df = synthetic_gold_features(50)
    with pytest.raises(ValueError, match="Empty test split: 50 train / 0 test rows"):
        split_data(df, split_at=pd.Timestamp("2030-01-01", tz="UTC"))


def test_tuning_objective_returns_auc_without_mlflow():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 1e-3,
              "reg_lambda": 1e-3}  # fmt: skip
    auc = objective(FixedTrial(params), train, test, log_to_mlflow=False)
    assert 0.5 < auc <= 1.0
```

In `test_quantile_split_keeps_time_order`, the 0.8 quantile of 101 hourly timestamps is exactly the 81st timestamp. So 80 rows fall strictly before it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_ml_train.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.ml'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/ml/__init__.py` as an empty file.

Create `pipeline/ml/names.py`:

```python
"""MLflow names shared by training, the Dagster assets and the app. No heavy imports."""

EXPERIMENT = "melee-matchup"
MODEL_NAME = "melee-matchup"
CHAMPION_ALIAS = "champion"
GOLD_SNAPSHOT_TAG = "gold_snapshot_id"
SHAP_ARTIFACT = "shap_metadata.json"
```

Create `pipeline/ml/train.py`. It moves `src/model.py`'s functions so they take DataFrames. The hyperparameters, the logistic-regression scaler wrapper (now a module-level class) and the evaluation are unchanged.

```python
"""Model training on gold.set_features (moved from src/model.py).

Models:
1. Baseline: Logistic regression on seed differential only
2. Full LR: Logistic regression on all features
3. XGBoost: Gradient boosting on all features

Uses a temporal train/test split (no random shuffle) to avoid leakage.
Run directly to print the comparison table without logging to MLflow.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from pyiceberg.catalog import Catalog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline.gold.features import FEATURE_COLUMNS
from pipeline.gold.tables import SET_FEATURES
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog, read_table
from pipeline.model_features import CORE_FEATURES, SEED_FEATURES, TARGET

logger = logging.getLogger(__name__)

# Train on everything before this time, test on the rest (the last ~9 months).
DEFAULT_SPLIT_AT = pd.Timestamp("2025-07-01", tz="UTC")
DEFAULT_MIN_SETS = 5


@dataclass
class ModelResult:
    """Stores evaluation results for a single model."""

    name: str
    accuracy: float
    log_loss_val: float
    brier_score: float
    roc_auc: float
    y_true: np.ndarray
    y_prob: np.ndarray
    model: object
    feature_names: list[str] = field(default_factory=list)
    feature_importances: np.ndarray | None = None


def load_features(catalog: Catalog) -> tuple[pd.DataFrame, int | None]:
    """gold.set_features in time order, plus the snapshot it was read from."""
    snapshot = current_snapshot_id(catalog, SET_FEATURES)
    df = read_table(catalog, SET_FEATURES, FEATURE_COLUMNS, snapshot_id=snapshot)
    return df.sort_values(["completed_at", "set_id"]).reset_index(drop=True), snapshot


def filter_min_sets(df: pd.DataFrame, min_sets: int) -> pd.DataFrame:
    """Keep sets where both players have at least min_sets prior sets."""
    kept = df[(df["p1_sets_played"] >= min_sets) & (df["p2_sets_played"] >= min_sets)]
    logger.info(f"{len(df):,} rows, {len(kept):,} after min_sets={min_sets} filter")
    return kept


def temporal_split(
    df: pd.DataFrame, split_at: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally — no random shuffle."""
    return df[df["completed_at"] < split_at].copy(), df[df["completed_at"] >= split_at].copy()


def quantile_split(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split at the q-th quantile of completed_at (for small datasets)."""
    return temporal_split(df, df["completed_at"].quantile(q))


def split_data(
    df: pd.DataFrame,
    *,
    split_at: pd.Timestamp | None = None,
    split_quantile: float | None = None,
    min_sets: int = DEFAULT_MIN_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = filter_min_sets(df, min_sets)
    if split_quantile is not None:
        train, test = quantile_split(data, split_quantile)
    else:
        train, test = temporal_split(data, DEFAULT_SPLIT_AT if split_at is None else split_at)
    if train.empty or test.empty:
        side = "train" if train.empty else "test"
        raise ValueError(
            f"Empty {side} split: {len(train)} train / {len(test)} test rows after "
            f"min_sets={min_sets}. Choose a different split_at or split_quantile."
        )
    logger.info(f"Train: {len(train):,} sets, Test: {len(test):,} sets")
    return train, test


def evaluate(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> ModelResult:
    """Evaluate a model and return structured results."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = brier_score_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    logger.info(f"[{name}] Acc={acc:.4f}  LogLoss={ll:.4f}  Brier={bs:.4f}  AUC={auc:.4f}")

    return ModelResult(
        name=name,
        accuracy=acc,
        log_loss_val=ll,
        brier_score=bs,
        roc_auc=auc,
        y_true=y_test.values,
        y_prob=y_prob,
        model=model,
        feature_names=feature_names,
        feature_importances=importances,
    )


def train_baseline(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """Baseline: logistic regression on seed differential only."""
    features = SEED_FEATURES
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train[TARGET])
    return evaluate("Baseline (Seed Only)", model, test[features], test[TARGET], features)


class ScaledModel:
    """Logistic regression behind a StandardScaler, with a predict_proba interface."""

    def __init__(self, scaler, model):
        self._scaler = scaler
        self._model = model
        self.coef_ = model.coef_

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))


def train_logistic(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """Full logistic regression on all core features."""
    features = CORE_FEATURES
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(train[features]), train[TARGET])
    return evaluate(
        "Logistic Regression (All Features)",
        ScaledModel(scaler, model),
        test[features],
        test[TARGET],
        features,
    )


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """XGBoost gradient boosting on all core features."""
    features = CORE_FEATURES
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(
        train[features],
        train[TARGET],
        eval_set=[(test[features], test[TARGET])],
        verbose=False,
    )
    return evaluate("XGBoost", model, test[features], test[TARGET], features)


def shap_metadata(model, X_sample: pd.DataFrame) -> dict:
    """SHAP base value and mean absolute SHAP per feature, for the app's explanations."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    mean_abs = np.abs(shap_values.values).mean(axis=0).tolist()
    return {
        "base_value": float(explainer.expected_value),
        "feature_names": list(X_sample.columns),
        "mean_abs_shap": dict(zip(X_sample.columns, mean_abs, strict=True)),
    }


def walk_forward_cv(
    df: pd.DataFrame, min_sets: int = DEFAULT_MIN_SETS, n_folds: int = 4
) -> pd.DataFrame:
    """Walk-forward time-series cross-validation for XGBoost.

    Splits data into n_folds+1 temporal chunks. For each fold k (1..n_folds),
    trains on chunks 0..k-1 and tests on chunk k.
    """
    data = filter_min_sets(df, min_sets)
    quantiles = [i / (n_folds + 1) for i in range(1, n_folds + 1)]
    cuts = data["completed_at"].quantile(quantiles).tolist()
    cuts.append(data["completed_at"].max() + pd.Timedelta(seconds=1))

    fold_results = []
    for fold_idx, split_at in enumerate(cuts):
        train = data[data["completed_at"] < split_at]
        if fold_idx < len(cuts) - 1:
            in_fold = (data["completed_at"] >= split_at) & (
                data["completed_at"] < cuts[fold_idx + 1]
            )
            test = data[in_fold]
        else:
            test = data[data["completed_at"] >= split_at]
        if len(test) == 0:
            continue
        result = train_xgboost(train, test)
        fold_results.append(
            {
                "Fold": fold_idx + 1,
                "Train Size": len(train),
                "Test Size": len(test),
                "Accuracy": result.accuracy,
                "Log Loss": result.log_loss_val,
                "Brier Score": result.brier_score,
                "ROC AUC": result.roc_auc,
            }
        )

    results_df = pd.DataFrame(fold_results)
    mean_row = results_df[["Accuracy", "Log Loss", "Brier Score", "ROC AUC"]].mean()
    mean_row["Fold"] = "Mean"
    mean_row["Train Size"] = ""
    mean_row["Test Size"] = ""
    return pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """Format results as a comparison table."""
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": f"{r.accuracy:.4f}",
                "Log Loss": f"{r.log_loss_val:.4f}",
                "Brier Score": f"{r.brier_score:.4f}",
                "ROC AUC": f"{r.roc_auc:.4f}",
            }
            for r in results
        ]
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    df, _ = load_features(load_lakehouse_catalog())
    train, test = split_data(df)
    results = [train_baseline(train, test), train_logistic(train, test), train_xgboost(train, test)]
    print("\n" + results_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
```

Create `pipeline/ml/tuning.py`:

```python
"""Optuna hyperparameter tuning for XGBoost with MLflow logging (moved from src/tuning.py).

Usage (inside the stack): python -m pipeline.ml.tuning --n-trials 50
"""

import argparse
import logging

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from pipeline.lakehouse import load_lakehouse_catalog
from pipeline.ml.train import DEFAULT_MIN_SETS, load_features, split_data
from pipeline.model_features import CORE_FEATURES, TARGET

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, train, test, log_to_mlflow: bool = True) -> float:
    """Single Optuna trial — train XGBoost with sampled hyperparams."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = XGBClassifier(
        **params, eval_metric="logloss", early_stopping_rounds=20, random_state=42
    )
    model.fit(
        train[CORE_FEATURES],
        train[TARGET],
        eval_set=[(test[CORE_FEATURES], test[TARGET])],
        verbose=False,
    )
    y_prob = model.predict_proba(test[CORE_FEATURES])[:, 1]
    auc = roc_auc_score(test[TARGET], y_prob)
    if log_to_mlflow:
        mlflow.set_experiment("melee-matchup-tuning")
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics({"roc_auc": auc, "log_loss": log_loss(test[TARGET], y_prob)})
    return auc


def tune(
    df: pd.DataFrame,
    n_trials: int = 50,
    min_sets: int = DEFAULT_MIN_SETS,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    train, test = split_data(df, min_sets=min_sets)
    study = optuna.create_study(direction="maximize", study_name="xgboost-melee")
    study.optimize(lambda trial: objective(trial, train, test, log_to_mlflow), n_trials=n_trials)
    logger.info(f"Best trial #{study.best_trial.number}: AUC={study.best_value:.4f}")
    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparams")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--min-sets", type=int, default=DEFAULT_MIN_SETS)
    args = parser.parse_args()
    features, _ = load_features(load_lakehouse_catalog())
    study = tune(features, n_trials=args.n_trials, min_sets=args.min_sets)
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_ml_train.py -v`
Expected: 7 passed, with no warnings. If a library emits a warning that comes from its own internals, add a narrowly scoped `filterwarnings` entry to `pyproject.toml` that names the exact message and module, like the existing dagster entry. Record it in the report.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/ml tests/pipeline/ml_data.py tests/pipeline/test_ml_train.py pyproject.toml
git commit -m "Move model training and tuning onto gold DataFrames"
```

---

### Task 8: Promotion rule and the MLflow registry flow

**Files:**
- Create: `pipeline/ml/promotion.py`, `pipeline/ml/registry.py`
- Modify: `tests/pipeline/conftest.py` (`mlflow_tracking` fixture)
- Test: `tests/pipeline/test_promotion.py`, `tests/pipeline/test_registry.py`

**Interfaces:**
- Consumes: the Task 7 train module and `pipeline.ml.names`.
- Produces:
  - `pipeline.ml.promotion`: `PromotionDecision(promote: bool, reason: str)`, and `decide_promotion(candidate_log_loss, baseline_log_loss, champion_log_loss: float | None) -> PromotionDecision`. The reasons are `below_baseline`, `no_champion`, `beats_or_ties_champion` and `worse_than_champion`.
  - `pipeline.ml.registry`:
    - `TrainingOutcome`, with fields `baseline`, `logistic`, `xgboost` (each a `ModelResult`), `version: str`, `run_id: str`, `decision`, `champion_version: str | None` and `champion_log_loss: float | None`;
    - `champion_version(client) -> ModelVersion | None`;
    - `champion_gold_snapshot(client) -> str | None`;
    - `train_and_register(features, gold_snapshot_id, *, split_at=None, split_quantile=None, min_sets=5) -> TrainingOutcome`.
  - Fixture `mlflow_tracking`: a SQLite tracking/registry URI in `tmp_path`, set as the global MLflow URI, with the `melee-matchup` experiment created and its artifacts in `tmp_path`. It returns the URI.

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/conftest.py`:

```python
@pytest.fixture
def mlflow_tracking(tmp_path):
    """A throwaway SQLite-backed MLflow tracking store and registry."""
    mlflow = pytest.importorskip("mlflow")
    from pipeline.ml.names import EXPERIMENT

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    return uri
```

Create `tests/pipeline/test_promotion.py`:

```python
from pipeline.ml.promotion import PromotionDecision, decide_promotion


def test_first_model_is_promoted():
    assert decide_promotion(0.50, 0.60, None) == PromotionDecision(True, "no_champion")


def test_better_model_is_promoted():
    assert decide_promotion(0.45, 0.60, 0.50) == PromotionDecision(True, "beats_or_ties_champion")


def test_equal_model_is_promoted():
    assert decide_promotion(0.50, 0.60, 0.50) == PromotionDecision(True, "beats_or_ties_champion")


def test_worse_model_is_not_promoted():
    assert decide_promotion(0.55, 0.60, 0.50) == PromotionDecision(False, "worse_than_champion")


def test_model_that_loses_to_the_baseline_is_never_promoted():
    assert decide_promotion(0.61, 0.60, None) == PromotionDecision(False, "below_baseline")
```

Create `tests/pipeline/test_registry.py`:

```python
import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from pipeline.ml.names import (  # noqa: E402
    CHAMPION_ALIAS,
    GOLD_SNAPSHOT_TAG,
    MODEL_NAME,
    SHAP_ARTIFACT,
)
from pipeline.ml.promotion import PromotionDecision  # noqa: E402
from pipeline.ml.registry import champion_gold_snapshot, train_and_register  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_first_model_becomes_champion(mlflow_tracking):
    outcome = train_and_register(synthetic_gold_features(), 111, split_quantile=0.7)
    assert outcome.decision == PromotionDecision(True, "no_champion")
    assert outcome.champion_version is None
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert champion.version == outcome.version


def test_version_carries_the_gold_snapshot_and_shap_artifact(mlflow_tracking):
    outcome = train_and_register(synthetic_gold_features(), 222, split_quantile=0.7)
    client = MlflowClient()
    assert client.get_model_version(MODEL_NAME, outcome.version).tags[GOLD_SNAPSHOT_TAG] == "222"
    run = client.get_run(outcome.run_id)
    assert run.data.tags[GOLD_SNAPSHOT_TAG] == "222"
    assert {"xgboost_log_loss", "baseline_log_loss", "logistic_roc_auc"} <= set(run.data.metrics)
    shap = mlflow.artifacts.load_dict(f"runs:/{outcome.run_id}/{SHAP_ARTIFACT}")
    assert shap["feature_names"] == CORE_FEATURES


def test_second_model_is_compared_with_the_champion(mlflow_tracking):
    first = train_and_register(synthetic_gold_features(), 1, split_quantile=0.7)
    second = train_and_register(synthetic_gold_features(), 2, split_quantile=0.7)
    assert second.champion_version == first.version
    assert second.champion_log_loss == pytest.approx(second.xgboost.log_loss_val)  # same data
    assert second.decision == PromotionDecision(True, "beats_or_ties_champion")
    assert champion_gold_snapshot(MlflowClient()) == "2"


def test_rejected_model_leaves_the_champion_alone(mlflow_tracking, monkeypatch):
    first = train_and_register(synthetic_gold_features(), 1, split_quantile=0.7)
    monkeypatch.setattr(
        "pipeline.ml.registry.decide_promotion",
        lambda *args: PromotionDecision(False, "worse_than_champion"),
    )
    second = train_and_register(synthetic_gold_features(), 2, split_quantile=0.7)
    client = MlflowClient()
    assert client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version == first.version
    tags = client.get_model_version(MODEL_NAME, second.version).tags
    assert tags["promotion"] == "worse_than_champion"
    assert champion_gold_snapshot(client) == "1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_promotion.py tests/pipeline/test_registry.py -v`
Expected: collection errors, `ModuleNotFoundError: No module named 'pipeline.ml.promotion'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/ml/promotion.py`:

```python
"""When a newly trained model may take the champion alias."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    candidate_log_loss: float, baseline_log_loss: float, champion_log_loss: float | None
) -> PromotionDecision:
    """A candidate must beat the seed baseline, then match or beat the champion.

    All log losses are measured on the candidate's own test split.
    """
    if not candidate_log_loss < baseline_log_loss:
        return PromotionDecision(False, "below_baseline")
    if champion_log_loss is None:
        return PromotionDecision(True, "no_champion")
    if candidate_log_loss <= champion_log_loss:
        return PromotionDecision(True, "beats_or_ties_champion")
    return PromotionDecision(False, "worse_than_champion")
```

Create `pipeline/ml/registry.py`:

```python
"""Train, log to MLflow, register a version, and decide whether it becomes champion."""

from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.metrics import log_loss

from pipeline.ml.names import (
    CHAMPION_ALIAS,
    EXPERIMENT,
    GOLD_SNAPSHOT_TAG,
    MODEL_NAME,
    SHAP_ARTIFACT,
)
from pipeline.ml.promotion import PromotionDecision, decide_promotion
from pipeline.ml.train import (
    DEFAULT_MIN_SETS,
    ModelResult,
    shap_metadata,
    split_data,
    train_baseline,
    train_logistic,
    train_xgboost,
)
from pipeline.model_features import CORE_FEATURES, TARGET

SHAP_SAMPLE_ROWS = 1000


@dataclass
class TrainingOutcome:
    baseline: ModelResult
    logistic: ModelResult
    xgboost: ModelResult
    version: str
    run_id: str
    decision: PromotionDecision
    champion_version: str | None
    champion_log_loss: float | None


def champion_version(client: MlflowClient) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:  # no registered model or no alias yet
        return None


def champion_gold_snapshot(client: MlflowClient) -> str | None:
    """The gold snapshot id the current champion was trained on."""
    champion = champion_version(client)
    return None if champion is None else champion.tags.get(GOLD_SNAPSHOT_TAG)


def _log_result(prefix: str, result: ModelResult) -> None:
    mlflow.log_metrics(
        {
            f"{prefix}_accuracy": result.accuracy,
            f"{prefix}_log_loss": result.log_loss_val,
            f"{prefix}_brier": result.brier_score,
            f"{prefix}_roc_auc": result.roc_auc,
        }
    )


def train_and_register(
    features: pd.DataFrame,
    gold_snapshot_id: int | None,
    *,
    split_at: pd.Timestamp | None = None,
    split_quantile: float | None = None,
    min_sets: int = DEFAULT_MIN_SETS,
) -> TrainingOutcome:
    train, test = split_data(
        features, split_at=split_at, split_quantile=split_quantile, min_sets=min_sets
    )
    snapshot_tag = "" if gold_snapshot_id is None else str(gold_snapshot_id)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train") as run:
        baseline = train_baseline(train, test)
        logistic = train_logistic(train, test)
        xgb = train_xgboost(train, test)
        mlflow.log_params(
            {
                "min_sets": min_sets,
                "split_at": "" if split_at is None else split_at.isoformat(),
                "split_quantile": "" if split_quantile is None else split_quantile,
                "train_size": len(train),
                "test_size": len(test),
                "features": ",".join(CORE_FEATURES),
            }
        )
        for prefix, result in (("baseline", baseline), ("logistic", logistic), ("xgboost", xgb)):
            _log_result(prefix, result)
        mlflow.set_tag(GOLD_SNAPSHOT_TAG, snapshot_tag)
        sample = test[CORE_FEATURES].head(SHAP_SAMPLE_ROWS)
        mlflow.log_dict(shap_metadata(xgb.model, sample), SHAP_ARTIFACT)
        info = mlflow.xgboost.log_model(xgb.model, name="model", registered_model_name=MODEL_NAME)

    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, GOLD_SNAPSHOT_TAG, snapshot_tag)

    champion = champion_version(client)
    champion_ll = None
    if champion is not None:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        champion_prob = model.predict_proba(test[CORE_FEATURES])[:, 1]
        champion_ll = float(log_loss(test[TARGET], champion_prob))

    decision = decide_promotion(xgb.log_loss_val, baseline.log_loss_val, champion_ll)
    client.set_model_version_tag(MODEL_NAME, version, "promotion", decision.reason)
    if decision.promote:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

    return TrainingOutcome(
        baseline=baseline,
        logistic=logistic,
        xgboost=xgb,
        version=version,
        run_id=run.info.run_id,
        decision=decision,
        champion_version=None if champion is None else str(champion.version),
        champion_log_loss=champion_ll,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_promotion.py tests/pipeline/test_registry.py -v`
Expected: 5 and 4 passed. The first MLflow test is slower (a few seconds) while the SQLite schema migrates.

If MLflow itself emits warnings (for example about a missing model signature or pip requirements inference), add a narrowly scoped `filterwarnings` entry to `pyproject.toml` for that exact message from `mlflow.*`. Record it in the report. Do not blanket-ignore warnings.

Run: `uv run pytest`
Expected: everything passes, with no warnings.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/ml/promotion.py pipeline/ml/registry.py tests/pipeline/conftest.py tests/pipeline/test_promotion.py tests/pipeline/test_registry.py pyproject.toml
git commit -m "Register trained models in MLflow and promote a champion behind the baseline gate"
```

---

### Task 9: MLflow, the mlflow database and pools in the Compose stack

**Files:**
- Create: `pipeline/dagster_defs/pools.py`
- Modify: `infra/docker-compose.yml`, `infra/docker-compose.smoke.yml`, `infra/postgres/init-databases.sql`, `infra/dagster/dagster.yaml`
- Test: `tests/pipeline/test_infra_config.py`

**Interfaces:**
- Produces:
  - `pipeline.dagster_defs.pools`: `STARTGG_API = "startgg_api"`, `SPARK_LAKEHOUSE = "spark_lakehouse"`, `GOLD = "gold"`, `TRAINING = "training"`, `ALL_POOLS`.
  - Compose service `mlflow`: `http://mlflow:5000` inside the network, `${MLFLOW_PORT:-5000}` on the host.
  - One-shot service `postgres-databases`, which creates any missing database.
  - `MLFLOW_TRACKING_URI=http://mlflow:5000` in `dagster-code` and `pipeline-tools`.
  - The `dagster-pools` service sets all four pools to 1.

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/test_infra_config.py`:

```python
from pipeline.dagster_defs.pools import ALL_POOLS


def _service(name: str) -> str:
    """The text of one top-level Compose service block (two-space indented key)."""
    start = COMPOSE.index(f"\n  {name}:\n")
    rest = COMPOSE[start + 1 :]
    following = re.search(r"\n  [a-z][a-z0-9-]*:\n", rest[1:])
    return rest if following is None else rest[: following.start() + 1]


def test_mlflow_uses_postgres_silo_and_one_worker():
    mlflow = _service("mlflow")
    assert "ghcr.io/mlflow/mlflow:v3.16.1-full" in mlflow
    assert "postgresql+psycopg2://" in mlflow and "@postgres:5432/mlflow" in mlflow
    assert "s3://mlflow" in mlflow and "--serve-artifacts" in mlflow
    assert '"1"' in mlflow.split("--workers", 1)[1].split("\n", 2)[1]
    assert "mlflow:5000" in mlflow  # allowed Host header for in-network clients


def test_mlflow_database_exists_on_new_and_existing_volumes():
    init_sql = Path("infra/postgres/init-databases.sql").read_text(encoding="utf-8")
    assert "CREATE DATABASE mlflow;" in init_sql
    assert "mlflow" in _service("postgres-databases")
    assert "local/mlflow" in _service("create-bucket")


def test_dagster_pools_service_sets_every_pool():
    pools = _service("dagster-pools")
    for pool in ALL_POOLS:
        assert f"concurrency set {pool} 1" in pools
    assert "spark_bronze" not in pools
```

Put `from pipeline.dagster_defs.pools import ALL_POOLS` with the other imports at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_infra_config.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.dagster_defs.pools'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/dagster_defs/pools.py`:

```python
"""Concurrency pools. Each is limited to one run (see the dagster-pools service)."""

STARTGG_API = "startgg_api"  # the start.gg rate limit is shared
SPARK_LAKEHOUSE = "spark_lakehouse"  # bronze, silver and maintenance: one Spark driver at a time
GOLD = "gold"
TRAINING = "training"
ALL_POOLS = (STARTGG_API, SPARK_LAKEHOUSE, GOLD, TRAINING)
```

Append to `infra/postgres/init-databases.sql`:

```sql
CREATE DATABASE mlflow;
```

In `infra/docker-compose.yml`:

1. Change `create-bucket`'s command line so it also creates the MLflow bucket:

```yaml
      - mc alias set local http://silo:9000 "$$SILO_ROOT_USER" "$$SILO_ROOT_PASSWORD" && mc mb --ignore-existing local/lakehouse && mc mb --ignore-existing local/mlflow
```

2. Add this service right after `postgres`:

```yaml
  postgres-databases:
    # init-databases.sql only runs on an empty volume; this also creates databases
    # added later (mlflow) on volumes that already exist.
    image: postgres:17.11
    environment:
      <<: *db-env
      PGPASSWORD: ${POSTGRES_PASSWORD:-melee}
    depends_on:
      postgres:
        condition: service_healthy
    entrypoint: /bin/bash
    command:
      - -c
      - |
        set -e
        for db in lakekeeper dagster mlflow; do
          if ! psql -h postgres -U "$$POSTGRES_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$$db'" | grep -q 1; then
            psql -h postgres -U "$$POSTGRES_USER" -d postgres -c "CREATE DATABASE $$db"
          fi
        done
    restart: "no"
```

3. Add this service right after `trino`:

```yaml
  mlflow:
    # The -full image includes mlflow[db,extras]: psycopg2-binary and boto3.
    image: ghcr.io/mlflow/mlflow:v3.16.1-full
    command:
      - mlflow
      - server
      - --host
      - 0.0.0.0
      - --port
      - "5000"
      - --workers
      - "1"
      - --backend-store-uri
      - postgresql+psycopg2://${POSTGRES_USER:-melee}:${POSTGRES_PASSWORD:-melee}@postgres:5432/mlflow
      - --artifacts-destination
      - s3://mlflow
      - --serve-artifacts
      # MLflow 3.5+ rejects unknown Host headers; containers call it as mlflow:5000.
      - --allowed-hosts
      - mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://silo:9000
      AWS_ACCESS_KEY_ID: ${SILO_ROOT_USER:-melee-admin}
      AWS_SECRET_ACCESS_KEY: ${SILO_ROOT_PASSWORD:-melee-admin-password}
      AWS_DEFAULT_REGION: us-east-1
    ports:
      - "${MLFLOW_PORT:-5000}:5000"
    mem_limit: 512m
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 5s
      timeout: 10s
      retries: 30
    depends_on:
      postgres-databases:
        condition: service_completed_successfully
      create-bucket:
        condition: service_completed_successfully
```

4. In `dagster-code`:
   - under `environment:`, add `MLFLOW_TRACKING_URI: http://mlflow:5000`;
   - under `depends_on:`, add:

```yaml
      mlflow:
        condition: service_healthy
```

5. Replace the `dagster-pools` command line with:

```yaml
      - dagster instance concurrency set startgg_api 1 && dagster instance concurrency set spark_lakehouse 1 && dagster instance concurrency set gold 1 && dagster instance concurrency set training 1
```

6. In `pipeline-tools`, add:

```yaml
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
```

In `infra/docker-compose.smoke.yml`, add:

```yaml
  mlflow:
    ports: !reset []
```

In `infra/dagster/dagster.yaml`, replace the two pool comment lines with:

```yaml
    # Every pool runs one at a time: startgg_api (shared API rate limit),
    # spark_lakehouse (one Spark driver; bronze checkpoints), gold, training.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_infra_config.py -v`
Expected: all pass, including the 3 new tests.

```bash
git add infra
git ls-files --eol infra | grep -v "w/lf" || echo "all LF"
```

Expected: `all LF`.

- [ ] **Step 5: Bring MLflow up on the dev stack**

```bash
export DAGSTER_WEBSERVER_PORT=3001
$MAKE up
curl -sf http://localhost:5000/health && echo
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-code python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); print([e.name for e in mlflow.search_experiments()])"
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-code dagster instance concurrency get --all
```

Expected:
- `make up` exits 0, and `curl` prints `OK`.
- The in-network call prints `['Default']`. That proves the `mlflow:5000` Host header is accepted and the Postgres backend works.
- The concurrency listing shows `startgg_api`, `spark_lakehouse`, `gold` and `training` with limit 1.

If the in-network call fails with an invalid Host header message, adjust `--allowed-hosts` to the exact host string the error names, and record it. If port 5000 is taken on the host, set `MLFLOW_PORT` in `.env`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/dagster_defs/pools.py infra tests/pipeline/test_infra_config.py
git commit -m "Add an MLflow server, its database and bucket, and the shared Spark pool"
```

---

### Task 10: Dagster silver and gold assets with declarative automation

**Files:**
- Modify: `pipeline/dagster_defs/resources.py`, `pipeline/dagster_defs/assets.py`, `pipeline/dagster_defs/automation.py`, `pipeline/dagster_defs/__init__.py`
- Create: `pipeline/dagster_defs/lakehouse_assets.py`
- Test: `tests/pipeline/test_dagster_lakehouse.py`

**Interfaces:**
- Consumes:
  - `run_silver` summary keys (Task 3) and the silver job CLI (Task 4);
  - `build_gold`, `gold_checks`, `SET_FEATURES`, `PLAYER_RATINGS`, `HEAD_TO_HEAD` (Task 6);
  - pools (Task 9);
  - `LakehouseSettings`, `load_lakehouse_catalog` (Task 1).
- Produces:
  - In `resources.py`:
    - `last_json_line(stdout, job) -> dict`, `parse_silver_summary(stdout) -> dict`;
    - `SparkJobResource` gains `silver_job_path`, `maintenance_job_path`, `silver_lock_dir="/checkpoints/silver"`, `maintenance_timeout_seconds=7200`, `.run_silver_job() -> dict` and `.run_maintenance_job() -> dict`;
    - `LakehouseResource(catalog_type, uri, warehouse, s3_endpoint, s3_access_key, s3_secret_key)` with `.get_catalog()`;
    - `lakehouse_from_env() -> LakehouseResource`;
    - `MlflowResource(tracking_uri="http://mlflow:5000")` with `.activate() -> MlflowClient`.
  - In `lakehouse_assets.py`: assets `silver_startgg` and `gold_startgg`, `drop_rate(summary) -> float`, `MAX_DROP_RATE = 0.10`.
  - In `automation.py`: sensor `lakehouse_automation`.
  - Definitions resource keys: `lakehouse`, `mlflow_tracking`.
  - `bronze_startgg` uses pool `spark_lakehouse`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_dagster_lakehouse.py`:

```python
import json

import dagster as dg
import pytest

from pipeline.dagster_defs import defs
from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import lakehouse_automation
from pipeline.dagster_defs.lakehouse_assets import drop_rate, gold_startgg, silver_startgg
from pipeline.dagster_defs.resources import (
    LakehouseResource,
    SparkJobResource,
    parse_silver_summary,
)
from tests.pipeline.lakehouse_data import silver_set, write_silver

ENTITIES = ("tournaments", "events", "players", "sets", "games")


def silver_summary(orphans=0, dropped=0, read=100, bad_winners=0):
    return {
        "tables": {e: {"inserted": 1, "updated": 0} for e in ENTITIES},
        "dropped": {"too_few_slots": dropped, "missing_player": 0, "not_completed": 0,
                    "unresolved_winner": 0},  # fmt: skip
        "bronze_rows_read": read,
        "bronze_tournament_rows_read": 0,
        "orphan_sets": orphans,
        "winner_not_player": bad_winners,
    }


class FakeSilverSpark(SparkJobResource):
    orphans: int = 0
    dropped: int = 0
    read: int = 100

    def run_silver_job(self):
        return silver_summary(self.orphans, self.dropped, self.read)


def check_results(**fake):
    result = dg.materialize([silver_startgg], resources={"spark_job": FakeSilverSpark(**fake)})
    return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}


def test_clean_silver_summary_passes_every_check():
    assert check_results() == {
        "no_orphan_sets": True,
        "drop_rate_below_10pct": True,
        "winner_is_a_player": True,
    }


def test_drop_rate_above_10pct_fails():
    assert check_results(dropped=11, read=100)["drop_rate_below_10pct"] is False
    assert drop_rate(silver_summary(dropped=11, read=100)) == pytest.approx(0.11)


def test_nothing_read_passes_the_drop_rate_check():
    assert check_results(dropped=0, read=0)["drop_rate_below_10pct"] is True


def test_orphan_sets_fail_the_check():
    assert check_results(orphans=2)["no_orphan_sets"] is False


def test_parse_silver_summary_requires_every_key():
    with pytest.raises(ValueError, match="missing"):
        parse_silver_summary('{"tables": {}}\n')
    assert parse_silver_summary("log\n" + json.dumps(silver_summary()))["orphan_sets"] == 0


def test_silver_job_command_and_environment(monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = json.dumps(silver_summary()) + "\n"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setenv("STARTGG_API_TOKEN", "secret-token-value")
    monkeypatch.setattr("pipeline.dagster_defs.resources.subprocess.run", fake_run)
    SparkJobResource().run_silver_job()
    ((command, kwargs),) = calls
    assert command == [
        "spark-submit",
        "/app/pipeline/spark_jobs/silver_job.py",
        "--lock-dir",
        "/checkpoints/silver",
    ]
    assert "STARTGG_API_TOKEN" not in kwargs["env"]


def test_gold_asset_builds_tables_and_reports_checks(sql_settings, sql_catalog):
    write_silver(
        sql_catalog,
        sets=[silver_set(1, 100, 200, 100, 1_000_000)],
        events=[{"event_id": 10, "tournament_id": 1}],
        tournaments=[{"tournament_id": 1, "name": "T", "num_attendees": 64}],
        players=[{"player_id": 100, "gamer_tag": "A"}, {"player_id": 200, "gamer_tag": "B"}],
    )
    lakehouse = LakehouseResource(
        catalog_type="sql", uri=sql_settings.uri, warehouse=sql_settings.warehouse
    )
    result = dg.materialize([gold_startgg], resources={"lakehouse": lakehouse})
    assert all(e.passed for e in result.get_asset_check_evaluations())
    (materialization,) = result.asset_materializations_for_node("gold_startgg")
    assert materialization.metadata["set_features_rows"].value == 1


def automation(asset):
    return next(iter(asset.specs)).automation_condition


def test_silver_and_gold_are_eager_and_ingest_is_not():
    assert automation(silver_startgg) is not None
    assert automation(gold_startgg) is not None
    assert automation(startgg_raw) is None
    assert automation(bronze_startgg) is None


def test_automation_sensor_runs_by_default_and_targets_silver_and_gold():
    assert lakehouse_automation.default_status == dg.DefaultSensorStatus.RUNNING
    selected = lakehouse_automation.asset_selection.resolve(defs.resolve_asset_graph())
    assert selected == {silver_startgg.key, gold_startgg.key}


def test_spark_jobs_share_one_pool():
    assert bronze_startgg.op.pool == "spark_lakehouse"
    assert silver_startgg.op.pool == "spark_lakehouse"
    assert gold_startgg.op.pool == "gold"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_dagster_lakehouse.py -v`
Expected: collection error, `ImportError` for `lakehouse_automation` or `pipeline.dagster_defs.lakehouse_assets`.

- [ ] **Step 3: Write the implementation**

Replace `pipeline/dagster_defs/resources.py` with the version below. `StartGGResource`, `KafkaResource`, `build_bronze_command`, the bronze command and the bronze error messages are unchanged, so the existing tests keep passing.

```python
"""Dagster resources: start.gg, Kafka, Spark jobs, the lakehouse catalog and MLflow."""

import json
import os
import subprocess
from datetime import datetime

import dagster as dg

from pipeline.ingest.fixtures import FixtureClient
from pipeline.ingest.kafka_publisher import KafkaPublisher
from pipeline.ingest.startgg import StartGGClient
from pipeline.lakehouse import LakehouseSettings, load_lakehouse_catalog

SUMMARY_KEYS = ("tournaments", "sets", "rejects")
SILVER_SUMMARY_KEYS = ("tables", "dropped", "bronze_rows_read", "orphan_sets", "winner_not_player")


class StartGGResource(dg.ConfigurableResource):
    """start.gg API access. With fixture_path set, replays a recorded week instead."""

    token: str
    fixture_path: str | None = None
    min_attendees: int = 50

    def get_client(self):
        if self.fixture_path:
            return FixtureClient(self.fixture_path)
        return StartGGClient(self.token)


class KafkaResource(dg.ConfigurableResource):
    bootstrap_servers: str = "kafka:9092"

    def get_publisher(self) -> KafkaPublisher:
        return KafkaPublisher(self.bootstrap_servers)


def build_bronze_command(
    job_path: str, bootstrap_servers: str, checkpoint_root: str, since: datetime | None = None
) -> list[str]:
    command = [
        "spark-submit",
        job_path,
        "--bootstrap-servers",
        bootstrap_servers,
        "--checkpoint-root",
        checkpoint_root,
    ]
    if since is not None:
        command += ["--since", since.isoformat()]
    return command


def last_json_line(stdout: str, job: str) -> dict:
    """A Spark job's JSON summary: its last non-empty stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{job} printed no summary")
    return json.loads(lines[-1])


def parse_job_summary(stdout: str) -> dict[str, int]:
    """Read the bronze job's JSON summary from the last non-empty stdout line."""
    summary = last_json_line(stdout, "Bronze job")
    missing = set(SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Bronze job summary is missing {sorted(missing)}: {json.dumps(summary)}")
    return {key: int(summary[key]) for key in SUMMARY_KEYS}


def parse_silver_summary(stdout: str) -> dict:
    summary = last_json_line(stdout, "Silver job")
    missing = set(SILVER_SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Silver job summary is missing {sorted(missing)}: {json.dumps(summary)}")
    return summary


class SparkJobResource(dg.ConfigurableResource):
    job_path: str = "/app/pipeline/spark_jobs/bronze_job.py"
    silver_job_path: str = "/app/pipeline/spark_jobs/silver_job.py"
    maintenance_job_path: str = "/app/pipeline/spark_jobs/maintenance_job.py"
    bootstrap_servers: str = "kafka:9092"
    checkpoint_root: str = "/checkpoints/bronze"
    silver_lock_dir: str = "/checkpoints/silver"
    timeout_seconds: int = 3600
    maintenance_timeout_seconds: int = 7200

    def _submit(self, job: str, command: list[str], timeout: int) -> str:
        """Run spark-submit and return stdout; fail the step on timeout or a non-zero exit."""
        # spark-submit never needs the start.gg token, so keep it out of its environment.
        env = {k: v for k, v in os.environ.items() if k != "STARTGG_API_TOKEN"}
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, env=env, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise dg.Failure(description=f"{job} Spark job timed out after {timeout}s") from None
        if result.returncode != 0:
            raise dg.Failure(
                description=f"{job} Spark job exited with code {result.returncode}",
                metadata={"stderr_tail": dg.MetadataValue.text(result.stderr[-5000:])},
            )
        return result.stdout

    def run_bronze_job(self, since: datetime | None = None) -> dict[str, int]:
        """Run the bronze job, counting rows committed after `since` (default: job start)."""
        command = build_bronze_command(
            self.job_path, self.bootstrap_servers, self.checkpoint_root, since
        )
        return parse_job_summary(self._submit("Bronze", command, self.timeout_seconds))

    def run_silver_job(self) -> dict:
        command = ["spark-submit", self.silver_job_path, "--lock-dir", self.silver_lock_dir]
        return parse_silver_summary(self._submit("Silver", command, self.timeout_seconds))

    def run_maintenance_job(self) -> dict:
        command = ["spark-submit", self.maintenance_job_path]
        stdout = self._submit("Maintenance", command, self.maintenance_timeout_seconds)
        return last_json_line(stdout, "Maintenance job")


class LakehouseResource(dg.ConfigurableResource):
    """The Iceberg catalog through PyIceberg. The defaults match the Compose stack."""

    catalog_type: str = "rest"
    uri: str = "http://lakekeeper:8181/catalog"
    warehouse: str = "melee"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    def get_catalog(self):
        return load_lakehouse_catalog(
            LakehouseSettings(
                catalog_type=self.catalog_type,
                uri=self.uri,
                warehouse=self.warehouse,
                s3_endpoint=self.s3_endpoint,
                s3_access_key=self.s3_access_key,
                s3_secret_key=self.s3_secret_key,
            )
        )


def lakehouse_from_env() -> LakehouseResource:
    settings = LakehouseSettings.from_env()
    return LakehouseResource(
        catalog_type=settings.catalog_type,
        uri=settings.uri,
        warehouse=settings.warehouse,
        s3_endpoint=settings.s3_endpoint,
        s3_access_key=settings.s3_access_key,
        # EnvVar keeps the secret out of the config Dagster shows in the UI.
        s3_secret_key=dg.EnvVar("S3_SECRET_KEY") if settings.s3_secret_key else None,
    )


class MlflowResource(dg.ConfigurableResource):
    tracking_uri: str = "http://mlflow:5000"

    def activate(self):
        """Point MLflow's fluent API at this server and return a client for it."""
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        return MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
```

In `pipeline/dagster_defs/assets.py`:
- add `from pipeline.dagster_defs.pools import SPARK_LAKEHOUSE, STARTGG_API`;
- change `pool="startgg_api"` to `pool=STARTGG_API`;
- change `pool="spark_bronze"` to `pool=SPARK_LAKEHOUSE`.

Create `pipeline/dagster_defs/lakehouse_assets.py`:

```python
"""Silver and gold assets. Both run on declarative automation after their upstream."""

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg
from pipeline.dagster_defs.pools import GOLD, SPARK_LAKEHOUSE
from pipeline.dagster_defs.resources import LakehouseResource, SparkJobResource
from pipeline.gold.build import build_gold, gold_checks
from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES

MAX_DROP_RATE = 0.10
SILVER_CHECKS = ("no_orphan_sets", "drop_rate_below_10pct", "winner_is_a_player")
GOLD_CHECKS = ("features_match_silver", "no_null_features", "completed_at_non_decreasing")


def drop_rate(summary: dict) -> float:
    """Dropped bronze set rows / bronze set rows read in this run (0 when nothing was read)."""
    read = summary["bronze_rows_read"]
    return sum(summary["dropped"].values()) / read if read else 0.0


@dg.asset(
    deps=[bronze_startgg],
    pool=SPARK_LAKEHOUSE,
    group_name="silver",
    kinds={"spark", "iceberg"},
    automation_condition=dg.AutomationCondition.eager(),
    retry_policy=dg.RetryPolicy(max_retries=2, delay=30, backoff=dg.Backoff.EXPONENTIAL),
    check_specs=[dg.AssetCheckSpec(name, asset="silver_startgg") for name in SILVER_CHECKS],
)
def silver_startgg(spark_job: SparkJobResource) -> dg.MaterializeResult:
    """New bronze rows, parsed and merged into the silver tables."""
    summary = spark_job.run_silver_job()
    rate = drop_rate(summary)
    metadata = {
        f"{entity}_{kind}": counts[kind]
        for entity, counts in summary["tables"].items()
        for kind in ("inserted", "updated")
    }
    metadata.update(
        {
            "bronze_rows_read": summary["bronze_rows_read"],
            "drop_rate": rate,
            "dropped": dg.MetadataValue.json(summary["dropped"]),
        }
    )
    return dg.MaterializeResult(
        metadata=metadata,
        check_results=[
            dg.AssetCheckResult(
                check_name="no_orphan_sets",
                passed=summary["orphan_sets"] == 0,
                metadata={"orphan_sets": summary["orphan_sets"]},
            ),
            dg.AssetCheckResult(
                check_name="drop_rate_below_10pct",
                passed=rate < MAX_DROP_RATE,
                metadata={"drop_rate": rate, **summary["dropped"]},
            ),
            dg.AssetCheckResult(
                check_name="winner_is_a_player",
                passed=summary["winner_not_player"] == 0,
                metadata={"winner_not_player": summary["winner_not_player"]},
            ),
        ],
    )


@dg.asset(
    deps=[silver_startgg],
    pool=GOLD,
    group_name="gold",
    kinds={"python", "iceberg"},
    automation_condition=dg.AutomationCondition.eager(),
    check_specs=[dg.AssetCheckSpec(name, asset="gold_startgg") for name in GOLD_CHECKS],
)
def gold_startgg(lakehouse: LakehouseResource) -> dg.MaterializeResult:
    """Feature, player and head-to-head tables recomputed from silver."""
    build = build_gold(lakehouse.get_catalog())
    return dg.MaterializeResult(
        metadata={
            "set_features_rows": len(build.frames[SET_FEATURES]),
            "player_ratings_rows": len(build.frames[PLAYER_RATINGS]),
            "head_to_head_rows": len(build.frames[HEAD_TO_HEAD]),
            "silver_sets_snapshot": str(build.silver_sets_snapshot),
            "set_features_snapshot": str(build.snapshots[SET_FEATURES]),
        },
        check_results=[
            dg.AssetCheckResult(check_name=name, passed=passed, metadata=metadata)
            for name, (passed, metadata) in gold_checks(build).items()
        ],
    )
```

In `pipeline/dagster_defs/automation.py`, add the import and the sensor:

```python
from pipeline.dagster_defs.lakehouse_assets import gold_startgg, silver_startgg

# Declarative automation sensors start STOPPED unless told otherwise.
lakehouse_automation = dg.AutomationConditionSensorDefinition(
    "lakehouse_automation",
    target=dg.AssetSelection.assets(silver_startgg, gold_startgg),
    default_status=dg.DefaultSensorStatus.RUNNING,
)
```

Replace `pipeline/dagster_defs/__init__.py` with:

```python
"""Dagster definitions for the pipeline."""

import os

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import (
    bronze_job,
    ingest_job,
    lakehouse_automation,
    land_bronze_after_ingest,
    refresh_recent_weeks,
)
from pipeline.dagster_defs.lakehouse_assets import gold_startgg, silver_startgg
from pipeline.dagster_defs.resources import (
    KafkaResource,
    MlflowResource,
    SparkJobResource,
    StartGGResource,
    lakehouse_from_env,
)

defs = dg.Definitions(
    assets=[startgg_raw, bronze_startgg, silver_startgg, gold_startgg],
    jobs=[ingest_job, bronze_job],
    sensors=[land_bronze_after_ingest, lakehouse_automation],
    schedules=[refresh_recent_weeks],
    resources={
        # EnvVar keeps the token out of the config Dagster shows in the UI.
        "startgg": StartGGResource(
            token=dg.EnvVar("STARTGG_API_TOKEN"),
            fixture_path=os.getenv("STARTGG_FIXTURE_PATH") or None,
        ),
        "kafka": KafkaResource(),
        "spark_job": SparkJobResource(),
        "lakehouse": lakehouse_from_env(),
        "mlflow_tracking": MlflowResource(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        ),
    },
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_dagster_lakehouse.py tests/pipeline/test_dagster_defs.py -v`
Expected: 10 passed in `test_dagster_lakehouse.py`, and every existing test in `test_dagster_defs.py` still passes.

If `defs.resolve_asset_graph()` or `.op.pool` is named differently in Dagster 1.13.25, use the equivalent public accessor and record it. The checked API is `dg.AutomationConditionSensorDefinition(name, *, target, default_status)`, and `OpDefinition.pool` exists.

Run: `uv run dagster definitions validate -m pipeline.dagster_defs`
Expected: validation succeeds.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/dagster_defs tests/pipeline/test_dagster_lakehouse.py
git commit -m "Add silver and gold Dagster assets on eager automation with data-quality checks"
```

---

### Task 11: Training asset, daily schedule and the model check

**Files:**
- Create: `pipeline/dagster_defs/training.py`
- Modify: `pipeline/dagster_defs/__init__.py`
- Test: `tests/pipeline/test_dagster_training.py`

**Interfaces:**
- Consumes:
  - `train_and_register`, `champion_gold_snapshot` (Task 8);
  - `load_features` (Task 7);
  - `LakehouseResource`, `MlflowResource`, `gold_startgg`, `TRAINING` (Tasks 9–10);
  - `replace_table`, `to_arrow`, `SET_FEATURES_SCHEMA` (Task 6).
- Produces:
  - `TrainingConfig(split_at="2025-07-01T00:00:00+00:00", split_quantile=None, min_sets=5)`;
  - asset `matchup_model`, with check `model_beats_baseline` and metadata keys `model_version`, `mlflow_run_id`, `gold_snapshot_id`, `promotion_decision`, `promoted`, `xgboost_log_loss`, `baseline_log_loss`, `champion_log_loss`, `champion_version_before`, `xgboost_accuracy`, `xgboost_roc_auc`;
  - job `train_matchup_model`;
  - schedule `daily_training`;
  - `should_train(gold_snapshot: int | None, champion_snapshot: str | None) -> str | None`, which returns a skip reason.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_dagster_training.py`:

```python
import dagster as dg
import pytest

pytest.importorskip("mlflow")

from pipeline.dagster_defs.resources import LakehouseResource, MlflowResource  # noqa: E402
from pipeline.dagster_defs.training import (  # noqa: E402
    daily_training,
    matchup_model,
    should_train,
)
from pipeline.gold.tables import SET_FEATURES, SET_FEATURES_SCHEMA  # noqa: E402
from pipeline.gold.writer import replace_table, to_arrow  # noqa: E402
from pipeline.lakehouse import current_snapshot_id  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


@pytest.mark.parametrize(
    ("gold", "champion", "expected"),
    [
        (None, None, "gold.set_features has no data yet"),
        (5, "5", "The champion was already trained on gold snapshot 5"),
        (6, "5", None),
    ],
)
def test_should_train(gold, champion, expected):
    assert should_train(gold, champion) == expected


@pytest.fixture
def resources(sql_settings, sql_catalog, mlflow_tracking):
    replace_table(
        sql_catalog, SET_FEATURES, to_arrow(synthetic_gold_features(), SET_FEATURES_SCHEMA), {}
    )
    return {
        "lakehouse": LakehouseResource(
            catalog_type="sql", uri=sql_settings.uri, warehouse=sql_settings.warehouse
        ),
        "mlflow_tracking": MlflowResource(tracking_uri=mlflow_tracking),
    }


RUN_CONFIG = {"ops": {"matchup_model": {"config": {"split_quantile": 0.7}}}}


def test_matchup_model_registers_and_reports_promotion(resources, sql_catalog):
    result = dg.materialize([matchup_model], resources=resources, run_config=RUN_CONFIG)
    (materialization,) = result.asset_materializations_for_node("matchup_model")
    metadata = {key: value.value for key, value in materialization.metadata.items()}
    assert metadata["promotion_decision"] == "no_champion"
    assert metadata["promoted"] is True
    assert metadata["model_version"] == "1"
    assert metadata["gold_snapshot_id"] == str(current_snapshot_id(sql_catalog, SET_FEATURES))
    (check,) = result.get_asset_check_evaluations()
    assert check.check_name == "model_beats_baseline" and check.passed


def test_schedule_trains_once_per_gold_snapshot(resources, sql_catalog):
    gold = current_snapshot_id(sql_catalog, SET_FEATURES)
    request = daily_training(dg.build_schedule_context(resources=resources))
    assert isinstance(request, dg.RunRequest) and request.run_key == f"gold-{gold}"
    dg.materialize([matchup_model], resources=resources, run_config=RUN_CONFIG)
    skip = daily_training(dg.build_schedule_context(resources=resources))
    assert isinstance(skip, dg.SkipReason)


def test_training_schedule_runs_daily_at_7_pacific():
    assert daily_training.cron_schedule == "0 7 * * *"
    assert daily_training.execution_timezone == "America/Los_Angeles"
    assert daily_training.default_status == dg.DefaultScheduleStatus.RUNNING
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_dagster_training.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pipeline.dagster_defs.training'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/dagster_defs/training.py`:

```python
"""Daily model training on gold, registered in MLflow with a gated champion alias."""

import dagster as dg
import pandas as pd

from pipeline.dagster_defs.lakehouse_assets import gold_startgg
from pipeline.dagster_defs.pools import TRAINING
from pipeline.dagster_defs.resources import LakehouseResource, MlflowResource
from pipeline.gold.tables import SET_FEATURES
from pipeline.lakehouse import current_snapshot_id


class TrainingConfig(dg.Config):
    split_at: str = "2025-07-01T00:00:00+00:00"
    # Set for small datasets (the smoke test): split at this quantile of completed_at.
    split_quantile: float | None = None
    min_sets: int = 5


@dg.asset(
    deps=[gold_startgg],
    pool=TRAINING,
    group_name="ml",
    kinds={"xgboost", "mlflow"},
    check_specs=[dg.AssetCheckSpec("model_beats_baseline", asset="matchup_model")],
)
def matchup_model(
    config: TrainingConfig, lakehouse: LakehouseResource, mlflow_tracking: MlflowResource
) -> dg.MaterializeResult:
    """XGBoost trained on gold.set_features, registered as a new MLflow model version."""
    from pipeline.ml.registry import train_and_register
    from pipeline.ml.train import load_features

    mlflow_tracking.activate()
    features, snapshot = load_features(lakehouse.get_catalog())
    outcome = train_and_register(
        features,
        snapshot,
        split_at=pd.Timestamp(config.split_at),
        split_quantile=config.split_quantile,
        min_sets=config.min_sets,
    )
    xgb, baseline = outcome.xgboost, outcome.baseline
    beats_baseline = bool(xgb.log_loss_val < baseline.log_loss_val)
    return dg.MaterializeResult(
        metadata={
            "model_version": outcome.version,
            "mlflow_run_id": outcome.run_id,
            "gold_snapshot_id": "" if snapshot is None else str(snapshot),
            "promotion_decision": outcome.decision.reason,
            "promoted": outcome.decision.promote,
            "xgboost_log_loss": xgb.log_loss_val,
            "baseline_log_loss": baseline.log_loss_val,
            "champion_log_loss": "none"
            if outcome.champion_log_loss is None
            else outcome.champion_log_loss,
            "champion_version_before": outcome.champion_version or "none",
            "xgboost_accuracy": xgb.accuracy,
            "xgboost_roc_auc": xgb.roc_auc,
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="model_beats_baseline",
                passed=beats_baseline,
                metadata={
                    "xgboost_log_loss": xgb.log_loss_val,
                    "baseline_log_loss": baseline.log_loss_val,
                },
            )
        ],
    )


train_job = dg.define_asset_job("train_matchup_model", selection=[matchup_model])


def should_train(gold_snapshot: int | None, champion_snapshot: str | None) -> str | None:
    """A reason to skip today's training, or None to train."""
    if gold_snapshot is None:
        return "gold.set_features has no data yet"
    if champion_snapshot == str(gold_snapshot):
        return f"The champion was already trained on gold snapshot {gold_snapshot}"
    return None


@dg.schedule(
    job=train_job,
    cron_schedule="0 7 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def daily_training(
    context: dg.ScheduleEvaluationContext,
    lakehouse: LakehouseResource,
    mlflow_tracking: MlflowResource,
):
    """Train when gold has changed since the champion was trained."""
    from pipeline.ml.registry import champion_gold_snapshot

    catalog = lakehouse.get_catalog()
    gold = current_snapshot_id(catalog, SET_FEATURES) if catalog.table_exists(SET_FEATURES) else None
    skip = should_train(gold, champion_gold_snapshot(mlflow_tracking.activate()))
    if skip:
        return dg.SkipReason(skip)
    return dg.RunRequest(run_key=f"gold-{gold}")
```

In `pipeline/dagster_defs/__init__.py`:
- import `daily_training`, `matchup_model` and `train_job` from `pipeline.dagster_defs.training`;
- add `matchup_model` to `assets`, `train_job` to `jobs` and `daily_training` to `schedules`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_dagster_training.py -v`
Expected: 6 passed, with no warnings.

Run: `uv run dagster definitions validate -m pipeline.dagster_defs` and `uv run pytest`
Expected: validation succeeds, and every test passes.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/dagster_defs tests/pipeline/test_dagster_training.py
git commit -m "Train and register the matchup model daily when gold changes"
```

---

### Task 12: Weekly Iceberg table maintenance

**Files:**
- Create: `pipeline/spark_jobs/maintenance_job.py`, `pipeline/dagster_defs/maintenance.py`
- Modify: `pipeline/dagster_defs/__init__.py`
- Test: `tests/pipeline/test_maintenance_sql.py`, `tests/pipeline/test_maintenance_job.py` (spark), `tests/pipeline/test_dagster_maintenance.py`

**Interfaces:**
- Consumes: `BRONZE_TABLES`, `REJECTS_TABLE`, `MERGE_ORDER`, `table_name`, `GOLD_TABLES`, `SparkJobResource.run_maintenance_job`, `SPARK_LAKEHOUSE`.
- Produces:
  - `pipeline.spark_jobs.maintenance_job`:
    - `MAINTAINED_TABLES`;
    - `maintenance_sql(catalog, table, now) -> dict[str, str]`;
    - `maintain_table(spark, catalog, table, now) -> {"files_before", "files_after", "errors"}`;
    - `run_maintenance(spark, tables=MAINTAINED_TABLES, now=None) -> {"tables": {...}}`;
    - `fatal_errors(summary) -> list[str]`;
    - a `spark-submit` entrypoint that exits 1 on fatal errors.
  - `pipeline.dagster_defs.maintenance`: op `run_lakehouse_maintenance` (pool `spark_lakehouse`), job `lakehouse_maintenance`, schedule `weekly_maintenance`, and `maintenance_totals(summary) -> tuple[int, int]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_maintenance_sql.py`:

```python
from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.maintenance_job import fatal_errors, maintenance_sql  # noqa: E402


def test_maintenance_sql_uses_the_catalog_and_utc_literals():
    now = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
    sql = maintenance_sql("lakekeeper", "bronze.startgg_sets", now)
    assert sql["rewrite_data_files"].startswith("CALL lakekeeper.system.rewrite_data_files(")
    assert "'target-file-size-bytes', '134217728'" in sql["rewrite_data_files"]
    assert "older_than => TIMESTAMP '2026-10-04 10:00:00'" in sql["expire_snapshots"]
    assert "retain_last => 5" in sql["expire_snapshots"]
    assert "older_than => TIMESTAMP '2026-10-08 10:00:00'" in sql["remove_orphan_files"]


def test_only_compaction_and_expiry_failures_are_fatal():
    summary = {
        "tables": {
            "a": {"files_before": 3, "files_after": 1, "errors": {"remove_orphan_files": "x"}},
            "b": {"files_before": 3, "files_after": 3, "errors": {"expire_snapshots": "y"}},
        }
    }
    assert fatal_errors(summary) == ["b: expire_snapshots"]
```

Create `tests/pipeline/test_maintenance_job.py`:

```python
import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.maintenance_job import run_maintenance  # noqa: E402


def test_maintenance_compacts_small_files(spark):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS maint")
    spark.sql("DROP TABLE IF EXISTS maint.events")
    spark.sql("CREATE TABLE maint.events (id BIGINT) USING iceberg")
    for i in range(6):
        spark.createDataFrame([(i,)], "id BIGINT").writeTo("maint.events").append()
    summary = run_maintenance(spark, tables=("maint.events",))
    result = summary["tables"]["maint.events"]
    assert result["errors"] == {}
    assert result["files_before"] == 6
    assert result["files_after"] < result["files_before"]
    assert spark.table("maint.events").count() == 6


def test_missing_tables_are_skipped(spark):
    assert run_maintenance(spark, tables=("maint.does_not_exist",)) == {"tables": {}}
```

Create `tests/pipeline/test_dagster_maintenance.py`:

```python
import dagster as dg

from pipeline.dagster_defs.maintenance import (
    lakehouse_maintenance,
    maintenance_totals,
    run_lakehouse_maintenance,
    weekly_maintenance,
)
from pipeline.dagster_defs.resources import SparkJobResource

SUMMARY = {
    "tables": {
        "bronze.startgg_sets": {"files_before": 40, "files_after": 2, "errors": {}},
        "gold.set_features": {"files_before": 3, "files_after": 3, "errors": {}},
    }
}


class FakeMaintenanceSpark(SparkJobResource):
    def run_maintenance_job(self):
        return SUMMARY


def test_maintenance_job_reports_file_counts():
    result = lakehouse_maintenance.execute_in_process(resources={"spark_job": FakeMaintenanceSpark()})
    assert result.success
    assert result.output_for_node("run_lakehouse_maintenance") == SUMMARY
    assert maintenance_totals(SUMMARY) == (43, 5)


def test_weekly_maintenance_runs_sunday_3am_pacific():
    assert weekly_maintenance.cron_schedule == "0 3 * * 0"
    assert weekly_maintenance.execution_timezone == "America/Los_Angeles"
    assert weekly_maintenance.default_status == dg.DefaultScheduleStatus.RUNNING


def test_maintenance_uses_the_spark_pool():
    assert run_lakehouse_maintenance.pool == "spark_lakehouse"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_maintenance_sql.py tests/pipeline/test_dagster_maintenance.py -v`
Expected: collection errors, `ModuleNotFoundError` for `maintenance_job` and `pipeline.dagster_defs.maintenance`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/spark_jobs/maintenance_job.py`:

```python
"""Weekly Iceberg maintenance: compact small files, expire old snapshots, remove orphans.

Run with spark-submit. The last line of stdout is a JSON summary. The exit code is 1
if compaction or snapshot expiry failed on any table; orphan-file cleanup failures
are reported but not fatal.
"""

import json
import sys
from datetime import UTC, datetime, timedelta

from pyspark.sql import SparkSession

from pipeline.gold.tables import GOLD_TABLES
from pipeline.spark_jobs.bronze import BRONZE_TABLES, REJECTS_TABLE
from pipeline.spark_jobs.silver import MERGE_ORDER, table_name

MAINTAINED_TABLES = (
    *BRONZE_TABLES.values(),
    REJECTS_TABLE,
    *(table_name(entity) for entity in MERGE_ORDER),
    *GOLD_TABLES,
)
TARGET_FILE_SIZE_BYTES = 128 * 1024 * 1024
SNAPSHOT_RETENTION = timedelta(days=7)
RETAIN_LAST = 5
ORPHAN_MIN_AGE = timedelta(days=3)
FATAL_PROCEDURES = frozenset({"rewrite_data_files", "expire_snapshots"})


def _literal(moment: datetime) -> str:
    # Spark reads TIMESTAMP literals in the session time zone, which the job pins to UTC.
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def maintenance_sql(catalog: str, table: str, now: datetime) -> dict[str, str]:
    expire_before = _literal(now - SNAPSHOT_RETENTION)
    orphans_before = _literal(now - ORPHAN_MIN_AGE)
    return {
        "rewrite_data_files": (
            f"CALL {catalog}.system.rewrite_data_files(table => '{table}', "
            f"options => map('target-file-size-bytes', '{TARGET_FILE_SIZE_BYTES}'))"
        ),
        "expire_snapshots": (
            f"CALL {catalog}.system.expire_snapshots(table => '{table}', "
            f"older_than => TIMESTAMP '{expire_before}', retain_last => {RETAIN_LAST})"
        ),
        "remove_orphan_files": (
            f"CALL {catalog}.system.remove_orphan_files(table => '{table}', "
            f"older_than => TIMESTAMP '{orphans_before}')"
        ),
    }


def file_count(spark: SparkSession, table: str) -> int:
    return int(spark.sql(f"SELECT count(*) AS n FROM {table}.files").first().n)


def maintain_table(spark: SparkSession, catalog: str, table: str, now: datetime) -> dict:
    before = file_count(spark, table)
    errors = {}
    for procedure, sql in maintenance_sql(catalog, table, now).items():
        try:
            spark.sql(sql).collect()
        except Exception as exc:  # report every table, then decide what is fatal
            errors[procedure] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"files_before": before, "files_after": file_count(spark, table), "errors": errors}


def run_maintenance(
    spark: SparkSession, tables: tuple[str, ...] = MAINTAINED_TABLES, now: datetime | None = None
) -> dict:
    now = now or datetime.now(UTC)
    catalog = spark.conf.get("spark.sql.defaultCatalog")
    return {
        "tables": {
            table: maintain_table(spark, catalog, table, now)
            for table in tables
            if spark.catalog.tableExists(table)
        }
    }


def fatal_errors(summary: dict) -> list[str]:
    return [
        f"{table}: {procedure}"
        for table, result in summary["tables"].items()
        for procedure in result["errors"]
        if procedure in FATAL_PROCEDURES
    ]


def main() -> None:
    spark = SparkSession.builder.appName("lakehouse-maintenance").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    summary = run_maintenance(spark)
    spark.stop()
    print(json.dumps(summary))
    fatal = fatal_errors(summary)
    if fatal:
        print(f"maintenance failed: {fatal}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Create `pipeline/dagster_defs/maintenance.py`:

```python
"""Weekly Iceberg maintenance job (compaction, snapshot expiry, orphan cleanup)."""

import dagster as dg

from pipeline.dagster_defs.pools import SPARK_LAKEHOUSE
from pipeline.dagster_defs.resources import SparkJobResource


def maintenance_totals(summary: dict) -> tuple[int, int]:
    """(data files before, data files after) across every maintained table."""
    tables = summary["tables"].values()
    return sum(t["files_before"] for t in tables), sum(t["files_after"] for t in tables)


@dg.op(pool=SPARK_LAKEHOUSE)
def run_lakehouse_maintenance(
    context: dg.OpExecutionContext, spark_job: SparkJobResource
) -> dg.Output[dict]:
    summary = spark_job.run_maintenance_job()
    for table, result in summary["tables"].items():
        if result["errors"]:
            context.log.warning(f"{table}: {result['errors']}")
    before, after = maintenance_totals(summary)
    return dg.Output(
        summary,
        metadata={
            "files_before": before,
            "files_after": after,
            "tables": dg.MetadataValue.json(summary["tables"]),
        },
    )


@dg.job
def lakehouse_maintenance():
    run_lakehouse_maintenance()


weekly_maintenance = dg.ScheduleDefinition(
    name="weekly_maintenance",
    job=lakehouse_maintenance,
    cron_schedule="0 3 * * 0",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
```

In `pipeline/dagster_defs/__init__.py`:
- import `lakehouse_maintenance` and `weekly_maintenance` from `pipeline.dagster_defs.maintenance`;
- add `lakehouse_maintenance` to `jobs` and `weekly_maintenance` to `schedules`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_maintenance_sql.py tests/pipeline/test_dagster_maintenance.py -v`
Expected: 2 and 3 passed.

Run: `$MAKE test-spark`
Expected: `test_maintenance_job.py` shows 2 passed, and all other Spark tests pass.

Run: `uv run dagster definitions validate -m pipeline.dagster_defs`
Expected: validation succeeds.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline && uv run ruff check pipeline tests
git add pipeline/spark_jobs/maintenance_job.py pipeline/dagster_defs tests/pipeline/test_maintenance_sql.py tests/pipeline/test_maintenance_job.py tests/pipeline/test_dagster_maintenance.py
git commit -m "Add weekly Iceberg compaction, snapshot expiry and orphan-file cleanup"
```

---

### Task 13: The app on gold and the MLflow champion

**Files:**
- Create: `pipeline/serving/__init__.py` (empty), `pipeline/serving/gold_reader.py`, `pipeline/serving/champion.py`
- Modify: `app.py`, `Dockerfile`, `infra/docker-compose.yml`, `infra/docker-compose.smoke.yml`, `tests/pipeline/test_infra_config.py`
- Test: `tests/pipeline/test_gold_reader.py`, `tests/pipeline/test_champion.py`, `tests/pipeline/test_app.py`

**Interfaces:**
- Consumes:
  - `read_table`, `load_lakehouse_catalog` (Task 1);
  - `PLAYER_RATINGS`, `HEAD_TO_HEAD`, `PLAYER_RATINGS_SCHEMA` (Task 6);
  - `MODEL_NAME`, `CHAMPION_ALIAS`, `SHAP_ARTIFACT` (Task 7);
  - `CORE_FEATURES`;
  - `train_and_register` (tests only).
- Produces:
  - `pipeline.serving.gold_reader`:
    - `load_player_ratings(catalog, min_sets=5) -> pd.DataFrame`, which adds `total_sets` and `win_rate` and sorts by `elo` descending;
    - `load_head_to_head(catalog) -> pd.DataFrame`;
    - `lookup_h2h(h2h, pid1, pid2) -> tuple[int, int]`.
  - `pipeline.serving.champion`:
    - `Champion(model, version: str, shap_metadata: dict)`;
    - `ChampionNotFound(RuntimeError)`;
    - `load_champion(tracking_uri) -> Champion`.
  - Compose service `app` (image `melee-app:dev`, port `${APP_PORT:-8501}`).

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_gold_reader.py`:

```python
import pandas as pd

from pipeline.gold.tables import PLAYER_RATINGS, PLAYER_RATINGS_SCHEMA
from pipeline.gold.writer import replace_table, to_arrow
from pipeline.serving.gold_reader import load_player_ratings, lookup_h2h

TS = pd.Timestamp("2025-01-06", tz="UTC")


def test_player_ratings_add_win_rate_and_filter_min_sets(sql_catalog):
    ratings = pd.DataFrame(
        [
            {"player_id": 1, "gamer_tag": "A", "elo": 1600.0, "sets_played": 10, "recent_wr": 0.6,
             "wins": 6, "losses": 4, "last_played_at": TS},
            {"player_id": 2, "gamer_tag": "B", "elo": 1700.0, "sets_played": 8, "recent_wr": 0.75,
             "wins": 6, "losses": 2, "last_played_at": TS},
            {"player_id": 3, "gamer_tag": "C", "elo": 1800.0, "sets_played": 2, "recent_wr": 1.0,
             "wins": 2, "losses": 0, "last_played_at": TS},
        ]
    )  # fmt: skip
    replace_table(sql_catalog, PLAYER_RATINGS, to_arrow(ratings, PLAYER_RATINGS_SCHEMA), {})
    players = load_player_ratings(sql_catalog)
    assert players["gamer_tag"].tolist() == ["B", "A"]  # C has too few sets; sorted by Elo
    assert players["total_sets"].tolist() == [8, 10]
    assert players["win_rate"].tolist() == [0.75, 0.6]


H2H = pd.DataFrame([{"player_a_id": 1, "player_b_id": 2, "a_wins": 3, "b_wins": 1, "total": 4}])


def test_lookup_h2h_in_both_orders():
    assert lookup_h2h(H2H, 1, 2) == (3, 1)
    assert lookup_h2h(H2H, 2, 1) == (1, 3)


def test_lookup_h2h_for_players_who_never_met():
    assert lookup_h2h(H2H, 1, 99) == (0, 0)
```

Create `tests/pipeline/test_champion.py`:

```python
import pytest

pytest.importorskip("mlflow")

from pipeline.ml.registry import train_and_register  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from pipeline.serving.champion import ChampionNotFound, load_champion  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_load_champion_returns_the_aliased_model(mlflow_tracking):
    features = synthetic_gold_features()
    outcome = train_and_register(features, 7, split_quantile=0.7)
    champion = load_champion(mlflow_tracking)
    assert champion.version == outcome.version
    assert champion.shap_metadata["feature_names"] == CORE_FEATURES
    assert champion.model.predict_proba(features[CORE_FEATURES].head(3)).shape == (3, 2)


def test_missing_champion_raises_champion_not_found(mlflow_tracking):
    with pytest.raises(ChampionNotFound, match="No 'champion' version of melee-matchup"):
        load_champion(mlflow_tracking)
```

Create `tests/pipeline/test_app.py`:

```python
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"


def test_app_reports_an_unreachable_catalog(monkeypatch):
    monkeypatch.setenv("LAKEKEEPER_URL", "http://127.0.0.1:9/catalog")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    (error,) = app.error
    assert error.value.startswith("The data platform is unavailable: the lakehouse catalog")
```

In `tests/pipeline/test_infra_config.py`:
- change the skip in `test_compose_images_are_pinned` to `if image in ("melee-pipeline:dev", "melee-app:dev"):`;
- add:

```python
def test_app_service_reads_gold_and_mlflow():
    app = _service("app")
    assert "image: melee-app:dev" in app
    assert "LAKEKEEPER_URL: http://lakekeeper:8181/catalog" in app
    assert "MLFLOW_TRACKING_URI: http://mlflow:5000" in app
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "data/" not in dockerfile
    assert "COPY pipeline/serving/" in dockerfile
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_gold_reader.py tests/pipeline/test_champion.py tests/pipeline/test_app.py -v`
Expected: collection errors, `ModuleNotFoundError: No module named 'pipeline.serving'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/serving/__init__.py` as an empty file.

Create `pipeline/serving/gold_reader.py`:

```python
"""Gold tables the app reads, through PyIceberg."""

import pandas as pd
from pyiceberg.catalog import Catalog

from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS
from pipeline.lakehouse import read_table

PLAYER_COLUMNS = (
    "player_id",
    "gamer_tag",
    "elo",
    "sets_played",
    "recent_wr",
    "wins",
    "losses",
    "last_played_at",
)
H2H_COLUMNS = ("player_a_id", "player_b_id", "a_wins", "b_wins", "total", "last_played_at")


def load_player_ratings(catalog: Catalog, min_sets: int = 5) -> pd.DataFrame:
    """Players with at least min_sets sets, strongest first, with career totals."""
    players = read_table(catalog, PLAYER_RATINGS, PLAYER_COLUMNS)
    players = players[players["sets_played"] >= min_sets].copy()
    players["total_sets"] = players["wins"] + players["losses"]
    players["win_rate"] = (players["wins"] / players["total_sets"]).where(
        players["total_sets"] > 0, 0.0
    )
    return players.sort_values("elo", ascending=False).reset_index(drop=True)


def load_head_to_head(catalog: Catalog) -> pd.DataFrame:
    return read_table(catalog, HEAD_TO_HEAD, H2H_COLUMNS)


def lookup_h2h(h2h: pd.DataFrame, pid1: int, pid2: int) -> tuple[int, int]:
    """(pid1's wins, pid2's wins) against each other."""
    a, b = min(pid1, pid2), max(pid1, pid2)
    row = h2h[(h2h["player_a_id"] == a) & (h2h["player_b_id"] == b)]
    if row.empty:
        return 0, 0
    record = row.iloc[0]
    a_wins, b_wins = int(record["a_wins"]), int(record["b_wins"])
    return (a_wins, b_wins) if pid1 == a else (b_wins, a_wins)
```

Create `pipeline/serving/champion.py`:

```python
"""Load the MLflow champion model and its SHAP metadata for the app."""

from dataclasses import dataclass

from pipeline.ml.names import CHAMPION_ALIAS, MODEL_NAME, SHAP_ARTIFACT
from pipeline.model_features import CORE_FEATURES


class ChampionNotFound(RuntimeError):
    """No model version holds the champion alias yet."""


@dataclass(frozen=True)
class Champion:
    model: object
    version: str
    shap_metadata: dict


def load_champion(tracking_uri: str) -> Champion:
    import mlflow
    import mlflow.xgboost
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        raise ChampionNotFound(
            f"No '{CHAMPION_ALIAS}' version of {MODEL_NAME} is registered yet"
        ) from exc
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    metadata = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{SHAP_ARTIFACT}")
    if metadata.get("feature_names") != CORE_FEATURES:
        raise ValueError(
            f"Champion v{version.version} was trained on {metadata.get('feature_names')}, "
            f"not {CORE_FEATURES}"
        )
    return Champion(model=model, version=str(version.version), shap_metadata=metadata)
```

In `app.py`, replace everything from the top of the file down to and including the line `explainer = load_shap_explainer()` with the block below. The rest of the file (the sidebar, both pages and the SHAP chart) is unchanged.

```python
"""MeleeMatchup — Streamlit App

Interactive predictions for competitive Melee tournament sets, served from the
gold tables and the MLflow champion model.
"""

import math
import os

import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from pipeline.lakehouse import load_lakehouse_catalog
from pipeline.model_features import CORE_FEATURES
from pipeline.serving.champion import load_champion
from pipeline.serving.gold_reader import load_head_to_head, load_player_ratings, lookup_h2h

UNAVAILABLE = "The data platform is unavailable"


@st.cache_resource(ttl=3600)
def load_gold():
    catalog = load_lakehouse_catalog()
    return load_player_ratings(catalog), load_head_to_head(catalog)


@st.cache_resource(ttl=3600)
def load_model():
    champion = load_champion(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    return champion.model, shap.TreeExplainer(champion.model)


FEATURE_LABELS = {
    "elo_diff": "Elo Rating Gap",
    "p1_expected": "Expected Win Rate (Elo)",
    "p1_sets_played": "P1 Experience",
    "p2_sets_played": "P2 Experience",
    "recent_wr_diff": "Recent Form Gap",
    "p1_h2h_wr": "Head-to-Head Record",
    "h2h_total": "H2H Sets Played",
    "seed_diff": "Seed Differential",
    "num_attendees": "Tournament Size",
}


def elo_expected(ra, rb):
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MeleeMatchup", page_icon="🎮", layout="wide")

try:
    players, h2h_df = load_gold()
except Exception as exc:  # show which part of the platform is down, never a traceback
    st.error(f"{UNAVAILABLE}: the lakehouse catalog could not be read ({type(exc).__name__}).")
    st.stop()
try:
    model, explainer = load_model()
except Exception as exc:
    st.error(f"{UNAVAILABLE}: the MLflow champion model could not be loaded ({type(exc).__name__}).")
    st.stop()
```

Further down in `app.py`, replace `get_h2h(h2h_df, p1["player_id"], p2["player_id"])` with `lookup_h2h(h2h_df, p1["player_id"], p2["player_id"])`.

Replace the root `Dockerfile` with:

```dockerfile
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-cache --extra ml --extra app

# Only the modules the app imports; it reads gold and MLflow, never local files.
COPY pipeline/__init__.py pipeline/model_features.py pipeline/lakehouse.py pipeline/
COPY pipeline/gold/__init__.py pipeline/gold/tables.py pipeline/gold/
COPY pipeline/ml/__init__.py pipeline/ml/names.py pipeline/ml/
COPY pipeline/serving/ pipeline/serving/
COPY app.py .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

In `infra/docker-compose.yml`, add this service after `dagster-daemon`. If Task 6 needed the explicit-S3 fallback, add the same three `S3_*` variables here too.

```yaml
  app:
    image: melee-app:dev
    build:
      context: ..
      dockerfile: Dockerfile
    environment:
      LAKEKEEPER_URL: http://lakekeeper:8181/catalog
      LAKEHOUSE_WAREHOUSE: melee
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "${APP_PORT:-8501}:8501"
    mem_limit: 768m
    depends_on:
      lakekeeper-warehouse:
        condition: service_completed_successfully
      mlflow:
        condition: service_healthy
```

In `infra/docker-compose.smoke.yml`, add:

```yaml
  app:
    ports: !reset []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_gold_reader.py tests/pipeline/test_champion.py tests/pipeline/test_app.py tests/pipeline/test_infra_config.py -v`
Expected:
- `test_gold_reader.py`: 3 passed;
- `test_champion.py`: 2 passed;
- `test_app.py`: 1 passed;
- `test_infra_config.py`: all passed.

Build the app image once to prove it no longer needs `data/`:

```bash
docker build -t melee-app:check -f Dockerfile . && docker rmi melee-app:check
```

Expected: the build succeeds from a clean checkout, with no `data/` directory.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline app.py && uv run ruff check pipeline tests app.py
git add pipeline/serving app.py Dockerfile infra tests/pipeline/test_gold_reader.py tests/pipeline/test_champion.py tests/pipeline/test_app.py tests/pipeline/test_infra_config.py
git ls-files --eol Dockerfile infra | grep -v "w/lf" || echo "all LF"
git commit -m "Serve the app from gold tables and the MLflow champion model"
```

---

### Task 14: Retire SQLite and `src/`

**Files:**
- Delete: `src/` (every file), `tests/test_database.py`
- Modify: `pyproject.toml` (drop `[project.scripts]`), `Makefile` (full replacement below), `.github/workflows/ci.yml` (lint scope), `.dockerignore`, `pipeline/ingest/queries.py` (docstring)
- Test: `tests/pipeline/test_retirement.py`

**Interfaces:**
- Consumes: everything that replaced `src/` (Tasks 5–8, 13).
- Produces:
  - `make train` / `make tune`: run inside `dagster-code` against gold.
  - `make app`: starts the `app` service.
  - CI lint scope: `tests/ app.py pipeline/ scripts/`.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_retirement.py`:

```python
import re
from pathlib import Path

DATA_PATH = re.compile(r"(?<![\w/])data/(raw|processed|app)\b")


def test_src_package_is_gone():
    assert not Path("src").exists()


def test_no_code_reads_sqlite_or_local_data_files():
    sources = [Path("app.py"), *Path("pipeline").rglob("*.py"), *Path("scripts").rglob("*.py")]
    offenders = [
        str(path)
        for path in sources
        if "sqlite3" in (text := path.read_text(encoding="utf-8")) or DATA_PATH.search(text)
    ]
    assert offenders == []


def test_build_files_no_longer_mention_src():
    for path in ("Makefile", "pyproject.toml", ".github/workflows/ci.yml", "Dockerfile"):
        assert not re.search(r"\bsrc[/.]", Path(path).read_text(encoding="utf-8")), path
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/pipeline/test_retirement.py -v`
Expected: 2 failed and 1 passed:
- `test_src_package_is_gone` fails because `src` exists;
- `test_build_files_no_longer_mention_src` fails on `Makefile`;
- `test_no_code_reads_sqlite_or_local_data_files` passes, because `app.py` no longer reads `data/app` after Task 13.

- [ ] **Step 3: Retire the files and update the build**

```bash
git rm -r src tests/test_database.py
```

In `pyproject.toml`, delete these two lines:

```toml
[project.scripts]
melee-collect = "src.collect:main"
```

Replace the `Makefile` with:

```makefile
SHELL := bash
.PHONY: install train tune app test lint format docker-build clean up down nuke ps logs check-lakehouse test-spark bronze-once silver-once gold-once smoke

# ── Setup ──────────────────────────────────────────────
install:
	uv sync --all-extras

# ── Local stack ────────────────────────────────────────
# .env is optional so a fresh clone can still run compose with the defaults.
COMPOSE := docker compose -f infra/docker-compose.yml $(if $(wildcard .env),--env-file .env)

up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

nuke:  ## Stops the stack and DELETES all Kafka, Silo, Postgres and checkpoint data
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps -a

logs:
	$(COMPOSE) logs -f --tail=100

check-lakehouse:
	$(COMPOSE) run --rm pipeline-tools spark-submit infra/spark/check_lakehouse.py
	out="$$($(COMPOSE) exec -T trino trino --execute "SELECT * FROM lakekeeper.healthcheck.ping")"; \
	  echo "$$out"; \
	  grep -q '"1","ok"' <<<"$$out" \
	  || { echo 'trino lakehouse check FAILED: expected row "1","ok"' >&2; exit 1; }
	@echo "trino lakehouse check ok"
	$(COMPOSE) exec -T trino trino --execute "DROP TABLE lakekeeper.healthcheck.ping"
	$(COMPOSE) exec -T trino trino --execute "DROP SCHEMA lakekeeper.healthcheck"

test-spark:
	$(COMPOSE) run --rm --no-deps pipeline-tools pytest -m spark -v tests/pipeline

smoke:
	bash scripts/smoke.sh

bronze-once:
	$(COMPOSE) run --rm pipeline-tools spark-submit pipeline/spark_jobs/bronze_job.py \
		--bootstrap-servers kafka:9092 --checkpoint-root /checkpoints/bronze

silver-once:
	$(COMPOSE) run --rm pipeline-tools spark-submit pipeline/spark_jobs/silver_job.py \
		--lock-dir /checkpoints/silver

gold-once:
	$(COMPOSE) run --rm pipeline-tools python -m pipeline.gold.build

# ── Modeling ───────────────────────────────────────────
# Training reads gold through the lakehouse catalog, so it runs inside the stack.
train:
	$(COMPOSE) exec -T dagster-code python -m pipeline.ml.train

tune:
	$(COMPOSE) exec -T dagster-code python -m pipeline.ml.tuning --n-trials 50

# ── App ────────────────────────────────────────────────
app:
	$(COMPOSE) up -d --build --wait app

# ── Quality ────────────────────────────────────────────
test:
	uv run pytest -v

lint:
	uv run ruff check tests/ app.py pipeline/ scripts/

format:
	uv run ruff format pipeline tests/pipeline tests/smoke scripts app.py

# ── Docker ─────────────────────────────────────────────
docker-build:
	docker build -t melee-app:dev .

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf mlruns/ .pytest_cache/ __pycache__/
	find . -name "*.pyc" -delete
```

In `.github/workflows/ci.yml`, change the lint step to:

```yaml
      - name: Lint
        run: uv run ruff check tests/ app.py pipeline/ scripts/
```

In `.dockerignore`, delete the `data/raw` and `data/processed` lines; nothing under `data/` exists or is copied any more.

In `pipeline/ingest/queries.py`, change the docstring's second line to `Originally in the retired src/queries.py.`

`notebooks/` still imports `src` and is left as historical analysis; Task 16's README change says so.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_retirement.py -v`
Expected: 3 passed.

Run: `uv run pytest` and `uv run ruff check tests/ app.py pipeline/ scripts/`
Expected: everything passes with no warnings, and ruff is clean.

Run: `uv run dagster definitions validate -m pipeline.dagster_defs`
Expected: validation succeeds.

- [ ] **Step 5: Commit**

```bash
uv run ruff format pipeline tests/pipeline tests/smoke scripts app.py
git add -A src tests/test_database.py pyproject.toml Makefile .github/workflows/ci.yml .dockerignore pipeline/ingest/queries.py tests/pipeline/test_retirement.py
git commit -m "Retire SQLite and src/; run training and the app against the lakehouse"
```

---

### Task 15: Extend the smoke test through silver, gold and the model

**Files:**
- Create: `tests/smoke/check_lakehouse.py`
- Modify: `scripts/smoke.sh`

**Interfaces:**
- Consumes:
  - the `silver_startgg`, `gold_startgg` and `matchup_model` asset names;
  - `TrainingConfig` fields `split_quantile` and `min_sets`;
  - `load_lakehouse_catalog`, `current_snapshot_id`, `load_head_to_head`, `lookup_h2h`;
  - the MLflow names;
  - `collect_week` and `FixtureClient`.
- Produces: `make smoke`, which now covers fixture → bronze → silver → gold → model, twice.

- [ ] **Step 1: Write the checker**

Create `tests/smoke/check_lakehouse.py`:

```python
"""Assert silver, gold and MLflow after the smoke test's runs.

Usage: python -m tests.smoke.check_lakehouse --fixture PATH --week KEY --copies N
"""

import argparse

import dagster as dg
from mlflow import MlflowClient

from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES
from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import FixtureClient
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog
from pipeline.ml.names import GOLD_SNAPSHOT_TAG, MODEL_NAME
from pipeline.serving.gold_reader import load_head_to_head, load_player_ratings, lookup_h2h


class CapturingPublisher:
    def __init__(self):
        self.set_payloads: list[dict] = []

    def publish(self, topic: str, key: str, value: dict) -> None:
        if value["entity"] == "set":
            self.set_payloads.append(value["payload"])


def slot_player_id(slot: dict):
    participants = (slot.get("entrant") or {}).get("participants") or [{}]
    return ((participants[0] or {}).get("player") or {}).get("id")


def expected_kept_sets(payloads: list[dict]) -> dict[int, dict]:
    """The silver set rules, reimplemented independently as the smoke test's oracle."""
    kept = {}
    for node in payloads:
        slots = node.get("slots") or []
        if len(slots) < 2:
            continue
        if not all(slot_player_id(slot) for slot in slots[:2]):
            continue
        if node.get("state") != 3 or not str(node.get("id")).isdigit():
            continue
        entrants = [(slot.get("entrant") or {}).get("id") for slot in slots[:2]]
        if node.get("winnerId") not in entrants:
            continue
        kept[int(node["id"])] = node
    return kept


def row_count(catalog, identifier: str) -> int:
    return catalog.load_table(identifier).scan().to_arrow().num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--week", required=True)
    parser.add_argument("--copies", type=int, required=True)
    args = parser.parse_args()

    publisher = CapturingPublisher()
    stats = collect_week(FixtureClient(args.fixture), publisher, args.week, run_id="check")
    kept = expected_kept_sets(publisher.set_payloads)
    games = {g["id"] for node in kept.values() for g in (node.get("games") or []) if g.get("id")}
    eligible = sum(1 for node in kept.values() if node.get("completedAt") is not None)
    assert kept and eligible, f"fixture has no eligible sets for {args.week}"

    catalog = load_lakehouse_catalog()
    # Replays must not change silver or gold: the counts are the same on every pass.
    exact = {
        "silver.tournaments": stats.tournaments,
        "silver.events": stats.events,
        "silver.sets": len(kept),
        "silver.games": len(games),
        SET_FEATURES: eligible,
    }
    for identifier, expected in exact.items():
        actual = row_count(catalog, identifier)
        assert actual == expected, f"{identifier}: {actual} rows, expected {expected}"
        print(f"{identifier}: {actual} rows")
    for identifier in ("silver.players", PLAYER_RATINGS, HEAD_TO_HEAD):
        actual = row_count(catalog, identifier)
        assert actual > 0, f"{identifier} is empty"
        print(f"{identifier}: {actual} rows")

    # The app's own readers must work against these tables.
    assert len(load_player_ratings(catalog, min_sets=1)) > 0, "app reader returned no players"
    first = next(iter(kept.values()))
    p1, p2 = (int(slot_player_id(slot)) for slot in first["slots"][:2])
    assert sum(lookup_h2h(load_head_to_head(catalog), p1, p2)) >= 1, f"no h2h for {p1} vs {p2}"

    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) == args.copies, f"{len(versions)} model versions, expected {args.copies}"
    latest = max(versions, key=lambda version: int(version.version))
    gold_snapshot = str(current_snapshot_id(catalog, SET_FEATURES))
    assert latest.tags.get(GOLD_SNAPSHOT_TAG) == gold_snapshot, (
        f"model v{latest.version} tagged {latest.tags.get(GOLD_SNAPSHOT_TAG)}, gold is {gold_snapshot}"
    )
    event = dg.DagsterInstance.get().get_latest_materialization_event(dg.AssetKey("matchup_model"))
    decision = event.asset_materialization.metadata["promotion_decision"].value
    print(f"model v{latest.version}: promotion decision {decision}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Extend `scripts/smoke.sh`**

Make these changes:
- Change the header comment's first line to `# End-to-end smoke test: recorded week -> Kafka -> bronze -> silver -> gold -> model, twice.`
- Replace the `up` line with:

```bash
"${COMPOSE[@]}" up -d --build --wait dagster-code trino mlflow
```

- Add after `in_code() { ... }`:

```bash
# One fixture week is too small for the default 2025-07-01 split.
TRAIN_CONFIG='{"ops": {"matchup_model": {"config": {"split_quantile": 0.8, "min_sets": 1}}}}'
```

- Replace the `for` loop with:

```bash
for copies in 1 2; do
  echo "--- ingestion pass $copies"
  in_code dagster asset materialize -m pipeline.dagster_defs --select startgg_raw --partition "$WEEK"
  in_code dagster asset materialize -m pipeline.dagster_defs --select bronze_startgg
  in_code python -m tests.smoke.check_bronze --fixture "$FIXTURE" --week "$WEEK" --copies "$copies"
  in_code dagster asset materialize -m pipeline.dagster_defs --select silver_startgg
  in_code dagster asset materialize -m pipeline.dagster_defs --select gold_startgg
  in_code dagster asset materialize -m pipeline.dagster_defs --select matchup_model \
    --config-json "$TRAIN_CONFIG"
  in_code python -m tests.smoke.check_lakehouse --fixture "$FIXTURE" --week "$WEEK" --copies "$copies"
done
```

`scripts/smoke.sh` must stay LF and executable: `git ls-files --eol scripts/smoke.sh` should show `w/lf`, and `git ls-files -s scripts/smoke.sh` should show mode `100755`.

- [ ] **Step 3: Lint, unit tests, and one local smoke run**

Run: `uv run ruff format tests/smoke && uv run ruff check tests/ scripts/` and `uv run pytest`
Expected: clean, and all tests pass.

Run the smoke test once, in the foreground, with a 45-minute timeout. Don't run it in the background, where Claude Code's low-memory reaper can kill it.

```bash
$MAKE smoke 2>&1 | tail -40
```

Expected:
- **Pass 1** prints:
  - `silver.tournaments: 20 rows`, `silver.events: 32 rows`, `silver.sets: 2927 rows`;
  - `silver.games: N rows` (N is 1105 for the current fixture, or whatever the distinct game ids of the kept sets are);
  - `gold.set_features: 2927 rows`;
  - non-zero players, ratings and head-to-head rows;
  - `model v1: promotion decision ...`.
- **Pass 2** prints the same silver and gold counts, plus `model v2: ...`.
- The run ends with `smoke test passed`.

If the host runs out of memory and the run is killed, do not loop. Record it in the report; CI's `spark-and-smoke` job in Task 16 is the authoritative run.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/check_lakehouse.py scripts/smoke.sh
git ls-files --eol scripts/smoke.sh
git commit -m "Extend the smoke test through silver, gold and model registration"
```

---

### Task 16: Verify the platform on the dev stack, document it, open the PR

**Files:**
- Create: `scripts/dagster_ops.py`
- Modify: `README.md`
- Test: `tests/pipeline/test_dagster_ops_script.py`

**Interfaces:**
- Consumes: every earlier task, and the running dev stack (Dagster UI on port 3001).
- Produces:
  - `scripts/dagster_ops.py` with subcommands `launch JOB`, `run-status RUN_ID`, `backfill --start --end` and `backfill-status ID`;
  - `weekly_partition_keys(start, end) -> list[str]`;
  - an updated README;
  - PR into `master`, after the user confirms.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_dagster_ops_script.py`:

```python
import importlib.util
from datetime import date
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("dagster_ops", Path("scripts/dagster_ops.py"))
dagster_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dagster_ops)


def test_backfill_covers_every_monday_inclusive():
    keys = dagster_ops.weekly_partition_keys(date(2018, 1, 1), date(2024, 12, 30))
    assert len(keys) == 366
    assert (keys[0], keys[1], keys[-1]) == ("2018-01-01", "2018-01-08", "2024-12-30")


def test_backfill_bounds_must_be_mondays():
    with pytest.raises(ValueError, match="Monday"):
        dagster_ops.weekly_partition_keys(date(2018, 1, 2), date(2024, 12, 30))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/pipeline/test_dagster_ops_script.py -v`
Expected: `FileNotFoundError` for `scripts/dagster_ops.py`.

- [ ] **Step 3: Write the script**

Create `scripts/dagster_ops.py`:

```python
"""Drive the local Dagster instance over GraphQL: launch jobs and backfills, poll status.

Usage:
  python scripts/dagster_ops.py launch train_matchup_model
  python scripts/dagster_ops.py run-status RUN_ID
  python scripts/dagster_ops.py backfill --start 2018-01-01 --end 2024-12-30
  python scripts/dagster_ops.py backfill-status BACKFILL_ID

Set DAGSTER_GRAPHQL_URL if the webserver isn't at http://localhost:3000/graphql.
"""

import argparse
import json
import os
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


def graphql(url: str, query: str, variables: dict) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps({"query": query, "variables": variables}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"]))
    return body["data"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("launch").add_argument("job")
    commands.add_parser("run-status").add_argument("run_id")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--start", type=date.fromisoformat, required=True)
    backfill.add_argument("--end", type=date.fromisoformat, required=True)
    backfill.add_argument("--asset", default="startgg_raw")
    commands.add_parser("backfill-status").add_argument("backfill_id")
    args = parser.parse_args()
    url = os.getenv("DAGSTER_GRAPHQL_URL", "http://localhost:3000/graphql")

    if args.command == "launch":
        selector = {"repositoryLocationName": LOCATION, "repositoryName": REPOSITORY, "jobName": args.job}
        result = graphql(url, LAUNCH_RUN, {"params": {"selector": selector, "runConfigData": {}}})
        print(json.dumps(result["launchRun"]))
    elif args.command == "run-status":
        print(json.dumps(graphql(url, RUN_STATUS, {"id": args.run_id})["runOrError"]))
    elif args.command == "backfill":
        keys = weekly_partition_keys(args.start, args.end)
        params = {
            "assetSelection": [{"path": [args.asset]}],
            "partitionNames": keys,
            "title": f"{args.asset} {keys[0]}..{keys[-1]}",
        }
        result = graphql(url, LAUNCH_BACKFILL, {"params": params})
        print(len(keys), "partitions:", json.dumps(result["launchPartitionBackfill"]))
    else:
        print(json.dumps(graphql(url, BACKFILL_STATUS, {"id": args.backfill_id})["partitionBackfillOrError"]))


if __name__ == "__main__":
    main()
```

Run: `uv run pytest tests/pipeline/test_dagster_ops_script.py -v`
Expected: 2 passed.

- [ ] **Step 4: Run the whole platform on the dev stack**

```bash
export DAGSTER_WEBSERVER_PORT=3001 DAGSTER_GRAPHQL_URL=http://localhost:3001/graphql
$MAKE up
```

Expected: every service healthy, including `mlflow` and `app`.

Automation should materialize `silver_startgg` and then `gold_startgg` within a few minutes. Silver already caught up in Task 4, so its run reads 0 rows. Check:

```bash
curl -s "$DAGSTER_GRAPHQL_URL" -H 'Content-Type: application/json' \
  -d '{"query":"{ runsOrError(limit: 10) { ... on Runs { results { jobName status assetSelection { path } } } } }"}'
```

Expected: recent runs for `silver_startgg` and `gold_startgg` with status `SUCCESS`.

If none appear within 10 minutes, start the sensor with `MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-daemon dagster sensor start lakehouse_automation -w /opt/dagster/workspace.yaml`, and record that it was stopped.

Train and register a model on the 2025 data:

```bash
uv run python scripts/dagster_ops.py launch train_matchup_model
uv run python scripts/dagster_ops.py run-status <runId from the previous line>
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-code python -c "
from mlflow import MlflowClient
v = MlflowClient('http://mlflow:5000').get_model_version_by_alias('melee-matchup', 'champion')
print(v.version, v.tags)"
```

Expected:
- The run reaches `SUCCESS`; re-run `run-status` until it does.
- The champion is version 1, tagged with `gold_snapshot_id`, `promotion: no_champion`.

Run maintenance once and confirm bronze compaction:

```bash
q() { MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T trino trino --execute "$1"; }
q 'SELECT count(*) FROM lakekeeper.bronze."startgg_sets$files"'
uv run python scripts/dagster_ops.py launch lakehouse_maintenance
uv run python scripts/dagster_ops.py run-status <runId>
q 'SELECT count(*) FROM lakekeeper.bronze."startgg_sets$files"'
```

Expected: the run succeeds, and the second file count is lower than the first. Record both numbers. If the run's metadata shows `remove_orphan_files` errors, for example vended credentials not allowing listing, record them; they are reported but not fatal by design.

Check the app:

```bash
curl -sf http://localhost:8501/_stcore/health && echo
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T app python -c "
from pipeline.lakehouse import load_lakehouse_catalog
from pipeline.serving.gold_reader import load_player_ratings
from pipeline.serving.champion import load_champion
print(len(load_player_ratings(load_lakehouse_catalog())), load_champion('http://mlflow:5000').version)"
test -e data && echo "data/ exists" || echo "no data/ directory"
```

Expected: `ok`, then a positive player count and champion version `1`, then `no data/ directory`. Ask the user to open http://localhost:8501 and spot-check a prediction. That check is not blocking.

- [ ] **Step 5: Update the README**

Replace the whole `## Data Platform (in progress)` section, from its heading down to the line before `## Dataset`, with:

````markdown
## Data Platform

Every record enters through Kafka and flows through a medallion lakehouse on
Apache Iceberg:

```
start.gg API --(Dagster, weekly partitions)--> Kafka --(Spark, availableNow)--> bronze
bronze --(Spark MERGE, incremental snapshots)--> silver --(Python, PyIceberg)--> gold
gold --> XGBoost (MLflow registry, champion alias) --> Streamlit app
```

| Component | Tool | Local URL |
|---|---|---|
| Orchestration | Dagster | http://localhost:3000 |
| Event backbone | Kafka (KRaft) + Kafka UI | http://localhost:8085 |
| Table catalog | Lakekeeper (Iceberg REST) | http://localhost:8181 |
| Object storage | Silo (S3-compatible) | http://localhost:9001 |
| Processing | Spark 4.0 (local mode), PyIceberg | n/a |
| SQL | Trino | `localhost:8090` |
| Experiments and registry | MLflow | http://localhost:5000 |
| App | Streamlit | http://localhost:8501 |

**Layers:**
- **bronze:** raw JSON envelopes, append-only.
- **silver:**
  - `tournaments`, `events`, `players`, `sets`, `games`;
  - deduplicated, typed;
  - upserted with `MERGE INTO`, where only an equal-or-newer bronze version wins;
  - read incrementally from bronze snapshots after a stored watermark.
- **gold:**
  - `set_features` (one row per set, features from earlier sets only), `player_ratings` and `head_to_head`;
  - fully recomputed, one Iceberg commit per table.

**Automation:**
- Silver and gold run on Dagster declarative automation after each bronze landing.
- The model trains daily at 07:00 PT when gold has changed. A new version only takes the `champion` alias if it beats the seed-only baseline, and matches or beats the current champion.
- Weekly maintenance (Sunday 03:00 PT) compacts small files, expires old snapshots and removes orphan files.

**Run it** (needs Docker with ~7 GB of memory and GNU make):

```bash
cp .env.example .env      # then set STARTGG_API_TOKEN
make up                   # build and start the stack
make check-lakehouse      # Spark -> Iceberg -> Trino round trip
```

If a port is taken, set `DAGSTER_WEBSERVER_PORT`, `MLFLOW_PORT` or `APP_PORT`
in `.env`.

**Using it:**
1. Backfill `startgg_raw` from the Dagster asset page, or run `python scripts/dagster_ops.py backfill --start 2025-01-06 --end 2025-03-31`.
2. The `land_bronze_after_ingest` sensor lands each finished week into bronze; silver and gold then follow automatically.
3. Then train with `python scripts/dagster_ops.py launch train_matchup_model`, or wait for the daily schedule.

**One-off commands:**
- `make bronze-once`, `make silver-once` and `make gold-once` run a single layer by hand.
- The Spark jobs refuse to start while a Dagster run holds their lock.
- `make train` prints the model comparison.
- `make tune` runs Optuna.

**Tests:**
- `make test`: unit, Dagster, PyIceberg and MLflow tests;
- `make test-spark`: Spark transforms and the silver MERGE, in the pipeline container;
- `make smoke`: a recorded week end to end through the model registry, with no API token needed.

**Failure handling:**
- Events that still fail after retries go to the `startgg.ingest.dlq` topic and fail the per-week `no_dlq_messages` check.
- Malformed messages go to `bronze.startgg_rejects` and fail `no_rejects`.
- Silver checks: orphan sets, a drop rate of 10% or more, and winners who aren't one of the two players.
- Gold checks: row parity with silver, null features, and time order.

Designs:
- [`docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md`](docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md)
- [`docs/superpowers/specs/2026-10-08-silver-gold-design.md`](docs/superpowers/specs/2026-10-08-silver-gold-design.md)
````

Replace the `## Project Structure` code block with:

````markdown
```
melee-matchup/
├── pipeline/
│   ├── ingest/             # start.gg client, weekly collection, Kafka publisher, fixtures
│   ├── spark_jobs/         # bronze landing, silver MERGE, Iceberg maintenance
│   ├── gold/               # Elo + feature pass, gold tables (PyIceberg)
│   ├── ml/                 # training, tuning, MLflow registry and promotion
│   ├── serving/            # gold reader and champion loader for the app
│   └── dagster_defs/       # assets, checks, sensors, schedules, resources
├── infra/                  # Docker Compose stack, Spark/Trino/Dagster config
├── scripts/                # smoke test, fixture recorder, Dagster GraphQL helper
├── tests/                  # unit, Spark, and smoke tests
├── notebooks/              # original EDA and modeling notebooks (read the retired SQLite DB)
├── app.py                  # Streamlit app on gold + MLflow
└── pyproject.toml
```
````

In `## How It Works`:
- Replace the `python -m src.collect ...` code block with `python scripts/dagster_ops.py backfill --start 2018-01-01 --end 2024-12-30`.
- Replace the `python -m src.features` block with `make gold-once   # or let Dagster's automation rebuild gold`.
- Replace the `python -m src.model ...` block with `make train   # prints the comparison table; Dagster registers models in MLflow`.
- In the collection bullets, change "**Idempotent reruns** (upsert logic + skip-existing ...)" to "**Idempotent reruns** (bronze keeps every copy; silver's MERGE keeps the newest version of each record)".

Replace `## Setup` through the end of its Quick Start code block with:

````markdown
## Setup

### Prerequisites

- Docker with ~7 GB of memory, GNU make, and [uv](https://docs.astral.sh/uv/)
- A [start.gg API token](https://start.gg/admin/profile/developer) (free)

### Installation

```bash
git clone https://github.com/justinko157/melee-matchup.git
cd melee-matchup
uv sync --all-extras
cp .env.example .env      # then paste your start.gg token
make up
```

### Quick Start

```bash
python scripts/dagster_ops.py backfill --start 2025-01-06 --end 2025-03-31   # ingest
python scripts/dagster_ops.py launch train_matchup_model                      # train
open http://localhost:8501                                                    # predict
```
````

In `## Tools & Technologies`, replace the list with:

```markdown
- **Ingestion**: start.gg GraphQL API, Kafka (KRaft), confluent-kafka
- **Lakehouse**: Apache Iceberg (Lakekeeper REST catalog, Silo S3), Spark 4.0, PyIceberg, Trino
- **Orchestration**: Dagster (partitions, sensors, declarative automation, asset checks)
- **Modeling**: scikit-learn, XGBoost, SHAP, Optuna
- **Experiment Tracking and Registry**: MLflow
- **App**: Streamlit
```

- [ ] **Step 6: Commit**

```bash
uv run ruff format scripts && uv run ruff check scripts tests
git add scripts/dagster_ops.py tests/pipeline/test_dagster_ops_script.py README.md
git commit -m "Document the lakehouse platform and add a Dagster GraphQL helper"
```

- [ ] **Step 7: STOP — ask the user before pushing**

Ask the user to confirm pushing `de/silver-gold` and opening a PR into `master` as `justinko157`. Once they confirm:

```bash
git push -u origin de/silver-gold
GH_TOKEN=$(gh auth token --user justinko157) gh pr create --base master --head de/silver-gold \
  --title "Silver and gold lakehouse layers, MLflow registry and the app on gold" --body-file "$TMPDIR/pr-body.md"
GH_TOKEN=$(gh auth token --user justinko157) gh pr checks --watch
```

Before running these, write the PR body to `$TMPDIR/pr-body.md`, outside the repo. It summarizes the layers, checks, automation, retirement and the dev-stack verification numbers from Step 4. It must not contain a Claude co-author line or "Generated with Claude Code".

Expected: `lint-and-unit` and `spark-and-smoke` both pass. If `spark-and-smoke` fails, fix it on the branch and push again; that push needs no new confirmation.

---

### Task 17: Backfill 2018–2024, retrain, and record the results

**Files:**
- Modify: `README.md` (results table), and the PR description

**Interfaces:**
- Consumes: `scripts/dagster_ops.py`, the running dev stack, and the PR from Task 16.

- [ ] **Step 1: STOP — confirm the long run with the user**

Ask the user to confirm all of the following before anything starts:
- The run takes roughly 15 hours. It is about 366 weekly partitions at around 2.5 minutes each, using the start.gg token.
- Their healthcare-data-platform containers may be stopped to free memory and restarted afterwards.
- Pushing the README update to the PR branch when it's done.

Report times in Pacific time.

- [ ] **Step 2: Pause silver/gold automation during the backfill**

Gold is a full recompute. Rebuilding it after every one of 366 bronze landings would waste hours and compete for `dagster-code` memory.

```bash
export DAGSTER_GRAPHQL_URL=http://localhost:3001/graphql
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-daemon dagster sensor stop lakehouse_automation -w /opt/dagster/workspace.yaml
```

Expected: the CLI reports the sensor stopped. If the CLI errors, ask the user to toggle the sensor off in the Dagster UI.

- [ ] **Step 3: Launch and monitor the backfill**

```bash
uv run python scripts/dagster_ops.py backfill --start 2018-01-01 --end 2024-12-30
```

Expected: `366 partitions: {"__typename": "LaunchBackfillSuccess", "backfillId": "..."}`. Record the backfill id and the start time.

Monitor with `uv run python scripts/dagster_ops.py backfill-status <id>`, using a Monitor or a bounded polling loop at a 20-minute interval or longer. Report progress every 10% and immediately on any failed partition. Bronze runs follow automatically through `land_bronze_after_ingest`.

Done means: the backfill status is `COMPLETED_SUCCESS` or `COMPLETED`, every partition `SUCCESS`, and no `land_bronze` run queued or started. If a partition fails, re-run just that week from the asset page, or with `backfill --start <week> --end <week>`; ingestion is idempotent.

- [ ] **Step 4: Catch up silver and gold, retrain, maintain**

```bash
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-daemon dagster sensor start lakehouse_automation -w /opt/dagster/workspace.yaml
```

Wait until one `silver_startgg` run and one `gold_startgg` run succeed (the run query is in Task 16, Step 4). Silver reads only the snapshots appended since its watermark. Then:

```bash
uv run python scripts/dagster_ops.py launch train_matchup_model
uv run python scripts/dagster_ops.py launch lakehouse_maintenance
```

Expected: both runs reach `SUCCESS`.

- [ ] **Step 5: Verify the data and the model**

```bash
q() { MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T trino trino --execute "$1"; }
q "SELECT year(completed_at) AS y, count(*) FROM lakekeeper.silver.sets GROUP BY 1 ORDER BY 1"
q "SELECT (SELECT count(*) FROM lakekeeper.silver.tournaments), (SELECT count(*) FROM lakekeeper.silver.events),
          (SELECT count(*) FROM lakekeeper.silver.players), (SELECT count(*) FROM lakekeeper.silver.sets),
          (SELECT count(*) FROM lakekeeper.silver.games)"
q "SELECT (SELECT count(*) FROM lakekeeper.gold.set_features),
          (SELECT count(*) FROM lakekeeper.silver.sets WHERE winner_player_id IS NOT NULL AND completed_at IS NOT NULL)"
MSYS_NO_PATHCONV=1 docker compose -f infra/docker-compose.yml --env-file .env exec -T dagster-code python -c "
from mlflow import MlflowClient
c = MlflowClient('http://mlflow:5000')
v = c.get_model_version_by_alias('melee-matchup', 'champion')
m = c.get_run(v.run_id).data.metrics
print(v.version, v.tags.get('gold_snapshot_id'), {k: round(m[k], 4) for k in sorted(m)})"
```

Expected:
- Silver has sets for every year from 2018 through 2026.
- Gold `set_features` equals the eligible silver count.
- The champion is tagged with the current gold snapshot.
- In the Dagster UI, every asset check on `silver_startgg`, `gold_startgg` and `matchup_model` passes.

Compare `xgboost_accuracy`, `xgboost_log_loss` and `xgboost_roc_auc` with the README's previous numbers (78.5%, 0.463, 0.818). A shortfall is reported, not a failure. The new pipeline uses `min_attendees` 50 against the old dataset's 100, and the test window now ends in October 2026.

- [ ] **Step 6: Record the results and update the PR**

In `README.md`, update the "Results at a Glance" table and the opening paragraph's accuracy/AUC numbers with the champion's baseline, logistic and XGBoost metrics from Step 5. Use 3–4 significant digits, as the existing table does. Then update the `## Dataset` table with the silver counts from Step 5, and its date-range sentence to "offline tournaments with 50+ attendees from January 2018 through <latest completed week>".

```bash
git add README.md
git commit -m "Record dataset counts and champion metrics after the 2018-2024 backfill"
git push
```

Update the PR description with:
- the backfill id and duration;
- the silver and gold row counts;
- the champion version and metrics next to the previous numbers;
- the maintenance file-count reduction.

No Claude attribution lines. Restart any containers that were stopped in Step 1.
