# Silver and Gold Layers, Model Training and App on Gold (DE Pivot Sub-project 2)

**Status:** approved design, 2026-10-08
**Builds on:** `docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md` (merged in PR #1)

## Goal

Turn the append-only bronze tables into cleaned, deduplicated silver tables and
model-ready gold tables, then move the existing ML model and Streamlit app onto
them. When this sub-project is done, nothing in the project reads SQLite or files
under `data/`.

## Why

The ingestion foundation lands raw start.gg JSON in Iceberg bronze. The model and
app still expect the retired SQLite database and local parquet exports. This
sub-project completes the medallion architecture and shows the skills the
healthcare repo does not:

- incremental, snapshot-based processing with Iceberg `MERGE INTO`;
- Iceberg table maintenance;
- PyIceberg as a lightweight reader and writer;
- Dagster declarative automation;
- an MLflow model registry with a gated promotion step.

dbt is deliberately not used, because the healthcare repo already covers it.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| Transform tool | Spark + Iceberg. The sequential Elo feature pass stays in Python and writes gold through PyIceberg. |
| History | Backfill 2018-01-01 to 2024-12-30 through the existing ingestion backfill as the final step. Design and tests don't depend on it. |
| How silver stays current | Incremental `MERGE INTO` from bronze, driven by Iceberg incremental append scans and a per-bronze-table watermark. |
| Gold refresh | Full recompute on every run, replaced in one commit per table. |
| How the app reads gold | PyIceberg against the Lakekeeper REST catalog. No Trino dependency. |
| Training | A Dagster asset with an MLflow server in Compose, model registry, and a `champion` alias set by a promotion rule. |

## Tables

### Silver (`silver.*`)

Each table is keyed by its start.gg id and upserted by MERGE.

| Table | Grain | Source | Columns (besides lineage) |
|---|---|---|---|
| `tournaments` | tournament | bronze tournament payload | `tournament_id`, `name`, `slug`, `num_attendees`, `start_at` (timestamp), `end_at`, `city`, `state`, `country_code`, `is_online` |
| `events` | Melee event | `payload.events[]` of tournaments, filtered to videogame id 1 | `event_id`, `tournament_id`, `name`, `slug`, `num_entrants` |
| `players` | player | set slots (`entrant.participants[0].player`) | `player_id`, `gamer_tag` (taken from the newest `source_ingested_at`) |
| `sets` | completed set | bronze set payload; `event_id` from `query_variables.eventId` | `set_id`, `event_id`, `round`, `full_round_text`, `phase_name`, `completed_at` (timestamp), `total_games`, `display_score`, `entrant1_id`, `entrant2_id`, `player1_id`, `player2_id`, `winner_player_id`, `player1_score`, `player2_score`, `player1_seed`, `player2_seed` |
| `games` | game | `payload.games[]` of sets | `game_id`, `set_id`, `game_number`, `winner_entrant_id`, `stage_id`, `stage_name`, `player1_character_id`, `player1_character_name`, `player2_character_id`, `player2_character_name` |

Lineage columns on every silver table:

- `source_ingested_at`: the bronze envelope's `ingested_at`.
- `source_snapshot_id`: the bronze snapshot the row was read from.
- `silver_updated_at`: when this job last wrote the row.

Parsing rules are carried over from `src/collector.py`:

- A set needs at least two slots, both with a player id, and `state = 3`.
- The winner is resolved by matching `winnerId` to the entrant ids.
- The seed is `seed.seedNum`, falling back to `entrant.initialSeedNum`.
- The score is `standing.stats.score.value`.
- Each game needs an id. Characters are matched to players by entrant id.

Sets that fail a rule are dropped, not errored. The job counts them per drop reason:

- `too_few_slots`
- `missing_player`
- `not_completed`
- `unresolved_winner`

**Excluded tournaments:**

- `EXCLUDED_TOURNAMENT_IDS` in `pipeline/spark_jobs/silver.py` lists fabricated brackets found after the 2018–2024 backfill.
- Silver skips those tournaments, their events, and the sets of those events.
- Excluded sets are counted as `excluded_sets`, apart from the drop reasons, and they are left out of the drop rate.
- Every run deletes any of their rows already in silver, and reports the counts as `purged`. A player is deleted only if every set they played was excluded.

**Partitioning:**

- `sets` and `games` by `months(completed_at)`. `games` carries `completed_at` from its set for this purpose.
- `tournaments`, `events` and `players` are unpartitioned.

### Gold (`gold.*`)

All gold tables are fully recomputed.

| Table | Grain | Consumer | Columns |
|---|---|---|---|
| `set_features` | eligible silver set | training | The same columns as the old `features.parquet`, built by `src/features.py`'s `FeatureBuilder` logic. Also `tournament_id`, `tournament_name`, `num_attendees`, and the target `p1_won`. |
| `player_ratings` | player | app | `player_id`, `gamer_tag`, `elo`, `sets_played`, `recent_wr` (last 30 sets), `wins`, `losses`, `last_played_at` |
| `head_to_head` | player pair, `player_a_id < player_b_id` | app | `player_a_id`, `player_b_id`, `a_wins`, `b_wins`, `total`, `last_played_at` |

An eligible set is a silver set with a non-null `winner_player_id` and `completed_at`.

Sets are processed in order of `(completed_at, set_id)`, so ties are broken the same way every time.

**Partitioning:**

- `set_features` by `years(completed_at)`.
- `player_ratings` and `head_to_head` are unpartitioned.

## Processing

### Silver job (`pipeline/spark_jobs/silver_job.py`, `pipeline/spark_jobs/silver.py`)

1. Take an exclusive non-blocking `flock` on `<checkpoint-root>/silver/.lock`, using the same pattern as the bronze job.
2. Create the `silver` namespace, the tables, and `silver._watermarks` if they are missing. The watermarks table has columns `bronze_table`, `snapshot_id` and `updated_at`.
3. For each bronze table, read the appended snapshots after the watermark up to the table's current snapshot, using `start-snapshot-id` / `end-snapshot-id`.
   - If there is no watermark, or the watermark snapshot is no longer in the table history, do a full scan instead.
   - If the current snapshot equals the watermark, there is nothing to do.
4. Parse with pure DataFrame functions in `silver.py`. Deduplicate to the newest version of each key by `(source_ingested_at, kafka_partition, kafka_offset)`.
5. Run one `MERGE INTO` per silver table, in parent-before-child order: tournaments, events, players, sets, games.
   - `WHEN MATCHED AND s.source_ingested_at >= t.source_ingested_at THEN UPDATE`
   - `WHEN NOT MATCHED THEN INSERT`
6. Update the watermarks with one MERGE into `silver._watermarks`, after all table MERGEs have committed.
7. Print a JSON summary as the last stdout line. It includes, per table, `{"inserted": n, "updated": n}`, plus `"dropped": {reason: n}` and `"bronze_rows_read": n`. The per-table counts come from a join against the target before the MERGE, not from the MERGE snapshot summaries, because copy-on-write summaries count every row in each rewritten file. `updated` counts every staged row whose bronze version is equal to or newer than the existing row's, so it includes same-value rewrites.

**Crash safety.** If the job fails after some MERGEs but before the watermark update, the next run re-reads the same snapshots. The `>=` rule turns already-written rows into same-value updates. Nothing goes backward.

### Gold (`pipeline/gold/`)

The gold builder is plain Python. It does not use Spark.

1. Load silver `sets`, `tournaments`, `events` and `players` through PyIceberg, selecting only the needed columns, into pandas.
2. Run the `EloEngine` and feature pass. These are moved unchanged from `src/features.py` into `pipeline/gold/features.py`, so the existing `tests/test_elo.py` and `tests/test_features.py` move with them.
3. Derive `player_ratings` and `head_to_head` from the final state of the same pass.
4. Write each table with PyIceberg `overwrite()`. That is one atomic commit per table, so readers see the old or the new table, never a mix.
5. Record the silver snapshot ids used in the gold table properties, as `melee.silver_sets_snapshot`.

### Training (`pipeline/ml/`)

`src/model.py`'s split, evaluation and model functions move to `pipeline/ml/train.py` and take a DataFrame instead of a path.

The `matchup_model` asset does the following:

1. Load `gold.set_features` via PyIceberg.
2. Make the temporal split as today, then train the baseline, logistic regression and XGBoost models.
3. Log to MLflow under experiment `melee-matchup`:
   - params;
   - metrics: `accuracy`, `log_loss`, `brier`, `roc_auc` for every model;
   - tag `gold_snapshot_id`;
   - artifacts: the SHAP metadata (as `export_shap_metadata` produces it today) and the XGBoost model.
4. Register the XGBoost model as a new version of the registered model `melee-matchup`.
5. Promotion:
   - A version is eligible only if its XGBoost log loss is below the baseline's on its own test split. This is the same rule as `model_beats_baseline`.
   - Compute the current `champion` version's log loss on the new run's test split.
   - Set the `champion` alias on an eligible new version if its log loss is less than or equal to the champion's, or if no champion exists.
   - Record both scores and the decision in the asset metadata.

Tuning (`tuning.py`) moves to `pipeline/ml/tuning.py` and stays a manual `make tune`.

### Table maintenance

The `lakehouse_maintenance` job runs weekly on Sunday at 03:00 PT, in the `spark_lakehouse` pool. For every bronze, silver and gold table it calls:

- `CALL lakekeeper.system.rewrite_data_files(...)` (target file size 128 MB);
- `expire_snapshots(older_than => now - 7 days, retain_last => 5)`;
- `remove_orphan_files(older_than => now - 3 days)`.

It then logs file counts before and after as metadata.

Expiring bronze snapshots older than a silver watermark is safe, because silver falls back to a full scan (step 3).

## Orchestration (Dagster)

| Asset / job | Runs | Pool | Checks |
|---|---|---|---|
| `silver_startgg` | `AutomationCondition.eager()` after `bronze_startgg` | `spark_lakehouse` | `no_orphan_sets`, `drop_rate_below_10pct`, `winner_is_a_player` |
| `gold_startgg` | `AutomationCondition.eager()` after `silver_startgg` | `spark_lakehouse` | `features_match_silver`, `no_null_features`, `completed_at_non_decreasing` |
| `matchup_model` | Daily schedule at 07:00 PT; skipped when the gold snapshot equals the champion's `gold_snapshot_id` tag | `spark_lakehouse` | `model_beats_baseline` |
| `lakehouse_maintenance` | Weekly schedule | `spark_lakehouse` | none (metadata only) |

The check details:

- **`no_orphan_sets`:** sets whose `event_id` is not in `silver.events` must be 0.
- **`drop_rate_below_10pct`:** dropped divided by bronze set rows read in this run must be under 10%. A run with 0 rows passes.
- **`winner_is_a_player`:** `winner_player_id` must be in (`player1_id`, `player2_id`) for every set.
- **`features_match_silver`:** the gold row count must equal the count of eligible silver sets.
- **`no_null_features`:** the model feature columns must contain no nulls.
- **`completed_at_non_decreasing`:** `completed_at` must not decrease in `set_features` row order.
- **`model_beats_baseline`:** XGBoost log loss must be below the baseline's.

The pool `spark_bronze` is renamed `spark_lakehouse` and shared by the bronze, silver, gold, training and maintenance runs (limit 1), because every run executes inside `dagster-code` (3 GB) and overlapping gold (~1.4 GB), Spark (~1.5 GB) and training (~1 GB) runs could OOM-kill a run worker. The `dagster-pools` service sets `startgg_api` and `spark_lakehouse` to 1.

The automation condition sensor must be enabled in the Dagster definitions.

## Services and app

### MLflow

- A new Compose service `mlflow`, with a pinned image tag chosen during planning.
- Backend store: Postgres database `mlflow`, added to `infra/postgres/init-databases.sql`.
- Artifacts: bucket `mlflow` on Silo, created by `create-bucket`.
- Host port `${MLFLOW_PORT:-5000}`.
- Memory limit 512 MB.

### App

- New Compose service `app`, built from the root `Dockerfile`, on host port `${APP_PORT:-8501}`.
- `app.py` reads `gold.player_ratings` and `gold.head_to_head` through PyIceberg (`pipeline/serving/gold_reader.py`).
- It loads `models:/melee-matchup@champion` and its SHAP artifact from MLflow.
- Configuration comes from environment variables with local defaults: `LAKEKEEPER_URL`, `LAKEHOUSE_WAREHOUSE`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `MLFLOW_TRACKING_URI`.
- Caching uses `st.cache_resource(ttl=3600)`.
- If the catalog or MLflow is unreachable, the page shows "The data platform is unavailable" plus the failing component, never a stack trace.
- The page's features and layout don't change.

## Retirement

These are removed:

- `src/database.py`, `src/collector.py`, `src/collect.py`, `src/api_client.py`, `src/queries.py`, `src/export_app_data.py`;
- `tests/test_database.py`;
- the `collect`, `features` and `export` Make targets;
- every `data/` path in code.

The `train` and `tune` Make targets keep working, against gold.

`src/features.py`, `src/model.py`, `src/tuning.py` and `src/validation.py` move into `pipeline/` as described above. Their tests move with them and keep passing. The validation schema becomes the gold checks plus a `pipeline/gold/schema.py` used by the tests.

When `src/` is empty it is deleted. The CI lint scope drops `src/`.

## Testing

### Unit tests (host and CI)

- Elo engine and feature pass:
  - the existing tests;
  - synthetic sequences proving each row uses only earlier sets;
  - deterministic tie-breaking on equal `completed_at`.
- `player_ratings` and `head_to_head` derivation.
- Gold writer and app reader against a PyIceberg `SqlCatalog` (SQLite plus local files, in a temporary directory).
- Watermark planning: no watermark, unchanged, advanced, and expired snapshot.
- Promotion rule: no champion, better, equal, and worse.
- Training on a small synthetic frame.
- Dagster definitions, checks, automation conditions, schedules and pools.

### Spark tests (`-m spark`, container and CI)

- Silver parsing: byes, missing players, non-completed sets, unresolved winner, game and character matching, non-Melee events filtered.
- Newest-version-wins dedupe.
- MERGE `>=` rule: an older bronze version never overwrites a newer one.
- Incremental read across two bronze snapshots.
- First-run full scan.
- Expired-watermark fallback.

### Smoke test (extends `make smoke`)

- **Pass 1:** fixture → bronze → silver → gold → model. Assertions:
  - silver sets equal the fixture's eligible sets;
  - gold `set_features` equals silver eligible sets;
  - `player_ratings` and `head_to_head` are non-empty;
  - MLflow has a registered version tagged with the gold snapshot id, and the asset metadata records a promotion decision. The test does not require a champion, because one fixture week may not beat the baseline;
  - the app's reader returns players and an h2h row for a known pair.
- **Pass 2 (replay):**
  - bronze doubles;
  - silver and gold counts are unchanged;
  - training runs again and promotion is evaluated.

CI keeps the `lint-and-unit` and `spark-and-smoke` jobs. The smoke job's timeout stays at 45 minutes.

## Failure handling

| Failure | Behavior |
|---|---|
| Silver job dies mid-run | The watermark is not advanced. The next run reprocesses, and the `>=` MERGE makes that idempotent. |
| Silver lock held | The job exits non-zero with "another silver job is running". Dagster retries twice, as bronze does. |
| Gold build fails | No overwrite has committed for tables not yet written, so readers keep the previous snapshot. The asset fails. |
| Training fails or `model_beats_baseline` fails | No promotion. The app keeps serving the current champion. |
| Catalog or MLflow unreachable from the app | A clear unavailable message names the component. |

## Resource budget

Docker has 7.3 GB.

- MLflow is about 300 MB (limit 512 MB) and the app about 500 MB (limit 768 MB), on top of the current stack.
- Gold's pandas pass over about 1.2 M sets (2018 onward) is about 1 GB. It runs inside `dagster-code`, which has a 3 GB limit.
- Spark silver runs in local mode inside `dagster-code`, never at the same time as bronze, gold or training, because of the shared pool.

## Done criteria

1. **Backfill and data.** The 2018-01-01 to 2024-12-30 backfill has completed. Silver contains tournaments, events, players, sets and games from 2018 onward. Gold `set_features` has one row per eligible silver set. All asset checks pass.
2. **Model.** MLflow has a `champion` version tagged with a gold snapshot id. Its test metrics are compared to the README's previous numbers (78.5% accuracy, 0.463 log loss, 0.818 AUC), and the comparison is recorded in the PR. Falling short is reported, not a failure.
3. **App.** The app at `http://localhost:8501` predicts a matchup using only gold and MLflow, and `data/` does not exist.
4. **Retirement.** The retired files and SQLite are gone, and CI passes.
5. **Maintenance.** `lakehouse_maintenance` has run once, and bronze's data file count dropped.

## Out of scope

- Continuous streaming (sub-project 3).
- AWS: Glue, S3, Athena (sub-project 4). This design only keeps the app and jobs configurable by environment variables.
- New model features or UI changes.
- Model serving beyond the Streamlit app.
