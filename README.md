# MeleeMatchup

**Predicting competitive Super Smash Bros Melee tournament set outcomes.**

Can we predict who wins a tournament set between two players? This project builds a full data platform — from API ingestion through a lakehouse to a served model — to answer that question from competitive Melee tournament results on start.gg.

On the data loaded into the platform so far (2025 onward), the champion XGBoost model reaches **78.4% accuracy** and **0.817 AUC**, against **74.4%** for a seed-only baseline. The original SQLite-era pipeline, trained on 550,000+ sets from 2018 through March 2026, reached 78.5% accuracy and 0.818 AUC.

## Results at a Glance

### Current platform (champion model version 1)

Data window: bronze weeks 2025-01-06 through 2026-10-05, with a temporal split at 2025-07-01 (train before, test after). These numbers will change once the 2018–2024 backfill is loaded.

| Model | Accuracy | Log Loss | Brier Score | ROC AUC |
|---|---|---|---|---|
| Baseline (Seed Only) | 74.4% | 0.540 | 0.177 | 0.755 |
| Logistic Regression | 77.5% | 0.478 | 0.155 | 0.806 |
| **XGBoost (champion)** | **78.4%** | **0.464** | **0.150** | **0.817** |

### Original SQLite-era pipeline (2018 – March 2026 data)

Trained before the move to the lakehouse, on 550,000+ sets across 2,200+ tournaments, with the same split date.

| Model | Accuracy | Log Loss | Brier Score | ROC AUC |
|---|---|---|---|---|
| Baseline (Seed Only) | 69.9% | 0.610 | 0.202 | 0.711 |
| Logistic Regression | 77.3% | 0.482 | 0.157 | 0.804 |
| **XGBoost** | **78.5%** | **0.463** | **0.150** | **0.818** |

### Key Findings

From the original analysis on the SQLite-era data:

- **Elo rating difference** is the single most predictive feature — a custom rating system built from historical results adds far more signal than tournament seeding alone
- **Recent form** (rolling 30-set win rate) is the second most important feature, capturing momentum and slumps
- **Head-to-head records** improve predictions for frequently-matched top players
- The model's biggest misses are **genuine community-recognized upsets** — not model bugs, but real surprises (e.g., Axe upset at Supernova 2025, Jmook losses at majors)

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
- Bronze, silver, gold, training and maintenance share the `spark_lakehouse` pool (limit 1): every run executes inside the 3 GB `dagster-code` container, so only one memory-heavy run goes at a time.

**Run it** (needs Docker with ~7 GB of memory and GNU make):

```bash
cp .env.example .env      # then set STARTGG_API_TOKEN
make up                   # build and start the stack
make check-lakehouse      # Spark -> Iceberg -> Trino round trip
```

If a port is taken, set `DAGSTER_WEBSERVER_PORT`, `MLFLOW_PORT` or `APP_PORT`
in `.env`. `scripts/dagster_ops.py` talks to http://localhost:3000/graphql by
default, so when you override `DAGSTER_WEBSERVER_PORT` also set
`DAGSTER_GRAPHQL_URL` (for example `http://localhost:3001/graphql`).

Each service has a memory limit in `infra/docker-compose.yml`. MLflow gets
768 MB and runs with `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, which keeps it
at about 420 MB when idle.

**Using it:**
1. Backfill `startgg_raw` from the Dagster asset page, or run `uv run python scripts/dagster_ops.py backfill --start 2025-01-06 --end 2025-03-31`.
2. The `land_bronze_after_ingest` sensor lands each finished week into bronze; silver and gold then follow automatically.
3. Then train with `uv run python scripts/dagster_ops.py launch train_matchup_model`, or wait for the daily schedule.

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

## Dataset

All data is collected from the [start.gg](https://start.gg) GraphQL API.

| | Lakehouse silver (2025-01-06 to 2026-10-05 weeks) | Original SQLite DB (Jan 2018 to Mar 2026) |
|---|---|---|
| Tournaments | 1,462 | 2,209 |
| Events | 2,450 | 4,026 |
| Players | 14,744 | 51,763 |
| Sets | 150,274 | 550,737 |
| Games | 129,404 | 138,968 |

The original SQLite database covered offline tournaments with 100+ attendees. The lakehouse counts will grow once the 2018–2024 backfill is loaded.

## Project Structure

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

## How It Works

### 1. Data Collection

A custom pipeline queries the start.gg GraphQL API to collect tournament brackets, set results, player info, seeds, and game-level data (character picks, stages where available).

```bash
uv run python scripts/dagster_ops.py backfill --start 2018-01-01 --end 2024-12-30
```

That range is 366 weekly partitions, about 15 hours at the start.gg API rate limit.

The pipeline handles:
- **Rate limiting** (80 requests/60s with automatic throttling)
- **Query complexity** (auto-reduces page size when hitting the 1000-object limit)
- **Pagination** (auto-paginates through results, chunks large date ranges to avoid the 10k entry cap)
- **Idempotent reruns** (bronze keeps every copy; silver's MERGE keeps the newest version of each record)

### 2. Feature Engineering

Features are computed **chronologically** — each set's features use only data available before that set was played (no data leakage).

| Feature | Description |
|---|---|
| `elo_diff` | Custom Elo rating difference (dynamic K-factor for newcomers) |
| `p1_expected` | Elo-derived win probability |
| `recent_wr_diff` | Rolling 30-set win rate difference |
| `p1_h2h_wr` | Historical head-to-head win rate |
| `seed_diff` | Tournament seed differential |
| `num_attendees` | Tournament size (proxy for competition level) |
| `p1_sets_played` / `p2_sets_played` | Experience (total sets in database) |

```bash
make gold-once   # or let Dagster's automation rebuild gold
```

### 3. Modeling

Three models compared using a **temporal train/test split** (train on pre-July 2025, test on post):

- **Baseline**: Logistic regression on seed differential only
- **Logistic Regression**: All features with standard scaling
- **XGBoost**: 300 trees, depth 5, early stopping — best performer

All experiments can be tracked with **MLflow**:

```bash
make train   # prints the comparison table; Dagster registers models in MLflow
```

## Exploratory Analysis Highlights

The EDA notebook (`notebooks/01_eda.ipynb`) covers:

- **Scene growth**: Tournament activity over time, impact of COVID and Slippi
- **Seeding accuracy**: Higher seeds win 74.4% of the time — a strong but beatable baseline
- **Player distribution**: Median player has 7 sets; 43% have 10+; heavy long tail
- **Top rivalries**: Zain vs Hungrybox is the most-played top-level matchup (42 sets, Zain leads 29-13)
- **Score patterns**: 78% of Bo3 sets are 2-0 sweeps; 51% of Bo5s are 3-0
- **Character data**: Only 8.1% of sets have game-level character picks (but 92% of those games have full data)

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
uv run python scripts/dagster_ops.py backfill --start 2025-01-06 --end 2025-03-31   # ingest
uv run python scripts/dagster_ops.py launch train_matchup_model                      # train
```

Then open http://localhost:8501 to predict a set.

## Tools & Technologies

- **Ingestion**: start.gg GraphQL API, Kafka (KRaft), confluent-kafka
- **Lakehouse**: Apache Iceberg (Lakekeeper REST catalog, Silo S3), Spark 4.0, PyIceberg, Trino
- **Orchestration**: Dagster (partitions, sensors, declarative automation, asset checks)
- **Modeling**: scikit-learn, XGBoost, SHAP, Optuna
- **Experiment Tracking and Registry**: MLflow
- **App**: Streamlit

## License

MIT
