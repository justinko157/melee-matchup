SHELL := bash
.PHONY: install collect features validate train tune export app test lint format docker-build docker-run clean up down nuke ps logs check-lakehouse test-spark bronze-once smoke

# ── Setup ──────────────────────────────────────────────
install:
	uv sync --all-extras

# ── Local stack ────────────────────────────────────────
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env

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
	$(COMPOSE) exec -T trino trino --execute "SELECT * FROM lakekeeper.healthcheck.ping"
	$(COMPOSE) exec -T trino trino --execute "DROP TABLE lakekeeper.healthcheck.ping"
	$(COMPOSE) exec -T trino trino --execute "DROP SCHEMA lakekeeper.healthcheck"

test-spark:
	$(COMPOSE) run --rm --no-deps pipeline-tools pytest -m spark -v tests/pipeline

smoke:
	bash scripts/smoke.sh

bronze-once:
	$(COMPOSE) run --rm pipeline-tools spark-submit pipeline/spark_jobs/bronze_job.py \
		--bootstrap-servers kafka:9092 --checkpoint-root /checkpoints/bronze

# ── Data Pipeline ──────────────────────────────────────
collect:
	uv run python -m src.collect --start-date 2018-01-01 --min-attendees 50

features:
	uv run python -m src.features

validate:
	uv run python -m src.validation

# ── Modeling ───────────────────────────────────────────
train:
	uv run python -m src.model

tune:
	uv run python -m src.tuning --n-trials 50

export:
	uv run python -m src.export_app_data

# ── App ────────────────────────────────────────────────
app:
	uv run streamlit run app.py

# ── Quality ────────────────────────────────────────────
test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/ app.py pipeline/ scripts/

format:
	uv run ruff format pipeline tests/pipeline tests/smoke scripts

# ── Docker ─────────────────────────────────────────────
docker-build:
	docker build -t melee-matchup .

docker-run:
	docker run -p 8501:8501 melee-matchup

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf mlruns/ .pytest_cache/ __pycache__/ src/__pycache__/
	find . -name "*.pyc" -delete
