SHELL := bash
.PHONY: install collect features validate train tune export app test lint docker-build docker-run clean up down nuke ps logs check-lakehouse test-spark

# ── Setup ──────────────────────────────────────────────
install:
	pip install -e ".[dev,ml,app]"

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

# ── Data Pipeline ──────────────────────────────────────
collect:
	python -m src.collect --start-date 2018-01-01 --min-attendees 50

features:
	python -m src.features

validate:
	python -m src.validation

# ── Modeling ───────────────────────────────────────────
train:
	python -m src.model

tune:
	python -m src.tuning --n-trials 50

export:
	python -m src.export_app_data

# ── App ────────────────────────────────────────────────
app:
	streamlit run app.py

# ── Quality ────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ app.py

format:
	ruff format src/ tests/ app.py

# ── Docker ─────────────────────────────────────────────
docker-build:
	docker build -t melee-matchup .

docker-run:
	docker run -p 8501:8501 melee-matchup

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf mlruns/ .pytest_cache/ __pycache__/ src/__pycache__/
	find . -name "*.pyc" -delete
