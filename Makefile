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
