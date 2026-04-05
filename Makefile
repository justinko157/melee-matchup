.PHONY: install collect features validate train tune export app test lint docker-build docker-run clean

# ── Setup ──────────────────────────────────────────────
install:
	pip install -e ".[dev,ml,app]"

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
