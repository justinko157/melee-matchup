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
