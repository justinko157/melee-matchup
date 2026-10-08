# One image for the Dagster code location, webserver, daemon and Spark jobs.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless curl procps \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    JAVA_HOME=/usr/lib/jvm/default-java \
    SPARK_CONF_DIR=/opt/spark-conf \
    PYSPARK_PYTHON=/opt/venv/bin/python \
    PYTHONPATH=/app \
    DAGSTER_HOME=/opt/dagster/home

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --extra pipeline --extra dev

COPY infra/spark/spark-defaults.conf /opt/spark-conf/spark-defaults.conf
COPY infra/spark/warm_ivy_cache.py /tmp/warm_ivy_cache.py
RUN python /tmp/warm_ivy_cache.py && mkdir -p /opt/dagster/home

COPY pipeline ./pipeline
COPY tests ./tests
COPY infra/spark ./infra/spark
