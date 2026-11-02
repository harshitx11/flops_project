# syntax=docker/dockerfile:1.7

# ---- base: Python + locked runtime dependencies --------------------------
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=120 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Dependencies first so this layer is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src

# ---- trainer: download the public dataset and train the model ------------
FROM base AS trainer
ARG SENTIMENT_MIN_ACCEPTED_F1=0.75
ENV SENTIMENT_MIN_ACCEPTED_F1=${SENTIMENT_MIN_ACCEPTED_F1}
RUN python -m sentiment_mlops.data.download_data \
    && python -m sentiment_mlops.models.train > /dev/null

# ---- runtime: API + frontend + accepted model only -----------------------
FROM base AS runtime

COPY frontend ./frontend
COPY --from=trainer /app/artifacts ./artifacts
COPY --from=trainer /app/reports ./reports

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"]

CMD ["uvicorn", "sentiment_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
