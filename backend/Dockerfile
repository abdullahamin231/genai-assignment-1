# syntax=docker/dockerfile:1
# FastAPI + ONNX Runtime inference backend.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

# libgomp1: OpenMP runtime used by onnxruntime / opencv wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Application code
COPY backend/ /app/backend/

# Shared corruption module (imported by the API, no training code pulled in)
COPY restoration/corruptions.py /app/restoration/corruptions.py

# Recorded evaluation results / Optuna summaries surfaced by GET /api/metrics and /api/optuna
COPY restoration/results/*.json /app/restoration/results/
COPY restoration/configs/*.json /app/restoration/configs/
COPY task4/results/*.json /app/task4/results/
COPY task4/configs/*.json /app/task4/configs/

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health >/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
