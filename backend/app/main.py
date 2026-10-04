"""FastAPI application: health check + the four assignment workspaces."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import health, tasks
from .config import CORS_ORIGINS, EXPECTED_MODELS, IMG_SIZE, MODEL_DIRS, REPO_ROOT
from .onnx_runtime import REGISTRY, ModelMissing
from .pipeline import BadRequest

log = logging.getLogger("genai.lab")
if not log.handlers:  # keep the startup summary visible under uvicorn's logging config
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False

app = FastAPI(
    title="GenAI Lab - Generative Image Restoration Suite",
    description=(
        "ONNX inference backend for Assignment #1: universal restoration (Task 1), "
        "hard-routed specialists (Task 2), soft mixture-of-experts (Task 3) and "
        "style-conditioned face-to-sketch generation (Task 4)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(tasks.router)


@app.exception_handler(BadRequest)
async def _bad_request(_req: Request, exc: BadRequest) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelMissing)
async def _missing(_req: Request, exc: ModelMissing) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "file": exc.meta.filename})


@app.exception_handler(Exception)
async def _boom(_req: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
    log.exception("unhandled error")
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) > 600:
        detail = detail[:600] + " ... (truncated)"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.on_event("startup")
def _startup() -> None:
    REGISTRY.load_all()
    status = REGISTRY.status()
    ready = [r["file"] for r in status if r["present"]]
    missing = [r["file"] for r in status if not r["present"]]
    log.info("repository: %s", REPO_ROOT)
    log.info("model search paths: %s", ", ".join(str(p) for p in MODEL_DIRS))
    log.info("models loaded: %s", ", ".join(ready) or "none")
    if missing:
        log.warning("models missing: %s", ", ".join(missing))


@app.get("/")
def root() -> dict:
    return {
        "name": "GenAI Lab API",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "GET  /api/health",
            "GET  /api/samples",
            "GET  /api/metrics/{task1|task2|task4}",
            "GET  /api/optuna",
            "POST /api/corrupt",
            "POST /api/universal-restoration",
            "POST /api/hard-routing",
            "POST /api/soft-mixture",
            "POST /api/face-to-sketch",
        ],
        "image_size": IMG_SIZE,
        "expected_models": [m.filename for m in EXPECTED_MODELS],
    }
