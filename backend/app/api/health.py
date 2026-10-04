"""Health check, recorded experiment metrics and bundled sample images."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from ..config import IMG_SIZE, REPO_ROOT, SAMPLES_DIR
from ..onnx_runtime import REGISTRY

router = APIRouter(prefix="/api", tags=["system"])

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

METRIC_FILES = {
    "task1": REPO_ROOT / "restoration" / "results" / "task1_test_metrics.json",
    "task2": REPO_ROOT / "restoration" / "results" / "task2_test_metrics.json",
    # published by restoration/task3_evaluate.py; 404 until that has been run
    "task3": REPO_ROOT / "restoration" / "results" / "task3_test_metrics.json",
    "task4": REPO_ROOT / "task4" / "results" / "task4_test_metrics.json",
}

OPTUNA_FILES = {
    "task1": REPO_ROOT / "restoration" / "configs" / "task1_optuna_summary.json",
    "task2": REPO_ROOT / "restoration" / "configs" / "task2_clf_optuna_summary.json",
    "task2_specialists": REPO_ROOT / "restoration" / "configs" / "task2_spec_optuna_summary.json",
    "task4": REPO_ROOT / "task4" / "configs" / "best_params.json",
}


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "onnxruntime": ort.__version__, "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import fastapi

        out["fastapi"] = fastapi.__version__
    except Exception:  # pragma: no cover
        pass
    return out


def _by_task(status: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {"task1": [], "task2": [], "task3": [], "task4": []}
    for row in status:
        grouped.setdefault(row["task"], []).append(row)
    return {
        task: {"ready": bool(rows) and all(r["present"] for r in rows),
               "ready_count": sum(1 for r in rows if r["present"]), "total": len(rows)}
        for task, rows in grouped.items()
    }


@router.get("/health")
def health() -> dict:
    REGISTRY.load_all()
    status = REGISTRY.status()
    ready = sum(1 for r in status if r["present"])
    return {
        "status": "ok",
        "ready": ready == len(status),
        "models_ready": ready,
        "models_total": len(status),
        "models": status,
        "tasks": _by_task(status),
        "search_paths": REGISTRY.search_paths(),
        "image_size": IMG_SIZE,
        "samples": len(_sample_files()),
        "versions": _versions(),
    }


@router.get("/metrics/{task}")
def task_metrics(task: str) -> dict:
    path = METRIC_FILES.get(task)
    if path is None or not path.is_file():
        raise HTTPException(404, f"No recorded metrics for '{task}'. Train and evaluate the task first.")
    return {"task": task, "source": str(path.relative_to(REPO_ROOT)), "metrics": _load(path)}


@router.get("/optuna")
def optuna_summaries() -> dict:
    found = {}
    for name, path in OPTUNA_FILES.items():
        if path.is_file():
            found[name] = _load(path)
    return {"studies": found}


def _load(path: Path):
    # Recorded metrics contain `NaN` (e.g. PSNR on a perfect clean match), which is
    # not valid JSON - map bare constants to null so the payload stays standard.
    def _hook(_constant: str):
        return None

    with open(path) as fh:
        return json.load(fh, parse_constant=_hook)


# --------------------------------------------------------------------------
# Bundled sample images (populated by scripts/prepare_samples.py)
# --------------------------------------------------------------------------
def _sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


@router.get("/samples")
def samples() -> dict:
    files = _sample_files()
    return {
        "samples": [{"id": p.stem, "label": p.stem.replace("_", " "), "url": f"/api/samples/{p.stem}"} for p in files],
        "directory": str(SAMPLES_DIR),
        "hint": "Run scripts/prepare_samples.py to bundle clean images from the dataset.",
    }


@router.get("/samples/{sample_id}")
def sample_image(sample_id: str) -> Response:
    for p in _sample_files():
        if p.stem == sample_id:
            return FileResponse(p, media_type=f"image/{p.suffix.lower().lstrip('.')}")
    raise HTTPException(404, f"Sample '{sample_id}' not found.")
