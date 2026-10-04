"""Central configuration and path discovery for the GenAI Lab backend.

Everything is resolved relative to the repository root so the same code works
both on the host (`uvicorn app.main:app` from ``backend/``) and inside the
Docker container (``/app``).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# --- paths -----------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]  # .../backend  (or /app/backend)
REPO_ROOT = BACKEND_ROOT.parent  # repository root    (or /app)
PKG_ROOT = Path(__file__).resolve().parent

IMG_SIZE = int(os.environ.get("IMG_SIZE", "128"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "12"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
ORT_PROVIDERS = [p.strip() for p in os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]

# Candidate locations for the exported ONNX files, searched in order.
MODEL_DIRS: list[Path] = []
for _p in (
    os.environ.get("MODEL_DIR"),
    REPO_ROOT / "models",
    REPO_ROOT / "restoration_outputs" / "models",  # RESTORATION_OUT / "models"
    REPO_ROOT / "task4_outputs" / "models",  # T4_OUT / "models"
    BACKEND_ROOT / "models",
):
    if not _p:
        continue
    _path = Path(_p).expanduser()
    if _path not in MODEL_DIRS:
        MODEL_DIRS.append(_path)

SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", BACKEND_ROOT / "samples"))

# Models the application expects, in the order they are presented in the UI.
@dataclass(frozen=True)
class ModelKey:
    key: str
    filename: str
    task: str
    title: str


EXPECTED_MODELS: tuple[ModelKey, ...] = (
    ModelKey("task1", "task1_universal_ae.onnx", "task1", "Universal restoration autoencoder"),
    ModelKey("task2_clf", "task2_classifier.onnx", "task2", "Corruption classifier"),
    ModelKey("task2_salt", "task2_specialist_salt.onnx", "task2", "Salt & pepper specialist"),
    ModelKey("task2_blur", "task2_specialist_blur.onnx", "task2", "Gaussian blur specialist"),
    ModelKey("task2_occlusion", "task2_specialist_occlusion.onnx", "task2", "Occlusion specialist"),
    ModelKey("task3", "task3_soft_moe.onnx", "task3", "Soft mixture-of-experts pipeline"),
    ModelKey("task4", "task4_generator.onnx", "task4", "Face-to-sketch generator"),
)

CLASS_NAMES = ("clean", "salt", "blur", "occlusion")
CLASS_LABELS = {
    "clean": "Clean",
    "salt": "Salt & Pepper",
    "blur": "Gaussian Blur",
    "occlusion": "Occlusion",
}
CLASS_ICONS = {
    "clean": "check_circle",
    "salt": "grain",
    "blur": "blur_on",
    "occlusion": "crop_square",
}

STYLE_NAMES = {0: "Style 1", 1: "Style 2", 2: "Style 3"}

# Make ``restoration/corruptions.py`` importable without pulling in the
# training-only modules (``restoration/config.py`` creates Colab paths on import).
_CORRUPTION_DIR = REPO_ROOT / "restoration"
if _CORRUPTION_DIR.is_dir() and str(_CORRUPTION_DIR) not in sys.path:
    sys.path.append(str(_CORRUPTION_DIR))
