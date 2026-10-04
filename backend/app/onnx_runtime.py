"""ONNX Runtime model registry.

Models are discovered from :data:`config.MODEL_DIRS`, loaded once and reused.
Input/output tensors are resolved by name *and* by shape so that the API keeps
working if Task 3's exporter chooses different tensor names (it is trained
independently of this backend).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import EXPECTED_MODELS, IMG_SIZE, MODEL_DIRS, ModelKey, ORT_PROVIDERS

IMAGE_INPUT_RANK = 4


def find_model(filename: str) -> Path | None:
    for d in MODEL_DIRS:
        p = d / filename
        if p.is_file():
            return p
    return None


def describe_shape(shape) -> list:
    out = []
    for s in shape:
        out.append(int(s) if isinstance(s, (int, np.integer)) and s > 0 else None)
    return out


@dataclass
class OnnxModel:
    meta: ModelKey
    path: Path
    session: ort.InferenceSession
    inputs: dict[str, list] = field(default_factory=dict)
    outputs: dict[str, list] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)

    # --- tensor resolution -------------------------------------------------
    @property
    def image_input(self) -> str:
        for name, shape in self.inputs.items():
            if len(shape) == IMAGE_INPUT_RANK and (shape[1] in (None, 1, 3)):
                return name
        return next(iter(self.inputs))

    @property
    def extra_inputs(self) -> dict[str, list]:
        return {k: v for k, v in self.inputs.items() if k != self.image_input}

    def outputs_of_rank(self, rank: int) -> dict[str, list]:
        return {k: v for k, v in self.outputs.items() if len(v) == rank}

    @property
    def image_outputs(self) -> dict[str, list]:
        return {k: v for k, v in self.outputs.items() if len(v) == IMAGE_INPUT_RANK}

    @property
    def vector_outputs(self) -> dict[str, list]:
        return {k: v for k, v in self.outputs.items() if len(v) == 2}

    def output_name(self, kind: str, preferred: tuple[str, ...] = ()) -> str | None:
        """Pick an output by name hints first, then by shape.

        ``kind`` is ``image`` (rank-4) or ``vector`` (rank-2, 4 wide).
        """
        pool = self.image_outputs if kind == "image" else self.vector_outputs
        if not pool:
            return None
        for hint in preferred:
            for name in pool:
                if hint in name.lower():
                    return name
        if kind == "vector":
            for name, shape in pool.items():
                if shape[1] in (None, 4):
                    return name
        else:
            for name, shape in pool.items():
                if shape[1] in (None, 3):
                    return name
        return next(iter(pool))

    # --- inference ---------------------------------------------------------
    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ordered = {k: v for k, v in feeds.items() if k in self.inputs}
        names = [o.name for o in self.session.get_outputs()]
        values = self.session.run(names, ordered)
        return dict(zip(names, values))

    def info(self) -> dict:
        return {
            "key": self.meta.key,
            "file": self.meta.filename,
            "title": self.meta.title,
            "task": self.meta.task,
            "path": str(self.path),
            "present": True,
            "loaded": True,
            "inputs": {k: describe_shape(v) for k, v in self.inputs.items()},
            "outputs": {k: describe_shape(v) for k, v in self.outputs.items()},
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, OnnxModel] = {}
        self._errors: dict[str, str] = {}

    # --- loading -----------------------------------------------------------
    def load(self, meta: ModelKey) -> OnnxModel | None:
        with self._lock:
            if meta.key in self._models:
                return self._models[meta.key]
            path = find_model(meta.filename)
            if path is None:
                self._errors.pop(meta.key, None)
                return None
            try:
                sess = ort.InferenceSession(str(path), providers=list(ORT_PROVIDERS))
                model = OnnxModel(
                    meta=meta,
                    path=path,
                    session=sess,
                    inputs={i.name: list(i.shape) for i in sess.get_inputs()},
                    outputs={o.name: list(o.shape) for o in sess.get_outputs()},
                )
            except Exception as exc:  # noqa: BLE001
                self._errors[meta.key] = f"{type(exc).__name__}: {exc}"
                return None
            self._models[meta.key] = model
            self._errors.pop(meta.key, None)
            return model

    def load_all(self) -> None:
        for meta in EXPECTED_MODELS:
            self.load(meta)

    # --- access ------------------------------------------------------------
    def get(self, key: str, required: bool = True) -> OnnxModel | None:
        meta = next((m for m in EXPECTED_MODELS if m.key == key), None)
        if meta is None:  # pragma: no cover - programming error
            raise KeyError(f"Unknown model key '{key}'")
        model = self._models.get(key) or self.load(meta)
        if model is None and required:
            raise ModelMissing(meta, self._errors.get(key))
        return model

    def status(self) -> list[dict]:
        rows = []
        for meta in EXPECTED_MODELS:
            model = self._models.get(meta.key)
            path = find_model(meta.filename)
            rows.append(
                {
                    "key": meta.key,
                    "file": meta.filename,
                    "title": meta.title,
                    "task": meta.task,
                    "present": path is not None,
                    "loaded": model is not None,
                    "path": str(path) if path else None,
                    "inputs": {k: describe_shape(v) for k, v in (model.inputs if model else {}).items()},
                    "outputs": {k: describe_shape(v) for k, v in (model.outputs if model else {}).items()},
                    "error": self._errors.get(meta.key),
                }
            )
        return rows

    def search_paths(self) -> list[str]:
        return [str(p) for p in MODEL_DIRS]

    def ready_for(self, task: str) -> bool:
        wanted = [m for m in EXPECTED_MODELS if m.task == task]
        return all(find_model(m.filename) is not None for m in wanted)


class ModelMissing(RuntimeError):
    def __init__(self, meta: ModelKey, detail: str | None = None) -> None:
        self.meta = meta
        msg = f"Model file '{meta.filename}' was not found."
        if detail:
            msg += f" ({detail})"
        msg += " Download the exported ONNX files and place them in one of: "
        msg += ", ".join(str(p) for p in MODEL_DIRS)
        super().__init__(msg)


REGISTRY = ModelRegistry()


def empty_feed(batch: int = 1) -> np.ndarray:
    return np.zeros((batch, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
