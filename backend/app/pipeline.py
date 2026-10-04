"""Shared request handling for the four task endpoints."""
from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np
from fastapi import UploadFile

from . import corrupt as C
from . import imaging, metrics
from .config import CLASS_LABELS, IMG_SIZE, MAX_UPLOAD_MB
from .onnx_runtime import OnnxModel

TASK_LABELS = {
    "task1": "Universal Restoration",
    "task2": "Hard-Routed Restoration",
    "task3": "Soft Mixture-of-Experts Restoration",
    "task4": "Face-to-Sketch Generator",
}


class BadRequest(ValueError):
    """Client-side error -> HTTP 400."""


async def read_image(file: UploadFile | None, field: str = "file") -> np.ndarray | None:
    if file is None:
        return None
    data = await file.read()
    if not data:
        return None
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise BadRequest(f"'{field}' exceeds the {MAX_UPLOAD_MB:g} MB upload limit.")
    try:
        return imaging.resize(imaging.decode(data))
    except imaging.ImageError as exc:
        raise BadRequest(str(exc)) from exc


def parse_corruptions(corruption: str | None) -> list[str]:
    """``"salt,blur"`` -> ``["salt", "blur"]`` (compound corruption for Task 3)."""
    raw = (corruption or "none").replace(";", ",").split(",")
    out = []
    for item in raw:
        name = item.strip().lower()
        if not name or name in ("none", "clean", ""):
            continue
        if name not in C.CLASS_IDX:
            raise BadRequest(f"Unknown corruption '{name}'. Use clean, salt, blur or occlusion.")
        if name not in out:
            out.append(name)
    return out


def corrupt_image(
    img: np.ndarray,
    corruption: str | None,
    level: int | None = None,
    severity: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    kinds = parse_corruptions(corruption)
    specs = C.build_specs(kinds, level=level, severity=severity, seed=seed)
    arr, desc = C.apply(img, specs)
    corrupted = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return corrupted, desc


def quality(reference: np.ndarray, produced: np.ndarray) -> dict:
    return metrics.compare(reference, produced)


def build_metrics(
    original: np.ndarray,
    corrupted: np.ndarray,
    output: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict:
    """Reference metrics when a clean target is supplied, otherwise input-vs-output."""
    if reference is not None:
        return {
            "mode": "reference",
            "input": quality(reference, corrupted),
            "output": quality(reference, output),
            "delta_ssim": round(quality(reference, output)["ssim"] - quality(reference, corrupted)["ssim"], 4),
            "delta_psnr": round(quality(reference, output)["psnr"] - quality(reference, corrupted)["psnr"], 3),
        }
    return {"mode": "input_output", "input": None, "output": quality(corrupted, output), "delta_ssim": None, "delta_psnr": None}


def run_autoencoder(model: OnnxModel, corrupted: np.ndarray) -> np.ndarray:
    feed = imaging.to_chw(corrupted, "unit").astype(np.float32)
    out = model.run({model.image_input: feed})
    name = model.output_name("image", ("restored", "output", "recon", "x_hat", "generated")) or next(iter(out))
    return imaging.from_chw(out[name], "unit")


def run_classifier(model: OnnxModel, corrupted: np.ndarray) -> np.ndarray:
    feed = imaging.to_chw(corrupted, "unit").astype(np.float32)
    out = model.run({model.image_input: feed})
    name = model.output_name("vector", ("prob", "softmax", "logit", "gate", "weight")) or next(iter(out))
    probs = np.asarray(out[name], dtype=np.float64).reshape(-1)
    if probs.size != 4:
        raise RuntimeError(f"Classifier returned {probs.size} values, expected 4.")
    total = float(probs.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-3) or probs.min() < 0:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    else:
        probs = probs / total
    return probs


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    tau = max(float(tau), 1e-4)
    z = x / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_soft_moe(model: OnnxModel, corrupted: np.ndarray, tau: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (restored, weights[4], routing info)."""
    feed = imaging.to_chw(corrupted, "unit").astype(np.float32)
    feeds = {model.image_input: feed}
    fed_tau = False
    for name in model.extra_inputs:
        low = name.lower()
        if low in ("tau", "temperature", "temp"):
            feeds[name] = np.array([1.0 if tau is None else float(tau)], dtype=np.float32)
            fed_tau = True
    out = model.run(feeds)

    img_name = model.output_name("image", ("restored", "output", "recon", "x_hat", "combined", "generated"))
    vec_name = model.output_name("vector", ("weight", "w", "gate", "routing", "prob", "logit"))
    if img_name is None:
        raise RuntimeError("The soft MoE model did not return an image-shaped output.")

    restored = imaging.from_chw(out[img_name], "unit")
    info = {"image_output": img_name, "weight_output": vec_name, "temperature": tau, "fed_tau": fed_tau}
    if vec_name is None:
        return restored, np.full(4, 0.25), info

    raw = np.asarray(out[vec_name], dtype=np.float64).reshape(-1)
    info["raw"] = [round(float(v), 5) for v in raw]
    if "logit" in vec_name.lower():
        # the model already consumed tau when it exposes a tau input
        weights = softmax(raw, 1.0 if fed_tau or tau is None else tau)
        info["applied"] = "softmax"
    else:
        total = float(raw.sum())
        weights = raw / total if abs(total - 1.0) > 1e-3 and total > 0 else raw
        weights = np.clip(weights, 0.0, None)
        s = float(weights.sum())
        weights = weights / s if s > 0 else np.full(4, 0.25)
        info["applied"] = "as_exported"
    return restored, weights.astype(np.float64), info


def run_generator(model: OnnxModel, photo: np.ndarray, style: int) -> np.ndarray:
    feed = imaging.to_chw(photo, "minus").astype(np.float32)
    feeds = {model.image_input: feed}
    for name in model.extra_inputs:
        low = name.lower()
        if low in ("style", "style_id", "label", "condition", "s"):
            feeds[name] = np.array([int(style)], dtype=np.int64)
    out = model.run(feeds)
    name = model.output_name("image", ("sketch", "output", "generated", "y_hat", "restored")) or next(iter(out))
    return imaging.from_chw(out[name], "minus")


def class_summary(index: int, probs: Iterable[float] | None = None) -> dict:
    key = C.CLASS_NAMES[index]
    row = {"index": index, "name": key, "label": CLASS_LABELS[key]}
    if probs is not None:
        row["prob"] = round(float(list(probs)[index]), 5)
    return row


def payload(
    *,
    task: str,
    original: np.ndarray,
    corrupted: np.ndarray,
    output: np.ndarray,
    corruption: dict,
    timing: dict,
    quality_metrics: dict,
    model: OnnxModel | None,
    extra: dict | None = None,
    error_reference: np.ndarray | None = None,
    include_error_map: bool = True,
) -> dict:
    if not include_error_map:
        error_map = None
    else:
        # against the clean reference when one was supplied, otherwise against the
        # corrupted network input (i.e. "what the model changed")
        target = error_reference if error_reference is not None else corrupted
        error_map = imaging.to_data_url(metrics.error_map(target, output))
    body = {
        "task": task,
        "workspace": TASK_LABELS[task],
        "input": imaging.to_data_url(original),
        "corrupted": imaging.to_data_url(corrupted),
        "output": imaging.to_data_url(output),
        "error_map": error_map,
        "error_reference": "reference" if error_reference is not None else "input",
        "corruption": corruption,
        "timing": timing,
        "metrics": quality_metrics,
        "size": [IMG_SIZE, IMG_SIZE],
        "model": {"file": model.meta.filename, "path": str(model.path)} if model else None,
    }
    if extra:
        body.update(extra)
    return body


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0
