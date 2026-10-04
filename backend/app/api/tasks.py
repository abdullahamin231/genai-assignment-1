"""Task endpoints: runtime corruption + the four model workspaces."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from .. import pipeline as P
from .. import imaging, metrics
from ..config import CLASS_LABELS, CLASS_NAMES, STYLE_NAMES
from ..onnx_runtime import REGISTRY
from ..pipeline import BadRequest

router = APIRouter(prefix="/api", tags=["tasks"])


def _opt_int(value: Optional[str], name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"Invalid value for '{name}'.") from exc


def _opt_float(value: Optional[str], name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"Invalid value for '{name}'.") from exc


def _opt_int_opt(value: Optional[str]) -> int | None:
    return _opt_int(value, "level")


async def _inputs(file: UploadFile, reference: Optional[UploadFile], corruption: str,
                  level: Optional[str], severity: Optional[str], seed: Optional[str]):
    original = await P.read_image(file)
    if original is None:
        raise BadRequest("An input image is required.")
    ref = await P.read_image(reference, "reference") if reference is not None else None
    corrupted, desc = P.corrupt_image(
        original,
        corruption,
        level=_opt_int_opt(level),
        severity=_opt_float(severity, "severity"),
        seed=_opt_int(seed, "seed"),
    )
    return original, ref, corrupted, desc


def _timing(started: float, inference_ms: float) -> dict:
    return {
        "inference_ms": round(float(inference_ms), 2),
        "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@router.post("/corrupt")
async def corrupt_only(
    file: UploadFile = File(...),
    corruption: str = Form("salt"),
    level: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
):
    """Apply a runtime corruption without running a model (used by the demo flow)."""
    original = await P.read_image(file)
    if original is None:
        raise BadRequest("An input image is required.")
    corrupted, desc = P.corrupt_image(
        original, corruption,
        level=_opt_int_opt(level),
        severity=_opt_float(severity, "severity"),
        seed=_opt_int(seed, "seed"),
    )
    return {
        "input": imaging.to_data_url(original),
        "corrupted": imaging.to_data_url(corrupted),
        "error_map": imaging.to_data_url(metrics.error_map(original, corrupted)),
        "corruption": desc,
        "metrics": metrics.compare(original, corrupted),
    }


@router.post("/universal-restoration")
async def universal_restoration(
    file: UploadFile = File(...),
    corruption: str = Form("none"),
    level: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
):
    """Task 1 - single universal denoising autoencoder."""
    started = time.perf_counter()
    original, ref, corrupted, desc = await _inputs(file, reference, corruption, level, severity, seed)
    model = REGISTRY.get("task1")
    output, ms = P.timed(P.run_autoencoder, model, corrupted)
    return P.payload(
        task="task1",
        original=original,
        corrupted=corrupted,
        output=output,
        corruption=desc,
        timing=_timing(started, ms),
        quality_metrics=P.build_metrics(original, corrupted, output, ref),
        model=model,
        error_reference=ref,
    )


@router.post("/hard-routing")
async def hard_routing(
    file: UploadFile = File(...),
    corruption: str = Form("none"),
    level: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    routing_mode: str = Form("predicted"),
    oracle_label: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
):
    """Task 2 - corruption classifier + hard-routed specialist autoencoders."""
    started = time.perf_counter()
    original, ref, corrupted, desc = await _inputs(file, reference, corruption, level, severity, seed)

    clf = REGISTRY.get("task2_clf")
    probs, ms_clf = P.timed(P.run_classifier, clf, corrupted)
    predicted = int(probs.argmax())

    mode = (routing_mode or "predicted").lower()
    if mode not in ("predicted", "oracle"):
        raise BadRequest("routing_mode must be 'predicted' or 'oracle'.")
    if mode == "oracle":
        label = (oracle_label or "").strip().lower()
        if label not in CLASS_NAMES:
            raise BadRequest("oracle_label must be one of: clean, salt, blur, occlusion.")
        target = CLASS_NAMES.index(label)
    else:
        target = predicted

    expert_file = None
    if target == 0:  # identity bypass for clean inputs
        output, ms_spec = corrupted.copy(), 0.0
        expert = "identity"
    else:
        keys = {1: "task2_salt", 2: "task2_blur", 3: "task2_occlusion"}
        spec_model = REGISTRY.get(keys[target])
        expert_file = spec_model.meta.filename
        output, ms_spec = P.timed(P.run_autoencoder, spec_model, corrupted)
        expert = spec_model.meta.filename.replace(".onnx", "")

    prob_rows = [
        {"index": i, "name": n, "label": CLASS_LABELS[n], "prob": round(float(probs[i]), 5)}
        for i, n in enumerate(CLASS_NAMES)
    ]
    return P.payload(
        task="task2",
        original=original,
        corrupted=corrupted,
        output=output,
        corruption=desc,
        timing={
            "classifier_ms": round(float(ms_clf), 2),
            "specialist_ms": round(float(ms_spec), 2),
            "inference_ms": round(float(ms_clf + ms_spec), 2),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
        },
        quality_metrics=P.build_metrics(original, corrupted, output, ref),
        model=clf,
        extra={
            "probs": prob_rows,
            "predicted": P.class_summary(predicted, probs),
            "selected": P.class_summary(target, probs),
            "confidence": round(float(probs[predicted]), 5),
            "routing_mode": mode,
            "expert": expert,
            "expert_model": expert_file,
            "identity_bypass": target == 0,
        },
        error_reference=ref,
    )


@router.post("/soft-mixture")
async def soft_mixture(
    file: UploadFile = File(...),
    corruption: str = Form("none"),
    level: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    tau: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
):
    """Task 3 - jointly trained soft mixture-of-experts pipeline."""
    started = time.perf_counter()
    original, ref, corrupted, desc = await _inputs(file, reference, corruption, level, severity, seed)
    tau_val = _opt_float(tau, "tau")

    model = REGISTRY.get("task3")
    (output, weights, routing), ms = P.timed(P.run_soft_moe, model, corrupted, tau_val)

    weight_rows = [
        {"index": i, "name": n, "label": CLASS_LABELS[n], "weight": round(float(weights[i]), 5)}
        for i, n in enumerate(CLASS_NAMES)
    ]
    dominant = int(weights.argmax())
    return P.payload(
        task="task3",
        original=original,
        corrupted=corrupted,
        output=output,
        corruption=desc,
        timing=_timing(started, ms),
        quality_metrics=P.build_metrics(original, corrupted, output, ref),
        model=model,
        extra={
            "weights": weight_rows,
            "dominant": P.class_summary(dominant, weights),
            "routing": routing,
            "tau": tau_val,
        },
        error_reference=ref,
    )


@router.post("/face-to-sketch")
async def face_to_sketch(
    file: UploadFile = File(...),
    style: str = Form("0"),
):
    """Task 4 - style-conditioned face-to-sketch conditional GAN."""
    started = time.perf_counter()
    photo = await P.read_image(file)
    if photo is None:
        raise BadRequest("A facial photograph is required.")
    style_idx = _opt_int(style, "style")
    if style_idx is None or not 0 <= style_idx <= 2:
        raise BadRequest("style must be 0, 1 or 2 (Style 1 / Style 2 / Style 3).")

    model = REGISTRY.get("task4")
    sketch, ms = P.timed(P.run_generator, model, photo, style_idx)
    blank = photo.copy()
    return P.payload(
        task="task4",
        original=photo,
        corrupted=blank,
        output=sketch,
        corruption={"type": "style", "label": STYLE_NAMES[style_idx], "style": style_idx},
        timing=_timing(started, ms),
        quality_metrics={"mode": "none", "input": None, "output": None, "delta_ssim": None, "delta_psnr": None},
        model=model,
        extra={
            "style": {"index": style_idx, "name": STYLE_NAMES[style_idx]},
            "pair": [imaging.to_data_url(photo), imaging.to_data_url(sketch)],
        },
        error_reference=None,
        include_error_map=False,
    )
