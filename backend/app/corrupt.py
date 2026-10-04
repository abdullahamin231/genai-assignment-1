"""Thin bridge to ``restoration/corruptions.py`` (the single source of truth).

The corruption definitions are shared with training so that the runtime
corruption offered through the UI matches the data the models were trained on.
"""
from __future__ import annotations

import numpy as np

from .config import IMG_SIZE

try:  # pragma: no cover - import path is prepared in config.py
    import corruptions as C
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "restoration/corruptions.py could not be imported. Run the backend from the "
        "repository (or use the Docker image, which vendors the file)."
    ) from exc

CLASS_NAMES = tuple(C.CLASS_NAMES)
CLASS_IDX = dict(C.CLASS_IDX)
LEVELS = tuple(C.LEVEL_NAMES)


def _interpolate_severity(kind: str, s: float) -> dict:
    """Map a continuous 0..1 severity to a spec inside the assignment ranges."""
    s = float(np.clip(s, 0.0, 1.0))
    rng = np.random.default_rng(int(np.random.default_rng().integers(0, 2**31 - 1)))
    if kind == "salt":
        return {"type": "salt", "p": round(0.02 + s * (0.15 - 0.02), 4), "seed": int(rng.integers(0, 2**31 - 1))}
    if kind == "blur":
        k = (3, 5, 7)[int(round(s * 2))]  # 0 -> 3, 0.5 -> 5, 1 -> 7
        return {"type": "blur", "k": int(k), "sigma": round(0.5 + s * (2.5 - 0.5), 3)}
    if kind == "occlusion":
        k = int(1 + round(s * 2))
        target = 0.10 + s * (0.35 - 0.10)
        rects, cov = C.make_occlusion_rects(rng, target, k, bounds=(0.10, 0.35))
        return {"type": "occlusion", "rects": rects, "coverage": round(float(cov), 4)}
    return {"type": "clean"}


def spec_for(kind: str, level: int | None = None, severity: float | None = None, seed: int | None = None) -> dict:
    """Build a deterministic corruption spec.

    ``level`` (0/1/2 = low/medium/high) uses the fixed test severities from the
    assignment; otherwise a continuous ``severity`` in [0, 1] is interpolated
    within the documented training ranges.
    """
    kind = "clean" if kind in ("", "none", "clean") else kind
    if kind not in CLASS_IDX:
        raise ValueError(f"Unknown corruption '{kind}'.")
    if kind == "clean":
        return {"type": "clean"}
    if level is not None:
        lv = int(np.clip(level, 0, 2))
        return C.fixed_spec(CLASS_IDX[kind], lv, int(seed) if seed is not None else 42)
    return _interpolate_severity(kind, 0.5 if severity is None else severity)


def build_specs(kinds: list[str], level: int | None, severity: float | None, seed: int | None) -> list[dict]:
    """One spec per requested corruption type (compound corruption is allowed for Task 3)."""
    seen: list[dict] = []
    for k in kinds:
        k = (k or "").strip().lower()
        if not k or k == "none":
            continue
        seen.append(spec_for(k, level=level, severity=severity, seed=seed))
    if not seen:
        seen.append({"type": "clean"})
    return seen


def apply(img: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, dict]:
    """Apply every spec in sequence. Returns (corrupted float image, description)."""
    out = np.ascontiguousarray(img.astype(np.float32) / 255.0)
    parts = []
    for spec in specs:
        out = C.apply_corruption(out, spec)
        parts.append(describe(spec))
    return out, {"chain": parts, "types": [p["type"] for p in parts]}


def describe(spec: dict) -> dict:
    """JSON-friendly description of a corruption spec (drops large rect lists)."""
    from .config import CLASS_LABELS

    kind = spec.get("type", "clean")
    info: dict = {"type": kind, "label": CLASS_LABELS.get(kind, kind.capitalize())}
    if kind == "salt":
        info.update(p=round(float(spec["p"]), 4), severity=severity_label((spec["p"] - 0.02) / 0.13))
    elif kind == "blur":
        info.update(k=int(spec["k"]), sigma=round(float(spec["sigma"]), 3))
    elif kind == "occlusion":
        info.update(
            coverage=round(float(spec.get("coverage", 0.0)), 4),
            rects=[[int(v) for v in r] for r in spec.get("rects", [])],
            n_rects=len(spec.get("rects", [])),
        )
    return info


def severity_label(frac: float) -> str:
    frac = float(frac)
    if frac < 0.34:
        return "low"
    if frac < 0.67:
        return "medium"
    return "high"


def clean() -> np.ndarray:  # pragma: no cover - trivial
    return np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
