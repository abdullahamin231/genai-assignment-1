"""Corruption definitions. Every corruption is fully described by a JSON-serialisable `spec` dict."""
import numpy as np
import torch
import torchvision.transforms.functional as TF

SIZE = 128
CORRUPTION_TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
TYPE_TO_ID = {t: i for i, t in enumerate(CORRUPTION_TYPES)}
SEVERITY_NAMES = ["low", "medium", "high"]

# training ranges
SP_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_FRAC_RANGE = (0.10, 0.35)
OCC_N_RANGE = (1, 3)

# fixed test severities
TEST_SP = [0.03, 0.08, 0.15]
TEST_BLUR = [(3, 0.7), (5, 1.5), (7, 2.5)]
TEST_OCC = [(1, 0.10), (2, 0.20), (3, 0.35)]


def _union_coverage(rects, size=SIZE):
    m = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return float(m.mean())


def _make_rects_once(rng, n, total_frac, equal, size=SIZE):
    total_area = total_frac * size * size
    fracs = np.full(n, 1.0 / n) if (equal or n == 1) else rng.dirichlet(np.full(n, 5.0))
    rects = []
    for f in fracs:
        area = total_area * f
        aspect = rng.uniform(0.5, 2.0)  # w / h
        w = int(round(np.sqrt(area * aspect)))
        h = int(round(area / max(w, 1)))
        w, h = int(np.clip(w, 1, size)), int(np.clip(h, 1, size))
        x0 = int(rng.integers(0, size - w + 1))
        y0 = int(rng.integers(0, size - h + 1))
        rects.append([x0, y0, x0 + w, y0 + h])
    return rects


def make_rects(rng, n, total_frac, equal=False, tol=0.01, tries=50):
    """Sample n rectangles whose UNION covers ~total_frac (re-samples when overlap shrinks coverage)."""
    rects = _make_rects_once(rng, n, total_frac, equal)
    for _ in range(tries):
        if abs(_union_coverage(rects) - total_frac) <= tol:
            break
        rects = _make_rects_once(rng, n, total_frac, equal)
    return rects


def _bin(v, lo, hi):
    r = (v - lo) / (hi - lo)
    return 0 if r < 1 / 3 else (1 if r < 2 / 3 else 2)


def sample_spec(rng, ctype):
    """Random spec following the training distribution (also used for the validation manifest)."""
    if ctype == "clean":
        return {"type": "clean", "severity": -1}
    if ctype == "salt_pepper":
        p = float(rng.uniform(*SP_RANGE))
        return {"type": ctype, "prob": p, "seed": int(rng.integers(2**31 - 1)),
                "severity": _bin(p, *SP_RANGE)}
    if ctype == "blur":
        k = int(rng.choice(BLUR_KERNELS))
        s = float(rng.uniform(*BLUR_SIGMA_RANGE))
        return {"type": ctype, "kernel": k, "sigma": s, "severity": _bin(s, *BLUR_SIGMA_RANGE)}
    if ctype == "occlusion":
        n = int(rng.integers(OCC_N_RANGE[0], OCC_N_RANGE[1] + 1))
        frac = float(rng.uniform(*OCC_FRAC_RANGE))
        rects = make_rects(rng, n, frac)
        return {"type": ctype, "n_rects": n, "target_frac": frac, "rects": rects,
                "coverage": _union_coverage(rects), "severity": _bin(frac, *OCC_FRAC_RANGE)}
    raise ValueError(ctype)


def fixed_spec(rng, ctype, level):
    """Spec for the fixed test severities (level 0/1/2 = low/medium/high)."""
    if ctype == "salt_pepper":
        return {"type": ctype, "prob": TEST_SP[level], "seed": int(rng.integers(2**31 - 1)), "severity": level}
    if ctype == "blur":
        k, s = TEST_BLUR[level]
        return {"type": ctype, "kernel": k, "sigma": s, "severity": level}
    if ctype == "occlusion":
        n, frac = TEST_OCC[level]
        rects = make_rects(rng, n, frac, equal=True)
        return {"type": ctype, "n_rects": n, "target_frac": frac, "rects": rects,
                "coverage": _union_coverage(rects), "severity": level}
    raise ValueError(ctype)


def apply_corruption(img: torch.Tensor, spec: dict) -> torch.Tensor:
    """img: float tensor (3,H,W) in [0,1]. Returns corrupted copy."""
    t = spec["type"]
    if t == "clean":
        return img.clone()
    if t == "salt_pepper":
        _, H, W = img.shape
        rng = np.random.default_rng(spec["seed"])
        noisy = rng.random((H, W)) < spec["prob"]
        salt = rng.random((H, W)) < 0.5
        out = img.clone()
        out[:, torch.from_numpy(noisy & salt)] = 1.0
        out[:, torch.from_numpy(noisy & ~salt)] = 0.0
        return out
    if t == "blur":
        k, s = spec["kernel"], spec["sigma"]
        return TF.gaussian_blur(img, [k, k], [s, s])
    if t == "occlusion":
        out = img.clone()
        for x0, y0, x1, y1 in spec["rects"]:
            out[:, y0:y1, x0:x1] = 0.0
        return out
    raise ValueError(t)
