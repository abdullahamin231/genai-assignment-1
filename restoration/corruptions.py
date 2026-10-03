"""Corruption functions. numpy + OpenCV only, so the backend can reuse them without torch."""
import numpy as np
import cv2

CLASS_NAMES = ["clean", "salt", "blur", "occlusion"]
CLASS_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
LEVEL_NAMES = ["low", "medium", "high"]
SIZE = 128

# Fixed test severities from the assignment
SALT_LEVELS = [0.03, 0.08, 0.15]
BLUR_LEVELS = [(3, 0.7), (5, 1.5), (7, 2.5)]
OCC_LEVELS = [(1, 0.10), (2, 0.20), (3, 0.35)]  # (num_rects, target coverage)


def _coverage(rects, size=SIZE):
    m = np.zeros((size, size), bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return float(m.mean())


def make_occlusion_rects(rng, target, k, size=SIZE, tol=0.02, bounds=None, tries=300):
    """k random rectangles whose UNION covers ~target of the image."""
    best = None
    for _ in range(tries):
        fracs = rng.dirichlet(np.full(k, 3.0)) * target
        rects = []
        for f in fracs:
            area = f * size * size
            ar = rng.uniform(0.5, 2.0)
            w = min(max(int(round(np.sqrt(area * ar))), 2), size)
            h = min(max(int(round(area / w)), 2), size)
            x0 = int(rng.integers(0, size - w + 1))
            y0 = int(rng.integers(0, size - h + 1))
            rects.append([x0, y0, x0 + w, y0 + h])
        cov = _coverage(rects, size)
        ok = abs(cov - target) <= tol and (bounds is None or bounds[0] <= cov <= bounds[1])
        if best is None or abs(cov - target) < abs(best[1] - target):
            best = (rects, cov)
        if ok:
            return rects, cov
    return best


def sample_spec(rng, t):
    """Random training-time corruption (assignment ranges). t = class index."""
    name = CLASS_NAMES[t]
    seed = int(rng.integers(0, 2**31 - 1))
    if name == "clean":
        return {"type": "clean"}
    if name == "salt":
        return {"type": "salt", "p": float(rng.uniform(0.02, 0.15)), "seed": seed}
    if name == "blur":
        return {"type": "blur", "k": int(rng.choice([3, 5, 7])), "sigma": float(rng.uniform(0.5, 2.5))}
    k = int(rng.integers(1, 4))
    target = float(rng.uniform(0.10, 0.35))
    rects, cov = make_occlusion_rects(rng, target, k, bounds=(0.10, 0.35))
    return {"type": "occlusion", "rects": rects, "coverage": cov}


def fixed_spec(t, level, seed):
    """Deterministic test-time corruption at one of the three fixed severities."""
    name = CLASS_NAMES[t]
    if name == "salt":
        return {"type": "salt", "p": SALT_LEVELS[level], "seed": seed}
    if name == "blur":
        k, s = BLUR_LEVELS[level]
        return {"type": "blur", "k": k, "sigma": s}
    k, target = OCC_LEVELS[level]
    rects, cov = make_occlusion_rects(np.random.default_rng(seed), target, k, tol=0.01)
    return {"type": "occlusion", "rects": rects, "coverage": cov}


def apply_corruption(img, spec):
    """img: float32 HxWx3 in [0,1]. Deterministic given spec. Returns float32 HxWx3."""
    t = spec["type"]
    if t == "clean":
        return img.copy()
    if t == "salt":
        rng = np.random.default_rng(spec["seed"])
        h, w = img.shape[:2]
        mask = rng.random((h, w)) < spec["p"]
        vals = rng.integers(0, 2, (h, w)).astype(np.float32)
        out = img.copy()
        out[mask] = vals[mask][:, None]
        return out
    if t == "blur":
        k, s = int(spec["k"]), float(spec["sigma"])
        return cv2.GaussianBlur(img, (k, k), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT_101)
    if t == "occlusion":
        out = img.copy()
        for x0, y0, x1, y1 in spec["rects"]:
            out[y0:y1, x0:x1] = 0.0
        return out
    raise ValueError(t)
