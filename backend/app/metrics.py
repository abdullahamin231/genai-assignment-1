"""Reference image-quality metrics (PSNR, SSIM) and error-map rendering.

Implemented with OpenCV only so the backend keeps a small dependency footprint.
SSIM follows Wang et al., 2004 with the standard 11x11 Gaussian window
(``sigma = 1.5``, ``C1 = (0.01 L)^2``, ``C2 = (0.03 L)^2``).
"""
from __future__ import annotations

import cv2
import numpy as np

_C1 = 0.01**2
_C2 = 0.03**2
_WINDOW = (11, 11)
_SIGMA = 1.5


def _f(img: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float64 in [0,1]."""
    x = np.asarray(img, dtype=np.float64)
    if x.max() > 1.0:
        x = x / 255.0
    return x


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (100.0 for identical images)."""
    x, y = _f(a), _f(b)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity over channels, in [0, 1]."""
    x, y = _f(a), _f(b)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    vals = []
    for c in range(x.shape[2]):
        ch1, ch2 = x[..., c], y[..., c]
        mu1 = cv2.GaussianBlur(ch1, _WINDOW, _SIGMA)
        mu2 = cv2.GaussianBlur(ch2, _WINDOW, _SIGMA)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = cv2.GaussianBlur(ch1 * ch1, _WINDOW, _SIGMA) - mu1_sq
        s2 = cv2.GaussianBlur(ch2 * ch2, _WINDOW, _SIGMA) - mu2_sq
        s12 = cv2.GaussianBlur(ch1 * ch2, _WINDOW, _SIGMA) - mu12
        num = (2 * mu12 + _C1) * (2 * s12 + _C2)
        den = (mu1_sq + mu2_sq + _C1) * (s1 + s2 + _C2)
        vals.append(float(np.mean(num / den)))
    return float(np.clip(np.mean(vals), -1.0, 1.0))


def error_map(reference: np.ndarray, produced: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Absolute per-pixel difference, amplified and colormapped for inspection."""
    x, y = _f(reference), _f(produced)
    diff = np.mean(np.abs(x - y), axis=2)
    heat = np.clip(diff * gain, 0.0, 1.0)
    heat = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def compare(reference: np.ndarray, produced: np.ndarray) -> dict:
    return {"psnr": round(psnr(reference, produced), 3), "ssim": round(ssim(reference, produced), 4)}
