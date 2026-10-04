"""Image decoding, model pre/post-processing and PNG encoding.

Mirrors the training pipeline exactly:

* Tasks 1-3 (``restoration/data.py``): RGB, resized to 128x128 with bicubic
  resampling, ``float32 / 255.0``, NCHW.
* Task 4 (``task4/data.py``): RGB photo, resized to 128x128 bicubic,
  ``float32 / 127.5 - 1``, NCHW; the generator returns a single-channel sketch
  in ``[-1, 1]``.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps

from .config import IMG_SIZE


class ImageError(ValueError):
    """Raised for malformed or unsupported uploads."""


def decode(data: bytes) -> np.ndarray:
    """bytes -> HxWx3 uint8 RGB."""
    if not data:
        raise ImageError("The uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the client
        raise ImageError("Could not decode the file as an image (PNG/JPG/WebP expected).") from exc
    return np.asarray(img, dtype=np.uint8)


def resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Bicubic resize, matching the dataset preparation used for training."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    pil = Image.fromarray(img).resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def to_chw(img: np.ndarray, mode: str = "unit") -> np.ndarray:
    """HxWx3 uint8 -> 1x3xHxW float32. ``mode`` is ``unit`` ([0,1]) or ``minus`` ([-1,1])."""
    x = img.astype(np.float32)
    if mode == "unit":
        x = x / 255.0
    elif mode == "minus":
        x = x / 127.5 - 1.0
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown mode {mode!r}")
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None, ...]


def from_chw(arr: np.ndarray, mode: str = "unit") -> np.ndarray:
    """1x3xHxW (or 1x1xHxW) float -> HxWxC uint8."""
    y = np.asarray(arr, dtype=np.float32)
    if y.ndim == 4:
        y = y[0]
    if mode == "unit":
        y = y * 255.0
    elif mode == "minus":
        y = (y + 1.0) * 127.5
    else:  # pragma: no cover
        raise ValueError(f"unknown mode {mode!r}")
    y = np.clip(y, 0, 255).astype(np.uint8)
    if y.shape[0] == 1:  # grayscale sketch -> RGB for a consistent UI
        y = np.repeat(y, 3, axis=0)
    return y.transpose(1, 2, 0)


def encode_png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode("ascii")


def from_data_url(url: str) -> np.ndarray:
    """Accept either a raw data URL or plain base64 (as sent by the browser)."""
    if "," in url and url.strip().startswith("data:"):
        url = url.split(",", 1)[1]
    try:
        return decode(base64.b64decode(url))
    except Exception as exc:  # noqa: BLE001
        raise ImageError("Could not decode the embedded image.") from exc
