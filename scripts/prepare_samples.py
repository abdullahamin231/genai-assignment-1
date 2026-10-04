#!/usr/bin/env python3
"""Bundle a handful of clean images into ``backend/samples`` for the UI picker.

Reads the cached Oxford-IIIT Pet test images produced by ``restoration/data.py``
(``$RESTORATION_OUT/cache/clean_test.npy``) when available, otherwise falls back
to torchvision (which downloads the dataset).

    python scripts/prepare_samples.py --n 8

The samples are small PNGs; add them to git only if you want them shipped with
the repository (they are ignored by default).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "backend" / "samples"

try:
    import config as rconfig  # restoration/config.py  (only if on the python path)

    CACHE = Path(rconfig.CACHE_DIR) / "clean_test.npy"
except Exception:  # noqa: BLE001
    CACHE = Path(os.environ.get("RESTORATION_OUT", REPO / "restoration_outputs")) / "cache" / "clean_test.npy"


def from_cache(n: int) -> list[np.ndarray]:
    arr = np.load(CACHE, mmap_mode="r")
    idx = np.linspace(0, len(arr) - 1, num=min(n, len(arr))).astype(int)
    return [np.asarray(arr[i]) for i in idx]


def from_torchvision(n: int) -> list[np.ndarray]:
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=str(REPO / "data" / "oxford_pets"), split="test", download=True)
    idx = np.linspace(0, len(ds) - 1, num=min(n, len(ds))).astype(int)
    return [np.asarray(ds[int(i)][0].convert("RGB")) for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of samples (default 8)")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    if CACHE.is_file():
        images = from_cache(args.n)
        src = f"cache: {CACHE}"
    else:
        sys.path.insert(0, str(REPO / "restoration"))
        try:
            images = from_torchvision(args.n)
            src = "torchvision (Oxford-IIIT Pet, official test split)"
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load the dataset: {exc}")
            print("The application still works - the UI simply hides the sample picker.")
            return 1

    for i, img in enumerate(images):
        path = args.dest / f"sample_{i:02d}.png"
        Image.fromarray(img).save(path)
        print(f"  + {path}")
    print(f"\n{len(images)} samples from {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
