"""Generate deterministic validation and test corruption manifests.

Usage: python -m src.data.make_manifests
Validation: one corruption per image, 4 conditions balanced, training distribution, seed 42.
Test: per image -> clean + 3 corruptions x 3 fixed severities (10 entries), seed 43.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .corruptions import CORRUPTION_TYPES, fixed_spec, sample_spec


def build_val(val_idx, seed):
    rng = np.random.default_rng(seed)
    n = len(val_idx)
    types = np.tile(np.arange(4), int(np.ceil(n / 4)))[:n]
    rng.shuffle(types)
    entries = []
    for idx, t in zip(val_idx, types):
        spec = sample_spec(rng, CORRUPTION_TYPES[int(t)])
        spec["image_idx"] = int(idx)
        entries.append(spec)
    return entries


def build_test(n_images, seed):
    rng = np.random.default_rng(seed)
    entries = []
    for idx in range(n_images):
        entries.append({"type": "clean", "severity": -1, "image_idx": idx})
        for ctype in CORRUPTION_TYPES[1:]:
            for level in range(3):
                spec = fixed_spec(rng, ctype, level)
                spec["image_idx"] = idx
                entries.append(spec)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="data/manifests")
    ap.add_argument("--val_seed", type=int, default=42)
    ap.add_argument("--test_seed", type=int, default=43)
    args = ap.parse_args()

    proc, out = Path(args.processed_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(proc / "split.json") as f:
        split = json.load(f)
    n_test = np.load(proc / "test_images.npy", mmap_mode="r").shape[0]  # only reads the shape

    val = build_val(split["val_idx"], args.val_seed)
    test = build_test(n_test, args.test_seed)

    with open(out / "val_manifest.json", "w") as f:
        json.dump({"meta": {"seed": args.val_seed, "n": len(val)}, "entries": val}, f)
    with open(out / "test_manifest.json", "w") as f:
        json.dump({"meta": {"seed": args.test_seed, "n": len(test)}, "entries": test}, f)
    print(f"val entries: {len(val)} | test entries: {len(test)}")


if __name__ == "__main__":
    main()
