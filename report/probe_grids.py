#!/usr/bin/env python3
"""Detect the cell grid of the evaluation montages so they can be re-tiled safely."""
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "restoration/results/task1_examples_12.png",
    "restoration/results/task1_failure_cases.png",
    "restoration/results/task2_examples.png",
    "restoration/results/task2_routing_failures.png",
    "restoration/results/task3_examples_12.png",
    "restoration/results/task3_failure_cases.png",
    "restoration/results/task3_mixed_corruptions.png",
    "task4/results/task4_test_best6.png",
    "task4/results/task4_test_worst6.png",
]

for rel in FILES:
    p = REPO / rel
    if not p.exists():
        print(f"{rel}: missing")
        continue
    a = np.asarray(Image.open(p).convert("L"), dtype=np.int16)
    h, w = a.shape
    # rows / columns that are almost entirely background (white or black gutters)
    white_rows = (a > 245).mean(axis=1)
    white_cols = (a > 245).mean(axis=0)
    black_rows = (a < 12).mean(axis=1)
    black_cols = (a < 12).mean(axis=0)
    gut_r = (white_rows > 0.90) | (black_rows > 0.90)
    gut_c = (white_cols > 0.90) | (black_cols > 0.90)

    def runs(mask):
        out, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start, i))
                start = None
        if start is not None:
            out.append((start, len(mask)))
        return out

    rb, cb = runs(gut_r), runs(gut_c)
    # interior gutters only (drop the frame)
    rb_i, cb_i = rb[1:-1], cb[1:-1]
    print(f"\n{Path(rel).name}: {w}x{h}")
    print(f"  col gutters (n={len(cb)}): {cb[:14]}")
    print(f"  row gutters (n={len(rb)}): {rb[:16]}")
    if len(cb_i) >= 2:
        widths = [b - a for a, b in cb]
        print(f"  col gutter widths: {widths}")
    if len(rb_i) >= 2:
        heights = [b - a for a, b in rb]
        print(f"  row gutter widths: {heights}")
