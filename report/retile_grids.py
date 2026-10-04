#!/usr/bin/env python3
"""Re-tile the tall evaluation montages into paper-friendly horizontal strips.

The evaluation scripts save one row per test image and one column per panel
(target / corrupted / restored / ...), which yields images like 924x2904 - far
too tall for a figure.  Transposing the block grid to panel-rows x
example-columns produces an aspect ratio near 2:1-3:1, which sits comfortably
across the full text width of a two-column IEEE page.

Cell boundaries are detected from the near-uniform gutters between blocks
rather than assumed, because each montage embeds its own caption strip and the
Task 3 montages carry an extra legend column that must be dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "figs"

# Minimum gutter width (px) that separates two blocks.  Captions inside a block
# produce 1-9 px gaps; block separators are 11 px or wider.
MIN_GUTTER = 11

# rel-path -> (n_rows, n_cols, output name)
JOBS = [
    ("restoration/results/task1_examples_12.png", 12, 4, "t1_examples.png"),
    ("restoration/results/task1_failure_cases.png", 6, 4, "t1_failures.png"),
    ("restoration/results/task2_examples.png", 12, 5, "t2_examples.png"),
    ("restoration/results/task2_routing_failures.png", 6, 5, "t2_failures.png"),
    ("restoration/results/task3_examples_12.png", 12, 5, "t3_examples.png"),
    ("restoration/results/task3_failure_cases.png", 6, 5, "t3_failures.png"),
    ("restoration/results/task3_mixed_corruptions.png", 6, 5, "t3_mixed.png"),
]


def gutters(mask: np.ndarray) -> list[tuple[int, int]]:
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


def bands(mask_len: int, runs: list[tuple[int, int]], n: int, axis: str,
          name: str) -> list[tuple[int, int]]:
    """Return exactly `n` cell bands along one axis."""
    majors = [(a, b) for a, b in runs if b - a >= MIN_GUTTER]
    if len(majors) > n + 1:
        majors = majors[: n + 1]
    if len(majors) < 2:
        raise ValueError(f"{name}: only {len(majors)} {axis} separators for {n} cells")

    out = [(majors[i][1], majors[i + 1][0]) for i in range(len(majors) - 1)]
    if len(out) < n:
        out.append((majors[-1][1], mask_len))
    if len(out) != n:
        raise ValueError(f"{name}: found {len(out)} {axis} bands, expected {n}")
    if any(b <= a for a, b in out):
        raise ValueError(f"{name}: degenerate {axis} band in {out}")
    return out


def retile(src: Path, rows: int, cols: int, dest_name: str) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    lum = np.asarray(im.convert("L"), dtype=np.int16)

    # a gutter column/row is almost entirely background (white margins, or the
    # black backdrop used behind the dark error maps)
    bg = (lum > 245) | (lum < 12)
    col_runs = gutters(bg.mean(axis=0) > 0.90)
    row_runs = gutters(bg.mean(axis=1) > 0.90)

    xs = bands(w, col_runs, cols, "column", src.name)
    ys = bands(h, row_runs, rows, "row", src.name)

    cw = max(b - a for a, b in xs)
    ch = max(b - a for a, b in ys)
    # transpose: example-columns across, panel-rows down
    out = Image.new("RGB", (rows * cw, cols * ch), (255, 255, 255))
    for panel in range(cols):          # destination row
        for ex in range(rows):         # destination column
            x0, x1 = xs[panel]
            y0, y1 = ys[ex]
            cell = im.crop((x0, y0, x1, y1))
            # normalise cell size so ragged caption strips do not offset the grid
            if cell.size != (cw, ch):
                padded = Image.new("RGB", (cw, ch), (255, 255, 255))
                padded.paste(cell, (0, 0))
                cell = padded
            out.paste(cell, (ex * cw, panel * ch))
    out.save(OUT / dest_name)
    print(f"  {src.name}: {w}x{h} -> {dest_name} {out.size}  (cells {cw}x{ch})")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, rows, cols, name in JOBS:
        src = REPO / rel
        if not src.exists():
            print(f"  -- missing {rel}")
            continue
        try:
            retile(src, rows, cols, name)
        except ValueError as e:
            print(f"  !! {e}")
