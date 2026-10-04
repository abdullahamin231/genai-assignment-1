#!/usr/bin/env python3
"""Inspect the exported ONNX models: locations, tensor names and shapes.

Useful before starting the app, and specifically to confirm the Task 3 export
signature (the backend resolves its image/weight outputs by shape, not by name).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.onnx_runtime import REGISTRY  # noqa: E402


def fmt_shape(shape) -> str:
    return "[" + ", ".join("?" if not isinstance(s, int) or s < 0 else str(s) for s in shape) + "]"


def main() -> int:
    REGISTRY.load_all()
    print("search paths:")
    for p in REGISTRY.search_paths():
        print(f"  - {p}")
    print()
    rows = REGISTRY.status()
    ok = True
    for row in rows:
        mark = "OK " if row["present"] and not row["error"] else ("ERR " if row["present"] else "MISS")
        print(f"[{mark}] {row['file']:<32} {row['title']}")
        if not row["present"]:
            ok = False
            continue
        if row["error"]:
            print(f"       error: {row['error']}")
            ok = False
            continue
        for name, shape in row["inputs"].items():
            print(f"       in  {name:<18} {fmt_shape(shape)}")
        for name, shape in row["outputs"].items():
            print(f"       out {name:<18} {fmt_shape(shape)}")
    print()
    print(f"{sum(1 for r in rows if r['present'])}/{len(rows)} model files present")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
