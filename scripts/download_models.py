#!/usr/bin/env python3
"""Fetch the exported ONNX models into ``models/``.

Two modes, both converging on the same layout the backend expects::

    # 1) copy from a folder you already downloaded (e.g. an unzipped Drive export)
    python scripts/download_models.py --from ~/Downloads/genai_models

    # 2) download from an HTTP base URL (GitHub release, Drive direct link, ...)
    python scripts/download_models.py --url https://example.com/models

Expected files (the same list used by the training/export scripts):

    task1_universal_ae.onnx          -> restoration_outputs/models
    task2_classifier.onnx            -> restoration_outputs/models
    task2_specialist_salt.onnx       -> restoration_outputs/models
    task2_specialist_blur.onnx       -> restoration_outputs/models
    task2_specialist_occlusion.onnx  -> restoration_outputs/models
    task3_soft_moe.onnx              -> restoration_outputs/models
    task4_generator.onnx             -> task4_outputs/models

Only the file *names* matter for the application; everything else is discovered
by the backend at startup.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST_DEFAULT = REPO / "models"

EXPECTED = [
    "task1_universal_ae.onnx",
    "task2_classifier.onnx",
    "task2_specialist_salt.onnx",
    "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx",
    "task3_soft_moe.onnx",
    "task4_generator.onnx",
]


def find_source_file(root: Path, filename: str) -> Path | None:
    """Search ``root`` recursively - downloads are often nested one level deep."""
    direct = root / filename
    if direct.is_file():
        return direct
    hits = sorted(root.rglob(filename))
    return hits[0] if hits else None


def copy_from(source: Path, dest: Path) -> list[str]:
    got = []
    for name in EXPECTED:
        src = find_source_file(source, name)
        if src is None:
            print(f"  - {name}: not found under {source}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / name)
        print(f"  + {name}: {src} -> {dest / name}")
        got.append(name)
    return got


def download_from(base: str, dest: Path) -> list[str]:
    base = base.rstrip("/")
    got = []
    for name in EXPECTED:
        url = f"{base}/{name}"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / name
        print(f"  * {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except (urllib.error.URLError, OSError) as exc:
            print(f"    failed: {exc}")
            target.unlink(missing_ok=True)
            continue
        print(f"    -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
        got.append(name)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", type=Path, help="folder to copy the .onnx files from")
    ap.add_argument("--url", help="HTTP base URL the .onnx files are published at")
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT, help=f"destination (default: {DEST_DEFAULT})")
    ap.add_argument("--verify", action="store_true", help="load every model with onnxruntime afterwards")
    args = ap.parse_args()

    if not args.source and not args.url:
        ap.error("choose one of --from / --url")

    print(f"Destination: {args.dest}")
    got = copy_from(args.source, args.dest) if args.source else download_from(args.url, args.dest)

    missing = [n for n in EXPECTED if n not in got]
    print(f"\n{len(got)}/{len(EXPECTED)} model files ready in {args.dest}")
    if missing:
        print("still missing: " + ", ".join(missing))

    if args.verify and got:
        import subprocess

        subprocess.run([sys.executable, str(Path(__file__).with_name("verify_models.py"))], check=False)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
