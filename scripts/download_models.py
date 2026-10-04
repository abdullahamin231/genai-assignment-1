#!/usr/bin/env python3
"""Fetch the exported ONNX models into ``models/``.

Three modes, all converging on the same layout the backend expects::

    # 1) the documented link: one zip on the GitHub release (recommended)
    python scripts/download_models.py --zip
    #    ...or any other archive URL
    python scripts/download_models.py --zip https://example.com/genai-models.zip

    # 2) copy from a folder you already downloaded (e.g. an unzipped Drive export)
    python scripts/download_models.py --from ~/Downloads/genai_models

    # 3) download individual .onnx files from an HTTP base URL
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

# The documented download link: all seven graphs in one archive on the
# repository's `models` release. Kept here so --zip needs no argument.
RELEASE_ZIP_URL = (
    "https://github.com/abdullahamin231/genai-assignment-1/releases/"
    "download/models/genai-models.zip"
)
RELEASE_ZIP_SHA256 = "5f531e7f4ed97d294fa393777f4cef3a0e7835b52f03e62a2f8a8857dcc644e9"

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


def download_zip(url: str, dest: Path, expect_sha: str | None = None) -> list[str]:
    """Fetch one archive and unpack the expected graphs into ``dest``.

    The GitHub release stores all seven models in a single zip because that
    keeps the documented link to one URL. The archive is verified against a
    pinned SHA-256 when the caller supplies one, then unpacked member by member
    so an archive with extra files cannot write outside ``dest``.
    """
    import hashlib
    import tempfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  * {url}")
    sha = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                while chunk := resp.read(1 << 20):
                    tmp.write(chunk)
                    sha.update(chunk)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"    failed: {exc}")
            tmp_path.unlink(missing_ok=True)
            return []

    digest = sha.hexdigest()
    if expect_sha and digest != expect_sha:
        print(f"    SHA-256 mismatch: got {digest}, expected {expect_sha}")
        tmp_path.unlink(missing_ok=True)
        return []
    print(f"    sha256 {digest} ({tmp_path.stat().st_size / 1e6:.1f} MB)")

    got = []
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            names = set(EXPECTED)
            for info in zf.infolist():
                base = Path(info.filename).name
                if info.is_dir() or base not in names:
                    continue
                with zf.open(info) as src, open(dest / base, "wb") as out:
                    shutil.copyfileobj(src, out)
                print(f"    -> {dest / base} ({(dest / base).stat().st_size / 1e6:.1f} MB)")
                got.append(base)
    except zipfile.BadZipFile as exc:
        print(f"    failed: {exc}")
        return []
    finally:
        tmp_path.unlink(missing_ok=True)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", type=Path, help="folder to copy the .onnx files from")
    ap.add_argument("--url", help="HTTP base URL the .onnx files are published at")
    ap.add_argument(
        "--zip",
        dest="zip_url",
        nargs="?",
        const=RELEASE_ZIP_URL,
        help=f"download one archive containing every model (default: {RELEASE_ZIP_URL})",
    )
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT, help=f"destination (default: {DEST_DEFAULT})")
    ap.add_argument("--verify", action="store_true", help="load every model with onnxruntime afterwards")
    args = ap.parse_args()

    if not args.source and not args.url and not args.zip_url:
        ap.error("choose one of --zip / --from / --url")

    print(f"Destination: {args.dest}")
    if args.source:
        got = copy_from(args.source, args.dest)
    elif args.zip_url:
        pinned = RELEASE_ZIP_SHA256 if args.zip_url == RELEASE_ZIP_URL else None
        got = download_zip(args.zip_url, args.dest, pinned)
    else:
        got = download_from(args.url, args.dest)

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
