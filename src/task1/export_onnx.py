"""Usage: python -m src.task1.export_onnx --ckpt outputs/task1/task1_best.pt"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.data.datasets import ManifestDataset
from src.task1.evaluate_test import MANIFESTS, PROC, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/task1/task1_best.pt")
    ap.add_argument("--out", default="models/task1_universal_restoration.onnx")
    ap.add_argument("--n_check", type=int, default=64)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, "cpu")
    dummy = torch.rand(1, 3, 128, 128)
    kw = dict(input_names=["input"], output_names=["restored"], opset_version=args.opset,
              dynamic_axes={"input": {0: "batch"}, "restored": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)  # legacy exporter, no extra deps
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, dummy, str(out), **kw)
    onnx.checker.check_model(onnx.load(str(out)))
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # --- consistency check on real corrupted test inputs (all condition types) ---
    ds = ManifestDataset(PROC / "test_images.npy", MANIFESTS / "test_manifest.json")
    idx = np.linspace(0, len(ds) - 1, args.n_check).astype(int)
    x = torch.stack([ds[int(i)]["input"] for i in idx])
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = np.abs(ref - got)

    # batch-size-1 path (what the app will use) and latency
    one_ref = ref[:1]
    t0 = time.perf_counter()
    for _ in range(20):
        one_got = sess.run(None, {"input": x[:1].numpy()})[0]
    ms = (time.perf_counter() - t0) / 20 * 1000

    result = {"n_checked": int(args.n_check), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
              "batch1_max_abs_diff": float(np.abs(one_ref - one_got).max()), "tolerance": args.tol,
              "cpu_ms_per_image_batch1": ms, "passed": bool(diff.max() < args.tol)}
    print(json.dumps(result, indent=2))
    with open(out.with_suffix(".verification.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump({"input_name": "input", "output_name": "restored", "shape": [None, 3, 128, 128],
                   "dtype": "float32", "range": [0.0, 1.0], "layout": "NCHW RGB", "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
