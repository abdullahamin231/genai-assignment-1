import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import config, data
from task3_moe import load_task3

OUT = Path(config.OUT_DIR)
(OUT / "models").mkdir(exist_ok=True)
path = OUT / "models/task3_soft_moe.onnx"

moe, ck = load_task3(OUT / "runs/task3_final/best.pt", "cpu")


class Wrap(nn.Module):
    """Whole pipeline in one graph: gate + 3 experts + identity branch + weighted sum."""

    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        y, w, _ = self.m(x)
        return y, w


wrap = Wrap(moe).eval()
kw = dict(input_names=["image"], output_names=["restored", "weights"],
          dynamic_axes={"image": {0: "batch"}, "restored": {0: "batch"}, "weights": {0: "batch"}}, opset_version=17)
dummy = torch.rand(1, 3, 128, 128)
try:
    torch.onnx.export(wrap, dummy, str(path), dynamo=False, **kw)
except TypeError:
    torch.onnx.export(wrap, dummy, str(path), **kw)
print("Exported to", path, f"({path.stat().st_size / 1e6:.1f} MB); weights order = [identity, salt, blur, occlusion]; tau = {moe.tau:.3f}")

ds = data.RestorationDataset("test")
x = torch.stack([ds[i][0] for i in range(0, 600, 50)])
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref_y, ref_w = wrap(x)
onnx_y, onnx_w = sess.run(None, {"image": x.numpy()})
dy = float(np.abs(ref_y.numpy() - onnx_y).max()); dw = float(np.abs(ref_w.numpy() - onnx_w).max())
print(f"max abs diff: restored {dy:.2e}, weights {dw:.2e}", "PASS" if max(dy, dw) < 1e-3 else "FAIL")
