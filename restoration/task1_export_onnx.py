import argparse, os
import numpy as np
import onnxruntime as ort
import torch

import config, data
from models import ConvAE

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{config.OUT_DIR}/runs/task1_final/best.pt")
ap.add_argument("--out", default=f"{config.OUT_DIR}/models/task1_universal_ae.onnx")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu")
cfg = ck["cfg"]
model = ConvAE(cfg["base_ch"], cfg["latent_ch"], cfg["dropout"])
model.load_state_dict(ck["model"]); model.eval()

kw = dict(input_names=["image"], output_names=["restored"],
          dynamic_axes={"image": {0: "batch"}, "restored": {0: "batch"}}, opset_version=17)
dummy = torch.rand(1, 3, 128, 128)
try:
    torch.onnx.export(model, dummy, args.out, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(model, dummy, args.out, **kw)
print("Exported to", args.out)

ds = data.RestorationDataset("test")
x = torch.stack([ds[i][0] for i in range(0, 600, 50)])  # mixed conditions/severities, batch of 12
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model(x).numpy()
out = sess.run(None, {"image": x.numpy()})[0]
d = float(np.abs(ref - out).max())
print(f"max abs diff PyTorch vs ONNX: {d:.2e}", "PASS" if d < 1e-3 else "FAIL")
