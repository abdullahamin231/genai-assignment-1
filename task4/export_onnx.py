import argparse
import numpy as np
import onnxruntime as ort
import torch

import config, data
from models import UNetGenerator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{config.OUT_DIR}/runs/final/best.pt")
ap.add_argument("--out", default=f"{config.OUT_DIR}/models/task4_generator.onnx")
args = ap.parse_args()

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu")
cfg = ck["cfg"]
G = UNetGenerator(cfg["base_ch"], cfg["emb_dim"], cfg["dropout"])
G.load_state_dict(ck["G"]); G.eval()

photo = torch.randn(1, 3, 128, 128)
style = torch.tensor([0], dtype=torch.long)
kwargs = dict(input_names=["photo", "style"], output_names=["sketch"],
              dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},
              opset_version=17)
try:
    torch.onnx.export(G, (photo, style), args.out, dynamo=False, **kwargs)
except TypeError:  # older torch without the dynamo flag
    torch.onnx.export(G, (photo, style), args.out, **kwargs)
print("Exported to", args.out)

# --- Verify ONNX vs PyTorch on real validation photos, all three styles ---
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
ds = data.get_dataset("val")
p = torch.stack([ds[i][0] for i in range(6)])
worst = 0.0
for k in range(config.NUM_STYLES):
    st = torch.full((6,), k, dtype=torch.long)
    with torch.no_grad():
        ref = G(p, st).numpy()
    out = sess.run(None, {"photo": p.numpy(), "style": st.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    worst = max(worst, diff)
    print(f"style {k}: max abs diff = {diff:.2e}")
print("PASS" if worst < 1e-3 else "FAIL", f"(worst diff {worst:.2e}, batch size 6 > export batch 1 -> dynamic batch OK)")
