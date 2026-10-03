import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import config, data
from models import ConvAE
from task2_clf import CorruptionClassifier, CHANNELS

OUT = Path(config.OUT_DIR)
(OUT / "models").mkdir(exist_ok=True)


class ProbWrap(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x).softmax(1)


def export(model, path, out_name):
    kw = dict(input_names=["image"], output_names=[out_name],
              dynamic_axes={"image": {0: "batch"}, out_name: {0: "batch"}}, opset_version=17)
    d = torch.rand(1, 3, 128, 128)
    try:
        torch.onnx.export(model, d, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, d, str(path), **kw)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


ds = data.RestorationDataset("test")
x = torch.stack([ds[i][0] for i in range(0, 600, 50)])   # 12 mixed conditions/severities

ck = torch.load(OUT / "runs/task2_classifier/best.pt", map_location="cpu")
clf = ProbWrap(CorruptionClassifier(CHANNELS[ck["cfg"]["channels"]], ck["cfg"]["dropout"]))
clf.m.load_state_dict(ck["model"]); clf.eval()
sess_clf = export(clf, OUT / "models/task2_classifier.onnx", "probs")
with torch.no_grad():
    ref_p = clf(x).numpy()
onnx_p = sess_clf.run(None, {"image": x.numpy()})[0]
print(f"classifier: max abs diff {np.abs(ref_p - onnx_p).max():.2e}")

spec_models, spec_sess = [], []
for name in ("salt", "blur", "occlusion"):
    c = torch.load(OUT / f"runs/task2_specialist_{name}/best.pt", map_location="cpu")
    m = ConvAE(c["cfg"]["base_ch"], c["cfg"]["latent_ch"], c["cfg"]["dropout"])
    m.load_state_dict(c["model"]); m.eval()
    s = export(m, OUT / f"models/task2_specialist_{name}.onnx", "restored")
    with torch.no_grad():
        ref = m(x).numpy()
    print(f"specialist {name}: max abs diff {np.abs(ref - s.run(None, {'image': x.numpy()})[0]).max():.2e}")
    spec_models.append(m); spec_sess.append(s)

# end-to-end hard-routed pipeline: ONNX vs PyTorch (clean -> identity bypass)
pred_t, pred_o = ref_p.argmax(1), onnx_p.argmax(1)
worst = 0.0
for i in range(len(x)):
    xi = x[i:i + 1]
    with torch.no_grad():
        out_t = xi.numpy() if pred_t[i] == 0 else spec_models[pred_t[i] - 1](xi).numpy()
    out_o = xi.numpy() if pred_o[i] == 0 else spec_sess[pred_o[i] - 1].run(None, {"image": xi.numpy()})[0]
    if pred_t[i] == pred_o[i]:
        worst = max(worst, float(np.abs(out_t - out_o).max()))
print(f"pipeline: routing agreement {np.mean(pred_t == pred_o):.2f}, max abs diff on routed outputs {worst:.2e}",
      "PASS" if worst < 1e-3 and np.all(pred_t == pred_o) else "CHECK")
print("Models in", OUT / "models")
