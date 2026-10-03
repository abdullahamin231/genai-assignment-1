import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config, data, trainer
import corruptions as C
from models import ConvAE

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{config.OUT_DIR}/runs/task1_final/best.pt")
args = ap.parse_args()

dev = trainer.DEVICE
ck = torch.load(args.ckpt, map_location=dev)
cfg = ck["cfg"]
model = ConvAE(cfg["base_ch"], cfg["latent_ch"], cfg["dropout"]).to(dev)
model.load_state_dict(ck["model"]); model.eval()

ds = data.RestorationDataset("test")
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
acc = {k: [] for k in ("out_l1", "out_psnr", "out_ssim", "in_l1", "in_psnr", "in_ssim")}
with torch.no_grad():
    for x, y, _, _ in loader:
        pred = model(x.to(dev)).cpu()
        for pre, (a, b) in (("out", (pred, y)), ("in", (x, y))):
            l1, ps, ss = trainer.per_sample_metrics(a, b)
            acc[f"{pre}_l1"].append(l1); acc[f"{pre}_psnr"].append(ps); acc[f"{pre}_ssim"].append(ss)

df = pd.DataFrame({k: torch.cat(v).numpy() for k, v in acc.items()})
df["type"] = [e["type"] for e in ds.entries]
df["level"] = [e["level"] for e in ds.entries]
df["idx"] = [e["idx"] for e in ds.entries]
df["severity"] = df["level"].map({-1: "-", 0: "low", 1: "medium", 2: "high"})

cols = ["out_l1", "out_psnr", "out_ssim", "in_psnr", "in_ssim"]
order = {"clean": 0, "salt": 1, "blur": 2, "occlusion": 3}
by_sev = df.groupby(["type", "severity"])[cols].mean().reset_index()
by_sev["o"] = by_sev["type"].map(order); by_sev["l"] = by_sev["severity"].map({"-": 0, "low": 1, "medium": 2, "high": 3})
by_sev = by_sev.sort_values(["o", "l"]).drop(columns=["o", "l"])
by_type = df.groupby("type")[cols].mean().reindex(list(order))
overall = df[cols].mean()
std_sev = df.groupby(["type", "severity"])[["out_l1", "out_psnr", "out_ssim"]].std()

print("\n=== Task 1 test results by condition and severity (mean) ===")
print(by_sev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n=== by corruption type ===")
print(by_type.to_string(float_format=lambda v: f"{v:.4f}"))
print("\n=== overall ===")
print(overall.to_string(float_format=lambda v: f"{v:.4f}"))

R = Path(config.RESULTS_DIR)
by_sev.to_csv(R / "task1_test_table.csv", index=False)
json.dump({"epoch": ck["epoch"], "cfg": cfg, "overall": overall.to_dict(),
           "by_type": by_type.to_dict("index"),
           "by_type_severity": by_sev.to_dict("records"),
           "std_by_type_severity": {f"{a}|{b}": r for (a, b), r in std_sev.to_dict("index").items()}},
          open(R / "task1_test_metrics.json", "w"), indent=2)

lookup = {(e["idx"], e["type"], e["level"]): i for i, e in enumerate(ds.entries)}


@torch.no_grad()
def grid(ds_idx, path):
    x = torch.stack([ds[i][0] for i in ds_idx]); y = torch.stack([ds[i][1] for i in ds_idx])
    p = model(x.to(dev)).cpu().clamp(0, 1)
    err = (p - y).abs().mean(1)
    n = len(ds_idx)
    fig, ax = plt.subplots(n, 4, figsize=(8.4, 2.2 * n))
    for r, i in enumerate(ds_idx):
        row = df.iloc[i]
        tag = row["type"] if row["type"] == "clean" else f"{row['type']} {row['severity']}"
        for c, (im, t) in enumerate(((y[r], "clean target"), (x[r], f"input ({tag})"),
                                     (p[r], f"restored  SSIM {row['out_ssim']:.2f} PSNR {row['out_psnr']:.1f}"))):
            ax[r, c].imshow(im.permute(1, 2, 0).numpy()); ax[r, c].set_title(t, fontsize=7)
        ax[r, 3].imshow(err[r].numpy(), cmap="inferno", vmin=0, vmax=0.5); ax[r, 3].set_title("abs error", fontsize=7)
        for a in ax[r]:
            a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


examples = [(10, "clean", -1)] + [(20 + 3 * j, t, lv) for j, t in enumerate(("salt", "blur", "occlusion")) for lv in range(3)] \
           + [(100, "salt", 2), (200, "occlusion", 2)]
grid([lookup[e] for e in examples], R / "task1_examples_12.png")
worst = df.sort_values("out_ssim").head(6)
grid(worst.index.tolist(), R / "task1_failure_cases.png")
print("\nWorst 6 by SSIM:\n", worst[["idx", "type", "severity", "out_ssim", "out_psnr", "in_ssim"]].to_string())

out = Path(config.OUT_DIR) / "results"; out.mkdir(exist_ok=True)
for f in R.glob("task1_*"):
    shutil.copy(f, out / f.name)
print("\nSaved to", R, "and", out)
