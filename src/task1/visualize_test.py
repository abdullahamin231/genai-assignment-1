"""Usage: python -m src.task1.visualize_test --ckpt outputs/task1/task1_best.pt"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.corruptions import CORRUPTION_TYPES, SEVERITY_NAMES
from src.data.datasets import ManifestDataset
from src.task1.evaluate_test import MANIFESTS, PROC, load_model


def nearest_median(sub, k=1):
    med = sub.ssim.median()
    return sub.iloc[(sub.ssim - med).abs().argsort()[:k]]


def plot_rows(rows, ds, model, device, path, vmax=0.5):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.3 * n), squeeze=False)
    im = None
    for r, (_, row) in enumerate(rows.iterrows()):
        b = ds[int(row.entry)]
        with torch.no_grad():
            out = model(b["input"].unsqueeze(0).to(device)).clamp(0, 1)[0].cpu()
        err = (out - b["target"]).abs().mean(0)
        for c, img in enumerate([b["target"], b["input"], out]):
            axes[r, c].imshow(img.permute(1, 2, 0).numpy())
        im = axes[r, 3].imshow(err.numpy(), cmap="inferno", vmin=0, vmax=vmax)
        axes[r, 0].set_ylabel(f"{row.type}/{row.sev}\nPSNR {row.psnr:.1f}\nSSIM {row.ssim:.3f}", fontsize=7)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(["Clean target", "Corrupted input", "Restored", "Abs. error"]):
        axes[0, c].set_title(t, fontsize=9)
    fig.colorbar(im, ax=axes[:, 3], shrink=0.5, pad=0.02)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/task1/task1_best.pt")
    ap.add_argument("--out_dir", default="outputs/task1/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device)
    ds = ManifestDataset(PROC / "test_images.npy", MANIFESTS / "test_manifest.json")
    df = pd.read_csv("outputs/task1/test_per_image.csv")

    # 12 representative examples: 3 clean + 9 corrupted (type x severity), nearest to the cell median SSIM
    reps = [nearest_median(df[df.type == "clean"], k=3)]
    for t in CORRUPTION_TYPES[1:]:
        for s in SEVERITY_NAMES:
            reps.append(nearest_median(df[(df.type == t) & (df.sev == s)]))
    reps = pd.concat(reps)
    plot_rows(reps.iloc[:6], ds, model, device, out / "examples_1.png")
    plot_rows(reps.iloc[6:], ds, model, device, out / "examples_2.png")

    # 4 failure cases: worst SSIM per corruption type + worst remaining entry
    corr = df[df.type != "clean"]
    fails = pd.concat([corr[corr.type == t].nsmallest(1, "ssim") for t in CORRUPTION_TYPES[1:]])
    fails = pd.concat([fails, corr.drop(fails.index).nsmallest(1, "ssim")])
    plot_rows(fails, ds, model, device, out / "failures.png")
    fails.to_csv(out / "failure_cases.csv", index=False)
    print(fails[["entry", "image_idx", "type", "sev", "psnr", "ssim"]].round(3).to_string())


if __name__ == "__main__":
    main()
