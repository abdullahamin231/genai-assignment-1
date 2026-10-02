"""Usage: python -m src.task1.evaluate_test --ckpt outputs/task1/task1_best.pt"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.losses import psnr_per_image, ssim_per_image
from src.data.corruptions import CORRUPTION_TYPES, SEVERITY_NAMES
from src.data.datasets import ManifestDataset
from src.task1.model import UniversalAE

PROC = Path("data/processed")
MANIFESTS = Path("data/manifests")


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = UniversalAE(cfg["base_channels"], cfg["latent_dim"], cfg["dropout"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/task1/task1_best.pt")
    ap.add_argument("--out_dir", default="outputs/task1")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device)
    ds = ManifestDataset(PROC / "test_images.npy", MANIFESTS / "test_manifest.json")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    cols = {k: [] for k in ["entry", "label", "severity", "psnr", "ssim", "l1", "in_psnr", "in_ssim"]}
    with torch.no_grad():
        for b in tqdm(loader, desc="test"):
            x, y = b["input"].to(device), b["target"].to(device)
            out = model(x).clamp(0, 1)
            cols["entry"].append(b["index"])
            cols["label"].append(b["label"])
            cols["severity"].append(b["severity"])
            cols["psnr"].append(psnr_per_image(out, y).cpu())
            cols["ssim"].append(ssim_per_image(out, y).cpu())
            cols["l1"].append((out - y).abs().flatten(1).mean(1).cpu())
            cols["in_psnr"].append(psnr_per_image(x, y).cpu())
            cols["in_ssim"].append(ssim_per_image(x, y).cpu())
    df = pd.DataFrame({k: torch.cat(v).numpy() for k, v in cols.items()})
    df["type"] = df.label.map(lambda i: CORRUPTION_TYPES[i])
    df["sev"] = df.severity.map(lambda s: "none" if s < 0 else SEVERITY_NAMES[s])
    df["image_idx"] = [ds.entries[i]["image_idx"] for i in df.entry]
    df.loc[df.type == "clean", ["in_psnr", "in_ssim"]] = np.nan  # input == target, baseline meaningless

    out = Path(args.out_dir)
    df.to_csv(out / "test_per_image.csv", index=False)

    agg = (df.groupby(["type", "sev"])[["psnr", "ssim", "l1", "in_psnr", "in_ssim"]].mean()
             .assign(n=df.groupby(["type", "sev"]).size()))
    order = [("clean", "none")] + [(t, s) for t in CORRUPTION_TYPES[1:] for s in SEVERITY_NAMES]
    agg = agg.loc[order]
    per_type = df.groupby("type")[["psnr", "ssim", "l1", "in_psnr", "in_ssim"]].mean().loc[CORRUPTION_TYPES]
    overall = df[["psnr", "ssim", "l1"]].mean().to_dict()

    agg.to_csv(out / "test_table.csv")
    per_type.to_csv(out / "test_table_by_type.csv")
    with open(out / "test_results.json", "w") as f:
        json.dump({"overall": overall, "by_type": per_type.to_dict("index"),
                   "by_type_severity": {f"{t}/{s}": r for (t, s), r in agg.to_dict("index").items()}}, f, indent=2)
    try:
        agg.to_latex(out / "test_table.tex", float_format="%.3f")
    except Exception as e:  # jinja2 missing etc.
        print("latex export skipped:", e)

    pd.set_option("display.width", 200)
    print(agg.round(4).to_string())
    print("\nBy type:\n", per_type.round(4).to_string())
    print("\nOverall:", {k: round(v, 4) for k, v in overall.items()})


if __name__ == "__main__":
    main()
