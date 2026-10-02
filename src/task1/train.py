"""Task 1 training. Usage:
  python -m src.task1.train --epochs 60 --params_json outputs/task1/best_params.json
"""
import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.losses import CombinedLoss, psnr_per_image, ssim_per_image
from src.data.corruptions import CORRUPTION_TYPES, SEVERITY_NAMES
from src.data.datasets import ManifestDataset, TrainDataset
from src.task1.model import UniversalAE

PROC = Path("data/processed")
MANIFESTS = Path("data/manifests")

DEFAULTS = dict(lr=1e-3, batch_size=64, latent_dim=256, base_channels=32, dropout=0.1,
                alpha=0.8, weight_decay=1e-4, epochs=40, num_workers=2, seed=42)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loaders(cfg):
    with open(PROC / "split.json") as f:
        split = json.load(f)
    train_ds = TrainDataset(PROC / "trainval_images.npy", split["train_idx"])
    val_ds = ManifestDataset(PROC / "trainval_images.npy", MANIFESTS / "val_manifest.json")
    nw = cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=nw,
                            pin_memory=True, persistent_workers=nw > 0)
    return train_loader, val_loader


def _summ(psnr, ssim, l1):
    s, p = float(ssim.mean()), float(psnr.mean())
    return {"psnr": p, "ssim": s, "l1": float(l1.mean()), "score": 0.5 * s + 0.5 * (p / 50.0), "n": int(len(psnr))}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, S, L, lab, sev = [], [], [], [], []
    for b in loader:
        x, y = b["input"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
        out = model(x).clamp(0, 1)
        P.append(psnr_per_image(out, y).cpu())
        S.append(ssim_per_image(out, y).cpu())
        L.append((out - y).abs().flatten(1).mean(1).cpu())
        lab.append(b["label"]); sev.append(b["severity"])
    P, S, L, lab, sev = map(torch.cat, (P, S, L, lab, sev))
    res = {"overall": _summ(P, S, L), "by_type": {}, "by_type_severity": {}}
    for t, name in enumerate(CORRUPTION_TYPES):
        m = lab == t
        if m.any():
            res["by_type"][name] = _summ(P[m], S[m], L[m])
        if t == 0:
            continue
        for s, sname in enumerate(SEVERITY_NAMES):
            ms = m & (sev == s)
            if ms.any():
                res["by_type_severity"][f"{name}/{sname}"] = _summ(P[ms], S[ms], L[ms])
    return res


def _log(metrics, step):
    if mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


def run_training(cfg, trial=None, out_dir=None, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_loader, val_loader = make_loaders(cfg)
    model = UniversalAE(cfg["base_channels"], cfg["latent_dim"], cfg["dropout"]).to(device)
    criterion = CombinedLoss(cfg["alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = {"overall": {"score": -1.0}}, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tl, tl1, tss, nb = 0.0, 0.0, 0.0, 0
        for b in train_loader:
            x, y = b["input"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss, l1, ss = criterion(out.float(), y)  # loss in fp32 for SSIM stability
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); tl1 += l1.item(); tss += ss.item(); nb += 1
        sched.step()

        val = evaluate(model, val_loader, device)
        o = val["overall"]
        row = {"epoch": epoch, "train_loss": tl / nb, "train_l1": tl1 / nb, "train_ssim": tss / nb,
               "val_psnr": o["psnr"], "val_ssim": o["ssim"], "val_l1": o["l1"], "val_score": o["score"]}
        history.append(row)
        _log({k: v for k, v in row.items() if k != "epoch"}, epoch)
        _log({f"val_psnr/{k}": v["psnr"] for k, v in val["by_type"].items()}, epoch)
        _log({f"val_ssim/{k}": v["ssim"] for k, v in val["by_type"].items()}, epoch)
        if verbose:
            print(f"[{epoch:03d}/{cfg['epochs']}] loss {row['train_loss']:.4f} | val PSNR {o['psnr']:.2f} "
                  f"SSIM {o['ssim']:.4f} score {o['score']:.4f}")

        if o["score"] > best["overall"]["score"]:
            best = {**val, "epoch": epoch}
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val},
                           Path(out_dir) / "task1_best.pt")

        if trial is not None:
            import optuna
            trial.report(o["score"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    best["history"] = history
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out_dir", default="outputs/task1")
    ap.add_argument("--run_name", default="task1_final")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.params_json:
        with open(args.params_json) as f:
            cfg.update(json.load(f))
    if args.epochs:
        cfg["epochs"] = args.epochs

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("task1_universal_restoration")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(cfg)
        best = run_training(cfg, out_dir=args.out_dir)
        out = Path(args.out_dir)
        summary = {k: v for k, v in best.items() if k != "history"}
        with open(out / "final_val_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(out / "history.json", "w") as f:
            json.dump(best["history"], f, indent=2)
        mlflow.log_artifact(str(out / "task1_best.pt"))
        mlflow.log_artifact(str(out / "final_val_results.json"))
        mlflow.log_artifact(str(out / "history.json"))
        print(json.dumps(summary["by_type"], indent=2))


if __name__ == "__main__":
    main()
