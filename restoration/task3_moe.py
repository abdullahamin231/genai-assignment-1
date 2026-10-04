"""Task 3: soft mixture-of-experts restoration (gate + 3 experts + identity branch)."""
import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchvision.utils import make_grid, save_image

import config, data
import corruptions as C
from models import ConvAE
from task2_clf import CorruptionClassifier, CHANNELS, BalancedTrainDS, BalancedBatchSampler
from trainer import DEVICE, per_sample_metrics

EXPERIMENT = "restoration_task3"
BRANCHES = ["identity", "salt", "blur", "occlusion"]


class SoftMoE(nn.Module):
    """y = w0*x + w1*E_salt(x) + w2*E_blur(x) + w3*E_occ(x),  w = softmax(gate(x) / tau)."""

    def __init__(self, gate, experts, tau=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.tau = float(tau)

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)
        outs = [x] + [e(x) for e in self.experts]
        y = sum(w[:, k].view(-1, 1, 1, 1) * outs[k] for k in range(4))
        return y, w, logits


def build_moe(arch, tau):
    gate = CorruptionClassifier(CHANNELS[arch["clf"]["channels"]], arch["clf"]["dropout"])
    experts = [ConvAE(e["base_ch"], e["latent_ch"], e["dropout"]) for e in arch["experts"]]
    return SoftMoE(gate, experts, tau)


def load_task2(device=DEVICE, tau=1.0):
    """Gate <- Task 2 classifier, experts <- Task 2 specialists."""
    out = Path(config.OUT_DIR)
    ck = torch.load(out / "runs/task2_classifier/best.pt", map_location=device)
    arch = {"clf": {"channels": ck["cfg"]["channels"], "dropout": ck["cfg"]["dropout"]}, "experts": []}
    sds = []
    for name in ("salt", "blur", "occlusion"):
        c = torch.load(out / f"runs/task2_specialist_{name}/best.pt", map_location=device)
        arch["experts"].append({k: c["cfg"][k] for k in ("base_ch", "latent_ch", "dropout")})
        sds.append(c["model"])
    moe = build_moe(arch, tau).to(device)
    moe.gate.load_state_dict(ck["model"])
    for e, sd in zip(moe.experts, sds):
        e.load_state_dict(sd)
    return moe, arch


def load_task3(path, device=DEVICE):
    ck = torch.load(path, map_location=device)
    moe = build_moe(ck["arch"], ck["cfg"]["tau"]).to(device)
    moe.load_state_dict(ck["model"])
    moe.eval()
    return moe, ck


def moe_loss(y, target, w, logits, labels, cfg):
    l1 = (y - target).abs().mean()
    ssim = ssim_fn(y, target, data_range=1.0)
    ce = F.cross_entropy(logits / cfg["tau"], labels)
    bal = ((w.mean(0) - 0.25) ** 2).sum()
    loss = cfg["alpha"] * l1 + (1 - cfg["alpha"]) * (1 - ssim) + cfg["lambda_c"] * ce + cfg["lambda_b"] * bal
    return loss, {"l1": l1, "ssim": ssim, "ce": ce, "bal": bal}


@torch.no_grad()
def evaluate_moe(model, loader, device=DEVICE):
    model.eval()
    L, P, S, W, Y = [], [], [], [], []
    for x, y, lab, _ in loader:
        out, w, _ = model(x.to(device))
        l1, ps, ss = per_sample_metrics(out, y.to(device))
        L.append(l1.cpu()); P.append(ps.cpu()); S.append(ss.cpu()); W.append(w.cpu()); Y.append(lab)
    model.train()
    L, P, S, W, Y = map(torch.cat, (L, P, S, W, Y))
    out = {"val_l1": L.mean().item(), "val_psnr": P.mean().item(), "val_ssim": S.mean().item(),
           "val_gate_acc": (W.argmax(1) == Y).float().mean().item()}
    for k, b in enumerate(BRANCHES):
        out[f"val_w_{b}"] = W[:, k].mean().item()
    for c, name in enumerate(C.CLASS_NAMES):
        m = Y == c
        out[f"val_ssim_{name}"] = S[m].mean().item()
        for k, b in enumerate(BRANCHES):
            out[f"val_w_{name}_to_{b}"] = W[m, k].mean().item()
    return out


@torch.no_grad()
def log_samples(model, ds, epoch, out_dir, device=DEVICE, n=12):
    model.eval()
    items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // n))][:n]
    x = torch.stack([it[0] for it in items])
    y = torch.stack([it[1] for it in items])
    out = model(x.to(device))[0].cpu()
    err = ((out - y).abs().mean(1, keepdim=True) * 4).clamp(0, 1).repeat(1, 3, 1, 1)
    grid = make_grid(torch.cat([x, out, y, err]), nrow=len(items), padding=2)  # input / restored / clean / error
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"samples_epoch{epoch:04d}.png"
    save_image(grid, path)
    mlflow.log_artifact(str(path), "samples")
    model.train()


def run_moe_training(cfg, epochs, warmup, run_name, trial=None, save=False, sample_every=0,
                     device=DEVICE, seed=config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    moe, arch = load_task2(device, tau=cfg["tau"])
    ds = BalancedTrainDS()
    loader = DataLoader(ds, batch_sampler=BalancedBatchSampler(len(ds), cfg["batch_size"]),
                        num_workers=2, persistent_workers=True)
    val_ds = data.RestorationDataset("val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    run_dir = Path(config.OUT_DIR) / "runs" / run_name
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)
    full_cfg = {**cfg, "warmup": warmup}
    best = {"objective": float("inf")}
    opt = sched = None
    nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params({**cfg, "epochs": epochs, "warmup": warmup})
        for epoch in range(1, epochs + 1):
            warm = epoch <= warmup
            if epoch == 1 or epoch == warmup + 1:
                for p in moe.experts.parameters():
                    p.requires_grad_(not warm)
                if warm:
                    opt = torch.optim.Adam(moe.gate.parameters(), lr=cfg["lr_warm"]); sched = None
                else:
                    opt = torch.optim.Adam(moe.parameters(), lr=cfg["lr_ft"])
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(1, epochs - warmup), eta_min=cfg["lr_ft"] * 0.05)
            moe.train()
            if warm:
                moe.experts.eval()      # frozen experts: keep BatchNorm statistics fixed

            sums = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "ce": 0.0, "bal": 0.0}
            for x, y, lab, _ in loader:
                x, y, lab = x.to(device, non_blocking=True), y.to(device, non_blocking=True), lab.to(device)
                out, w, logits = moe(x)
                loss, parts = moe_loss(out, y, w, logits, lab, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sums["loss"] += loss.item()
                for k, v in parts.items():
                    sums[k] += v.item()
            if sched is not None:
                sched.step()

            n = len(loader)
            metrics = {f"train_{k}": v / n for k, v in sums.items()}
            metrics.update(evaluate_moe(moe, val_loader, device))
            metrics["objective"] = metrics["val_l1"] + (1 - metrics["val_ssim"])
            metrics["phase_warmup"] = float(warm)
            metrics["lr"] = opt.param_groups[0]["lr"]
            mlflow.log_metrics(metrics, step=epoch)

            means = [metrics[f"val_w_{b}"] for b in BRANCHES]
            collapsed = (not warm) and (min(means) < 0.05 or max(means) > 0.6)
            print(f"[{run_name}] ep {epoch:2d} {'warm' if warm else 'ft  '} | val_l1 {metrics['val_l1']:.4f} "
                  f"ssim {metrics['val_ssim']:.4f} gate_acc {metrics['val_gate_acc']:.4f} | "
                  f"mean w {[round(m, 3) for m in means]}{'  COLLAPSE' if collapsed else ''}")

            eligible = (not warm) or warmup >= epochs
            if eligible and not collapsed and metrics["objective"] < best["objective"]:
                best = {**metrics, "epoch": epoch}
                if save:
                    torch.save({"model": moe.state_dict(), "cfg": full_cfg, "arch": arch, "epoch": epoch,
                                "metrics": {k: float(v) for k, v in metrics.items()}}, run_dir / "best.pt")
            if save and sample_every and (epoch % sample_every == 0 or epoch == 1):
                log_samples(moe, val_ds, epoch, run_dir / "samples", device)
                torch.save({"model": moe.state_dict(), "cfg": full_cfg, "arch": arch, "epoch": epoch},
                           run_dir / "last.pt")

            if trial is not None:
                if collapsed:
                    trial.set_user_attr("pruned_reason", "collapse")
                    raise optuna.TrialPruned()
                trial.report(metrics["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if isinstance(v, (int, float))})
    return best
