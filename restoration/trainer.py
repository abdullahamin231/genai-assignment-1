"""Generic autoencoder training loop (Task 1 universal AE; Task 2 specialists via `types`)."""
import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchvision.utils import make_grid, save_image

import config
import corruptions as C
import data
from models import ConvAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_mlflow(experiment):
    mlflow.set_tracking_uri(f"sqlite:///{config.OUT_DIR}/mlflow.db")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(Path(config.OUT_DIR) / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)


def per_sample_metrics(pred, target):
    p = pred.clamp(0, 1)
    l1 = (p - target).abs().flatten(1).mean(1)
    mse = ((p - target) ** 2).flatten(1).mean(1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    ssim = ssim_fn(p, target, data_range=1.0, reduction="none")
    return l1, psnr, ssim


def restoration_loss(pred, target, alpha):
    l1 = (pred - target).abs().mean()
    ssim = ssim_fn(pred, target, data_range=1.0)
    return alpha * l1 + (1 - alpha) * (1 - ssim), l1, ssim


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    model.eval()
    L, P, S, Y = [], [], [], []
    for x, y, lab, _ in loader:
        l1, ps, ss = per_sample_metrics(model(x.to(device)), y.to(device))
        L.append(l1.cpu()); P.append(ps.cpu()); S.append(ss.cpu()); Y.append(lab)
    model.train()
    L, P, S, Y = map(torch.cat, (L, P, S, Y))
    out = {"val_l1": L.mean().item(), "val_psnr": P.mean().item(), "val_ssim": S.mean().item()}
    for c, name in enumerate(C.CLASS_NAMES):
        m = Y == c
        if m.any():
            out[f"val_ssim_{name}"] = S[m].mean().item()
            out[f"val_psnr_{name}"] = P[m].mean().item()
    return out


@torch.no_grad()
def log_samples(model, ds, epoch, out_dir, device=DEVICE, n=12):
    model.eval()
    items = [ds[i] for i in range(min(n, len(ds)))]
    x = torch.stack([it[0] for it in items]).to(device)
    y = torch.stack([it[1] for it in items]).to(device)
    p = model(x)
    err = ((p - y).abs().mean(1, keepdim=True) * 4).clamp(0, 1).repeat(1, 3, 1, 1)
    grid = make_grid(torch.cat([x, p, y, err]).cpu(), nrow=len(items), padding=2)  # rows: input/restored/clean/error x4
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"samples_epoch{epoch:04d}.png"
    save_image(grid, path)
    mlflow.log_artifact(str(path), "samples")
    model.train()


def run_training(cfg, epochs, run_name, experiment, types=(0, 1, 2, 3), trial=None,
                 save=False, sample_every=10, device=DEVICE, seed=config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader = data.get_loaders(cfg["batch_size"], types=types)
    val_ds = data.RestorationDataset("val", types)

    model = ConvAE(cfg["base_ch"], cfg["latent_ch"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)

    run_dir = Path(config.OUT_DIR) / "runs" / run_name
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)

    best = {"objective": float("inf")}
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params({**cfg, "epochs": epochs, "types": list(types)})
        for epoch in range(1, epochs + 1):
            tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
            for x, y, _, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss, l1, ssim = restoration_loss(model(x), y, cfg["alpha"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot["loss"] += loss.item(); tot["l1"] += l1.item(); tot["ssim"] += ssim.item()
            sched.step()

            n = len(train_loader)
            metrics = {f"train_{k}": v / n for k, v in tot.items()}
            metrics.update(evaluate(model, val_loader, device))
            metrics["objective"] = metrics["val_l1"] + (1 - metrics["val_ssim"])
            metrics["lr"] = sched.get_last_lr()[0]
            mlflow.log_metrics(metrics, step=epoch)

            if metrics["objective"] < best["objective"]:
                best = {**metrics, "epoch": epoch}
                if save:
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                                "types": list(types), "metrics": {k: float(v) for k, v in metrics.items()}},
                               run_dir / "best.pt")
            if save and sample_every and (epoch % sample_every == 0 or epoch == 1):
                log_samples(model, val_ds, epoch, run_dir / "samples", device)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "epoch": epoch},
                           run_dir / "last.pt")
            if epoch % 5 == 0 or epoch == 1:
                print(f"[{run_name}] ep {epoch:3d} | train_loss {metrics['train_loss']:.4f} | "
                      f"val_l1 {metrics['val_l1']:.4f} val_ssim {metrics['val_ssim']:.4f} "
                      f"val_psnr {metrics['val_psnr']:.2f} | obj {metrics['objective']:.4f}")

            if trial is not None:
                trial.report(metrics["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if isinstance(v, (int, float))})
    return best
