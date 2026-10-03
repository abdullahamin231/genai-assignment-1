import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchvision.utils import make_grid, save_image

import config
import data
from models import UNetGenerator, PatchDiscriminator, init_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_mlflow(experiment="task4_face2sketch"):
    mlflow.set_tracking_uri(f"sqlite:///{config.OUT_DIR}/mlflow.db")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(Path(config.OUT_DIR) / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)


def per_sample_metrics(fake, real):
    f, r = (fake.clamp(-1, 1) + 1) / 2, (real + 1) / 2
    l1 = (f - r).abs().flatten(1).mean(1)
    mse = ((f - r) ** 2).flatten(1).mean(1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    ssim = ssim_fn(f, r, data_range=1.0, reduction="none")
    return l1, psnr, ssim


@torch.no_grad()
def evaluate(G, loader, device=DEVICE):
    G.eval()
    acc = {"l1": [], "psnr": [], "ssim": []}
    for p, s, st in loader:
        p, s, st = p.to(device), s.to(device), st.to(device)
        l1, ps, ss = per_sample_metrics(G(p, st), s)
        acc["l1"].append(l1.cpu()); acc["psnr"].append(ps.cpu()); acc["ssim"].append(ss.cpu())
    G.train()
    return {f"val_{k}": torch.cat(v).mean().item() for k, v in acc.items()}


def triplet_grid(p, s, f):
    rep = lambda x: x.repeat(1, 3, 1, 1)
    return make_grid(torch.cat([p, rep(s), rep(f)]) * 0.5 + 0.5, nrow=p.shape[0], padding=2)


@torch.no_grad()
def log_samples(G, epoch, out_dir, device=DEVICE, n=8):
    G.eval()
    ds = data.get_dataset("val")
    p = torch.stack([ds[i][0] for i in range(n)]).to(device)
    s = torch.stack([ds[i][1] for i in range(n)]).to(device)
    st = torch.tensor([ds[i][2] for i in range(n)], device=device)
    grid1 = triplet_grid(p, s, G(p, st))  # rows: photo / real sketch / generated
    p4 = p[:4]
    rows = [p4] + [G(p4, torch.full((4,), k, device=device)).repeat(1, 3, 1, 1) for k in range(3)]
    grid2 = make_grid(torch.cat(rows) * 0.5 + 0.5, nrow=4, padding=2)  # rows: photo / style 0 / 1 / 2
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in (("fixed", grid1), ("styles", grid2)):
        path = out_dir / f"{name}_epoch{epoch:04d}.png"
        save_image(g, path)
        mlflow.log_artifact(str(path), f"samples/{name}")
    G.train()


def run_training(cfg, epochs, run_name, trial=None, save=False, sample_every=10, device=DEVICE):
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    train_loader, val_loader = data.get_loaders(cfg["batch_size"])

    G = UNetGenerator(cfg["base_ch"], cfg["emb_dim"], cfg["dropout"]).to(device)
    D = PatchDiscriminator(cfg["base_ch"], cfg["emb_dim"]).to(device)
    G.apply(init_weights); D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    bce, l1_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()

    run_dir = Path(config.OUT_DIR) / "runs" / run_name
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)

    best = {"objective": float("inf")}
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params({**cfg, "epochs": epochs})
        for epoch in range(1, epochs + 1):
            sums = dict(d_real=0.0, d_fake=0.0, g_adv=0.0, g_l1=0.0)
            for p, s, st in train_loader:
                p, s, st = p.to(device), s.to(device), st.to(device)
                fake = G(p, st)

                # Discriminator
                real_logits = D(p, s, st)
                fake_logits = D(p, fake.detach(), st)
                d_real = bce(real_logits, torch.ones_like(real_logits))
                d_fake = bce(fake_logits, torch.zeros_like(fake_logits))
                opt_d.zero_grad(set_to_none=True)
                (0.5 * (d_real + d_fake)).backward()
                opt_d.step()

                # Generator
                fake_logits = D(p, fake, st)
                g_adv = bce(fake_logits, torch.ones_like(fake_logits))
                g_l1 = l1_loss(fake, s)
                opt_g.zero_grad(set_to_none=True)
                (g_adv + cfg["lambda_l1"] * g_l1).backward()
                opt_g.step()

                for k, v in zip(sums, (d_real, d_fake, g_adv, g_l1)):
                    sums[k] += v.item()

            n = len(train_loader)
            metrics = {f"train_{k}": v / n for k, v in sums.items()}
            metrics.update(evaluate(G, val_loader, device))
            metrics["objective"] = metrics["val_l1"] + (1 - metrics["val_ssim"])
            mlflow.log_metrics(metrics, step=epoch)

            if metrics["objective"] < best["objective"]:
                best = {**metrics, "epoch": epoch}
                if save:
                    torch.save({"G": G.state_dict(), "cfg": cfg, "epoch": epoch,
                                "metrics": {k: float(v) for k, v in metrics.items()}},
                               run_dir / "best.pt")
            if save and sample_every and (epoch % sample_every == 0 or epoch == 1):
                log_samples(G, epoch, run_dir / "samples", device)
                torch.save({"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg, "epoch": epoch},
                           run_dir / "last.pt")
            if epoch % 5 == 0 or epoch == 1:
                print(f"[{run_name}] ep {epoch:3d} | d_real {metrics['train_d_real']:.3f} "
                      f"d_fake {metrics['train_d_fake']:.3f} g_adv {metrics['train_g_adv']:.3f} "
                      f"g_l1 {metrics['train_g_l1']:.3f} | val_l1 {metrics['val_l1']:.4f} "
                      f"val_ssim {metrics['val_ssim']:.4f} val_psnr {metrics['val_psnr']:.2f}")

            if trial is not None:
                trial.report(metrics["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if isinstance(v, (int, float))})
    return best
