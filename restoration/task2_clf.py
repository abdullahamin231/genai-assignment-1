"""Task 2 corruption classifier: model, balanced sampler, metrics, training loop."""
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Sampler

import config, data
import corruptions as C
from models import conv_block
from trainer import DEVICE

EXPERIMENT = "restoration_task2"
CHANNELS = {"small": [16, 32, 64, 128], "medium": [32, 64, 128, 256],
            "tapered": [32, 64, 128, 128], "wide": [48, 96, 192, 256]}


class CorruptionClassifier(nn.Module):
    """4 stages of (conv-BN-ReLU x2, maxpool) -> global average pool -> dropout -> linear(4). Outputs logits."""

    def __init__(self, chs, dropout=0.2, n_classes=4):
        super().__init__()
        layers, i = [], 3
        for o in chs:
            layers += [conv_block(i, o), conv_block(o, o), nn.MaxPool2d(2)]
            i = o
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(i, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class BalancedTrainDS(data.RestorationDataset):
    """Train set where the caller picks the corruption class: key = (image_index, class)."""

    def __init__(self):
        super().__init__("train")

    def __getitem__(self, key):
        i, t = key
        img = self.clean[i].astype(np.float32) / 255.0
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        spec = C.sample_spec(rng, int(t))
        return data._t(C.apply_corruption(img, spec)), data._t(img), int(t), -1


class BalancedBatchSampler(Sampler):
    """Every batch holds exactly batch_size/4 samples of each of the 4 classes."""

    def __init__(self, n, batch_size):
        assert batch_size % 4 == 0
        self.n, self.per, self.nb = n, batch_size // 4, n // batch_size

    def __len__(self):
        return self.nb

    def __iter__(self):
        for _ in range(self.nb):
            batch = []
            for t in range(4):
                batch += [(i, t) for i in torch.randint(0, self.n, (self.per,)).tolist()]
            yield batch


@torch.no_grad()
def predict(model, loader, device=DEVICE):
    model.eval()
    P, Y = [], []
    for x, _, lab, _ in loader:
        P.append(model(x.to(device)).softmax(1).cpu()); Y.append(lab)
    model.train()
    return torch.cat(P), torch.cat(Y)


def clf_metrics(probs, y):
    pred, y = probs.argmax(1).numpy(), y.numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    out = {"acc": accuracy_score(y, pred), "macro_precision": p.mean(), "macro_recall": r.mean(), "macro_f1": f.mean()}
    for k, name in enumerate(C.CLASS_NAMES):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = p[k], r[k], f[k]
    return {k: float(v) for k, v in out.items()}


def run_clf_training(cfg, epochs, run_name, trial=None, save=False, device=DEVICE, seed=config.SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    ds = BalancedTrainDS()
    loader = DataLoader(ds, batch_sampler=BalancedBatchSampler(len(ds), cfg["batch_size"]),
                        num_workers=2, persistent_workers=True)
    val_loader = DataLoader(data.RestorationDataset("val"), batch_size=128, shuffle=False)

    model = CorruptionClassifier(CHANNELS[cfg["channels"]], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    ce = nn.CrossEntropyLoss()

    run_dir = Path(config.OUT_DIR) / "runs" / run_name
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)
    best = {"objective": -1.0}
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params({**cfg, "epochs": epochs})
        for epoch in range(1, epochs + 1):
            tl = tc = tn = 0
            for x, _, lab, _ in loader:
                x, lab = x.to(device, non_blocking=True), lab.to(device)
                logits = model(x)
                loss = ce(logits, lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tl += loss.item() * len(lab); tc += (logits.argmax(1) == lab).sum().item(); tn += len(lab)
            sched.step()
            probs, y = predict(model, val_loader, device)
            m = clf_metrics(probs, y)
            metrics = {"train_loss": tl / tn, "train_acc": tc / tn, **{f"val_{k}": v for k, v in m.items()}}
            mlflow.log_metrics(metrics, step=epoch)
            obj = m["macro_f1"]
            if obj > best["objective"]:
                best = {**metrics, "objective": obj, "epoch": epoch}
                if save:
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "metrics": metrics},
                               run_dir / "best.pt")
            if epoch % 5 == 0 or epoch == 1:
                print(f"[{run_name}] ep {epoch:3d} | loss {metrics['train_loss']:.4f} train_acc {metrics['train_acc']:.4f} "
                      f"| val_acc {m['acc']:.4f} val_macro_f1 {m['macro_f1']:.4f}")
            if trial is not None:
                trial.report(obj, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if isinstance(v, (int, float))})
    return best
