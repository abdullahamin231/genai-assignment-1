import json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config, data, trainer
import corruptions as C
from models import ConvAE
from task2_clf import CorruptionClassifier, CHANNELS, clf_metrics

R, OUT, dev = Path(config.RESULTS_DIR), Path(config.OUT_DIR), trainer.DEVICE
NAMES = C.CLASS_NAMES

ck = torch.load(OUT / "runs/task2_classifier/best.pt", map_location=dev)
clf = CorruptionClassifier(CHANNELS[ck["cfg"]["channels"]], ck["cfg"]["dropout"]).to(dev)
clf.load_state_dict(ck["model"]); clf.eval()
specs = []
for name in ("salt", "blur", "occlusion"):
    c = torch.load(OUT / f"runs/task2_specialist_{name}/best.pt", map_location=dev)
    m = ConvAE(c["cfg"]["base_ch"], c["cfg"]["latent_ch"], c["cfg"]["dropout"]).to(dev)
    m.load_state_dict(c["model"]); m.eval(); specs.append(m)


@torch.no_grad()
def route_all(x):
    """Returns class probs and the 4 candidate outputs: [identity, salt, blur, occlusion] (clean = identity bypass)."""
    probs = clf(x).softmax(1)
    outs = torch.stack([x] + [m(x).clamp(0, 1) for m in specs])
    return probs, outs


ds = data.RestorationDataset("test")
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
rec = {f"{p}_{k}": [] for p in ("or", "pr", "in") for k in ("l1", "psnr", "ssim")}
P, Y = [], []
with torch.no_grad():
    for x, y, lab, _ in loader:
        x, y, lab = x.to(dev), y.to(dev), lab.to(dev)
        probs, outs = route_all(x)
        pred = probs.argmax(1)
        ar = torch.arange(len(x), device=dev)
        for pre, out in (("or", outs[lab, ar]), ("pr", outs[pred, ar]), ("in", x)):
            l1, ps, ss = trainer.per_sample_metrics(out, y)
            rec[f"{pre}_l1"].append(l1.cpu()); rec[f"{pre}_psnr"].append(ps.cpu()); rec[f"{pre}_ssim"].append(ss.cpu())
        P.append(probs.cpu()); Y.append(lab.cpu())
P, Y = torch.cat(P), torch.cat(Y)

df = pd.DataFrame({k: torch.cat(v).numpy() for k, v in rec.items()})
df["type"] = [e["type"] for e in ds.entries]
df["level"] = [e["level"] for e in ds.entries]
df["idx"] = [e["idx"] for e in ds.entries]
df["severity"] = df["level"].map({-1: "-", 0: "low", 1: "medium", 2: "high"})
df["label"], df["pred"] = Y.numpy(), P.argmax(1).numpy()
for k, n in enumerate(NAMES):
    df[f"p_{n}"] = P[:, k].numpy()
for c in ("or_psnr", "pr_psnr", "in_psnr"):      # PSNR undefined for identity-bypassed clean inputs
    df.loc[df.type == "clean", c] = np.nan

# ---- classifier results
cm = clf_metrics(P, Y)
print("=== Classifier (test) ===")
print(f"accuracy {cm['acc']:.4f} | macro precision {cm['macro_precision']:.4f} "
      f"recall {cm['macro_recall']:.4f} F1 {cm['macro_f1']:.4f}")
per_class = pd.DataFrame({n: [cm[f"precision_{n}"], cm[f"recall_{n}"], cm[f"f1_{n}"]] for n in NAMES},
                         index=["precision", "recall", "f1"]).T
print(per_class.to_string(float_format=lambda v: f"{v:.4f}"))
cmat = confusion_matrix(df.label, df.pred, labels=[0, 1, 2, 3], normalize="true")
fig, ax = plt.subplots(figsize=(5, 4.4))
ax.imshow(cmat, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(NAMES)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmat[i, j]:.2f}", ha="center", va="center", color="white" if cmat[i, j] > .5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Normalized confusion matrix (test)")
plt.tight_layout(); plt.savefig(R / "task2_confusion_matrix.png", dpi=130); plt.close()

order = {"clean": 0, "salt": 1, "blur": 2, "occlusion": 3}
sev_o = {"-": 0, "low": 1, "medium": 2, "high": 3}
acc_sev = df.assign(ok=(df.label == df.pred)).groupby(["type", "severity"])["ok"].mean().reset_index()
acc_sev = acc_sev.sort_values(by=["type", "severity"], key=lambda s: s.map(order if s.name == "type" else sev_o))
print("\nClassifier accuracy by condition/severity:\n", acc_sev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---- restoration results: oracle vs predicted routing
cols = ["or_ssim", "pr_ssim", "or_psnr", "pr_psnr", "or_l1", "pr_l1", "in_ssim", "in_psnr"]
by_sev = df.groupby(["type", "severity"])[cols].mean().reset_index()
by_sev = by_sev.sort_values(by=["type", "severity"], key=lambda s: s.map(order if s.name == "type" else sev_o))
by_type = df.groupby("type")[cols].mean().reindex(list(order))
print("\n=== Hard-routed restoration (test): oracle (or_) vs predicted (pr_) ===")
print(by_sev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nby type:\n", by_type.to_string(float_format=lambda v: f"{v:.4f}"))
overall = df[cols].mean()
print("\noverall (PSNR over corrupted inputs only):\n", overall.to_string(float_format=lambda v: f"{v:.4f}"))

# ---- failure analysis: misrouted samples that lost the most SSIM vs oracle
mis = df[df.label != df.pred].copy()
mis["gap"] = mis.or_ssim - mis.pr_ssim
worst = mis.sort_values("gap", ascending=False).head(6)
print(f"\nMisrouted: {len(mis)}/{len(df)} ({100 * len(mis) / len(df):.2f}%)")
print(pd.crosstab([df.type, df.severity], df.pred.map(dict(enumerate(NAMES)))).to_string())
print("\nWorst misrouting cases:\n", worst[["idx", "type", "severity", "pred", "or_ssim", "pr_ssim", "gap"]].to_string())

lookup = {(e["idx"], e["type"], e["level"]): i for i, e in enumerate(ds.entries)}


@torch.no_grad()
def show(ds_idx, path):
    x = torch.stack([ds[i][0] for i in ds_idx]).to(dev)
    y = torch.stack([ds[i][1] for i in ds_idx])
    lab = torch.tensor([ds[i][2] for i in ds_idx], device=dev)
    probs, outs = route_all(x)
    pred = probs.argmax(1)
    ar = torch.arange(len(ds_idx), device=dev)
    o_or, o_pr, x = outs[lab, ar].cpu(), outs[pred, ar].cpu(), x.cpu()
    fig, ax = plt.subplots(len(ds_idx), 5, figsize=(10.5, 2.3 * len(ds_idx)))
    for r, i in enumerate(ds_idx):
        row = df.iloc[i]
        tag = row["type"] if row["type"] == "clean" else f"{row['type']} {row['severity']}"
        err = (o_pr[r] - y[r]).abs().mean(0)
        panels = [(y[r], "clean target"), (x[r], f"input ({tag})"),
                  (o_or[r], f"oracle: {NAMES[row['label']]}\nSSIM {row['or_ssim']:.2f}"),
                  (o_pr[r], f"predicted: {NAMES[row['pred']]} (p={probs[r].max():.2f})\nSSIM {row['pr_ssim']:.2f}"),
                  (err, "abs error (predicted)")]
        for c, (im, t) in enumerate(panels):
            a = ax[r, c]
            if c == 4:
                a.imshow(im.numpy(), cmap="inferno", vmin=0, vmax=0.5)
            else:
                a.imshow(im.permute(1, 2, 0).numpy())
            a.set_title(t, fontsize=7); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


examples = [(10, "clean", -1)] + [(20 + 3 * j, t, lv) for j, t in enumerate(("salt", "blur", "occlusion")) for lv in range(3)] \
           + [(100, "salt", 2), (200, "occlusion", 2)]
show([lookup[e] for e in examples], R / "task2_examples.png")
if len(worst):
    show(worst.index.tolist(), R / "task2_routing_failures.png")

per_class.to_csv(R / "task2_classifier_per_class.csv")
by_sev.to_csv(R / "task2_test_table.csv", index=False)
acc_sev.to_csv(R / "task2_classifier_acc_by_severity.csv", index=False)
json.dump({"classifier": cm, "confusion_normalized": cmat.tolist(), "overall": overall.to_dict(),
           "by_type": by_type.to_dict("index"), "misrouted": int(len(mis)), "n": int(len(df))},
          open(R / "task2_test_metrics.json", "w"), indent=2)
(OUT / "results").mkdir(exist_ok=True)
for f in R.glob("task2_*"):
    shutil.copy(f, OUT / "results" / f.name)
print("\nSaved to", R, "and", OUT / "results")
