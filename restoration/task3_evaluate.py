import json, shutil
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
from task3_moe import load_task2, load_task3, BRANCHES

R, OUT, dev = Path(config.RESULTS_DIR), Path(config.OUT_DIR), trainer.DEVICE
ORDER = {"clean": 0, "salt": 1, "blur": 2, "occlusion": 3}
SEV = {"-": 0, "low": 1, "medium": 2, "high": 3}
moe, ck = load_task3(OUT / "runs/task3_final/best.pt", dev)
base, _ = load_task2(dev)          # Task 2 hard-routing system (untouched classifier + specialists)
base.eval()
print("Task 3 best epoch:", ck["epoch"], "| cfg:", ck["cfg"])


@torch.no_grad()
def hard(x):
    pred = base.gate(x).argmax(1)
    outs = torch.stack([x] + [e(x).clamp(0, 1) for e in base.experts])
    return outs[pred, torch.arange(len(x), device=x.device)], pred


def sort_idx(df):
    return df.sort_values(by=["type", "severity"], key=lambda s: s.map(ORDER if s.name == "type" else SEV))


# ---------------- test set: soft vs hard vs input ----------------
ds = data.RestorationDataset("test")
loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
rec = {f"{p}_{k}": [] for p in ("s", "h", "in") for k in ("l1", "psnr", "ssim")}
W = []
with torch.no_grad():
    for x, y, lab, _ in loader:
        x, y = x.to(dev), y.to(dev)
        ys, w, _ = moe(x)
        yh, _ = hard(x)
        for pre, out in (("s", ys.clamp(0, 1)), ("h", yh), ("in", x)):
            l1, ps, ss = trainer.per_sample_metrics(out, y)
            rec[f"{pre}_l1"].append(l1.cpu()); rec[f"{pre}_psnr"].append(ps.cpu()); rec[f"{pre}_ssim"].append(ss.cpu())
        W.append(w.cpu())
W = torch.cat(W).numpy()
df = pd.DataFrame({k: torch.cat(v).numpy() for k, v in rec.items()})
df["type"] = [e["type"] for e in ds.entries]
df["level"] = [e["level"] for e in ds.entries]
df["idx"] = [e["idx"] for e in ds.entries]
df["severity"] = df["level"].map({-1: "-", 0: "low", 1: "medium", 2: "high"})
df["label"] = df["type"].map(ORDER)
wcols = [f"w_{b}" for b in BRANCHES]
for k, c in enumerate(wcols):
    df[c] = W[:, k]
df["pred"] = W.argmax(1)
df["entropy"] = -(W * np.log(W + 1e-9)).sum(1)
for c in ("h_psnr", "in_psnr"):                       # undefined (inf) for bypassed / untouched clean images
    df.loc[df.type == "clean", c] = np.nan

cols = ["s_ssim", "h_ssim", "s_psnr", "h_psnr", "s_l1", "h_l1", "in_ssim", "in_psnr"]
by_sev = sort_idx(df.groupby(["type", "severity"])[cols].mean().reset_index())
by_type = df.groupby("type")[cols].mean().reindex(list(ORDER))
corr = df[df.type != "clean"]
print("\n=== Soft MoE (s_) vs Task 2 hard routing (h_), test set, by condition/severity ===")
print(by_sev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nby type:\n", by_type.to_string(float_format=lambda v: f"{v:.4f}"))
overall = {"s_ssim": df.s_ssim.mean(), "h_ssim": df.h_ssim.mean(), "in_ssim": df.in_ssim.mean(),
           "s_l1": df.s_l1.mean(), "h_l1": df.h_l1.mean(),
           "s_psnr_corrupted_only": corr.s_psnr.mean(), "h_psnr_corrupted_only": corr.h_psnr.mean()}
print("\noverall:", {k: round(float(v), 4) for k, v in overall.items()})

# ---------------- gating behaviour ----------------
w_sev = sort_idx(df.groupby(["type", "severity"])[wcols].mean().reset_index())
print("\n=== Mean gate weights per true condition and severity ===")
print(w_sev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
acc = (df.pred == df.label)
print(f"\ngate argmax accuracy: {acc.mean():.4f}")
print(sort_idx(df.assign(ok=acc).groupby(["type", "severity"])["ok"].mean().reset_index()).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

usage = []
for k, b in enumerate(BRANCHES):
    unrel = df[df.label != k]
    row = {"branch": b, "mean_weight": df[wcols[k]].mean(), "argmax_share": float((df.pred == k).mean()),
           "argmax_share_on_unrelated_inputs": float((unrel.pred == k).mean()),
           "mean_weight_on_unrelated_inputs": unrel[wcols[k]].mean()}
    row["inactive"] = bool(row["mean_weight"] < 0.05 and row["argmax_share"] < 0.02)
    row["dominates_unrelated"] = bool(row["argmax_share_on_unrelated_inputs"] > 0.10)
    usage.append(row)
usage = pd.DataFrame(usage)
print("\n=== Expert usage (inactive: mean w<0.05 and argmax<2%; dominates unrelated: argmax on >10% of unrelated inputs) ===")
print(usage.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nsamples with max weight > 0.95: {(W.max(1) > 0.95).mean():.3f} | with max weight < 0.60: {(W.max(1) < 0.60).mean():.3f}")

# heatmap + distribution figures
fig, ax = plt.subplots(figsize=(6, 6))
M = w_sev[wcols].values
ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(BRANCHES, rotation=30)
ax.set_yticks(range(len(w_sev))); ax.set_yticklabels([f"{t} {s}" if s != "-" else t for t, s in zip(w_sev.type, w_sev.severity)])
for i in range(M.shape[0]):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] < 0.6 else "black", fontsize=8)
ax.set_title("Mean gate weights by true condition and severity (test)")
plt.tight_layout(); plt.savefig(R / "task3_routing_heatmap.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
types = list(ORDER)
for k, b in enumerate(BRANCHES):
    axs[k].boxplot([df[df.type == t][wcols[k]].values for t in types], labels=types, showfliers=False)
    axs[k].set_title(f"weight on {b} branch"); axs[k].tick_params(axis="x", rotation=30)
axs[0].set_ylabel("gate weight")
plt.tight_layout(); plt.savefig(R / "task3_weight_distribution.png", dpi=130); plt.close()


# ---------------- qualitative figures ----------------
@torch.no_grad()
def show(x, y, tags, path):
    x, y = x.to(dev), y.to(dev)
    ys, w, _ = moe(x)
    ys = ys.clamp(0, 1)
    yh, ph = hard(x)
    ssm = trainer.per_sample_metrics(ys, y)[2]
    ssh = trainer.per_sample_metrics(yh, y)[2]
    err = (ys - y).abs().mean(1).cpu().numpy()
    n = len(x)
    fig, ax = plt.subplots(n, 6, figsize=(13, 2.2 * n))
    ax = np.atleast_2d(ax)
    toimg = lambda t: t.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    for r in range(n):
        panels = [(y[r], "clean target"), (x[r], f"input ({tags[r]})"),
                  (yh[r], f"hard-routed [{C.CLASS_NAMES[ph[r]]}]\nSSIM {ssh[r]:.2f}"),
                  (ys[r], f"soft MoE\nSSIM {ssm[r]:.2f}")]
        for c, (im, t) in enumerate(panels):
            ax[r, c].imshow(toimg(im)); ax[r, c].set_title(t, fontsize=7); ax[r, c].axis("off")
        ax[r, 4].imshow(err[r], cmap="inferno", vmin=0, vmax=0.5); ax[r, 4].set_title("abs error (soft)", fontsize=7); ax[r, 4].axis("off")
        ax[r, 5].bar(range(4), w[r].cpu().numpy(), color=["#888", "#d62728", "#1f77b4", "#2ca02c"])
        ax[r, 5].set_ylim(0, 1); ax[r, 5].set_xticks(range(4)); ax[r, 5].set_xticklabels(["id", "salt", "blur", "occ"], fontsize=7)
        ax[r, 5].set_title("gate weights", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def show_ds(ds_idx, path):
    x = torch.stack([ds[i][0] for i in ds_idx]); y = torch.stack([ds[i][1] for i in ds_idx])
    tags = [df.iloc[i]["type"] if df.iloc[i]["type"] == "clean" else f"{df.iloc[i]['type']} {df.iloc[i]['severity']}" for i in ds_idx]
    show(x, y, tags, path)


lookup = {(e["idx"], e["type"], e["level"]): i for i, e in enumerate(ds.entries)}
examples = [(10, "clean", -1)] + [(20 + 3 * j, t, lv) for j, t in enumerate(("salt", "blur", "occlusion")) for lv in range(3)] \
           + [(100, "salt", 2), (200, "occlusion", 2)]
show_ds([lookup[e] for e in examples], R / "task3_examples_12.png")
spread = df.sort_values("entropy", ascending=False).head(6)
show_ds(spread.index.tolist(), R / "task3_distributed_weights.png")
print("\nMost distributed gate weights:\n", spread[["idx", "type", "severity"] + wcols + ["s_ssim", "h_ssim"]].to_string(float_format=lambda v: f"{v:.3f}"))
worst = df.sort_values("s_ssim").head(6)
show_ds(worst.index.tolist(), R / "task3_failure_cases.png")
print("\nWorst soft-MoE cases:\n", worst[["idx", "type", "severity", "s_ssim", "h_ssim", "in_ssim"] + wcols].to_string(float_format=lambda v: f"{v:.3f}"))

# ---------------- mixed corruptions (two corruptions on one image) ----------------
clean_test = data.get_clean("test")
mixed_rows, ex_x, ex_y, ex_tag = [], [], [], []
for a, b in (("blur", "salt"), ("salt", "occlusion"), ("blur", "occlusion")):
    xs, ys_ = [], []
    for i in range(200):
        img = clean_test[i].astype(np.float32) / 255
        s1 = C.fixed_spec(C.CLASS_IDX[a], 1, 5000 + 2 * i)
        s2 = C.fixed_spec(C.CLASS_IDX[b], 1, 5001 + 2 * i)
        xs.append(C.apply_corruption(C.apply_corruption(img, s1), s2).transpose(2, 0, 1)); ys_.append(img.transpose(2, 0, 1))
    x = torch.from_numpy(np.stack(xs)).to(dev); y = torch.from_numpy(np.stack(ys_)).to(dev)
    with torch.no_grad():
        outs, w, _ = moe(x); outs = outs.clamp(0, 1)
        outh, _ = hard(x)
    row = {"combo": f"{a}+{b} (medium)"}
    for pre, o in (("in", x), ("hard", outh), ("soft", outs)):
        _, ps, ss = trainer.per_sample_metrics(o, y)
        row[f"{pre}_ssim"], row[f"{pre}_psnr"] = ss.mean().item(), ps.mean().item()
    for k, br in enumerate(BRANCHES):
        row[f"w_{br}"] = w[:, k].mean().item()
    mixed_rows.append(row)
    ex_x.append(x[:2].cpu()); ex_y.append(y[:2].cpu()); ex_tag += [f"{a}+{b}"] * 2
mixed = pd.DataFrame(mixed_rows)
print("\n=== Mixed corruptions (200 test images each): input vs hard routing vs soft MoE ===")
print(mixed.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
show(torch.cat(ex_x), torch.cat(ex_y), ex_tag, R / "task3_mixed_corruptions.png")

# ---------------- save ----------------
by_sev.to_csv(R / "task3_test_table.csv", index=False)
w_sev.to_csv(R / "task3_gate_weights_by_condition.csv", index=False)
usage.to_csv(R / "task3_expert_usage.csv", index=False)
mixed.to_csv(R / "task3_mixed_corruptions.csv", index=False)
json.dump({"epoch": ck["epoch"], "cfg": ck["cfg"], "overall": {k: float(v) for k, v in overall.items()},
           "gate_accuracy": float(acc.mean()), "by_type": by_type.to_dict("index")},
          open(R / "task3_test_metrics.json", "w"), indent=2)
(OUT / "results").mkdir(exist_ok=True)
for f in R.glob("task3_*"):
    shutil.copy(f, OUT / "results" / f.name)
print("\nSaved to", R, "and", OUT / "results")
