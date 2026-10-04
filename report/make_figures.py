#!/usr/bin/env python3
"""Generate report figures: training/validation curves and Optuna study plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CURVES = json.loads((FIGS / "curves.json").read_text())

# Stitch palette
CYAN = "#00e5ff"
CORAL = "#ff7d7d"
VIOLET = "#b39ddb"
AMBER = "#ffca58"
GREEN = "#69f0ae"
BG = "#0b0e13"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _xy(key: str, series: dict):
    return series[key]["steps"], series[key]["values"]


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Task 1 ----
def task1_curves():
    d = CURVES["task1"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.1))

    x, y = _xy("train_loss", d); ax1.plot(x, y, color=CYAN, lw=1.4, label="train loss")
    x, y = _xy("val_l1", d);     ax1.plot(x, y, color=CORAL, lw=1.4, label="val L1")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("(a) Training and validation loss")
    ax1.legend()

    x, y = _xy("val_psnr_clean", d); ax2.plot(x, y, color="#80deea", lw=1.2, label="clean")
    for key, col, lab in [
        ("val_psnr_salt", AMBER, "salt & pepper"),
        ("val_psnr_blur", VIOLET, "gaussian blur"),
        ("val_psnr_occlusion", CORAL, "occlusion"),
    ]:
        x, y = _xy(key, d)
        ax2.plot(x, y, color=col, lw=1.2, alpha=0.9, label=lab)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("PSNR (dB)")
    ax2.set_ylim(8, 30)
    ax2b = ax2.twinx()
    x, y = _xy("val_ssim", d); ax2b.plot(x, y, color=GREEN, lw=1.8, label="val SSIM")
    ax2b.set_ylabel("SSIM", color=GREEN); ax2b.tick_params(axis="y", colors=GREEN)
    ax2b.grid(False); ax2b.set_ylim(0.55, 1.0)
    ax2.set_title("(b) Validation PSNR by corruption + SSIM")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", ncol=2)
    save(fig, "task1_curves.png")


# --------------------------------------------------------------- Task 2 -----
def task2_curves():
    d = CURVES["task2_clf"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.0))

    x, y = _xy("train_loss", d); ax1.plot(x, y, color=CYAN, lw=1.5, label="train CE")
    x, y = _xy("val_acc", d); ax1.plot(x, y, color=CORAL, lw=1.5, label="val accuracy")
    x, y = _xy("val_macro_f1", d); ax1.plot(x, y, color=GREEN, lw=1.5, label="val macro-F1")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss / score")
    ax1.set_title("(a) Classifier loss and validation scores"); ax1.legend(loc="center right")

    x, y = _xy("train_acc", d); ax2.plot(x, y, color=CYAN, lw=1.3, alpha=0.7, label="train acc")
    x, y = _xy("val_acc", d);   ax2.plot(x, y, color=CORAL, lw=1.6, label="val acc")
    ax2.axhline(0.9977, color="#555", ls="--", lw=1)
    ax2.annotate("test acc = 99.77%", xy=(0.55, 0.9977), xytext=(0.5, 0.955),
                 fontsize=8, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#777", lw=0.9))
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_ylim(0.85, 1.005)
    ax2.set_title("(b) Accuracy convergence"); ax2.legend(loc="lower right")
    save(fig, "task2_curves.png")


# --------------------------------------------------------------- Task 3 -----
def task3_curves():
    d = CURVES["task3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.0))

    x, y = _xy("train_loss", d); ax1.plot(x, y, color=CYAN, lw=1.6, label="train joint loss")
    x, y = _xy("val_l1", d); ax1.plot(x, y, color=CORAL, lw=1.6, label="val L1")
    ax1.axvspan(-0.5, 2.5, color=AMBER, alpha=0.18, zorder=0)
    ax1.text(1.0, ax1.get_ylim()[1] * 0.92, "gate warm-up\n(experts frozen)",
             ha="center", fontsize=7.5, color="#8a6d00")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("(a) Warm-up then joint fine-tuning"); ax1.legend(loc="upper right")

    x, y = _xy("val_ssim", d); ax2.plot(x, y, color=GREEN, lw=1.8, label="val SSIM")
    ax2b = ax2.twinx()
    x, y = _xy("val_gate_acc", d)
    ax2b.plot(x, y, color=VIOLET, lw=1.6, ls="--", label="gate accuracy")
    ax2b.set_ylabel("gate accuracy", color=VIOLET)
    ax2b.tick_params(axis="y", colors=VIOLET); ax2b.grid(False); ax2b.set_ylim(0.4, 1.02)
    ax2.axvspan(-0.5, 2.5, color=AMBER, alpha=0.18, zorder=0)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("SSIM")
    ax2.set_title("(b) Reconstruction SSIM and routing accuracy")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right")
    save(fig, "task3_curves.png")


# --------------------------------------------------------------- Task 4 -----
def task4_curves():
    d = CURVES["task4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.0))

    x, y = _xy("train_g_adv", d); ax1.plot(x, y, color=CYAN, lw=1.2, label="G adv")
    x, y = _xy("train_g_l1", d);  ax1.plot(x, y, color=CORAL, lw=1.2, label="G L1 x100")
    x, y = _xy("train_d_real", d); ax1.plot(x, y, color=GREEN, lw=1.2, ls="--", label="D real")
    x, y = _xy("train_d_fake", d); ax1.plot(x, y, color=VIOLET, lw=1.2, ls="--", label="D fake")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("(a) Adversarial training dynamics"); ax1.legend(ncol=2)

    x, y = _xy("val_ssim", d); ax2.plot(x, y, color=GREEN, lw=1.8, label="val SSIM")
    ax2b = ax2.twinx()
    x, y = _xy("val_psnr", d)
    ax2b.plot(x, y, color=CYAN, lw=1.5, ls="--", label="val PSNR")
    x, y = _xy("val_l1", d)
    ax2b.plot(x, y, color=CORAL, lw=1.5, ls=":", label="val L1")
    ax2b.set_yscale("log"); ax2b.grid(False)
    ax2b.set_ylabel("PSNR (dB) / L1 (log)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("SSIM")
    ax2.set_title("(b) Validation reconstruction quality")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right")
    save(fig, "task4_curves.png")


# --------------------------------------------------------------- Optuna -----
def optuna_figures():
    specs = [
        ("restoration/configs/task1_optuna_trials.csv", "task1", True),
        ("restoration/configs/task2_clf_optuna_trials.csv", "task2_clf", False),
        ("restoration/configs/task2_spec_optuna_trials.csv", "task2_spec", True),
        ("restoration/configs/task3_optuna_trials.csv", "task3", True),
        ("task4/configs/optuna_trials.csv", "task4", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    axes = axes.ravel()
    summary = []
    for ax, (rel, name, minimise) in zip(axes, specs):
        rows = list(csv.DictReader((REPO / rel).open()))
        vals, states = [], []
        for r in rows:
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            vals.append(v); states.append(r.get("state", "COMPLETE"))
        # Optuna's reported best is over COMPLETE trials only - a pruned trial can
        # transiently hold a better intermediate value (e.g. task2_spec trial 15).
        done = [v for v, s in zip(vals, states) if s == "COMPLETE"]
        if not done:
            done = vals
        best = min(done) if minimise else max(done)
        xs = range(1, len(vals) + 1)
        cols = [AMBER if s == "PRUNED" else ("#bdbdbd" if s == "FAIL" else CYAN)
                for s in states]
        ax.scatter(xs, vals, c=cols, s=22, edgecolors="#333", linewidths=0.4, zorder=3)
        run = []
        cur = None
        for v, s in zip(vals, states):
            if s != "COMPLETE":
                run.append(cur)
                continue
            cur = v if cur is None else (min(cur, v) if minimise else max(cur, v))
            run.append(cur)
        ax.plot(range(1, len(vals) + 1), run, color="#1565c0", lw=1.4, zorder=2)
        bidx = (done.index(best) if minimise else len(done) - 1 - done[::-1].index(best))
        # map the COMPLETE-only index back onto the full trial list
        complete_positions = [i for i, s in enumerate(states) if s == "COMPLETE"]
        marker_x = complete_positions[bidx] + 1
        ax.scatter([marker_x], [best], marker="*", s=130, color="#d32f2f",
                   zorder=4, edgecolors="white", linewidths=0.6)
        ax.set_title(f"{name}  (best {best:.4g})", fontsize=9)
        ax.set_xlabel("trial")
        ax.set_ylabel("objective" if minimise else "macro-F1")
        n_pruned = states.count("PRUNED")
        summary.append((name, len(vals), n_pruned, best))
    axes[-1].axis("off")
    axes[-1].add_artist(Line2D([], [], marker="o", ls="", color=CYAN, label="completed"))
    axes[-1].add_artist(Line2D([], [], marker="o", ls="", color=AMBER, label="pruned"))
    axes[-1].add_artist(Line2D([], [], marker="*", ls="", color="#d32f2f", ms=12, label="best"))
    axes[-1].legend(loc="center left", fontsize=9)
    axes[-1].set_title("Optuna studies (all four tasks)", fontsize=9)
    save(fig, "optuna_studies.png")
    return summary


if __name__ == "__main__":
    task1_curves()
    task2_curves()
    task3_curves()
    task4_curves()
    s = optuna_figures()
    print("\nstudy summary:")
    for row in s:
        print("  ", row)
