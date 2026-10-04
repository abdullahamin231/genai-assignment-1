#!/usr/bin/env python3
"""Pull per-epoch train/val curves for the final run of each task out of MLflow."""
import sqlite3, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

TASK4_DB = REPO / "drive" / "task4_outputs" / "mlflow.db"
REST_DB = REPO / "drive" / "mlflow.db"


def series(db, run, key):
    con = sqlite3.connect(db)
    rows = con.execute(
        "select step, value from metrics where run_uuid=? and key=? order by step", (run, key)
    ).fetchall()
    con.close()
    return {"steps": [r[0] for r in rows], "values": [r[1] for r in rows]}


def tags(db, run):
    con = sqlite3.connect(db)
    try:
        rows = con.execute("select key, value from tags where run_uuid=?", (run,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return dict(rows)


TARGETS = {
    "task1": (REST_DB, "7abac8e9c832453aaefe67e41f2d58f2", "task1_final"),
    "task2_clf": (REST_DB, "1ab5833c537046ae9d207f3a114f5f6e", "task2_classifier"),
    "task3": (REST_DB, "1a72aaca674c4a4fab49f1ece41da9da", "task3_final"),
    "task4": (TASK4_DB, "2cb3ca41f3134f46aef9b81047100e2f", "final"),
}

KEYS = {
    "task1": ["train_loss", "val_l1", "val_ssim", "train_l1", "train_ssim",
              "val_psnr", "val_psnr_clean", "val_psnr_salt", "val_psnr_blur", "val_psnr_occlusion"],
    "task2_clf": ["train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1"],
    "task3": ["train_loss", "val_l1", "val_ssim", "val_acc", "val_gate_acc"],
    "task4": ["train_g_adv", "train_g_l1", "train_d_real", "train_d_fake",
              "val_l1", "val_ssim", "val_psnr"],
}

out = {}
for name, (db, run, label) in TARGETS.items():
    out[name] = {"run": run, "label": label, "tags": tags(db, run)}
    for k in KEYS[name]:
        s = series(db, run, k)
        if s["steps"]:
            out[name][k] = s
    print(f"{name}: " + ", ".join(
        f"{k}({len(out[name][k]['steps'])})" for k in KEYS[name] if k in out[name]
    ))

(OUT / "curves.json").write_text(json.dumps(out, indent=1))
print("wrote", OUT / "curves.json")
