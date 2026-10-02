"""Usage: python -m src.task1.optuna_search --n_trials 30 --epochs 8"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import optuna

from src.task1.train import DEFAULTS, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=8, help="epochs per trial (short budget)")
    ap.add_argument("--out_dir", default="outputs/task1")
    ap.add_argument("--study_name", default="task1_universal_ae")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("task1_optuna")

    def objective(trial):
        cfg = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "latent_dim": trial.suggest_categorical("latent_dim", [64, 128, 256, 512]),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 48, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": args.epochs,
        }
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(cfg)
            try:
                res = run_training(cfg, trial=trial, verbose=False)
            except optuna.TrialPruned:
                mlflow.set_tag("pruned", "true")
                raise
            o = res["overall"]
            mlflow.log_metrics({"best_val_score": o["score"], "best_val_psnr": o["psnr"], "best_val_ssim": o["ssim"]})
        return o["score"]

    study = optuna.create_study(
        study_name=args.study_name, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///{out / 'optuna_task1.db'}", load_if_exists=True,
    )
    with mlflow.start_run(run_name="optuna_study"):
        study.optimize(objective, n_trials=args.n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_score", study.best_value)

    best = {**study.best_params, "epochs": 60}  # longer budget for the final run
    with open(out / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    print("Best trial:", study.best_trial.number, study.best_value)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
