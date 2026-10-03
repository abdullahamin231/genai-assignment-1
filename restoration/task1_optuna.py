import argparse, json
import mlflow, optuna
import config, trainer

EXPERIMENT = "restoration_task1"
ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=30)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--study", default="task1_universal_ae")
args = ap.parse_args()

SEARCH_SPACE = {
    "lr": "loguniform [1e-4, 3e-3]",
    "batch_size": "categorical {16, 32, 64}",
    "latent_ch": "categorical {8, 16, 32, 64} (latent = latent_ch x 8 x 8)",
    "base_ch": "categorical {32, 48, 64}",
    "dropout": "uniform [0.0, 0.3] step 0.05",
    "alpha": "uniform [0.5, 0.95] step 0.05 (L1 weight; SSIM weight = 1 - alpha)",
}


def objective(trial):
    cfg = dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        latent_ch=trial.suggest_categorical("latent_ch", [8, 16, 32, 64]),
        base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        alpha=trial.suggest_float("alpha", 0.5, 0.95, step=0.05),
    )
    res = trainer.run_training(cfg, args.epochs, f"t1_trial_{trial.number}", EXPERIMENT,
                               trial=trial, save=False, sample_every=0)
    return res["objective"]


trainer.setup_mlflow(EXPERIMENT)
study = optuna.create_study(
    study_name=args.study, storage=f"sqlite:///{config.OUT_DIR}/optuna_restoration.db",
    load_if_exists=True, direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=config.SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))

with mlflow.start_run(run_name="t1_optuna_study"):
    study.optimize(objective, n_trials=args.n_trials)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    summary = {"search_space": SEARCH_SPACE, "objective": "val_l1 + (1 - val_ssim) on the validation manifest",
               "trial_epochs": args.epochs, "completed": len(done), "pruned": len(pruned),
               "best_value": study.best_value, "best_params": study.best_params}
    mlflow.log_dict(summary, "study_summary.json")

json.dump(study.best_params, open(f"{config.CONFIG_DIR}/task1_best_params.json", "w"), indent=2)
json.dump(summary, open(f"{config.CONFIG_DIR}/task1_optuna_summary.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"{config.CONFIG_DIR}/task1_optuna_trials.csv", index=False)
print(f"completed={len(done)} pruned={len(pruned)} best={study.best_value:.4f}")
print("best params:", study.best_params)
