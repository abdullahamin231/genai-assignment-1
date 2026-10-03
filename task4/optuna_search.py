import argparse, json
import mlflow, optuna
import config, trainer

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=30)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--study", default="task4_cgan")
args = ap.parse_args()


def objective(trial):
    cfg = dict(
        lr_g=trial.suggest_float("lr_g", 1e-4, 1e-3, log=True),
        lr_d=trial.suggest_float("lr_d", 5e-5, 5e-4, log=True),
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        emb_dim=trial.suggest_categorical("emb_dim", [8, 16, 32, 64]),
        lambda_l1=trial.suggest_categorical("lambda_l1", [10, 50, 100, 200]),
    )
    res = trainer.run_training(cfg, args.epochs, f"trial_{trial.number}", trial=trial,
                               save=False, sample_every=0)
    return res["objective"]


trainer.setup_mlflow()
study = optuna.create_study(
    study_name=args.study, storage=f"sqlite:///{config.OUT_DIR}/optuna_task4.db",
    load_if_exists=True, direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=config.SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))

with mlflow.start_run(run_name="optuna_study"):
    study.optimize(objective, n_trials=args.n_trials)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    mlflow.log_dict({"completed": len(done), "pruned": len(pruned),
                     "best_value": study.best_value, "best_params": study.best_params},
                    "study_summary.json")

best = dict(config.DEFAULT_PARAMS); best.update(study.best_params)
json.dump(best, open(f"{config.CONFIG_DIR}/best_params.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"{config.CONFIG_DIR}/optuna_trials.csv", index=False)
print(f"completed={len(done)} pruned={len(pruned)} best={study.best_value:.4f}")
print("best params:", best)
