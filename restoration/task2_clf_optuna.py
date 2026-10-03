import argparse, json
import mlflow, optuna
import config, trainer, task2_clf

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=25)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--study", default="task2_classifier")
args = ap.parse_args()

SEARCH_SPACE = {"lr": "loguniform [1e-4, 3e-3]", "batch_size": "categorical {16, 32, 64} (balanced: bs/4 per class)",
                "channels": f"categorical {task2_clf.CHANNELS}", "dropout": "uniform [0.0, 0.5] step 0.1",
                "weight_decay": "loguniform [1e-6, 1e-2]"}


def objective(trial):
    cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
               batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
               channels=trial.suggest_categorical("channels", list(task2_clf.CHANNELS)),
               dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
               weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    return task2_clf.run_clf_training(cfg, args.epochs, f"t2_clf_trial_{trial.number}", trial=trial)["objective"]


trainer.setup_mlflow(task2_clf.EXPERIMENT)
study = optuna.create_study(study_name=args.study, storage=f"sqlite:///{config.OUT_DIR}/optuna_restoration.db",
                            load_if_exists=True, direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=config.SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
with mlflow.start_run(run_name="t2_clf_optuna_study"):
    study.optimize(objective, n_trials=args.n_trials)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    summary = {"search_space": SEARCH_SPACE, "objective": "val macro-F1 (maximise)", "trial_epochs": args.epochs,
               "completed": len(done), "pruned": len(pruned), "best_value": study.best_value,
               "best_params": study.best_params}
    mlflow.log_dict(summary, "study_summary.json")
json.dump(study.best_params, open(f"{config.CONFIG_DIR}/task2_clf_best_params.json", "w"), indent=2)
json.dump(summary, open(f"{config.CONFIG_DIR}/task2_clf_optuna_summary.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"{config.CONFIG_DIR}/task2_clf_optuna_trials.csv", index=False)
print(f"completed={len(done)} pruned={len(pruned)} best={study.best_value:.4f}")
print("best params:", study.best_params)
