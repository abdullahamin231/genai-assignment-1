import argparse, json
import mlflow, optuna
import config, trainer, task3_moe

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=12)
ap.add_argument("--epochs", type=int, default=8, help="total epochs per trial (including warm-up)")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--study", default="task3_soft_moe")
args = ap.parse_args()

SEARCH_SPACE = {
    "lr_ft": "loguniform [1e-5, 5e-4] (joint fine-tuning LR)",
    "tau": "loguniform [0.5, 2.0] (gate temperature)",
    "lambda_c": "loguniform [0.01, 1.0] (gate cross-entropy weight)",
    "lambda_b": "loguniform [1e-3, 1.0] (routing-balance weight)",
    "alpha": "uniform [0.3, 0.9] step 0.05 (L1 weight; SSIM weight = 1 - alpha)",
    "batch_size": "categorical {16, 32} (class-balanced batches)",
    "lr_warm": "fixed 3e-4 (gate warm-up LR)",
}


def objective(trial):
    cfg = dict(
        lr_ft=trial.suggest_float("lr_ft", 1e-5, 5e-4, log=True),
        tau=trial.suggest_float("tau", 0.5, 2.0, log=True),
        lambda_c=trial.suggest_float("lambda_c", 0.01, 1.0, log=True),
        lambda_b=trial.suggest_float("lambda_b", 1e-3, 1.0, log=True),
        alpha=trial.suggest_float("alpha", 0.3, 0.9, step=0.05),
        batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        lr_warm=3e-4,
    )
    res = task3_moe.run_moe_training(cfg, args.epochs, args.warmup, f"t3_trial_{trial.number}", trial=trial)
    return res["objective"]


trainer.setup_mlflow(task3_moe.EXPERIMENT)
study = optuna.create_study(
    study_name=args.study, storage=f"sqlite:///{config.OUT_DIR}/optuna_restoration.db",
    load_if_exists=True, direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=config.SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=args.warmup))

with mlflow.start_run(run_name="t3_optuna_study"):
    study.optimize(objective, n_trials=args.n_trials)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    collapsed = [t for t in pruned if t.user_attrs.get("pruned_reason") == "collapse"]
    best = {"lr_warm": 3e-4, **study.best_params}
    summary = {"search_space": SEARCH_SPACE,
               "objective": "val_l1 + (1 - val_ssim) at the best post-warm-up epoch",
               "trial_epochs": args.epochs, "trial_warmup_epochs": args.warmup,
               "completed": len(done), "pruned": len(pruned), "pruned_for_routing_collapse": len(collapsed),
               "best_trial": study.best_trial.number, "best_value": study.best_value, "best_params": best}
    mlflow.log_dict(summary, "study_summary.json")

json.dump(best, open(f"{config.CONFIG_DIR}/task3_best_params.json", "w"), indent=2)
json.dump(summary, open(f"{config.CONFIG_DIR}/task3_optuna_summary.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"{config.CONFIG_DIR}/task3_optuna_trials.csv", index=False)
print(f"completed={len(done)} pruned={len(pruned)} (collapse={len(collapsed)}) best={study.best_value:.4f}")
print("best params:", best)
