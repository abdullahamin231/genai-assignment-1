import argparse, json
import numpy as np
import mlflow, optuna
import config, trainer

EXPERIMENT = "restoration_task2"
SPECIALISTS = [("salt", 1), ("blur", 2), ("occlusion", 3)]

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=20)
ap.add_argument("--epochs", type=int, default=8, help="epochs per specialist per trial")
ap.add_argument("--study", default="task2_specialists_shared")
args = ap.parse_args()

SEARCH_SPACE = {"lr": "loguniform [3e-4, 3e-3]", "batch_size": "categorical {16, 32, 64}",
                "latent_ch": "categorical {16, 32, 64, 128} (latent = latent_ch x 8 x 8)",
                "base_ch": "categorical {32, 48, 64, 96}",
                "alpha": "uniform [0.3, 0.9] step 0.05 (L1 weight; SSIM weight = 1 - alpha)",
                "dropout": "fixed 0.0"}


def objective(trial):
    cfg = dict(lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
               batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
               latent_ch=trial.suggest_categorical("latent_ch", [16, 32, 64, 128]),
               base_ch=trial.suggest_categorical("base_ch", [32, 48, 64, 96]),
               alpha=trial.suggest_float("alpha", 0.3, 0.9, step=0.05), dropout=0.0)
    objs = []
    for k, (name, t) in enumerate(SPECIALISTS):
        res = trainer.run_training(cfg, args.epochs, f"t2_spec_trial{trial.number}_{name}", EXPERIMENT,
                                   types=(t,), save=False, sample_every=0, seed=config.SEED + k)
        objs.append(res["objective"])
        trial.report(float(np.mean(objs)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("per_specialist", dict(zip([n for n, _ in SPECIALISTS], objs)))
    return float(np.mean(objs))


trainer.setup_mlflow(EXPERIMENT)
study = optuna.create_study(study_name=args.study, storage=f"sqlite:///{config.OUT_DIR}/optuna_restoration.db",
                            load_if_exists=True, direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=config.SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
with mlflow.start_run(run_name="t2_spec_optuna_study"):
    study.optimize(objective, n_trials=args.n_trials)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    best = dict(study.best_params); best["dropout"] = 0.0
    summary = {"search_space": SEARCH_SPACE,
               "objective": "mean over the 3 specialists of (val_l1 + 1 - val_ssim)",
               "trial_epochs_per_specialist": args.epochs, "completed": len(done), "pruned": len(pruned),
               "best_value": study.best_value, "best_params": best}
    mlflow.log_dict(summary, "study_summary.json")
json.dump(best, open(f"{config.CONFIG_DIR}/task2_spec_best_params.json", "w"), indent=2)
json.dump(summary, open(f"{config.CONFIG_DIR}/task2_spec_optuna_summary.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"{config.CONFIG_DIR}/task2_spec_optuna_trials.csv", index=False)
print(f"completed={len(done)} pruned={len(pruned)} best={study.best_value:.4f}")
print("best params:", best)
