import argparse, json
import config, trainer

EXPERIMENT = "restoration_task2"
SPECIALISTS = [("salt", 1), ("blur", 2), ("occlusion", 3)]

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=f"{config.CONFIG_DIR}/task2_spec_best_params.json")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--only", choices=[n for n, _ in SPECIALISTS], default=None)
args = ap.parse_args()

params = json.load(open(args.params))
print("Shared architecture / hyperparameters:", params)
trainer.setup_mlflow(EXPERIMENT)
for k, (name, t) in enumerate(SPECIALISTS):
    if args.only and name != args.only:
        continue
    print(f"\n=== specialist: {name} ===")
    res = trainer.run_training(params, args.epochs, f"task2_specialist_{name}", EXPERIMENT, types=(t,),
                               save=True, sample_every=10, seed=config.SEED + 10 + k)
    print(f"{name} best:", res)
