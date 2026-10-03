import argparse, json
import config, trainer

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=None, help="json with hyperparameters (e.g. configs/best_params.json)")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--run-name", default="final")
ap.add_argument("--sample-every", type=int, default=10)
args = ap.parse_args()

params = dict(config.DEFAULT_PARAMS)
if args.params:
    params.update(json.load(open(args.params)))
print("Hyperparameters:", params)

trainer.setup_mlflow()
res = trainer.run_training(params, args.epochs, args.run_name, save=True, sample_every=args.sample_every)
print("Best:", res)
