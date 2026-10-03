import argparse, json
import config, trainer

DEFAULTS = dict(lr=1e-3, batch_size=32, base_ch=64, latent_ch=32, dropout=0.1, alpha=0.8)
EXPERIMENT = "restoration_task1"

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=None)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--run-name", default="task1_final")
ap.add_argument("--sample-every", type=int, default=10)
args = ap.parse_args()

params = dict(DEFAULTS)
if args.params:
    params.update(json.load(open(args.params)))
print("Hyperparameters:", params)

trainer.setup_mlflow(EXPERIMENT)
res = trainer.run_training(params, args.epochs, args.run_name, EXPERIMENT, save=True, sample_every=args.sample_every)
print("Best:", res)
