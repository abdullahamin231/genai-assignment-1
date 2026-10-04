import argparse, json
import config, trainer, task3_moe

# Starting values from the assignment (alpha=0.8 -> lambda_s=0.2, lambda_c=0.1, lambda_b=0.01)
DEFAULTS = dict(lr_ft=1e-4, tau=1.0, lambda_c=0.1, lambda_b=0.01, alpha=0.8, batch_size=32, lr_warm=3e-4)

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=None)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--run-name", default="task3_final")
ap.add_argument("--sample-every", type=int, default=5)
args = ap.parse_args()

params = dict(DEFAULTS)
if args.params:
    params.update(json.load(open(args.params)))
print("Hyperparameters:", params)

trainer.setup_mlflow(task3_moe.EXPERIMENT)
res = task3_moe.run_moe_training(params, args.epochs, args.warmup, args.run_name,
                                 save=True, sample_every=args.sample_every)
print("Best:", res)
