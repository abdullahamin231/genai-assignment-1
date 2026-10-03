import argparse, json
import trainer, task2_clf

DEFAULTS = dict(lr=1e-3, batch_size=32, channels="medium", dropout=0.2, weight_decay=1e-4)

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=None)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--run-name", default="task2_classifier")
args = ap.parse_args()

params = dict(DEFAULTS)
if args.params:
    params.update(json.load(open(args.params)))
print("Hyperparameters:", params)
trainer.setup_mlflow(task2_clf.EXPERIMENT)
print("Best:", task2_clf.run_clf_training(params, args.epochs, args.run_name, save=True))
