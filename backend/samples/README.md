# Sample images (optional)

If this folder contains images (`*.png`, `*.jpg`, `*.jpeg`, `*.webp`), they are
listed by `GET /api/samples` and appear in the frontend's **Sample picker**, so the
evaluator can click through presets instead of uploading a file.

The folder is mounted read-only into the container (`/app/backend/samples`), so
files can be added while the stack is running — no rebuild needed.

To populate it from the Oxford-IIIT Pet test split:

```bash
python scripts/prepare_samples.py --n 8
```

When the folder is empty the picker hides itself and the workspaces fall back to
upload / webcam only. Nothing here is required for the application to run.
