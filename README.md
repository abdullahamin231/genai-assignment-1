# Generative Image Restoration Suite — GenAI Assignment #1

A single browser application that runs all four assignment tasks against exported
ONNX models: a universal restoration autoencoder, a hard-routed specialist system
with a learned corruption classifier, a jointly trained soft mixture-of-experts,
and a style-conditioned face-to-sketch generator.

React + Tailwind frontend (built from the Google Stitch design), FastAPI backend,
ONNX Runtime inference, Docker Compose deployment.

| | |
|---|---|
| **Frontend** | React 18, Vite 5, Tailwind CSS 3, React Router |
| **Backend** | FastAPI, ONNX Runtime, OpenCV, Pillow, NumPy |
| **Models** | 7 exported ONNX graphs (4 tasks) |
| **Deployment** | Docker Compose — one command |
| **Design source** | `stitch_generative_image_restoration_suite/` (Google Stitch) |

---

## 1. Quick start (Docker Compose — the one documented command)

**Prerequisites:** Docker with the Compose plugin. No Python, Node, or VS Code needed.

### Step 1 — get the model files

The seven `.onnx` files are **not** stored in git. Fetch them with the helper
script (see `models/README.md` for the full table of expected files):

```bash
# from a folder you already downloaded (unzipped Drive/GitHub export)
python scripts/download_models.py --from ~/Downloads/genai_models

# ...or directly from an HTTP base URL
python scripts/download_models.py --url https://YOUR-HOST/models
```

or simply copy the seven files by hand into `models/`:

```
models/
├── task1_universal_ae.onnx
├── task2_classifier.onnx
├── task2_specialist_salt.onnx
├── task2_specialist_blur.onnx
├── task2_specialist_occlusion.onnx
├── task3_soft_moe.onnx
└── task4_generator.onnx
```

Check them before starting:

```bash
python scripts/verify_models.py     # -> "7/7 model files present"
```

> **No weights yet?** `python scripts/make_dummy_models.py` writes
> structurally-valid random-weight models so you can smoke-test the UI. They are
> git-ignored and produce meaningless output — delete them before the real run.

### Step 2 — start everything

```bash
docker compose up --build
```

### Step 3 — open the app

| What | URL |
|---|---|
| **Application** | <http://localhost:8080> |
| API health / model status | <http://localhost:8000/api/health> |
| Interactive API docs (Swagger) | <http://localhost:8000/docs> |

Stop with `Ctrl-C` (or `docker compose down` for a detached stack).

**Configuration** (all optional, set in `.env` or on the command line):

| Variable | Default | Meaning |
|---|---|---|
| `APP_PORT` | `8080` | host port for the web UI |
| `API_PORT` | `8000` | host port for the FastAPI service |
| `ORT_PROVIDERS` | `CPUExecutionProvider` | comma-separated ONNX Runtime providers |
| `MAX_UPLOAD_MB` | `12` | maximum upload size |

---

## 2. What each workspace does

All four tasks live behind one router in a single application — the tabs in the
header switch workspace without a page reload.

### Task 1 — Universal Restoration (`/`)
One convolutional autoencoder with a bottleneck latent is trained on **all three**
corruptions jointly (salt & pepper, Gaussian blur, rectangular occlusion) and is
never told which one was applied. Upload an image, apply a corruption at runtime,
and the model restores it. Optionally upload a clean reference image to score the
output against ground truth instead of against the network input.

### Task 2 — Hard Routing (`/hard-routing`)
A 4-way corruption classifier (clean / salt / blur / occlusion) reads the input and
commits to **exactly one** specialist — or to the identity branch when the input is
classified as clean. The panel shows the softmax probabilities, the predicted class,
the selected expert and the confidence, with a switchable **oracle mode** that
routes from a ground-truth label instead. The recorded results table below reports
accuracy / macro-F1 for the classifier plus an oracle-vs-predicted SSIM comparison
per condition.

### Task 3 — Soft Mixture-of-Experts (`/soft-moe`)
A gating network emits a **continuous** weight for every branch, and the output is
the weighted blend `Σ wᵢ · expertᵢ(x)`. Compound or ambiguous corruptions are
blended rather than hard-routed. The workspace visualises the four weights as
live bars and exposes the softmax temperature τ.

### Task 4 — Face to Sketch (`/face-to-sketch`)
A U-Net generator conditioned on a learned categorical style embedding
`s ∈ {1,2,3}` maps a facial photograph to one of the three FS2K sketch styles.
Upload or capture a photo with the webcam, pick a style, and download the result.
Only the generator runs at inference — the PatchGAN discriminator is training-only.

---

## 3. Backend API

Every endpoint is prefixed with `/api`. The frontend talks to the same paths
through an nginx reverse proxy (Docker) or the Vite dev proxy (local dev).

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | service status + per-model presence/loaded/tensor shapes |
| `GET` | `/api/metrics/{task1\|task2\|task3\|task4}` | recorded test-set metrics (`404` until the evaluation script has run) |
| `GET` | `/api/optuna` | Optuna best-params + study summaries per task |
| `GET` | `/api/samples` | bundled sample images (`backend/samples/`) |
| `POST` | `/api/corrupt` | apply a runtime corruption only (no inference) |
| `POST` | `/api/universal-restoration` | Task 1 |
| `POST` | `/api/hard-routing` | Task 2 — classifier + selected specialist |
| `POST` | `/api/soft-mixture` | Task 3 — gate weights + blended output |
| `POST` | `/api/face-to-sketch` | Task 4 — style-conditioned generation |

Common form fields: `file` (required, PNG/JPG/WebP/…, ≤ `MAX_UPLOAD_MB`),
`corruption` ∈ `clean | salt | blur | occlusion`, `level` ∈ `0 | 1 | 2`
(the three fixed test severities) **or** a continuous `severity` in `[0,1]`
interpolated inside the training ranges, plus `reference` (optional clean image
for reference-based PSNR/SSIM) and `style` (Task 4, `0|1|2`).

Every task response carries `input` / `corrupted` / `output` (and `error_map`) as
data URLs, a `metrics` block, a `timing` block (`inference_ms`, `total_ms`, plus
`classifier_ms`/`specialist_ms` for Task 2), and the `model` file + path used.
`/api/corrupt` is the exception: it returns only `input` / `corrupted` / `metrics`
plus the applied `corruption` description, since no model runs.

Out-of-range inputs are clamped rather than rejected: `level` outside 0–2 becomes
the nearest of {0, 1, 2} and `severity` outside [0,1] clips to the endpoint.

Example:

```bash
curl -X POST http://localhost:8000/api/universal-restoration \
  -F file=@photo.png -F corruption=salt -F level=1
```

### Runtime corruptions

Corruptions are applied **server-side at request time** using the exact definitions
the models were trained with (`restoration/corruptions.py`) — nothing is baked into
the uploaded file, so the evaluator can apply unseen damage to unseen images.

| Type | `level=0` | `level=1` | `level=2` |
|---|---|---|---|
| Salt & pepper `salt` | `p = 0.03` | `p = 0.08` | `p = 0.15` |
| Gaussian blur `blur` | `(k=3, σ=0.7)` | `(k=5, σ=1.5)` | `(k=7, σ=2.5)` |
| Occlusion `occlusion` | 10 % masked | 20 % masked | 35 % masked |

Passing a non-integer `severity` in `[0,1]` instead of `level` interpolates
continuously across those ranges; `clean` passes the image through untouched.

---

## 4. Repository layout

```
.
├── docker-compose.yml          # one-command deployment
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── samples/                # optional preset images for the sample picker
│   └── app/
│       ├── main.py             # FastAPI app, CORS, exception handlers
│       ├── config.py           # path + model discovery, corruption sys.path bridge
│       ├── imaging.py          # decode / resize / encode helpers
│       ├── corrupt.py          # severity model over restoration/corruptions.py
│       ├── metrics.py          # PSNR, SSIM, error map
│       ├── onnx_runtime.py     # model registry; shape-based tensor resolution
│       ├── pipeline.py         # shared inference + response assembly
│       └── api/
│           ├── health.py       # health / metrics / optuna / samples
│           └── tasks.py        # the four task endpoints + /api/corrupt
├── frontend/
│   ├── Dockerfile              # node build -> nginx with /api proxy
│   ├── nginx.conf
│   ├── tailwind.config.js      # Stitch design tokens
│   └── src/
│       ├── pages/              # Universal, HardRouting, SoftMoE, FaceToSketch
│       ├── components/         # Shell, panels, controls, dropzone, webcam, ...
│       ├── hooks/              # useImage, useWorkspace
│       └── lib/                # api client, formatters
├── models/                     # ONNX files go here (git-ignored)
├── scripts/
│   ├── download_models.py      # fetch the 7 ONNX files (--from / --url)
│   ├── verify_models.py        # inspect signatures, 7/7 check
│   ├── make_dummy_models.py    # random-weight stand-ins for smoke tests
│   └── prepare_samples.py      # seed backend/samples/ from the test split
├── restoration/                # Tasks 1-3: data, corruptions, models, training
│   ├── corruptions.py          # shared by training AND the API
│   ├── configs/                # Optuna best params + summaries (served by /api/optuna)
│   └── results/                # test metrics (served by /api/metrics)
└── task4/                      # Task 4: FS2K cGAN
    ├── configs/  results/      # same roles as above
    └── export_onnx.py
```

---

## 5. Running without Docker (development)

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

The backend resolves model paths at startup by searching, in order: `$MODEL_DIR`,
`models/`, `restoration_outputs/models/`, `task4_outputs/models/`,
`backend/models/`. Only the **file names** matter.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, /api proxied to :8000
```

`VITE_API_TARGET` overrides the proxy target; `PORT` changes the dev port.
`npm run build` writes `frontend/dist/`, which the Docker image serves via nginx.

---

## 6. Training, evaluation and export (Tasks 1–4)

All scripts run from the repository root. Optuna is used for hyperparameter
optimization in every task and MLflow records experiments.

```bash
# --- Task 1: universal autoencoder -----------------------------------------
python restoration/task1_optuna.py    --n-trials 30
python restoration/task1_train.py     --params restoration/configs/task1_best_params.json
python restoration/task1_evaluate.py                 # -> restoration/results/
python restoration/task1_export_onnx.py              # -> .../models/task1_universal_ae.onnx

# --- Task 2: classifier + specialists --------------------------------------
python restoration/task2_clf_optuna.py --n-trials 30
python restoration/task2_clf_train.py  --params restoration/configs/task2_clf_best_params.json
python restoration/task2_spec_optuna.py --n-trials 30
python restoration/task2_spec_train.py
python restoration/task2_evaluate.py
python restoration/task2_export_onnx.py              # -> classifier + 3 specialists

# --- Task 3: soft mixture-of-experts ---------------------------------------
# NOTE: these land with the Task 3 push; names follow the Task 1/2 convention.
python restoration/task3_optuna.py    --n-trials 30
python restoration/task3_train.py
python restoration/task3_evaluate.py                 # publishes restoration/results/task3_*.json
python restoration/task3_export_onnx.py              # -> .../models/task3_soft_moe.onnx

# --- Task 4: face-to-sketch cGAN -------------------------------------------
python task4/optuna_search.py  --n-trials 30
python task4/train.py          --params task4/configs/best_params.json
python task4/evaluate.py
python task4/export_onnx.py                           # -> .../models/task4_generator.onnx
```

Recorded metrics and Optuna summaries are picked up automatically by
`GET /api/metrics` and `GET /api/optuna` — the workspaces display them in the
"Recorded results" panels with no extra wiring.

### Notes

- **Task 3** exports a single joint graph containing the gate and all four
  branches. Its final tensor names were still being finalised, so the backend
  resolves them by **shape as well as name** (rank-4 → image, rank-2 × 4 →
  weights/logits). Run `python scripts/verify_models.py` to see what was resolved.
  Until `task3_evaluate.py` has run, `/api/metrics/task3` returns `404` and the
  workspace shows a "No recorded test results yet" card — inference still works.
- **ONNX ↔ PyTorch parity** is checked by comparing the export script's
  max-abs-difference report against the reference forward pass.
- **Data:** Tasks 1–3 use the Oxford-IIIT Pet train split (80/20, seed 42) and the
  official test split for final evaluation; Task 4 uses FS2K. Datasets and
  checkpoints are **not** committed (`data/`, `*.pt`, `mlruns/` are ignored).

---

## 7. Design

The interface was designed in **Google Stitch**; the original design export is
kept in `stitch_generative_image_restoration_suite/` (screenshots plus the
generated `code.html`), and `stitch.png` shows the reference layout.

The implementation follows the Stitch tokens defined for the workbench:

- dark surface palette (`surface-container-*` ramp) with `#00e5ff` primary-container accent
- type: **Space Grotesk** (headlines), **Geist** (body), **JetBrains Mono** (metrics/code)
- Material Symbols iconography, 12-column responsive workspace grid
- per-task accent: Task 1 primary, Task 2 secondary, Task 3 primary-container, Task 4 tertiary

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `7/7 models` shows red in the header | the `.onnx` files are missing — see §1 Step 1, then `docker compose restart backend` |
| `404` on `/api/metrics/task3` | expected until `restoration/task3_evaluate.py` has run |
| Port already in use | `APP_PORT=8090 API_PORT=8010 docker compose up --build` |
| Frontend shows "API: not ready" | `curl http://localhost:8000/api/health` and check the `models[].error` field |
| Stale build after editing the frontend | `docker compose up --build --force-recreate` |
| Icons show as raw text (e.g. `download`) | fonts come from the Google Fonts CDN — the machine needs internet access on first load; icons render correctly once cached |
