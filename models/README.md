# ONNX model files

The seven exported ONNX graphs are **not** committed to git (the assignment forbids
uploading very large model files directly to GitHub). Drop them into this folder
before starting the containers — the backend reads them from `./models` at startup
and reports their status on `GET /api/health`.

## Expected files

| # | File | Task | Role | Input → Output |
|---|------|------|------|----------------|
| 1 | `task1_universal_ae.onnx` | 1 | Universal restoration autoencoder | `image [1,3,128,128]` → `restored [1,3,128,128]` |
| 2 | `task2_classifier.onnx` | 2 | Corruption classifier (4-way gate) | `image [1,3,128,128]` → `probs [1,4]` |
| 3 | `task2_specialist_salt.onnx` | 2 | Salt & pepper specialist | `image [1,3,128,128]` → `restored [1,3,128,128]` |
| 4 | `task2_specialist_blur.onnx` | 2 | Gaussian blur specialist | `image [1,3,128,128]` → `restored [1,3,128,128]` |
| 5 | `task2_specialist_occlusion.onnx` | 2 | Occlusion specialist | `image [1,3,128,128]` → `restored [1,3,128,128]` |
| 6 | `task3_soft_moe.onnx` | 3 | Soft mixture-of-experts joint graph | `image [1,3,128,128]` → image + gate weights |
| 7 | `task4_generator.onnx` | 4 | Style-conditioned U-Net generator | `image [1,3,128,128]` (+ style) → `sketch [1,3,128,128]` |

Task 3's exact export signature was still being finalised when this README was
written, so the backend resolves its image and weight tensors **by shape as well as
by name** (rank-4 tensor = image, rank-2 × 4 = gate weights/logits). Whatever the
final export names are, the pipeline will pick them up; run
`python scripts/verify_models.py` to see the resolved signature.

## Getting the files

**Option A — copy from a downloaded archive**

```bash
python scripts/download_models.py --from ~/Downloads/genai_models
```

**Option B — download from an HTTP base URL** (GitHub release, Drive direct link, ...)

```bash
python scripts/download_models.py --url https://example.com/models
```

**Option C — place them manually** in `models/` (or in `restoration_outputs/models/`
or `task4_outputs/models/`, which the backend also searches).

## Verify

```bash
python scripts/verify_models.py
```

Expected output: `7/7 model files present`, with an `OK` row per file showing the
resolved tensor names and shapes.

## Local smoke test without the real weights

```bash
python scripts/make_dummy_models.py   # writes structurally-valid random-weight ONNX files
```

This is only for wiring/UI testing — the outputs are meaningless.
`models/*.onnx` is git-ignored, so these placeholders are never committed.
