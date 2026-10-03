import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.environ.get("PET_DIR", "/content/data/oxford_pets")
OUT_DIR = os.environ.get("RESTORATION_OUT", "/content/drive/MyDrive/genai-assignment-1/restoration_outputs")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
MANIFEST_DIR = os.path.join(ROOT, "manifests")
RESULTS_DIR = os.path.join(ROOT, "results")

IMG_SIZE = 128
SEED = 42
VAL_FRAC = 0.2

for d in (RAW_DIR, OUT_DIR, CACHE_DIR, MANIFEST_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
