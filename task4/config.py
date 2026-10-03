import os

DATA_DIR = "/content/data/FS2K"
OUT_DIR = "/content/drive/MyDrive/genai-assignment-1/task4_outputs"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

IMG_SIZE = 128
SEED = 42
NUM_STYLES = 3
VAL_FRAC = 0.15

DEFAULT_PARAMS = dict(lr_g=2e-4, lr_d=2e-4, batch_size=16, base_ch=64,
                      dropout=0.3, emb_dim=16, lambda_l1=100.0)

for d in (OUT_DIR, CONFIG_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
