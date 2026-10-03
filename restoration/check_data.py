import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config, data
import corruptions as C

sp = data.get_splits()
print("split sizes (train/val/test):", len(sp["train_idx"]), len(sp["val_idx"]), len(data.get_clean("test")))
assert not set(sp["train_idx"]) & set(sp["val_idx"])

for s in ("val", "test"):
    man = data.get_manifest(s)
    print(s, "manifest:", len(man), dict(Counter(e["type"] for e in man)))

test = data.get_manifest("test")
clean = data.get_clean("test")
for e in test[:300]:
    img = clean[e["idx"]].astype(np.float32) / 255
    assert np.array_equal(C.apply_corruption(img, e), C.apply_corruption(img, e))
print("determinism OK")

cov, salt = defaultdict(list), defaultdict(list)
for e in test:
    if e["type"] == "occlusion":
        cov[e["level"]].append(e["coverage"])
    elif e["type"] == "salt" and e["idx"] < 200:
        img = clean[e["idx"]].astype(np.float32) / 255
        salt[e["level"]].append((C.apply_corruption(img, e) != img).any(-1).mean())
for lv in range(3):
    print(f"occlusion {C.LEVEL_NAMES[lv]}: coverage mean={np.mean(cov[lv]):.3f} "
          f"min={np.min(cov[lv]):.3f} max={np.max(cov[lv]):.3f} (target {C.OCC_LEVELS[lv][1]})")
    print(f"salt {C.LEVEL_NAMES[lv]}: pixels changed={np.mean(salt[lv]):.3f} (p={C.SALT_LEVELS[lv]})")

# figure 1: fixed test severities
lookup = {(e["idx"], e["type"], e["level"]): e for e in test}
idxs = [0, 50, 300]
cols = [("clean", -1)] + [(t, lv) for t in ("salt", "blur", "occlusion") for lv in range(3)]
fig, ax = plt.subplots(len(idxs), len(cols), figsize=(2 * len(cols), 2.1 * len(idxs)))
for r, i in enumerate(idxs):
    img = clean[i].astype(np.float32) / 255
    for c, (t, lv) in enumerate(cols):
        ax[r, c].imshow(np.clip(C.apply_corruption(img, lookup[(i, t, lv)]), 0, 1))
        ax[r, c].axis("off")
        if r == 0:
            ax[r, c].set_title(t if lv < 0 else f"{t} {C.LEVEL_NAMES[lv]}", fontsize=8)
plt.tight_layout()
p1 = Path(config.RESULTS_DIR) / "data_test_severities.png"
plt.savefig(p1, dpi=120); plt.close()

# figure 2: dynamic training corruption
ds = data.RestorationDataset("train")
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for j in range(8):
    x, y, lab, _ = ds[j]
    ax[0, j].imshow(x.permute(1, 2, 0).numpy()); ax[0, j].set_title(C.CLASS_NAMES[lab], fontsize=9)
    ax[1, j].imshow(y.permute(1, 2, 0).numpy())
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout()
p2 = Path(config.RESULTS_DIR) / "data_train_dynamic.png"
plt.savefig(p2, dpi=120); plt.close()

for p in (p1, p2):
    shutil.copy(p, Path(config.OUT_DIR) / p.name)
print("saved figures:", p1, p2)
