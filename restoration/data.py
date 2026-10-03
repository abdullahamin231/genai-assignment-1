"""Shared data pipeline for Tasks 1-3: Oxford-IIIT Pet, 80/20 split (seed 42), manifests, datasets."""
import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
import corruptions as C

SPLITS_PATH = Path(config.MANIFEST_DIR) / "splits.json"


# ---------- clean images (cached as uint8 arrays on Drive) ----------
def _official(split):
    """split: 'trainval' or 'test' -> uint8 array N x 128 x 128 x 3 (RGB)."""
    path = Path(config.CACHE_DIR) / f"clean_{split}.npy"
    if path.exists():
        return np.load(path)
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root=config.RAW_DIR, split=split, target_types="category", download=True)
    S = config.IMG_SIZE
    arr = np.stack([np.asarray(ds[i][0].convert("RGB").resize((S, S), Image.BICUBIC))
                    for i in tqdm(range(len(ds)), desc=f"load {split}")])
    np.save(path, arr)
    return arr


def get_splits():
    if SPLITS_PATH.exists():
        return json.load(open(SPLITS_PATH))
    n = len(_official("trainval"))
    perm = np.random.RandomState(config.SEED).permutation(n)
    n_val = int(round(config.VAL_FRAC * n))
    sp = {"seed": config.SEED, "n_trainval": n,
          "val_idx": sorted(perm[:n_val].tolist()), "train_idx": sorted(perm[n_val:].tolist())}
    json.dump(sp, open(SPLITS_PATH, "w"))
    return sp


@lru_cache(maxsize=None)
def get_clean(split):
    """split in {'train','val','test'} -> uint8 N x H x W x 3."""
    if split == "test":
        return _official("test")
    sp = get_splits()
    return _official("trainval")[sp["train_idx" if split == "train" else "val_idx"]]


# ---------- deterministic manifests (val + test) ----------
def build_manifests(force=False):
    val_p = Path(config.MANIFEST_DIR) / "val_manifest.json"
    test_p = Path(config.MANIFEST_DIR) / "test_manifest.json"
    if val_p.exists() and test_p.exists() and not force:
        return

    # Validation: exactly balanced over the 4 conditions, severities sampled from training ranges
    n = len(get_clean("val"))
    rng = np.random.default_rng(config.SEED)
    types = np.tile([0, 1, 2, 3], n // 4 + 1)[:n]
    rng.shuffle(types)
    val = []
    for i, t in enumerate(types):
        spec = C.sample_spec(rng, int(t))
        spec.update(idx=i, level=-1)
        val.append(spec)

    # Test: clean + 3 corruptions x 3 fixed severities per image
    n = len(get_clean("test"))
    test = []
    for i in range(n):
        test.append({"type": "clean", "idx": i, "level": -1})
        for t in (1, 2, 3):
            for lv in range(3):
                seed = config.SEED * 1_000_000 + i * 16 + t * 3 + lv
                spec = C.fixed_spec(t, lv, seed)
                spec.update(idx=i, level=lv, seed=seed)
                test.append(spec)

    for path, man in ((val_p, val), (test_p, test)):
        json.dump(man, open(path, "w"), separators=(",", ":"))
    mdir = Path(config.OUT_DIR) / "manifests"
    mdir.mkdir(exist_ok=True)
    for f in (SPLITS_PATH, val_p, test_p):
        shutil.copy(f, mdir / f.name)


@lru_cache(maxsize=None)
def get_manifest(split):
    return json.load(open(Path(config.MANIFEST_DIR) / f"{split}_manifest.json"))


# ---------- torch datasets ----------
def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class RestorationDataset(Dataset):
    """train: new random condition+severity on every load (equal prob over `types`).
    val/test: deterministic, driven by the manifest.
    types: subset of class indices {0 clean, 1 salt, 2 blur, 3 occlusion} (e.g. (1,) for a salt specialist).
    Returns (corrupted, clean, class_label, severity_level) with images float32 CHW in [0,1]."""

    def __init__(self, split, types=(0, 1, 2, 3)):
        self.split = split
        self.train = split == "train"
        self.types = list(types)
        self.clean = get_clean(split)
        if self.train:
            self.entries = None
        else:
            man = get_manifest(split)
            self.entries = [e for e in man if C.CLASS_IDX[e["type"]] in self.types]

    def __len__(self):
        return len(self.clean) if self.train else len(self.entries)

    def __getitem__(self, i):
        if self.train:
            img = self.clean[i].astype(np.float32) / 255.0
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            spec = C.sample_spec(rng, int(rng.choice(self.types)))
            level = -1
        else:
            e = self.entries[i]
            img = self.clean[e["idx"]].astype(np.float32) / 255.0
            spec, level = e, e["level"]
        return _t(C.apply_corruption(img, spec)), _t(img), C.CLASS_IDX[spec["type"]], level


def get_loaders(batch_size, num_workers=2, types=(0, 1, 2, 3)):
    tr = DataLoader(RestorationDataset("train", types), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    va = DataLoader(RestorationDataset("val", types), batch_size=64, shuffle=False)
    return tr, va


def get_test_loader(batch_size=64, types=(0, 1, 2, 3)):
    return DataLoader(RestorationDataset("test", types), batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    get_splits()
    build_manifests(force=True)
    print("train/val/test:", len(get_clean("train")), len(get_clean("val")), len(get_clean("test")))
    print("manifests written to", config.MANIFEST_DIR, "and", Path(config.OUT_DIR) / "manifests")
