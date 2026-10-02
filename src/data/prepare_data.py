"""Download Oxford-IIIT Pet, resize to 128x128 RGB, cache as .npy, make 80/20 split (seed 42).

Usage: python -m src.data.prepare_data
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

SIZE = 128
SEED = 12938129123


def load_split(root: str, split: str):
    ds = OxfordIIITPet(root=root, split=split, target_types="category", download=True)
    imgs = np.zeros((len(ds), SIZE, SIZE, 3), dtype=np.uint8)
    labels = np.zeros(len(ds), dtype=np.int64)
    for i in tqdm(range(len(ds)), desc=f"processing {split}"):
        img, y = ds[i]
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        imgs[i] = np.asarray(img)
        labels[i] = y
    return imgs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # official "trainval" = development data; official "test" stays untouched until the end
    tv_imgs, tv_labels = load_split(args.raw_dir, "trainval")
    te_imgs, te_labels = load_split(args.raw_dir, "test")

    np.save(out / "trainval_images.npy", tv_imgs)
    np.save(out / "trainval_labels.npy", tv_labels)
    np.save(out / "test_images.npy", te_imgs)
    np.save(out / "test_labels.npy", te_labels)

    # 80/20 split of trainval with seed 42 (shared by Tasks 1, 2, 3)
    n = len(tv_imgs)
    perm = np.random.RandomState(SEED).permutation(n)
    n_train = int(0.8 * n)
    split = {
        "seed": SEED,
        "n_trainval": n,
        "train_idx": sorted(perm[:n_train].tolist()),
        "val_idx": sorted(perm[n_train:].tolist()),
    }
    with open(out / "split.json", "w") as f:
        json.dump(split, f)

    print(f"trainval: {n} | train: {len(split['train_idx'])} | val: {len(split['val_idx'])} | test: {len(te_imgs)}")


if __name__ == "__main__":
    main()
