import json, random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS_PATH = Path(config.CONFIG_DIR) / "splits.json"


def _key(p: Path):
    return f"{p.parent.name}/{p.stem}".lower()


def _build_index(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(_key(p), p)
    return idx


def _load_anno(root, name):
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    with open(hits[0]) as f:
        return json.load(f)


def _pair_paths(rel, idx):
    parts = Path(rel).with_suffix("").parts[-2:]
    k = "/".join(parts).lower()
    photo = idx.get(k)
    if photo is None:
        raise FileNotFoundError(f"Photo not found for '{rel}' (key '{k}')")
    c1 = k.replace("photo", "sketch")
    c2 = c1.replace("image", "sketch")
    for c in (c1, c2):
        if c in idx:
            return photo, idx[c]
    raise FileNotFoundError(f"Sketch not found for '{rel}'; tried '{c1}' and '{c2}'")


def build_splits(force=False):
    if SPLITS_PATH.exists() and not force:
        return json.load(open(SPLITS_PATH))
    root = Path(config.DATA_DIR)
    idx = _build_index(root)

    def make(anno):
        items = []
        for a in anno:
            ph, sk = _pair_paths(a["image_name"], idx)
            items.append(dict(photo=str(ph.relative_to(root)),
                              sketch=str(sk.relative_to(root)),
                              style=int(a["style"])))
        return items

    train_all = make(_load_anno(root, "anno_train.json"))
    test = make(_load_anno(root, "anno_test.json"))

    styles = sorted({it["style"] for it in train_all + test})
    assert len(styles) == config.NUM_STYLES, f"Expected 3 styles, got {styles}"
    remap = {s: i for i, s in enumerate(styles)}
    for it in train_all + test:
        it["style"] = remap[it["style"]]

    tr, va = train_test_split(train_all, test_size=config.VAL_FRAC,
                              random_state=config.SEED,
                              stratify=[it["style"] for it in train_all])
    splits = {"train": tr, "val": va, "test": test,
              "style_map": {str(k): v for k, v in remap.items()}}
    json.dump(splits, open(SPLITS_PATH, "w"), indent=1)
    return splits


def _load(rel, mode):
    S = config.IMG_SIZE
    im = Image.open(Path(config.DATA_DIR) / rel).convert(mode).resize((S, S), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
    return t.permute(2, 0, 1) if mode == "RGB" else t.unsqueeze(0)


def _paired_aug(p, s):
    """Identical spatial transforms for photo and sketch."""
    if random.random() < 0.5:
        p, s = p.flip(-1), s.flip(-1)
    if random.random() < 0.7:
        S = p.shape[-1]
        h = int(S * random.uniform(0.85, 1.0))
        top, left = random.randint(0, S - h), random.randint(0, S - h)
        p = p[:, top:top + h, left:left + h]
        s = s[:, top:top + h, left:left + h]
        p = F.interpolate(p[None], size=(S, S), mode="bilinear", align_corners=False)[0]
        s = F.interpolate(s[None], size=(S, S), mode="bilinear", align_corners=False)[0]
    return p, s


class FS2KPairs(Dataset):
    def __init__(self, items, train=False):
        self.train = train
        self.photos = [_load(it["photo"], "RGB") for it in items]
        self.sketches = [_load(it["sketch"], "L") for it in items]
        self.styles = [it["style"] for it in items]

    def __len__(self):
        return len(self.styles)

    def __getitem__(self, i):
        p = self.photos[i].float() / 127.5 - 1
        s = self.sketches[i].float() / 127.5 - 1
        if self.train:
            p, s = _paired_aug(p, s)
        return p, s, self.styles[i]


@lru_cache(maxsize=None)
def get_dataset(split):
    splits = build_splits()
    return FS2KPairs(splits[split], train=(split == "train"))


def get_loaders(batch_size, num_workers=2):
    tr = DataLoader(get_dataset("train"), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    va = DataLoader(get_dataset("val"), batch_size=32, shuffle=False)
    return tr, va


if __name__ == "__main__":
    sp = build_splits(force=True)
    for k in ("train", "val", "test"):
        counts = np.bincount([it["style"] for it in sp[k]], minlength=3)
        print(k, len(sp[k]), "style counts:", counts.tolist())
    print("style map:", sp["style_map"])
