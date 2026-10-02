"""Training dataset (runtime corruption) and manifest dataset (deterministic val/test)."""
import json

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .corruptions import CORRUPTION_TYPES, TYPE_TO_ID, apply_corruption, sample_spec


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)


class TrainDataset(Dataset):
    """Each __getitem__ picks one of 4 conditions with equal probability and a fresh random severity.

    Geometric augmentation (horizontal/vertical flip + 90-degree rotations, i.e. the
    dihedral group) is applied to the clean image *before* the corruption, so input and
    target stay perfectly aligned and the corruption masks/blur are drawn in the
    augmented frame. Only training data is augmented; the validation/test manifests are
    deterministic and untouched.
    """

    def __init__(self, images_path, indices, augment=True):
        self.images = np.load(images_path)
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _augment(img, rng):
        if rng.random() < 0.5:
            img = TF.hflip(img)
        if rng.random() < 0.5:
            img = TF.vflip(img)
        k = int(rng.integers(0, 4))
        if k:
            img = torch.rot90(img, k, [1, 2])
        return img.contiguous()

    def __getitem__(self, i):
        clean = to_tensor(self.images[self.indices[i]])
        rng = np.random.default_rng()  # fresh OS entropy -> new corruption on every load
        if self.augment:
            clean = self._augment(clean, rng)
        ctype = CORRUPTION_TYPES[int(rng.integers(0, 4))]
        spec = sample_spec(rng, ctype)
        return {
            "input": apply_corruption(clean, spec),
            "target": clean,
            "label": TYPE_TO_ID[ctype],
            "severity": spec["severity"],
        }


class ManifestDataset(Dataset):
    """Deterministic corruptions loaded from a manifest JSON (validation / test)."""

    def __init__(self, images_path, manifest_path):
        self.images = np.load(images_path)
        with open(manifest_path) as f:
            self.entries = json.load(f)["entries"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        spec = self.entries[i]
        clean = to_tensor(self.images[spec["image_idx"]])
        return {
            "input": apply_corruption(clean, spec),
            "target": clean,
            "label": TYPE_TO_ID[spec["type"]],
            "severity": spec["severity"],
            "index": i,
        }
