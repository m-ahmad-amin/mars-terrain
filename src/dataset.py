from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .config import IGNORE_INDEX, NUM_CLASSES

SPLITS = {
    "train": ("labels", "train"),
    "min1": ("labels", "test", "masked-gold-min1-100agree"),
    "min2": ("labels", "test", "masked-gold-min2-100agree"),
    "min3": ("labels", "test", "masked-gold-min3-100agree"),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _dir_index(folder: Path, product: str | None = None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for path in folder.iterdir():
        if not path.is_file():
            continue
        key = path.stem.replace(product, "EDR") if product else path.stem
        out[key] = path
    return out


def paired_samples(ncam: Path, split: str) -> list[tuple[str, Path, Path]]:
    edr = _dir_index(ncam / "images" / "edr")
    label_dir = ncam.joinpath(*SPLITS[split])
    pairs = []
    for path in sorted(label_dir.glob("*.png")):
        stem = path.stem.removesuffix("_merged")
        image = edr.get(stem)
        if image is not None:
            pairs.append((stem, image, path))
    return pairs


def is_empty_label(path: Path) -> bool:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return not np.any(arr != IGNORE_INDEX)


def load_index(ncam: Path, split: str, cache_dir: Path | None, skip_empty: bool = True) -> list[tuple[str, Path, Path]]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{split}_index.json"
        if cache_path.is_file():
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
            return [(r["stem"], Path(r["image"]), Path(r["label"])) for r in rows]

    pairs = paired_samples(ncam, split)
    if skip_empty:
        pairs = [p for p in pairs if not is_empty_label(p[2])]
    if cache_path is not None:
        payload = [{"stem": s, "image": str(i), "label": str(l)} for s, i, l in pairs]
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return pairs


class NavCamDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, Path, Path]],
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        stem, image_path, label_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        if label.mode != "L":
            label = label.convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        x = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(label, dtype=np.int64))
        return {"image": x, "label": y, "stem": stem}


def make_loader(
    pairs: list[tuple[str, Path, Path]],
    image_size: int,
    batch_size: int,
    augment: bool,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = NavCamDataset(pairs, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def class_weights(pairs: list[tuple[str, Path, Path]], max_files: int = 800) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    if len(pairs) > max_files:
        idxs = np.linspace(0, len(pairs) - 1, max_files, dtype=int)
        subset = [pairs[i] for i in idxs]
    else:
        subset = pairs
    for _, _, label_path in subset:
        arr = np.array(Image.open(label_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        for cls in range(NUM_CLASSES):
            counts[cls] += np.sum(arr == cls)
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.clip(freq, 1e-6, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
