from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import IGNORE_INDEX
from .dataset import load_index, make_loader
from .model import build_model
from .paths import find_ncam, output_dir, project_root
from .visualize import denormalize, overlay


def _has_class(path: Path, cls: int) -> bool:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return bool(np.any(arr == cls))


def pick_pairs(pairs: list, n: int = 8) -> list:
    buckets = {3: [], 2: [], 1: [], 0: []}
    for item in pairs:
        label = item[2]
        for cls in (3, 2, 1, 0):
            if _has_class(label, cls):
                buckets[cls].append(item)
                break
    chosen = []
    for cls, quota in ((3, 2), (2, 2), (1, 2), (0, 2)):
        for item in buckets[cls][:quota]:
            if item not in chosen:
                chosen.append(item)
        if len(chosen) >= n:
            break
    for item in pairs:
        if len(chosen) >= n:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:n]


def stack_row(image: np.ndarray, expert: np.ndarray, pred: np.ndarray) -> Image.Image:
    left = overlay(image, expert)
    right = overlay(image, pred)
    h, w = image.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = image
    canvas[:, w : 2 * w] = left
    canvas[:, 2 * w :] = right
    return Image.fromarray(canvas)


@torch.no_grad()
def save_previews(
    ncam: Path | None = None,
    project: Path | None = None,
    n: int = 8,
) -> list[Path]:
    project = project or project_root()
    ncam = ncam or find_ncam(project)
    out = output_dir(project)
    dest = out / "preds"
    dest.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(out / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cache = Path("/content/cache") if Path("/content").is_dir() else out / "cache"
    pairs = pick_pairs(load_index(ncam, "min1", cache, skip_empty=True), n)
    loader = make_loader(pairs, 512, 1, False, 0, False)

    saved = []
    for batch in loader:
        image = batch["image"].to(device)
        pred = model(image)["out"].argmax(1)[0].cpu().numpy()
        rgb = denormalize(batch["image"][0])
        expert = batch["label"][0].numpy()
        stem = batch["stem"][0]
        path = dest / f"{stem}.jpg"
        stack_row(rgb, expert, pred).save(path, quality=85)
        labeled = expert != IGNORE_INDEX
        print(
            stem,
            "expert",
            sorted(int(c) for c in np.unique(expert[labeled])) if labeled.any() else [],
            "pred",
            sorted(int(c) for c in np.unique(pred)),
        )
        saved.append(path)
    return saved
