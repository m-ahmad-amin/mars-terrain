from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .config import IGNORE_INDEX
from .dataset import IMAGENET_MEAN, IMAGENET_STD

NAV_RGB = {
    0: (210, 160, 90),
    1: (150, 150, 155),
    2: (230, 200, 70),
    3: (220, 45, 45),
}


def denormalize(image: torch.Tensor) -> np.ndarray:
    x = image.detach().cpu().clone()
    for t, m, s in zip(x, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def colorize(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls, color in NAV_RGB.items():
        rgb[label == cls] = color
    return rgb


def overlay(image: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = colorize(label)
    out = image.copy()
    mask = label != IGNORE_INDEX
    out[mask] = (image[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    return out


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
