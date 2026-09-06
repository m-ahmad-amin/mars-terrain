from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_NAMES
from .metrics import confusion_matrix, scores_from_cm


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    cm = None
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)["out"]
        pred = logits.argmax(1)
        batch_cm = confusion_matrix(pred, labels)
        cm = batch_cm if cm is None else cm + batch_cm
    if cm is None:
        return {"mIoU": 0.0, "pixel_acc": 0.0, **{f"IoU_{n}": 0.0 for n in CLASS_NAMES}}
    return scores_from_cm(cm)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def format_metrics(metrics: dict[str, float]) -> str:
    parts = [f"mIoU {metrics['mIoU']:.3f}", f"acc {metrics['pixel_acc']:.3f}"]
    for name in CLASS_NAMES:
        parts.append(f"{name} {metrics[f'IoU_{name}']:.3f}")
    return "  ".join(parts)
