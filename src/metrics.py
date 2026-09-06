from __future__ import annotations

import torch

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    pred = pred[valid].view(-1)
    target = target[valid].view(-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=target.device)
    if pred.numel() == 0:
        return cm
    idx = target * num_classes + pred
    cm += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return cm


def scores_from_cm(cm: torch.Tensor) -> dict[str, float]:
    cm = cm.cpu().float()
    tp = cm.diag()
    denom = cm.sum(1) + cm.sum(0) - tp
    iou = torch.where(denom > 0, tp / denom.clamp(min=1e-8), torch.zeros_like(tp))
    present = denom > 0
    miou = iou[present].mean().item() if present.any() else 0.0
    pixel_acc = (tp.sum() / cm.sum().clamp(min=1e-8)).item()
    out = {"mIoU": miou, "pixel_acc": pixel_acc}
    for i, name in enumerate(CLASS_NAMES):
        out[f"IoU_{name}"] = iou[i].item()
    return out
