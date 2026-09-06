from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model
from .paths import output_dir, project_root


def load_segmenter(project: Path | None = None, device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = output_dir(project or project_root()) / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing {ckpt_path}")
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


@torch.no_grad()
def predict_label(model: torch.nn.Module, image: Image.Image, device: torch.device, size: int = 512) -> np.ndarray:
    rgb = image.convert("RGB")
    w, h = rgb.size
    x = TF.normalize(TF.to_tensor(TF.resize(rgb, [size, size])), IMAGENET_MEAN, IMAGENET_STD)
    logits = model(x.unsqueeze(0).to(device))["out"][0]
    pred = logits.argmax(0).cpu().numpy().astype(np.uint8)
    return np.array(Image.fromarray(pred, mode="L").resize((w, h), Image.Resampling.NEAREST))
