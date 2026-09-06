from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .config import IGNORE_INDEX, TrainConfig
from .dataset import class_weights, load_index, make_loader
from .evaluate import evaluate, format_metrics, save_metrics
from .model import build_model
from .paths import output_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _same_run(saved: dict, cfg: TrainConfig) -> bool:
    return saved.get("max_train") == cfg.max_train and saved.get("image_size") == cfg.image_size


def train(
    ncam: Path,
    cfg: TrainConfig | None = None,
    project: Path | None = None,
) -> dict[str, float]:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = output_dir(project)
    cache = Path("/content/cache") if Path("/content").is_dir() else out / "cache"
    ckpt_dir = out / "checkpoints"
    log_path = out / "logs" / "train.json"

    train_pairs = load_index(ncam, "train", cache, skip_empty=True)
    val_pairs = load_index(ncam, "min1", cache, skip_empty=True)
    if cfg.max_train:
        train_pairs = train_pairs[: cfg.max_train]
        val_pairs = val_pairs[: max(32, cfg.max_train // 8)]

    print(f"train {len(train_pairs)}  val {len(val_pairs)}  device {device}")
    train_loader = make_loader(train_pairs, cfg.image_size, cfg.batch_size, True, cfg.num_workers, True)
    val_loader = make_loader(val_pairs, cfg.image_size, cfg.batch_size, False, cfg.num_workers, False)

    weights = class_weights(train_pairs).to(device)
    model = build_model().to(device)
    criterion = CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    history: list[dict] = []
    best = {"mIoU": -1.0}
    last_path = ckpt_dir / "last.pt"
    if last_path.is_file():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        saved = ckpt.get("config", {})
        if cfg.resume and _same_run(saved, cfg):
            if "optim" in ckpt:
                optim.load_state_dict(ckpt["optim"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            history = list(ckpt.get("history", []))
            best = dict(ckpt.get("best", {"mIoU": -1.0}))
            print(f"resume from epoch {start_epoch}")
        else:
            print("loaded smoke/other weights, starting full run at epoch 1")

    remaining = max(cfg.epochs - start_epoch + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=remaining)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_dict = model(images)
                loss = criterion(out_dict["out"], labels)
                if "aux" in out_dict:
                    loss = loss + 0.4 * criterion(out_dict["aux"], labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            steps += 1
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = running / max(steps, 1)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(f"epoch {epoch}: loss {metrics['loss']:.4f}  {format_metrics(metrics)}")

        improved = metrics["mIoU"] > best.get("mIoU", -1.0)
        if improved:
            best = metrics

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "metrics": metrics,
            "best": best,
            "history": history,
            "config": cfg.__dict__,
        }
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, ckpt_dir / "best.pt")
        save_metrics({"best": best, "history": history}, log_path)

    return best
