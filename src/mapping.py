from __future__ import annotations

import numpy as np

from .config import IGNORE_INDEX

CELL_M = 0.2
BEV_ROWS = 110
BEV_COLS = 80
Z_NEAR = 1.4
Z_FAR = 22.0
HFOV_DEG = 36.0

COST = {0: 1.0, 1: 2.2, 2: 5.5, 3: 90.0}
UNKNOWN_COST = 16.0
BLOCKED = 80.0


def inflate_rocks(label: np.ndarray, radius: int = 1) -> np.ndarray:
    rock = label == 3
    if radius <= 0 or not rock.any():
        return label
    out = label.copy()
    h, w = label.shape
    ys, xs = np.where(rock)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = 3
    return out


def image_to_bev(
    label: np.ndarray,
    rgb: np.ndarray | None = None,
    horizon: float = 0.34,
) -> tuple[np.ndarray, np.ndarray]:
    if label.ndim == 3:
        label = label[:, :, 0]
    h, w = label.shape[:2]
    rows, cols = BEV_ROWS, BEV_COLS
    v0 = int(horizon * h)
    if v0 >= h - 2:
        v0 = h // 3

    vv, uu = np.mgrid[v0 + 1 : h, 0:w]
    t = (vv - v0) / max(h - 1 - v0, 1)
    z = np.clip(Z_NEAR / np.maximum(t, Z_NEAR / Z_FAR), Z_NEAR, Z_FAR)
    iz = ((z - Z_NEAR) / CELL_M).astype(np.int32)
    half = z * np.tan(np.radians(HFOV_DEG))
    x = ((uu + 0.5) / w - 0.5) * 2.0 * half
    ix = ((x + cols * CELL_M / 2.0) / CELL_M).astype(np.int32)
    valid = (iz >= 0) & (iz < rows) & (ix >= 0) & (ix < cols)
    cls = label[v0 + 1 : h, :].astype(np.int32)
    ok = valid & (cls >= 0) & (cls <= 3)

    votes = np.zeros((rows, cols, 4), dtype=np.int32)
    np.add.at(votes, (iz[ok], ix[ok], cls[ok]), 1)
    bev = np.full((rows, cols), IGNORE_INDEX, dtype=np.uint8)
    filled = votes.sum(axis=2) > 0
    bev[filled] = np.argmax(votes, axis=2).astype(np.uint8)[filled]

    bev_rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    if rgb is not None:
        acc = np.zeros((rows, cols, 3), dtype=np.float64)
        cnt = np.zeros((rows, cols), dtype=np.int32)
        patch = rgb[v0 + 1 : h]
        np.add.at(acc, (iz[valid], ix[valid]), patch[valid])
        np.add.at(cnt, (iz[valid], ix[valid]), 1)
        nz = cnt > 0
        bev_rgb[nz] = (acc[nz] / cnt[nz, None]).astype(np.uint8)
    return inflate_rocks(bev, 1), bev_rgb


def cost_from_label(label: np.ndarray) -> np.ndarray:
    cost = np.full(label.shape, UNKNOWN_COST, dtype=np.float32)
    for cls, value in COST.items():
        cost[label == cls] = value
    return cost
