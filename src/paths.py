from __future__ import annotations

import os
from pathlib import Path

NCAM_MARKERS = (Path("images") / "edr", Path("labels") / "train")


def project_root() -> Path:
    env = os.environ.get("MARS_TERRAIN_ROOT")
    if env:
        return Path(env)
    candidates = [
        Path("/content/drive/MyDrive/mars-terrain"),
        Path(r"G:\My Drive\mars-terrain"),
        Path(r"D:\GitHub\mars-terrain"),
    ]
    here = Path(__file__).resolve().parent.parent
    candidates.insert(0, here)
    for path in candidates:
        if path.is_dir():
            return path
    return here


def find_ncam(root: Path | None = None) -> Path:
    root = root or project_root()
    candidates = [
        Path("/content/data/ncam"),
        Path("/content/data/msl/ncam"),
        root / "data" / "ncam",
        root / "ncam",
        root / "data" / "msl" / "ncam",
        root / "data" / "ai4mars-dataset-merged-0.6" / "msl" / "ncam",
        Path(r"D:\GitHub\mars-terrain\data\ai4mars-dataset-merged-0.6\msl\ncam"),
    ]
    for path in candidates:
        if all((path / marker).is_dir() for marker in NCAM_MARKERS):
            return path
    raise FileNotFoundError(
        "MSL NavCam folder not found. Extract ai4mars-msl-ncam.tar so that "
        "images/edr and labels/train exist under ncam/."
    )


def archive_path(root: Path | None = None) -> Path | None:
    root = root or project_root()
    for path in (
        root / "ai4mars-msl-ncam.tar",
        root / "data" / "ai4mars-msl-ncam.tar",
        Path("/content/drive/MyDrive/mars-terrain/ai4mars-msl-ncam.tar"),
    ):
        if path.is_file():
            return path
    return None


def output_dir(root: Path | None = None) -> Path:
    path = (root or project_root()) / "outputs"
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "cache").mkdir(parents=True, exist_ok=True)
    (path / "preds").mkdir(parents=True, exist_ok=True)
    return path
