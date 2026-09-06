from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from .paths import archive_path, find_ncam


def extract_archive(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest)
    ncam = dest / "ncam"
    if not (ncam / "images" / "edr").is_dir():
        matches = list(dest.rglob("images/edr"))
        if matches:
            ncam = matches[0].parent.parent
    return ncam


def ensure_local_ncam(drive_root: Path, local_root: Path = Path("/content/data")) -> Path:
    try:
        return find_ncam(drive_root)
    except FileNotFoundError:
        pass
    try:
        return find_ncam(local_root)
    except FileNotFoundError:
        pass

    tar = archive_path(drive_root)
    if tar is None:
        raise FileNotFoundError("ai4mars-msl-ncam.tar not found on Drive.")

    local_tar = local_root / tar.name
    local_root.mkdir(parents=True, exist_ok=True)
    if not local_tar.is_file():
        shutil.copy2(tar, local_tar)
    return extract_archive(local_tar, local_root)
