from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image

from .dataset import load_index
from .infer import load_segmenter, predict_label
from .mapping import image_to_bev
from .paths import find_ncam, output_dir, project_root
from .sim import RoverSim
from .visualize import overlay

ADVANCE_Y = 0.74


class Mission:
    def __init__(self) -> None:
        self.project = project_root()
        self.ncam = find_ncam(self.project)
        cache = output_dir(self.project) / "cache"
        self.pairs = load_index(self.ncam, "min1", cache, skip_empty=True)
        if not self.pairs:
            self.pairs = load_index(self.ncam, "train", cache, skip_empty=True)
        self.model = None
        self.device = None
        self.use_model = False
        self.continuous = True
        self.mode = "autonav"
        self.index = 0
        self.sites = 1
        self.events: list[str] = []
        self.rgb: np.ndarray | None = None
        self.pred: np.ndarray | None = None
        self.stem = ""
        self.sim: RoverSim | None = None
        self._load(0)

    def enable_model(self) -> str:
        if self.model is None:
            self.model, self.device = load_segmenter(self.project)
        self.use_model = True
        self._load(self.index)
        return "model"

    def start(self, index: int | None = None, mode: str = "autonav", continuous: bool = True) -> dict:
        self.mode = mode
        self.continuous = continuous
        self.sites = 1
        self.events = [f"site 1 · {mode}"]
        self._load(index if index is not None else self.index)
        return self.snapshot()

    def _load(self, index: int) -> None:
        self.index = index % len(self.pairs)
        stem, image_path, label_path = self.pairs[self.index]
        rgb = np.array(Image.open(image_path).convert("RGB"))
        expert = np.array(Image.open(label_path))
        if expert.ndim == 3:
            expert = expert[:, :, 0]
        if self.use_model and self.model is not None:
            pred = predict_label(self.model, Image.fromarray(rgb), self.device)
            src = "model"
        else:
            pred = expert
            src = "labels"
        bev, _ = image_to_bev(pred, rgb)
        self.stem = stem
        self.rgb = rgb
        self.pred = pred
        self.source = src
        self.sim = RoverSim(bev, self.mode)
        self.events.append(f"loaded {stem}")

    def set_goal(self, y: int, x: int) -> dict:
        assert self.sim
        self.sim.set_goal(y, x)
        self.events.append(f"goal ({y},{x})")
        return self.snapshot()

    def step(self) -> dict:
        assert self.sim
        before = len(self.sim.path)
        moved = self.sim.step()
        if self.sim.hazard:
            self.events.append("hazard · guarded stop")
        if self.sim.replans and moved and len(self.sim.path) != before:
            self.events.append("replan")
        if self.sim.reached:
            self.events.append("goal reached")
        if self._should_advance():
            self.sites += 1
            self.events.append(f"next photo · site {self.sites}")
            self._load(self.index + 1)
        return self.snapshot()

    def _should_advance(self) -> bool:
        if not self.continuous or self.sim is None:
            return False
        far = self.sim.y >= int(self.sim.rows * ADVANCE_Y)
        return bool(self.sim.reached or (self.sim.stuck and not self.sim.hazard) or far)

    def snapshot(self) -> dict:
        sim = self.sim
        assert sim and self.rgb is not None and self.pred is not None
        cam = overlay(self.rgb, self.pred, 0.42)
        buf = io.BytesIO()
        Image.fromarray(cam).resize((640, 640), Image.Resampling.BILINEAR).save(buf, format="JPEG", quality=78)
        status = "drive"
        if sim.hazard:
            status = "hazard"
        elif sim.reached:
            status = "goal"
        elif sim.stuck:
            status = "stuck"
        return {
            "stem": self.stem,
            "index": self.index,
            "total": len(self.pairs),
            "sites": self.sites,
            "mode": sim.mode,
            "source": self.source,
            "continuous": self.continuous,
            "status": status,
            "rows": sim.rows,
            "cols": sim.cols,
            "rover": [sim.y, sim.x],
            "heading": sim.heading,
            "goal": [sim.goal[0], sim.goal[1]],
            "path": sim.path,
            "oracle": sim.oracle,
            "belief": base64.b64encode(sim.belief.tobytes()).decode("ascii"),
            "seen": base64.b64encode(np.packbits(sim.seen.ravel())).decode("ascii"),
            "seen_frac": float(sim.seen.mean()),
            "path_len": len(sim.path),
            "oracle_len": len(sim.oracle),
            "replans": max(0, sim.replans - 1),
            "cam": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "events": self.events[-14:],
        }


_mission: Mission | None = None


def get_mission() -> Mission:
    global _mission
    if _mission is None:
        _mission = Mission()
    return _mission
