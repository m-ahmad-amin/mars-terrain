"""Local AutoNav demo: NavCam + belief map + A* replan.

Run from the repo root:

    python -m src.demo
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .config import IGNORE_INDEX
from .dataset import load_index
from .infer import load_segmenter, predict_label
from .mapping import image_to_bev
from .paths import find_ncam, output_dir, project_root
from .sim import RANGE_CELLS, RoverSim
from .visualize import NAV_RGB, overlay

DEFAULT_STEM = "NLA_409036993EDR_F0051662NCAM05703M1"
CAM_W, MAP_W = 480, 400
PLAY_MS = 70


def find_pair(ncam: Path, stem: str | None):
    cache = output_dir() / "cache"
    pairs = load_index(ncam, "min1", cache, skip_empty=True)
    if not pairs:
        pairs = load_index(ncam, "train", cache, skip_empty=True)
    if stem:
        for item in pairs:
            if item[0] == stem:
                return item, pairs
    for item in pairs:
        if item[0] == DEFAULT_STEM:
            return item, pairs
    return pairs[0], pairs


def load_scene(ncam: Path, pair, model, device, use_model: bool):
    stem, image_path, label_path = pair
    rgb = np.array(Image.open(image_path).convert("RGB"))
    expert = np.array(Image.open(label_path))
    if expert.ndim == 3:
        expert = expert[:, :, 0]
    if use_model and model is not None:
        pred = predict_label(model, Image.fromarray(rgb), device)
    else:
        pred = expert
    bev, bev_rgb = image_to_bev(pred, rgb)
    return stem, rgb, pred, bev, bev_rgb


def draw_cam(rgb: np.ndarray, pred: np.ndarray, width: int) -> Image.Image:
    vis = overlay(rgb, pred, 0.45)
    im = Image.fromarray(vis)
    h = int(width * im.height / im.width)
    return im.resize((width, h), Image.Resampling.BILINEAR)


def draw_map(sim: RoverSim, bev_rgb: np.ndarray, width: int) -> Image.Image:
    rows, cols = sim.belief.shape
    scale = max(3, width // cols)
    w, h = cols * scale, rows * scale
    img = Image.new("RGB", (w, h), (22, 22, 26))
    px = img.load()
    belief = sim.belief
    seen = sim.seen
    for y in range(rows):
        dy = rows - 1 - y
        for x in range(cols):
            if not seen[y, x]:
                color = (32, 32, 38)
            else:
                cls = int(belief[y, x])
                color = NAV_RGB.get(cls, (50, 50, 50))
                if cls == IGNORE_INDEX:
                    base = bev_rgb[y, x]
                    color = tuple(int(c * 0.35) for c in base) if base.any() else (40, 40, 46)
            x0, y0 = x * scale, dy * scale
            for yy in range(y0, y0 + scale):
                for xx in range(x0, x0 + scale):
                    px[xx, yy] = color

    draw = ImageDraw.Draw(img)

    def cell(y: int, x: int) -> tuple[int, int]:
        return x * scale + scale // 2, (rows - 1 - y) * scale + scale // 2

    if len(sim.oracle) > 1:
        draw.line([cell(y, x) for y, x in sim.oracle], fill=(80, 140, 80), width=max(1, scale // 3))
    if len(sim.path) > 1:
        draw.line([cell(y, x) for y, x in sim.path], fill=(40, 210, 230), width=max(2, scale // 2))

    gy, gx = sim.goal
    cx, cy = cell(gy, gx)
    r = max(4, scale)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(250, 210, 40), width=2)

    rx, ry = cell(sim.y, sim.x)
    import math

    ang = -sim.heading
    pts = []
    for a, rad in ((0, 2.2 * scale), (2.4, 1.1 * scale), (-2.4, 1.1 * scale)):
        pts.append((rx + rad * math.sin(ang + a), ry - rad * math.cos(ang + a)))
    draw.polygon(pts, fill=(255, 255, 255), outline=(0, 0, 0))

    half = 0.54
    reach = RANGE_CELLS * scale
    left = ang - half
    right = ang + half
    draw.line(
        [
            (rx, ry),
            (rx + reach * math.sin(left), ry - reach * math.cos(left)),
            (rx + reach * math.sin(right), ry - reach * math.cos(right)),
            (rx, ry),
        ],
        fill=(180, 180, 200),
        width=1,
    )
    return img.resize((width, int(width * h / w)), Image.Resampling.NEAREST)


class Demo(tk.Tk):
    def __init__(self, stem: str | None, labels_only: bool) -> None:
        super().__init__()
        self.title("mars-terrain AutoNav")
        self.project = project_root()
        self.ncam = find_ncam(self.project)
        self.model = None
        self.device = None
        self.use_model = not labels_only
        self.playing = False
        self.photos: list[ImageTk.PhotoImage] = []

        try:
            if self.use_model:
                self.status_seed = "loading model…"
                self.update_idletasks()
                self.model, self.device = load_segmenter(self.project)
        except FileNotFoundError:
            self.use_model = False

        self.pair, self.pairs = find_pair(self.ncam, stem)
        self.pair_i = self.pairs.index(self.pair)
        self._build()
        self.reload()

    def _build(self) -> None:
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        ttk.Button(bar, text="Play", command=self.toggle).pack(side="left", padx=2)
        ttk.Button(bar, text="Step", command=self.step_once).pack(side="left", padx=2)
        ttk.Button(bar, text="Reset", command=self.reset).pack(side="left", padx=2)
        ttk.Button(bar, text="Next image", command=self.next_image).pack(side="left", padx=8)
        self.src_var = tk.StringVar(value="model" if self.use_model else "labels")
        ttk.Label(bar, textvariable=self.src_var).pack(side="left", padx=8)
        ttk.Label(bar, text="click map to set goal    space play    n next").pack(side="right")

        pics = ttk.Frame(self, padding=8)
        pics.pack(fill="both", expand=True)
        left = ttk.Frame(pics)
        left.pack(side="left", padx=6)
        ttk.Label(left, text="NavCam + terrain").pack()
        self.cam_panel = tk.Label(left, background="#111")
        self.cam_panel.pack()
        right = ttk.Frame(pics)
        right.pack(side="left", padx=6)
        ttk.Label(right, text="belief map   cyan=AutoNav   green=oracle").pack()
        self.map_panel = tk.Label(right, background="#111")
        self.map_panel.pack()
        self.map_panel.bind("<Button-1>", self.on_click)

        self.info = ttk.Label(self, padding=8, justify="left")
        self.info.pack(fill="x")
        self.bind("<space>", lambda _e: self.toggle())
        self.bind("n", lambda _e: self.next_image())
        self.bind("r", lambda _e: self.reset())

    def reload(self) -> None:
        self.playing = False
        self.info.config(text="projecting map…")
        self.update_idletasks()
        stem, rgb, pred, bev, bev_rgb = load_scene(
            self.ncam, self.pair, self.model, self.device, self.use_model
        )
        self.stem = stem
        self.rgb, self.pred = rgb, pred
        self.bev_rgb = bev_rgb
        self.sim = RoverSim(bev)
        self.src_var.set("model" if self.use_model else "labels")
        self.redraw()

    def redraw(self) -> None:
        cam = draw_cam(self.rgb, self.pred, CAM_W)
        mp = draw_map(self.sim, self.bev_rgb, MAP_W)
        self.photos = [ImageTk.PhotoImage(cam), ImageTk.PhotoImage(mp)]
        self.cam_panel.config(image=self.photos[0])
        self.map_panel.config(image=self.photos[1])
        s = self.sim
        seen = float(s.seen.mean())
        extra = ""
        if s.oracle and s.path:
            extra = f"   path {len(s.path)}   oracle {len(s.oracle)}"
        state = "goal" if s.reached else ("stuck" if s.stuck else "drive")
        self.info.config(
            text=f"{self.stem}   {state}   seen {seen:.0%}   pose ({s.y},{s.x}) → {s.goal}{extra}"
        )
        self.title(f"mars-terrain AutoNav — {self.stem} [{state}]")

    def toggle(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.tick()

    def tick(self) -> None:
        if not self.playing:
            return
        moved = self.sim.step()
        self.redraw()
        if moved and not self.sim.reached and not self.sim.stuck:
            self.after(PLAY_MS, self.tick)
        else:
            self.playing = False

    def step_once(self) -> None:
        self.playing = False
        self.sim.step()
        self.redraw()

    def reset(self) -> None:
        self.playing = False
        self.sim.reset()
        self.redraw()

    def next_image(self) -> None:
        self.playing = False
        self.pair_i = (self.pair_i + 1) % len(self.pairs)
        self.pair = self.pairs[self.pair_i]
        self.reload()

    def on_click(self, event: tk.Event) -> None:
        w = self.map_panel.winfo_width()
        h = self.map_panel.winfo_height()
        rows, cols = self.sim.belief.shape
        if w <= 1 or h <= 1:
            return
        x = int(event.x / w * cols)
        y_disp = int(event.y / h * rows)
        y = rows - 1 - y_disp
        self.sim.set_goal(y, x)
        self.redraw()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default=None)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()
    Demo(args.stem, args.labels).mainloop()


if __name__ == "__main__":
    main()
