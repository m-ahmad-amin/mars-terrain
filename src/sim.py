from __future__ import annotations

import math

import numpy as np

from .config import IGNORE_INDEX
from .mapping import BLOCKED, UNKNOWN_COST, cost_from_label
from .planner import astar

FOV_DEG = 62.0
RANGE_CELLS = 28
MODES = ("autonav", "blind", "guarded")


def auto_goal(cost: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    rows, cols = cost.shape
    sy, sx = start
    for y in range(rows - 3, sy, -1):
        xs = list(range(cols))
        xs.sort(key=lambda x: abs(x - sx))
        for x in xs:
            if cost[y, x] < 6:
                return y, x
    return min(rows - 8, 78), cols // 2


class RoverSim:
    def __init__(self, gt_label: np.ndarray, mode: str = "autonav") -> None:
        if mode not in MODES:
            mode = "autonav"
        self.mode = mode
        self.gt_label = gt_label
        self.gt_cost = cost_from_label(gt_label)
        self.rows, self.cols = gt_label.shape
        self.belief = np.full(gt_label.shape, IGNORE_INDEX, dtype=np.uint8)
        self.seen = np.zeros(gt_label.shape, dtype=bool)
        self.y = 3
        self.x = self.cols // 2
        self.heading = 0.0
        self.goal = auto_goal(self.gt_cost, (self.y, self.x))
        self.path: list[tuple[int, int]] = []
        self.oracle: list[tuple[int, int]] = []
        self.reached = False
        self.stuck = False
        self.hazard = False
        self.replans = 0
        if mode == "blind":
            self.belief = gt_label.copy()
            self.seen[:] = True
        else:
            self.sense()
        self.replan()

    def set_goal(self, y: int, x: int) -> None:
        self.goal = (int(np.clip(y, 0, self.rows - 1)), int(np.clip(x, 0, self.cols - 1)))
        self.reached = False
        self.stuck = False
        self.replan()

    def reset(self) -> None:
        self.__init__(self.gt_label, self.mode)

    def sense(self) -> bool:
        changed = False
        half = math.radians(FOV_DEG) / 2.0
        y0, x0 = self.y, self.x
        if not self.seen[y0, x0]:
            self.belief[y0, x0] = self.gt_label[y0, x0]
            self.seen[y0, x0] = True
            changed = True
        rmax = RANGE_CELLS
        y_lo = max(0, y0 - rmax)
        y_hi = min(self.rows, y0 + rmax + 1)
        x_lo = max(0, x0 - rmax)
        x_hi = min(self.cols, x0 + rmax + 1)
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                dy, dx = y - y0, x - x0
                dist = math.hypot(dy, dx)
                if dist > rmax or dist < 1e-6:
                    continue
                ang = math.atan2(dx, dy) - self.heading
                ang = (ang + math.pi) % (2 * math.pi) - math.pi
                if abs(ang) > half:
                    continue
                if not self.seen[y, x]:
                    self.belief[y, x] = self.gt_label[y, x]
                    self.seen[y, x] = True
                    changed = True
        return changed

    def belief_cost(self) -> np.ndarray:
        cost = cost_from_label(self.belief)
        cost[~self.seen] = UNKNOWN_COST
        return cost

    def replan(self) -> None:
        start = (self.y, self.x)
        cost = self.gt_cost if self.mode == "blind" else self.belief_cost()
        self.path = astar(cost, start, self.goal)
        self.oracle = astar(self.gt_cost, start, self.goal)
        self.stuck = len(self.path) < 2 and start != self.goal
        self.reached = start == self.goal
        self.replans += 1

    def step(self) -> bool:
        self.hazard = False
        if self.reached or self.stuck:
            return False
        if len(self.path) < 2:
            self.replan()
            if len(self.path) < 2:
                self.stuck = True
                return False
        ny, nx = self.path[1]
        if self.mode == "guarded" and (self.gt_cost[ny, nx] >= BLOCKED or self.gt_label[ny, nx] == 3):
            self.belief[ny, nx] = self.gt_label[ny, nx]
            self.seen[ny, nx] = True
            self.hazard = True
            self.stuck = True
            return False
        self.heading = math.atan2(nx - self.x, ny - self.y)
        self.y, self.x = ny, nx
        if self.mode != "blind" and self.sense():
            before = self.path
            self.replan()
            if self.path != before:
                pass
        else:
            self.path = self.path[1:]
        if (self.y, self.x) == self.goal:
            self.reached = True
            self.path = [(self.y, self.x)]
        return True
