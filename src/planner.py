from __future__ import annotations

import heapq

import numpy as np

from .mapping import BLOCKED

NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.4142),
    (-1, 1, 1.4142),
    (1, -1, 1.4142),
    (1, 1, 1.4142),
)


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = cost.shape
    sy, sx = start
    gy, gx = goal
    if not (0 <= sy < rows and 0 <= sx < cols and 0 <= gy < rows and 0 <= gx < cols):
        return []
    if cost[sy, sx] >= BLOCKED or cost[gy, gx] >= BLOCKED:
        return []

    def h(y: int, x: int) -> float:
        return abs(y - gy) + abs(x - gx)

    openq: list[tuple[float, int, int, int]] = [(h(sy, sx), 0, sy, sx)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {start: 0.0}
    seen: set[tuple[int, int]] = set()
    tie = 1

    while openq:
        _, _, y, x = heapq.heappop(openq)
        if (y, x) in seen:
            continue
        seen.add((y, x))
        if (y, x) == goal:
            path = [(y, x)]
            while (y, x) in came:
                y, x = came[(y, x)]
                path.append((y, x))
            path.reverse()
            return path
        for dy, dx, step in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                continue
            cell = float(cost[ny, nx])
            if cell >= BLOCKED:
                continue
            ng = gscore[(y, x)] + step * 0.5 * (float(cost[y, x]) + cell)
            prev = gscore.get((ny, nx))
            if prev is None or ng < prev:
                gscore[(ny, nx)] = ng
                came[(ny, nx)] = (y, x)
                heapq.heappush(openq, (ng + h(ny, nx), tie, ny, nx))
                tie += 1
    return []
