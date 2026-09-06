function b64bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function wrapPi(a) {
  a = (a + Math.PI) % (Math.PI * 2);
  if (a < 0) a += Math.PI * 2;
  return a - Math.PI;
}

const IGNORE = 255;
const COST = { 0: 1, 1: 2.2, 2: 5.5, 3: 90 };
const UNKNOWN_COST = 16;
const BLOCKED = 80;
const FOV_DEG = 62;
const RANGE_CELLS = 28;
const ADVANCE_Y = 0.74;
const NEIGHBORS = [
  [-1, 0, 1],
  [1, 0, 1],
  [0, -1, 1],
  [0, 1, 1],
  [-1, -1, 1.4142],
  [-1, 1, 1.4142],
  [1, -1, 1.4142],
  [1, 1, 1.4142],
];

function costFromLabel(label) {
  const cost = new Float32Array(label.length);
  for (let i = 0; i < label.length; i++) {
    const cls = label[i];
    cost[i] = cls in COST ? COST[cls] : UNKNOWN_COST;
  }
  return cost;
}

function autoGoal(cost, rows, cols, sy, sx) {
  for (let y = rows - 3; y > sy; y--) {
    const xs = Array.from({ length: cols }, (_, x) => x).sort((a, b) => Math.abs(a - sx) - Math.abs(b - sx));
    for (const x of xs) {
      if (cost[y * cols + x] < 6) return [y, x];
    }
  }
  return [Math.min(rows - 8, 78), (cols / 2) | 0];
}

function astar(cost, rows, cols, start, goal) {
  const [sy, sx] = start;
  const [gy, gx] = goal;
  if (sy < 0 || sx < 0 || gy < 0 || gx < 0 || sy >= rows || sx >= cols || gy >= rows || gx >= cols) return [];
  if (cost[sy * cols + sx] >= BLOCKED || cost[gy * cols + gx] >= BLOCKED) return [];
  const key = (y, x) => y * cols + x;
  const h = (y, x) => Math.abs(y - gy) + Math.abs(x - gx);
  const heap = [[h(sy, sx), 0, sy, sx]];
  const came = new Int32Array(rows * cols).fill(-1);
  const gscore = new Float32Array(rows * cols).fill(Infinity);
  const seen = new Uint8Array(rows * cols);
  gscore[key(sy, sx)] = 0;
  let tie = 1;
  while (heap.length) {
    let best = 0;
    for (let i = 1; i < heap.length; i++) if (heap[i][0] < heap[best][0]) best = i;
    const cur = heap[best];
    heap[best] = heap[heap.length - 1];
    heap.pop();
    const y = cur[2];
    const x = cur[3];
    const k = key(y, x);
    if (seen[k]) continue;
    seen[k] = 1;
    if (y === gy && x === gx) {
      const path = [[y, x]];
      let cy = y;
      let cx = x;
      while (came[key(cy, cx)] >= 0) {
        const prev = came[key(cy, cx)];
        cy = (prev / cols) | 0;
        cx = prev % cols;
        path.push([cy, cx]);
      }
      path.reverse();
      return path;
    }
    for (const [dy, dx, step] of NEIGHBORS) {
      const ny = y + dy;
      const nx = x + dx;
      if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) continue;
      const nk = key(ny, nx);
      const cell = cost[nk];
      if (cell >= BLOCKED) continue;
      const ng = gscore[k] + step * 0.5 * (cost[k] + cell);
      if (ng < gscore[nk]) {
        gscore[nk] = ng;
        came[nk] = k;
        heap.push([ng + h(ny, nx), tie++, ny, nx]);
      }
    }
  }
  return [];
}

function packSeen(seen) {
  const out = new Uint8Array(Math.ceil(seen.length / 8));
  for (let i = 0; i < seen.length; i++) {
    if (seen[i]) out[i >> 3] |= 1 << (7 - (i & 7));
  }
  let s = "";
  for (let i = 0; i < out.length; i++) s += String.fromCharCode(out[i]);
  return btoa(s);
}

function bytesB64(arr) {
  let s = "";
  for (let i = 0; i < arr.length; i++) s += String.fromCharCode(arr[i]);
  return btoa(s);
}

class RoverSim {
  constructor(label, rows, cols, mode) {
    this.mode = ["autonav", "blind", "guarded"].includes(mode) ? mode : "autonav";
    this.gt = label;
    this.rows = rows;
    this.cols = cols;
    this.gtCost = costFromLabel(label);
    this.belief = new Uint8Array(label.length).fill(IGNORE);
    this.seen = new Uint8Array(label.length);
    this.y = 3;
    this.x = (cols / 2) | 0;
    this.heading = 0;
    this.goal = autoGoal(this.gtCost, rows, cols, this.y, this.x);
    this.path = [];
    this.oracle = [];
    this.reached = false;
    this.stuck = false;
    this.hazard = false;
    this.replans = 0;
    if (this.mode === "blind") {
      this.belief = new Uint8Array(label);
      this.seen.fill(1);
    } else {
      this.sense();
    }
    this.replan();
  }

  setGoal(y, x) {
    this.goal = [Math.max(0, Math.min(this.rows - 1, y | 0)), Math.max(0, Math.min(this.cols - 1, x | 0))];
    this.reached = false;
    this.stuck = false;
    this.replan();
  }

  sense() {
    let changed = false;
    const half = (FOV_DEG * Math.PI) / 180 / 2;
    const { y: y0, x: x0, cols, rows } = this;
    const mark = (y, x) => {
      const i = y * cols + x;
      if (!this.seen[i]) {
        this.belief[i] = this.gt[i];
        this.seen[i] = 1;
        changed = true;
      }
    };
    mark(y0, x0);
    const yLo = Math.max(0, y0 - RANGE_CELLS);
    const yHi = Math.min(rows, y0 + RANGE_CELLS + 1);
    const xLo = Math.max(0, x0 - RANGE_CELLS);
    const xHi = Math.min(cols, x0 + RANGE_CELLS + 1);
    for (let y = yLo; y < yHi; y++) {
      for (let x = xLo; x < xHi; x++) {
        const dy = y - y0;
        const dx = x - x0;
        const dist = Math.hypot(dy, dx);
        if (dist > RANGE_CELLS || dist < 1e-6) continue;
        const ang = wrapPi(Math.atan2(dx, dy) - this.heading);
        if (Math.abs(ang) > half) continue;
        mark(y, x);
      }
    }
    return changed;
  }

  beliefCost() {
    const cost = costFromLabel(this.belief);
    for (let i = 0; i < cost.length; i++) if (!this.seen[i]) cost[i] = UNKNOWN_COST;
    return cost;
  }

  replan() {
    const start = [this.y, this.x];
    const cost = this.mode === "blind" ? this.gtCost : this.beliefCost();
    this.path = astar(cost, this.rows, this.cols, start, this.goal);
    this.oracle = astar(this.gtCost, this.rows, this.cols, start, this.goal);
    this.stuck = this.path.length < 2 && (this.y !== this.goal[0] || this.x !== this.goal[1]);
    this.reached = this.y === this.goal[0] && this.x === this.goal[1];
    this.replans += 1;
  }

  step() {
    this.hazard = false;
    if (this.reached || this.stuck) return false;
    if (this.path.length < 2) {
      this.replan();
      if (this.path.length < 2) {
        this.stuck = true;
        return false;
      }
    }
    const [ny, nx] = this.path[1];
    const ni = ny * this.cols + nx;
    if (this.mode === "guarded" && (this.gtCost[ni] >= BLOCKED || this.gt[ni] === 3)) {
      this.belief[ni] = this.gt[ni];
      this.seen[ni] = 1;
      this.hazard = true;
      this.stuck = true;
      return false;
    }
    this.heading = Math.atan2(nx - this.x, ny - this.y);
    this.y = ny;
    this.x = nx;
    if (this.mode !== "blind" && this.sense()) this.replan();
    else this.path = this.path.slice(1);
    if (this.y === this.goal[0] && this.x === this.goal[1]) {
      this.reached = true;
      this.path = [[this.y, this.x]];
    }
    return true;
  }
}

class DemoMission {
  constructor(scenes) {
    this.scenes = scenes;
    this.index = 0;
    this.sites = 1;
    this.mode = "autonav";
    this.continuous = true;
    this.sim = null;
    this.scene = null;
    this.load(0);
  }

  load(index) {
    this.index = ((index % this.scenes.length) + this.scenes.length) % this.scenes.length;
    this.scene = this.scenes[this.index];
    const label = b64bytes(this.scene.label);
    this.sim = new RoverSim(label, this.scene.rows, this.scene.cols, this.mode);
  }

  start(index, mode, continuous) {
    this.mode = mode;
    this.continuous = continuous;
    this.sites = 1;
    this.load(index ?? this.index);
    return this.snapshot();
  }

  setGoal(y, x) {
    this.sim.setGoal(y, x);
    return this.snapshot();
  }

  step() {
    this.sim.step();
    if (this.shouldAdvance()) {
      this.sites += 1;
      this.load(this.index + 1);
    }
    return this.snapshot();
  }

  shouldAdvance() {
    if (!this.continuous || !this.sim) return false;
    const far = this.sim.y >= ((this.sim.rows * ADVANCE_Y) | 0);
    return this.sim.reached || (this.sim.stuck && !this.sim.hazard) || far;
  }

  snapshot() {
    const s = this.sim;
    let status = "drive";
    if (s.hazard) status = "hazard";
    else if (s.reached) status = "goal";
    else if (s.stuck) status = "stuck";
    let seenN = 0;
    for (let i = 0; i < s.seen.length; i++) if (s.seen[i]) seenN += 1;
    return {
      stem: this.scene.stem,
      index: this.index,
      total: this.scenes.length,
      sites: this.sites,
      source: "labels",
      status,
      rows: s.rows,
      cols: s.cols,
      rover: [s.y, s.x],
      heading: s.heading,
      goal: s.goal,
      path: s.path,
      oracle: s.oracle,
      belief: bytesB64(s.belief),
      seen: packSeen(s.seen),
      seen_frac: seenN / s.seen.length,
      path_len: s.path.length,
      cam: new URL(this.scene.cam, document.baseURI).href,
    };
  }
}
