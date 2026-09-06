const TERRAIN = {
  0: [196, 154, 102],
  1: [138, 138, 142],
  2: [210, 186, 86],
  3: [148, 74, 50],
};

const $ = (id) => document.getElementById(id);
const state = {
  playing: false,
  timer: null,
  data: null,
  layout: null,
  mission: null,
};

function b64bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function unpackBits(b64, n) {
  const packed = b64bytes(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (packed[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

function hash(x, y) {
  let n = x * 374761393 + y * 668265263;
  n = (n ^ (n >> 13)) * 1274126177;
  return ((n ^ (n >> 16)) >>> 0) / 4294967295;
}

function apply(data) {
  state.data = data;
  $("wait").classList.add("off");
  $("cam").src = data.cam;
  $("camMeta").textContent = `${data.source} · ${data.stem}`;
  $("stStatus").textContent = data.status;
  $("stSite").textContent = `site ${data.sites} · ${data.index + 1}/${data.total}`;
  $("stSeen").textContent = `seen ${(data.seen_frac * 100).toFixed(0)}%`;
  $("stPath").textContent = `path ${data.path_len}`;
  $("stNote").textContent = "click the map to drop a goal";
  drawMap(data);
}

function viewBounds(data) {
  const size = 26;
  const [ry, rx] = data.rover;
  const [gy, gx] = data.goal;
  const dy = gy - ry;
  const dx = gx - rx;
  const dist = Math.hypot(dy, dx) || 1;
  const bias = Math.min(7, dist * 0.22);
  const cy = Math.round(ry + (dy / dist) * bias);
  const cx = Math.round(rx + (dx / dist) * bias);
  let y0 = cy - Math.floor(size / 2);
  let x0 = cx - Math.floor(size / 2);
  let y1 = y0 + size - 1;
  let x1 = x0 + size - 1;
  if (x0 < 0) {
    x1 -= x0;
    x0 = 0;
  }
  if (y0 < 0) {
    y1 -= y0;
    y0 = 0;
  }
  if (x1 > data.cols - 1) {
    x0 -= x1 - (data.cols - 1);
    x1 = data.cols - 1;
  }
  if (y1 > data.rows - 1) {
    y0 -= y1 - (data.rows - 1);
    y1 = data.rows - 1;
  }
  return {
    x0: Math.max(0, x0),
    y0: Math.max(0, y0),
    x1: Math.min(data.cols - 1, x1),
    y1: Math.min(data.rows - 1, y1),
  };
}

function fitLayout(w, h, data) {
  const { x0, y0, x1, y1 } = viewBounds(data);
  const cols = x1 - x0 + 1;
  const rows = y1 - y0 + 1;
  const padL = 36;
  const padT = 26;
  const padR = 20;
  const padB = 20;
  const cell = Math.max(10, Math.floor(Math.min((w - padL - padR) / cols, (h - padT - padB) / rows)));
  const ox = padL + Math.floor((w - padL - padR - cols * cell) / 2);
  const oy = padT + Math.floor((h - padT - padB - rows * cell) / 2);
  return { ox, oy, cell, rows, cols, w, h, x0, y0, x1, y1, worldRows: data.rows, worldCols: data.cols };
}

function cellXY(layout, y, x) {
  return [
    layout.ox + (x - layout.x0) * layout.cell + layout.cell / 2,
    layout.oy + (layout.y1 - y) * layout.cell + layout.cell / 2,
  ];
}

function drawBackdrop(ctx, layout) {
  const { w, h, ox, oy, cell, rows, cols } = layout;
  ctx.fillStyle = "#0a0a0b";
  ctx.fillRect(0, 0, w, h);

  const gw = cols * cell;
  const gh = rows * cell;
  ctx.fillStyle = "#111214";
  ctx.fillRect(ox, oy, gw, gh);

  ctx.strokeStyle = "rgba(255,255,255,0.055)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= cols; x++) {
    const px = ox + x * cell + 0.5;
    ctx.moveTo(px, oy);
    ctx.lineTo(px, oy + gh);
  }
  for (let y = 0; y <= rows; y++) {
    const py = oy + y * cell + 0.5;
    ctx.moveTo(ox, py);
    ctx.lineTo(ox + gw, py);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(230,228,220,0.58)";
  ctx.font = "11px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  const tick = cols > 16 ? 2 : 1;
  for (let i = 0; i <= cols; i += tick) {
    ctx.fillText(String(i), ox + i * cell, oy - 7);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= rows; i += tick) {
    ctx.fillText(String(rows - i), ox - 8, oy + i * cell);
  }
}

function shade(rgb, n) {
  const t = (n - 0.5) * 36;
  return `rgb(${rgb[0] + t | 0},${rgb[1] + t | 0},${rgb[2] + t | 0})`;
}

function drawTiles(ctx, layout, belief, seen, worldCols) {
  const { ox, oy, cell, x0, y0, x1, y1 } = layout;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * worldCols + x;
      if (!seen[i]) continue;
      const cls = belief[i];
      const base = cls === 3 ? [168, 132, 86] : TERRAIN[cls] || [48, 48, 52];
      const px = ox + (x - x0) * cell;
      const py = oy + (y1 - y) * cell;
      const n = hash(x, y);
      ctx.fillStyle = shade(base, n);
      ctx.fillRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
      const speck = hash(x + 9, y + 4);
      ctx.fillStyle = shade(base, speck * 0.65 + 0.1);
      ctx.fillRect(px + cell * 0.18, py + cell * 0.22, cell * 0.38, cell * 0.28);
      if (cls === 1) {
        ctx.fillStyle = "rgba(40,38,36,0.2)";
        ctx.fillRect(px + cell * 0.55, py + cell * 0.12, cell * 0.28, cell * 0.2);
        ctx.fillRect(px + cell * 0.1, py + cell * 0.6, cell * 0.22, cell * 0.16);
      }
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(px + 1, py + 1, cell - 2, 1);
      ctx.fillStyle = "rgba(0,0,0,0.18)";
      ctx.fillRect(px + 1, py + cell - 2, cell - 2, 1);
    }
  }
}

function boulder(ctx, cx, cy, rx, ry, n, big) {
  const pts = 7 + ((n * 4) | 0);
  ctx.beginPath();
  for (let i = 0; i < pts; i++) {
    const a = (i / pts) * Math.PI * 2 + n * 4.2;
    const j = hash(((cx + i * 17) | 0), ((cy + i * 9) | 0));
    const px = cx + Math.cos(a) * rx * (0.72 + j * 0.38);
    const py = cy + Math.sin(a) * ry * (0.68 + j * 0.4);
    if (i) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(0,0,0,0.38)";
  ctx.fill();

  ctx.save();
  ctx.translate(0, -ry * (big ? 0.28 : 0.18));
  ctx.beginPath();
  for (let i = 0; i < pts; i++) {
    const a = (i / pts) * Math.PI * 2 + n * 4.2;
    const j = hash(((cx + i * 17) | 0), ((cy + i * 9) | 0));
    const px = cx + Math.cos(a) * rx * (0.72 + j * 0.38);
    const py = cy + Math.sin(a) * ry * (0.68 + j * 0.4);
    if (i) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
  }
  ctx.closePath();
  const r = 128 + n * 36;
  const g = 62 + n * 22;
  const b = 40 + n * 12;
  ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
  ctx.fill();
  ctx.fillStyle = `rgba(${(r + 50) | 0},${(g + 28) | 0},${(b + 16) | 0},0.55)`;
  ctx.beginPath();
  ctx.ellipse(cx - rx * 0.18, cy - ry * 0.28, rx * 0.34, ry * 0.22, -0.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawRocks(ctx, layout, belief, seen, worldCols) {
  const { ox, oy, cell, x0, y0, x1, y1 } = layout;
  const marks = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * worldCols + x;
      if (!seen[i]) continue;
      const cls = belief[i];
      const n = hash(x, y + 3);
      if (cls === 3) marks.push({ y, x, kind: "big", n });
      else if (cls === 1 && n > 0.34) marks.push({ y, x, kind: "mid", n });
      else if (cls === 0 && n > 0.86) marks.push({ y, x, kind: "peb", n });
    }
  }
  marks.sort((a, b) => b.y - a.y);
  for (const m of marks) {
    const px = ox + (m.x - x0) * cell + cell / 2;
    const py = oy + (y1 - m.y) * cell + cell / 2;
    if (m.kind === "big") {
      boulder(ctx, px - cell * 0.12, py + cell * 0.1, cell * 0.48, cell * 0.34, m.n, true);
      boulder(ctx, px + cell * 0.22, py - cell * 0.02, cell * 0.38, cell * 0.28, 1 - m.n, true);
      boulder(ctx, px + cell * 0.02, py - cell * 0.18, cell * 0.3, cell * 0.22, hash(m.x, m.y + 11), true);
    } else if (m.kind === "mid") {
      boulder(ctx, px + (m.n - 0.5) * cell * 0.22, py + cell * 0.06, cell * 0.4, cell * 0.28, m.n, false);
      if (m.n > 0.62) {
        boulder(ctx, px - cell * 0.2, py - cell * 0.02, cell * 0.24, cell * 0.18, 1 - m.n, false);
      }
    } else {
      boulder(ctx, px, py + cell * 0.1, cell * 0.14, cell * 0.1, m.n, false);
    }
  }
}

function strokePath(ctx, pts, layout, dashed) {
  if (!pts || pts.length < 2) return;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const [x, y] = cellXY(layout, p[0], p[1]);
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  });
  ctx.setLineDash(dashed ? [7, 6] : []);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawRoutes(ctx, data, layout) {
  const w = Math.max(2, layout.cell * 0.28);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  if (data.oracle && data.oracle.length > 1) {
    ctx.save();
    ctx.strokeStyle = "rgba(80, 230, 255, 0.38)";
    ctx.lineWidth = w;
    strokePath(ctx, data.oracle, layout, true);
    ctx.restore();
  }

  if (data.path && data.path.length > 1) {
    ctx.save();
    ctx.shadowColor = "#2de8ff";
    ctx.shadowBlur = 14;
    ctx.strokeStyle = "#2de8ff";
    ctx.lineWidth = w + 1.4;
    strokePath(ctx, data.path, layout, false);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#9ff6ff";
    ctx.lineWidth = Math.max(1.4, w * 0.55);
    strokePath(ctx, data.path, layout, false);
    ctx.restore();
  }
}

function drawPin(ctx, x, y, label) {
  ctx.save();
  ctx.translate(x, y);
  ctx.shadowColor = "rgba(255, 225, 74, 0.7)";
  ctx.shadowBlur = 12;
  ctx.fillStyle = "#ffe14a";
  ctx.beginPath();
  ctx.moveTo(0, 11);
  ctx.bezierCurveTo(-9, 2, -8, -10, 0, -12);
  ctx.bezierCurveTo(8, -10, 9, 2, 0, 11);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#1a1608";
  ctx.beginPath();
  ctx.arc(0, -5, 2.6, 0, Math.PI * 2);
  ctx.fill();
  if (label) {
    ctx.font = "700 12px Segoe UI, sans-serif";
    ctx.fillStyle = "#ffe14a";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur = 4;
    ctx.fillText(label, 12, -4);
  }
  ctx.restore();
}

function wheel(ctx, x, y, s) {
  ctx.fillStyle = "#1c1c1e";
  ctx.strokeStyle = "#0a0a0a";
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.roundRect(x - s * 0.17, y - s * 0.13, s * 0.34, s * 0.26, 2);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "rgba(220,220,220,0.22)";
  ctx.beginPath();
  ctx.moveTo(x - s * 0.08, y);
  ctx.lineTo(x + s * 0.08, y);
  ctx.stroke();
}

function drawRover(ctx, x, y, heading, cell) {
  const s = Math.max(13, cell * 1.55);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);

  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.beginPath();
  ctx.ellipse(1, s * 0.12, s * 0.78, s * 0.4, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#5a5448";
  ctx.lineWidth = Math.max(1.6, s * 0.08);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-s * 0.58, -s * 0.46);
  ctx.lineTo(-s * 0.36, 0);
  ctx.lineTo(-s * 0.58, s * 0.46);
  ctx.moveTo(s * 0.58, -s * 0.46);
  ctx.lineTo(s * 0.36, 0);
  ctx.lineTo(s * 0.58, s * 0.46);
  ctx.stroke();

  wheel(ctx, -s * 0.58, -s * 0.46, s);
  wheel(ctx, -s * 0.6, 0, s);
  wheel(ctx, -s * 0.58, s * 0.46, s);
  wheel(ctx, s * 0.58, -s * 0.46, s);
  wheel(ctx, s * 0.6, 0, s);
  wheel(ctx, s * 0.58, s * 0.46, s);

  ctx.fillStyle = "#cbb892";
  ctx.strokeStyle = "#3b3428";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.roundRect(-s * 0.36, -s * 0.48, s * 0.72, s * 0.96, 3);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#b7a57a";
  ctx.fillRect(-s * 0.28, -s * 0.38, s * 0.56, s * 0.22);
  ctx.fillStyle = "#9e8d68";
  ctx.fillRect(-s * 0.22, s * 0.08, s * 0.44, s * 0.28);

  ctx.strokeStyle = "#8a8170";
  ctx.lineWidth = Math.max(1.4, s * 0.07);
  ctx.beginPath();
  ctx.moveTo(0, -s * 0.48);
  ctx.lineTo(0, -s * 0.72);
  ctx.stroke();
  ctx.fillStyle = "#6e6758";
  ctx.beginPath();
  ctx.arc(0, -s * 0.78, s * 0.09, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#7fe7ff";
  ctx.beginPath();
  ctx.arc(0, -s * 0.78, s * 0.035, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#d7cdb6";
  ctx.beginPath();
  ctx.ellipse(s * 0.14, s * 0.42, s * 0.11, s * 0.07, 0.35, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawMap(data) {
  const canvas = $("map");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w < 20 || h < 20) return;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const layout = fitLayout(w, h, data);
  state.layout = layout;
  const belief = b64bytes(data.belief);
  const seen = unpackBits(data.seen, data.rows * data.cols);

  drawBackdrop(ctx, layout);
  drawTiles(ctx, layout, belief, seen, data.cols);
  drawRocks(ctx, layout, belief, seen, data.cols);
  drawRoutes(ctx, data, layout);

  const goalIn =
    data.goal[0] >= layout.y0 &&
    data.goal[0] <= layout.y1 &&
    data.goal[1] >= layout.x0 &&
    data.goal[1] <= layout.x1;
  if (goalIn) {
    const [gpx, gpy] = cellXY(layout, data.goal[0], data.goal[1]);
    drawPin(ctx, gpx, gpy, "GOAL");
  }

  const [rx, ry] = cellXY(layout, data.rover[0], data.rover[1]);
  drawRover(ctx, rx, ry, data.heading, layout.cell);

  const g = ctx.createRadialGradient(w * 0.5, h * 0.45, h * 0.2, w * 0.5, h * 0.5, h * 0.78);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, "rgba(0,0,0,0.28)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function start() {
  stop();
  apply(
    state.mission.start(
      Number($("scenes").value || 0),
      $("mode").value,
      $("continuous").checked
    )
  );
}

function step() {
  apply(state.mission.step());
}

function stop() {
  state.playing = false;
  $("play").textContent = "Play";
  if (state.timer) clearTimeout(state.timer);
}

function play() {
  if (state.playing) {
    stop();
    return;
  }
  state.playing = true;
  $("play").textContent = "Pause";
  const tick = () => {
    if (!state.playing) return;
    step();
    state.timer = setTimeout(tick, Math.max(40, 280 - Number($("speed").value)));
  };
  tick();
}

function onMapClick(ev) {
  if (!state.data || !state.layout || !state.mission) return;
  const rect = $("map").getBoundingClientRect();
  const { ox, oy, cell, x0, y0, x1, y1, worldRows, worldCols } = state.layout;
  const vx = Math.floor((ev.clientX - rect.left - ox) / cell);
  const vy = Math.floor((ev.clientY - rect.top - oy) / cell);
  const x = x0 + vx;
  const y = y1 - vy;
  if (x < 0 || y < 0 || x >= worldCols || y >= worldRows) return;
  if (vx < 0 || vy < 0 || x > x1 || y < y0) return;
  apply(state.mission.setGoal(y, x));
}

$("play").addEventListener("click", play);
$("step").addEventListener("click", () => {
  stop();
  step();
});
$("mode").addEventListener("change", start);
$("continuous").addEventListener("change", start);
$("map").addEventListener("click", onMapClick);
$("scenes").addEventListener("change", start);
window.addEventListener("resize", () => {
  if (state.data) drawMap(state.data);
});
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") {
    e.preventDefault();
    play();
  }
});

fetch("data/scenes.json")
  .then((r) => {
    if (!r.ok) throw new Error("missing scenes");
    return r.json();
  })
  .then((pack) => {
    state.mission = new DemoMission(pack.scenes);
    $("scenes").innerHTML = pack.scenes
      .map((s, i) => `<option value="${i}">${String(i).padStart(3, "0")}  ${s.stem}</option>`)
      .join("");
    start();
  })
  .catch((err) => {
    $("wait").textContent = "could not load demo";
    $("stStatus").textContent = "error";
    $("stNote").textContent = String(err);
  });
