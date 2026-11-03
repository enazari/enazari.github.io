/* Cars on the landing page, with the simulator, the network and the options of the car-swarm demo.
   Every car is a physical car: cars never overlap, they collide with each other and with the boxes, and a dragged box shoves them.
   Cars appear at the start of the right or the top road (or at the mouse), and from their first moment a driver takes them to the
   tunnel of one of the selected projects. Two drivers share the one world, and both see exactly the same thing: the network's
   picture of the world around the car (walls, other cars and their velocities, the route) and its own speed and last action.
   R drives by hand-written rules read off that picture; RL is a 16k-parameter diffusion network (v3 in policy/: behaviour cloning,
   DAgger, RL, distilled from the 32k model, then RL again), run in plain JavaScript (cars/net.js, no ONNX runtime). They are dragged
   onto the "ground" in the options. Nothing else steers them; the roads only shape the route, which prefers the asphalt as a GPS
   prefers roads. Guardrails line both sides of every road and close off the island: they are solid walls like the boxes, open only
   where the way out ends. The "cars" button holds the options. Loaded last. */
(async () => {
  const main = document.querySelector("main.home"), rb = main && main.querySelector(".rb");
  if (!main || (rb && !rb.querySelector("path[data-r]"))) return;                        // the roundabout is optional (build.py ROUNDABOUT)
  const load = src => new Promise((ok, no) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = no; document.head.append(s); });
  try { await Promise.all([load("cars/sim.js?v=43006601"), load("cars/net.js?v=ad2aba05")]); } catch (_) { return; }   // offline or blocked: the page simply has no cars
  // The drivers. Both get the same input per car (Sim.buildObs3): a 5 x 32 x 32 picture in the car's frame and 3 numbers.
  const DRIVERS = [
    { tag: "R", name: "Rule-based", note: "hand-written rules that read the same picture as the network: aim along the route layer, brake for car pixels ahead, a coin flip when two cars block each other", color: "#455a64" },
    { tag: "L", name: "Learned", note: "a 16k-parameter diffusion policy: copied a planner, learned to recover from its own mistakes (DAgger), then learned from the outcome with reinforcement learning (arriving, contacts, walls, getting stuck)", color: "#e8590c", file: "cars/policy_16k_rl.json?v=31df3203" },
  ];
  const nets = [];                                     // per learned driver, loaded the first time it is put on the ground
  const loadDriver = async q => nets[q] || !DRIVERS[q].file ? nets[q] : (nets[q] = PolicyNet.create(await (await fetch(DRIVERS[q].file)).json()));
  let ground = [1];                                    // the drivers on the ground (indices into DRIVERS): RL alone to start
  try { await loadDriver(1); } catch (_) { return; }

  const { CS, DT, SUB, DTC, MAXST } = Sim;
  // Page px are simulator px: the cars are 22 x 11 px, as in the demo and in training.
  const MAXCARS = 100;                 // hard limit on cars; the limit for this machine is measured, see "load" below
  const SW = 44, SD = 42, TUNLEN = 50; // tunnel bay (the notch cut into a project box, .tn in the CSS); px a car drives on inside before it is gone
  const ARR = Sim.GOAL_R, STUCK = 20;  // px from the middle of the bay that counts as arrived (18, as in training: core.py GOAL_R); seconds without progress before a car tries another tunnel
  const STEPS = 8, K = 24, HC = 8;     // denoising steps; plans drawn in the one-car view; plan length in 0.25 s steps

  // ---------------------------------------------------------------- options (the demo's, plus where cars come from)
  let paused = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let showViz = true, allMode = true, live = false, EXEC = 1, rate = 1, inspecting = false, source = 0;      // source: 0 the gates in the name (the default), 1 the roundabout, 2 the mouse, 3 rush hour (gates and roundabout, taking turns)

  // ---------------------------------------------------------------- the page as the simulator sees it
  // The cars' world is the whole page, from its top left corner and as wide as the screen, so boxes can be dragged to any edge.
  // It is kept in simulator units, the size the network was trained on (cars 22 x 11), and drawn at SCL page px per unit:
  // 1 on screens 1200 px wide or more, less on smaller ones (0.5 on a phone). The CSS sizes the roundabout and the tunnel
  // notches with the same rule (35vw = 420 px x width / 1200), so on every screen the page and the cars agree.
  let OX = 0, OY = 0, W = 0, H = 0, SCL = 1;
  let walls = [], obsWalls = [], wvel = [], owners = [], bays = [], tun = [], prev = null, mapKey = "";
  let blocked = null, tight = null, gw = 0, gh = 0, fields = [], todo = [];
  function geometry() {
    const m = main.getBoundingClientRect(), r = rb ? rb.getBoundingClientRect() : m;
    OX = -scrollX; OY = -scrollY;                                                              // the page's top left corner, in screen coordinates
    SCL = Math.min(1, Math.max(0.8, innerWidth / 1200));          // phones: at least 0.8 page px per unit, so cars stay visible (0.5 made them 13 px long)
    W = document.documentElement.clientWidth / SCL; H = (Math.max(m.bottom, r.bottom) + scrollY + 8) / SCL;
  }
  const colorCache = new WeakMap();
  function colors(e) {                                                              // a box's tunnel colour and a darker shade of it, read once per box
    let c = colorCache.get(e);
    if (!c) {
      const color = getComputedStyle(e).getPropertyValue("--c").trim() || "hsl(20,90%,48%)", m = color.match(/hsl\(\s*([\d.]+)[,\s]+([\d.]+)%[,\s]+([\d.]+)%\s*\)/);
      c = { color, dark: m ? `hsl(${m[1]},${m[2]}%,${(+m[3] * 0.8).toFixed(1)}%)` : color };
      colorCache.set(e, c);
    }
    return c;
  }
  function readPage(dt) {
    const els = [...main.querySelectorAll(".bx")];
    const rel = e => { const r = e.getBoundingClientRect(); return [(r.left - OX) / SCL, (r.top - OY) / SCL, (r.right - OX) / SCL, (r.bottom - OY) / SCL]; };
    const now = els.map(rel);
    walls = []; wvel = []; owners = []; bays = []; tun = [];
    els.forEach((e, i) => {
      const [x0, y0, x1, y1] = now[i], p = prev && prev[i];
      const v = p && dt > 0 ? [(x0 - p[0]) / dt, (y0 - p[1]) / dt] : [0, 0];     // a box being dragged shoves the cars it hits
      if (!e.classList.contains("tn")) { walls.push(now[i]); wvel.push(v); owners.push(undefined); return; }
      const k = tun.length, mx = (x0 + x1) / 2, sw = SW / 2, top = y1 - SD;
      walls.push([x0, y0, mx - sw, y1], [mx + sw, y0, x1, y1], [mx - sw, y0, mx + sw, top]);
      wvel.push(v, v, v); owners.push(k, k, k);
      bays.push([mx - sw, top, mx + sw, y1 + 8]);
      tun.push({ goal: [mx, (top + y1) / 2], bay: [mx - sw, top, mx + sw, y1], ...colors(e) });
    });
    obsWalls = walls;                                                               // the boxes; the picture adds the guardrails from their mask
    walls = walls.concat(rails);                                                    // the physics and the contact count get boxes and guardrails
    prev = now;
    const key = obsWalls.map(r => r.map(Math.round).join(",")).join(";") + "|" + Math.round(W) + "x" + Math.round(H) + "|" + rkey;
    if (key !== mapKey) { mapKey = key; todo = []; blocked = null; }                // the map changed: route fields are rebuilt, one per frame
  }
  // Route fields. The demo keeps routes 15 px from every box, which would close the narrower gaps of this page. Here a gap is closed
  // only where the car would not fit; the cells the demo would close stay open but cost 3 times as much, so routes prefer wide gaps.
  const TIGHT = 8, NARROW = 3, ROADC = 0.4;                                         // ROADC: cost of a step on the asphalt, relative to plain ground
  let roadCells = null;
  function buildFields() {                                                          // at most one Dijkstra per frame, so dragging a box never stalls the page
    if (!blocked) {
      blocked = Sim.buildBlocked(obsWalls, W, H, Sim.INFLATE); tight = Sim.buildBlocked(obsWalls, W, H, TIGHT); ({ gw, gh } = Sim.grid(W, H));
      markRails(blocked); markRails(tight);
      roadCells = asphaltCells(CS, 0);                                                        // the route hint prefers the roads; the island in the middle of the ring is closed
      const [cx0, cy0, r0] = ring;
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) if (Math.hypot((i + 0.5) * CS - cx0, (j + 0.5) * CS - cy0) < r0 - 4) blocked[j * gw + i] = tight[j * gw + i] = 1;
      todo = tun.map((_, k) => k); fields = fields.slice(0, tun.length); return;
    }
    const k = todo.shift(); if (k === undefined) return;
    const g = tun[k].goal;
    fields[k] = dijkstra(Math.min(gw - 1, Math.trunc(g[0] / CS)), Math.min(gh - 1, Math.trunc(g[1] / CS)));
  }
  function dijkstra(gx, gy) {
    const dist = new Float64Array(gw * gh).fill(1e9), hd = [], hi = [];
    const push = (d, i) => {
      hd.push(d); hi.push(i); let k = hd.length - 1;
      while (k > 0) { const p = (k - 1) >> 1; if (hd[p] <= d) break; hd[k] = hd[p]; hi[k] = hi[p]; k = p; }
      hd[k] = d; hi[k] = i;
    };
    const pop = () => {
      const top = hi[0], d = hd.pop(), i = hi.pop(), n = hd.length;
      if (n) {
        let k = 0;
        for (;;) { let c = 2 * k + 1; if (c >= n) break; if (c + 1 < n && hd[c + 1] < hd[c]) c++; if (hd[c] >= d) break; hd[k] = hd[c]; hi[k] = hi[c]; k = c; }
        hd[k] = d; hi[k] = i;
      }
      return top;
    };
    const s = gy * gw + gx; dist[s] = 0; push(0, s);
    while (hd.length) {
      const d = hd[0], idx = pop();
      if (d > dist[idx]) continue;
      const y = (idx / gw) | 0, x = idx - y * gw;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy, ni = ny * gw + nx;
        if ((!dx && !dy) || nx < 0 || ny < 0 || nx >= gw || ny >= gh || tight[ni]) continue;
        if (dx && dy && (tight[y * gw + nx] || tight[ny * gw + x])) continue;
        const nd = d + (dx && dy ? CS * 1.4142 : CS) * (blocked[ni] ? NARROW : roadCells[ni] ? ROADC : 1);
        if (nd < dist[ni]) { dist[ni] = nd; push(nd, ni); }
      }
    }
    return dist;
  }
  const ready = () => blocked && !todo.length && fields.length >= tun.length;
  function cost(f, x, y) {                                                          // route length to the tunnel from (x, y), from the nearest open ground within about 50 px; 1e9 when cut off
    if (!f) return 1e9;
    const ci = Math.trunc(x / CS), cj = Math.trunc(y / CS);
    for (let r = 2; r <= 6; r += 2) {
      let best = 1e9;
      for (let j = cj - r; j <= cj + r; j++) for (let i = ci - r; i <= ci + r; i++) if (i >= 0 && j >= 0 && i < gw && j < gh) best = Math.min(best, f[j * gw + i]);
      if (best < 1e8) return best;
    }
    return 1e9;
  }
  const reachable = (x, y) => tun.map((_, k) => k).filter(k => cost(fields[k], x, y) < 1e8);
  function nearestFree(px, py) {
    const cx = Math.min(gw - 1, Math.max(0, Math.trunc(px / CS))), cy = Math.min(gh - 1, Math.max(0, Math.trunc(py / CS)));
    for (let r = 0; r < 30; r++) {
      let best = null, bd = 1e9;
      for (let j = cy - r; j <= cy + r; j++) for (let i = cx - r; i <= cx + r; i++) {
        if (i < 0 || j < 0 || i >= gw || j >= gh || Math.max(Math.abs(i - cx), Math.abs(j - cy)) !== r || blocked[j * gw + i]) continue;
        const d = (i - cx) ** 2 + (j - cy) ** 2; if (d < bd) { bd = d; best = [i, j]; }
      }
      if (best) return [(best[0] + 0.5) * CS, (best[1] + 0.5) * CS];
    }
    return null;
  }

  // ---------------------------------------------------------------- the roads: two lanes in, round the ring, one lane out
  let routes = {}, rkey = "";
  function buildRoutes() {
    if (!rb) return;                                                                       // no roundabout: no roads, no guardrails
    const r = rb.getBoundingClientRect(), k = r.width / 612 / SCL, rx = (r.left - OX) / SCL, ry = (r.top - OY) / SCL;   // k: simulator units per unit of the drawing
    const key = [rx, ry, k, W, H].map(v => v.toFixed(1)).join();
    if (key === rkey) return; rkey = key;
    const map = (x, y) => [rx + x * k, ry + (459 - y) * k];                                 // drawing coordinates (before its vertical flip) to simulator units
    const lane = (pts, off) => pts.map((p, i) => {                                         // keep to the right-hand side of the road
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)], dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy) || 1;
      return [p[0] - dy / n * off, p[1] + dx / n * off];
    });
    const road = (name, rev) => {
      const p = rb.querySelector(`path[data-r="${name}"]`), L = p.getTotalLength(), pts = [];
      for (let s = 0; s <= L; s += 3) { const q = p.getPointAtLength(s); pts.push(map(q.x, q.y)); }
      return lane(rev ? pts.reverse() : pts, 11 * k);
    };
    const inside = p => p[0] > 16 && p[1] > 16 && p[0] < W - 16 && p[1] < H - 16;
    const C = map(297, 226), R = 137 * k;                                                   // ring centre and the outer lane
    const out = road("out-left", true).filter(p => p[0] > rx + 60 * k);           // leaves the ring, ends where the road fades out
    for (const name of ["in-right", "in-top"]) {
      let inn = road(name, name === "in-right"); const s0 = inn.findIndex(inside); inn = inn.slice(Math.max(0, s0));   // starts inside the page
      const a = inn[inn.length - 1], b = out[0];
      let a0 = Math.atan2(a[1] - C[1], a[0] - C[0]), a1 = Math.atan2(b[1] - C[1], b[0] - C[0]);
      while (a1 >= a0) a1 -= 2 * Math.PI;                                                   // anticlockwise on screen, as on a right-hand-traffic roundabout
      const ring = []; for (let t = a0; t > a1; t -= 3 / R) ring.push([C[0] + R * Math.cos(t), C[1] + R * Math.sin(t)]);
      const pts = inn.concat(ring, out), cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      routes[name] = { pts, cum, ring0: cum[inn.length], merge: ring[0] || a };
    }
    // The asphalt: discs along every road of the drawing, and the ring as a band. Routes prefer it, as a GPS prefers roads.
    const E = out[out.length - 1];
    discs = []; let gap = null;
    for (const p of rb.querySelectorAll("g[mask] path[stroke-width]")) {                    // the roads (not the dashed centre lines, which have no width)
      const L = p.getTotalLength(), rr = +p.getAttribute("stroke-width") / 2 * k, exit = p.dataset.r === "out-left";
      let last = null;
      for (let s = 0; s <= L; s += 4) {
        const q = p.getPointAtLength(s), [px, py] = map(q.x, q.y);
        if (exit && px < E[0]) { last = [px, py]; continue; }                             // the faded stub at the end of the way out is plain ground
        if (exit && !gap && last) { const dx = last[0] - px, dy = last[1] - py, n = Math.hypot(dx, dy) || 1; gap = { x: px, y: py, dx: dx / n, dy: dy / n, r: rr }; }
        discs.push([px, py, rr]);
      }
    }
    ring = [C[0], C[1], 94 * k, 152 * k];
    asphaltGrid = {};                                                                      // grids are made on demand for the current size
    buildRails(rx, ry, k, gap);
  }

  // ---------------------------------------------------------------- guardrails
  // Off-road ground within RAILW of the asphalt, and the whole island, on a RG grid over the drawing, merged into rectangles:
  // solid walls for the physics, a wall for the route fields, and cells the network sees in its map like those of a box.
  // At the end of the way out (gap) the rails stop, so the cars can leave the roundabout there and only there.
  const RG = 4, RAILW = 12, RAIL_TIGHT = 5;
  let rails = [], railMask = null, rnx = 0, rny = 0, rx0 = 0, ry0 = 0, railG = null;
  function onRoad(x, y, grow) {
    const [cx0, cy0, r0, r1] = ring, d = Math.hypot(x - cx0, y - cy0);
    if (d < r0) return false;
    if (d < r1 + grow) return true;
    for (const [px, py, r] of discs) {
      const R = r + grow, dx = x - px; if (dx > R || dx < -R) continue;
      const dy = y - py; if (dy > R || dy < -R) continue;
      if (dx * dx + dy * dy < R * R) return true;
    }
    return false;
  }
  function buildRails(rx, ry, k, gap) {
    const open = (x, y) => gap && (x - gap.x) * gap.dx + (y - gap.y) * gap.dy > -2 && Math.hypot(x - gap.x, y - gap.y) < gap.r + RAILW + 8;
    const [cx0, cy0, r0] = ring;
    rx0 = Math.max(0, rx); ry0 = Math.max(0, ry);
    const x1 = Math.min(W, rx + 612 * k), y1 = Math.min(H, ry + 459 * k);
    rnx = Math.max(0, Math.ceil((x1 - rx0) / RG)); rny = Math.max(0, Math.ceil((y1 - ry0) / RG));
    const m = new Uint8Array(rnx * rny);
    for (let j = 0; j < rny; j++) for (let i = 0; i < rnx; i++) {
      const x = rx0 + (i + 0.5) * RG, y = ry0 + (j + 0.5) * RG;
      if (Math.hypot(x - cx0, y - cy0) < r0) m[j * rnx + i] = 1;                            // the island
      else if (!open(x, y) && !onRoad(x, y, 0) && onRoad(x, y, RAILW)) m[j * rnx + i] = 1;
    }
    railMask = m; rails = [];
    const used = new Uint8Array(m.length);                                                 // greedy merge into as few rectangles as possible
    for (let j = 0; j < rny; j++) for (let i = 0; i < rnx; i++) {
      if (!m[j * rnx + i] || used[j * rnx + i]) continue;
      let i1 = i; while (i1 + 1 < rnx && m[j * rnx + i1 + 1] && !used[j * rnx + i1 + 1]) i1++;
      let j1 = j;
      grow: while (j1 + 1 < rny) { for (let q = i; q <= i1; q++) if (!m[(j1 + 1) * rnx + q] || used[(j1 + 1) * rnx + q]) break grow; j1++; }
      for (let jj = j; jj <= j1; jj++) for (let q = i; q <= i1; q++) used[jj * rnx + q] = 1;
      rails.push([rx0 + i * RG, ry0 + j * RG, rx0 + (i1 + 1) * RG, ry0 + (j1 + 1) * RG]);
    }
    drawRails(rx, ry, k, open);
  }
  function markRails(b) {                                                                  // route-field cells closed by a guardrail (a car centre keeps RAIL_TIGHT from it)
    for (const [a0, b0, a1, b1] of rails)
      for (let j = Math.max(0, Math.floor((b0 - RAIL_TIGHT) / CS)); j <= Math.min(gh - 1, Math.floor((b1 + RAIL_TIGHT) / CS)); j++)
        for (let i = Math.max(0, Math.floor((a0 - RAIL_TIGHT) / CS)); i <= Math.min(gw - 1, Math.floor((a1 + RAIL_TIGHT) / CS)); i++) {
          const cx = (i + 0.5) * CS, cy = (j + 0.5) * CS;
          if (cx > a0 - RAIL_TIGHT && cx < a1 + RAIL_TIGHT && cy > b0 - RAIL_TIGHT && cy < b1 + RAIL_TIGHT) b[j * gw + i] = 1;
        }
  }
  function drawRails(rx, ry, k, open) {                                                    // a thin rail along the edge of the asphalt, drawn in the roundabout's own SVG
    const back = (x, y) => [((x - rx) / k).toFixed(1), (459 - (y - ry) / k).toFixed(1)]; // simulator units to drawing coordinates
    const x1 = rx + 612 * k, y1 = ry + 459 * k, inBox = (x, y) => x > rx && y > ry && x < x1 && y < y1;
    const [cx0, cy0, r0, r1] = ring;
    let d = "";
    const dot = (x, y) => { if (inBox(x, y) && !open(x, y)) { const [a, b] = back(x, y); d += `M${a} ${b}h0`; } };
    for (const [px, py, r] of discs) {
      const n = Math.ceil(2 * Math.PI * r / 1.6);
      for (let t = 0; t < n; t++) {
        const a = t * 2 * Math.PI / n, ux = Math.cos(a), uy = Math.sin(a);
        if (!onRoad(px + ux * (r + 1), py + uy * (r + 1), 0) && Math.hypot(px + ux * r - cx0, py + uy * r - cy0) > r0) dot(px + ux * r, py + uy * r);
      }
    }
    for (const R of [r0, r1]) {
      const n = Math.ceil(2 * Math.PI * R / 1.6);
      for (let t = 0; t < n; t++) {
        const a = t * 2 * Math.PI / n, ux = Math.cos(a), uy = Math.sin(a);
        if (R === r0 || !onRoad(cx0 + ux * (R + 1), cy0 + uy * (R + 1), 0)) dot(cx0 + ux * R, cy0 + uy * R);
      }
    }
    if (!railG) {
      railG = document.createElementNS("http://www.w3.org/2000/svg", "path");
      railG.setAttribute("fill", "none"); railG.setAttribute("stroke-linecap", "round"); railG.setAttribute("class", "rail");
      rb.querySelector("svg g[transform]").append(railG);
    }
    railG.setAttribute("stroke-width", (2.2 / k).toFixed(2)); railG.setAttribute("d", d);
  }
  let discs = [], ring = [0, 0, 0, 0], asphaltGrid = {};
  // cells of a grid (cell size cs) whose centre is within `grow` px of the asphalt
  function asphaltCells(cs, grow) {
    const key = cs + "|" + grow + "|" + W + "x" + H;
    if (asphaltGrid[key]) return asphaltGrid[key];
    const nx = Math.ceil(W / cs), ny = Math.ceil(H / cs), g = new Uint8Array(nx * ny);
    const mark = (x0, y0, x1, y1, test) => {
      for (let j = Math.max(0, Math.floor(y0 / cs)); j <= Math.min(ny - 1, Math.floor(y1 / cs)); j++)
        for (let i = Math.max(0, Math.floor(x0 / cs)); i <= Math.min(nx - 1, Math.floor(x1 / cs)); i++)
          if (!g[j * nx + i] && test((i + 0.5) * cs, (j + 0.5) * cs)) g[j * nx + i] = 1;
    };
    for (const [px, py, r] of discs) { const R = r + grow; mark(px - R, py - R, px + R, py + R, (cx, cy) => (cx - px) ** 2 + (cy - py) ** 2 < R * R); }
    const [cx0, cy0, r0, r1] = ring, R1 = r1 + grow, R0 = Math.max(0, r0 - grow);
    mark(cx0 - R1, cy0 - R1, cx0 + R1, cy0 + R1, (cx, cy) => { const d = Math.hypot(cx - cx0, cy - cy0); return d < R1 && d > R0; });
    asphaltGrid[key] = g; return g;
  }

  // ---------------------------------------------------------------- cars: one simulator for all of them
  const N = MAXCARS;
  const S = { X: new Float64Array(N), Y: new Float64Array(N), TH: new Float64Array(N), V: new Float64Array(N), state: new Int32Array(N), grp: new Int8Array(N) };   // state 1 driving, 2 inside a tunnel
  const B = { bx: new Float64Array(N), by: new Float64Array(N), om: new Float64Array(N) };
  // t: the tunnel it heads for; q: its driver (index into DRIVERS); d: a learned driver (q > 0)
  const car = Array.from({ length: N }, () => ({ t: 0, q: 1, d: true, chunk: new Float32Array(16), ptr: 99, exec: 1, prevA: [0, 0], since: 0, best: 1e9, tun: 0, alpha: 1, trail: [] }));
  const goals = car.map(() => [0, 0]), prevA = car.map(c => c.prevA);
  const vt = new Float64Array(N), dl = new Float64Array(N);
  let spawnT = 1, simT = 0;
  const mk = () => ({ arrived: 0, tripSum: 0, cc: 0, cd: 0, rerouted: 0 });
  const stats = { by: DRIVERS.map(mk), t0: 0, cap: 4, load: 0 };                             // by: one set of counters per driver, since t0 (s)
  const st = k => stats.by[car[k].q];
  const contact = { cc: new Set(), cd: new Set() };
  const sparks = [];                                                                         // impact rings: [x, y, strength, age]
  const mouse = { x: -1, y: -1, in: false };
  let held = null;                                                                           // a car being carried by the pointer: {k, id, samples}

  const freeSlot = () => S.state.findIndex(s => s === 0);
  function active() { let n = 0; for (let k = 0; k < N; k++) if (S.state[k]) n++; return n; }
  const clear = (x, y, g, r) => { for (let j = 0; j < N; j++) if (S.state[j] && (g === undefined || S.grp[j] === g) && Math.hypot(S.X[j] - x, S.Y[j] - y) < r) return false; return true; };
  function place(k, x, y, th, v, g, props) {
    S.X[k] = x; S.Y[k] = y; S.TH[k] = th; S.V[k] = v; S.state[k] = 1; S.grp[k] = g; B.bx[k] = B.by[k] = B.om[k] = 0;
    Object.assign(car[k], { ptr: 99, exec: EXEC, since: 0, best: 1e9, tun: 0, t0: simT, at: 0, ax: undefined, trail: [], stranded: false }, props);
    car[k].d = car[k].q > 0;
    car[k].prevA[0] = car[k].prevA[1] = 0; delete allViz[k];
  }
  function spawn() {
    if (!tun.length || !ready()) return;
    if (source === 2) return spawnAtMouse();
    if (source === 0) return spawnAtGates();
    if (!rb && (source === 1 || source === 3)) return spawnAtGates();
    if (source === 3) { rush ^= 1; return rush ? spawnAtGates() : spawnAtRing(); }        // rush hour: cars from the name and the roundabout, so they meet head-on
    spawnAtRing();
  }
  let rush = 0;
  function spawnAtRing() {
    const rt = routes[Math.random() < 0.5 ? "in-right" : "in-top"];                         // a new car appears at the start of the right or the top road; the driver has it from there
    if (!rt) return;
    const [x, y] = rt.pts[0];
    if (obsWalls.some(w => Sim.rectDist(x, y, w[0], w[1], w[2], w[3]) < 12)) return;          // a box lies on the road there
    spawnAt([x, y], Math.atan2(rt.pts[3][1] - y, rt.pts[3][0] - x));
  }
  let turn = 0;                                                                              // the drivers on the ground take turns: each gets every n-th new car
  const nextDriver = () => ground[turn % ground.length];
  function spawnAt(p, th) {                                                                  // one car, of the driver whose turn it is
    if (!ground.length || active() + 1 > stats.cap || !clear(p[0], p[1], undefined, 30)) return;
    const k = freeSlot(); if (k < 0) return;
    const t = (Math.random() * tun.length) | 0, [x, y] = p, rp = Sim.routePoints(fields[t], gw, gh, x, y, tun[t].goal[0], tun[t].goal[1]);
    place(k, x, y, th === undefined ? Math.atan2(rp[1][1] - y, rp[1][0] - x) : th, 0, 0, { t, q: nextDriver(), alpha: 0 }); turn++;
  }
  // ---------------------------------------------------------------- the gates: the arches of the h and the n in the name
  // A car rolls out from under the arch (state 3, drawn clipped so it seems to come from inside the letter, like the tunnel
  // animation in reverse) and becomes a car of the world, driven by its driver, once it is clear of the name's box.
  const h1 = main.querySelector(".intro h1"), gateEls = h1 ? [...h1.querySelectorAll(".gate")] : [], blEl = h1 && h1.querySelector(".bl");
  const glyphs = {}, gcv = document.createElement("canvas"), gctx = gcv.getContext("2d", { willReadFrequently: true });
  function glyph(ch, font, size) {                                     // the arch of a glyph, measured once per font: counter [a, b] from the pen position, arch height over the baseline (px)
    const key = ch + "|" + font;
    if (glyphs[key]) return glyphs[key];
    const w = Math.ceil(size * 1.5), h = Math.ceil(size * 1.5), base = Math.round(size * 1.1);
    gcv.width = w; gcv.height = h; gctx.font = font; gctx.fillStyle = "#000"; gctx.textBaseline = "alphabetic"; gctx.fillText(ch, 0, base);
    const px = gctx.getImageData(0, 0, w, h).data, ink = (x, y) => px[(y * w + x) * 4 + 3] > 100;
    const row = Math.round(base - size * 0.12);                        // just above the baseline: the two stems and the opening between them
    let a = -1, b = -1, inRun = false, runs = [];
    for (let x = 0; x < w; x++) { const i = ink(x, row); if (i && !inRun) runs.push([x, x]); if (i) runs[runs.length - 1][1] = x; inRun = i; }
    if (runs.length >= 2) { a = runs[0][1] + 1; b = runs[1][0]; }
    let arch = 0;
    if (a >= 0) { const cx = Math.round((a + b) / 2); for (let y = row; y > 0; y--) if (ink(cx, y)) { arch = base - y - 1; break; } }
    return glyphs[key] = a >= 0 && arch > 0 ? { a, b, arch } : null;
  }
  function gates() {                                                   // where the gates are now, in simulator units (the name box can be dragged)
    if (!h1 || !blEl) return [];
    const cs = getComputedStyle(h1), size = parseFloat(cs.fontSize), font = `${cs.fontStyle} ${cs.fontWeight} ${size}px ${cs.fontFamily}`;
    const base = blEl.getBoundingClientRect().bottom, box = h1.getBoundingClientRect(), out = [];
    gateEls.forEach((e, id) => {
      const g = glyph(e.textContent, font, size); if (!g) return;
      const r = e.getBoundingClientRect(), x = r.left + (g.a + g.b) / 2;
      out.push({ id, x: (x - OX) / SCL, top: (base - g.arch - OY) / SCL, hw: (g.b - g.a) / 2 / SCL, bottom: (box.bottom - OY) / SCL });
    });
    return out;
  }
  let gateNow = [];
  function spawnAtGates() {
    const gs = gateNow; if (!gs.length || !ground.length || active() + 1 > stats.cap) return;
    const free = gs.filter(g => !S.state.some((st, j) => st === 3 && car[j].gate === g.id));   // one car at a time per gate: a free one
    if (!free.length) return;
    const g = free[(Math.random() * free.length) | 0], t = (Math.random() * tun.length) | 0;
    const k = freeSlot(); if (k < 0) return;
    place(k, g.x, g.top, Math.PI / 2, 0, 0, { t, q: nextDriver(), alpha: 1, gate: g.id, em: 0, out: false }); turn++;
    S.state[k] = 3;
  }
  // Gate rule, so that a car never overlaps another: it rolls down inside the name box (where no other car can be) until its front
  // reaches the box's bottom edge, waits there until no driving car is within GATE_CLEAR of the strip it will roll out over, then rolls
  // out at ROLL px/s and is handed to its driver standing still (as in training, env.py _spawn: 30 px clear, speed 0). GATE_CLEAR is the
  // contact distance (12) plus how far a car at top speed can come while it rolls out (29 px at ROLL), so nobody can reach it meanwhile.
  const ROLL = 90, HALF_L = 13, GATE_CLEAR = Math.max(30, 12 + Sim.VMAX * (HALF_L + 16) / ROLL + 4);
  function emerge() {                                                  // every 0.05 s: cars under the arches roll on; out of the box they are cars of the world
    for (let k = 0; k < N; k++) if (S.state[k] === 3) {
      const c = car[k], g = gateNow.find(q => q.id === c.gate);
      if (!g) { S.state[k] = 0; continue; }
      const wait = g.bottom - HALF_L - g.top, end = g.bottom + 16 - g.top;   // distances below the arch top: front at the box edge; handover point
      if (c.em < wait) { c.em = Math.min(wait, c.em + 45 * DT); }
      else if (!c.out) {
        let busy = false;
        for (let j = 0; j < N && !busy; j++) if ((S.state[j] === 1 || (S.state[j] === 3 && car[j].out)) && S.grp[j] === S.grp[k]) {   // driving cars, and cars rolling out of the other gate
          const dy = Math.max(0, Math.max(g.bottom - HALF_L - S.Y[j], S.Y[j] - (g.bottom + 16)));   // distance from car j to the strip x = g.x, y in [box edge - 13, handover]
          busy = Math.hypot(S.X[j] - g.x, dy) < GATE_CLEAR;
        }
        if (!busy && !Sim.cardHit(g.x, g.bottom + 16, Math.PI / 2, walls, W, H, 0)) c.out = true;   // clear, and no box lies in front of the gate
      }
      if (c.out) c.em = Math.min(end, c.em + ROLL * DT);
      S.X[k] = g.x; S.Y[k] = g.top + c.em; S.TH[k] = Math.PI / 2;
      if (c.em >= end) { S.state[k] = 1; S.V[k] = 0; c.t0 = simT; }               // handed to its driver standing still, as in training
    }
  }
  function spawnAtMouse() {                                                                  // as in the demo: cars appear at the mouse
    if (!mouse.in) return;
    const p = nearestFree(mouse.x + (Math.random() - 0.5) * 6, mouse.y + (Math.random() - 0.5) * 6);
    if (p) spawnAt(p);
  }

  // ---------------------------------------------------------------- drivers
  // Both drivers get the same input (Sim.buildObs3, policy/core.py build_obs3): the picture of the world around the car on the
  // route field of its tunnel, the boxes and the guardrails. The rule driver below uses nothing else.
  const railObs = () => railMask && { mask: railMask, rnx, rny, rx0, ry0 };
  // The rule driver (policy/policy.py rule_pic_act, prio "coin"): it aims at the open cell with the lowest route value it can
  // reach within 25 px of its own cell (walking through open cells, so it never jumps a guardrail), slows down near the goal when
  // the goal is in the picture, and brakes for car pixels in its lane ahead. The picture says nothing about who goes first, so a
  // blocked car goes anyway with chance COIN_P per decision (a coin-flip tie-break).
  const HALF = 15.5, EF = Float64Array.from({ length: 32 }, (_, q) => (q - HALF) * 6 + 40), EL = Float64Array.from({ length: 32 }, (_, r) => (r - HALF) * 6);
  const C0 = [16, 9], COIN_P = 0.1, VMAX = Sim.VMAX;                                         // C0: (row, column) of the car's own cell
  const rpic = new Float32Array(5 * 1024), rvec = new Float32Array(3);
  function routeTarget(route) {                                                              // policy.py _route_target
    const open = c => route[c] < 0.999;
    let [r0, q0] = C0;
    if (!open(r0 * 32 + q0)) {                                                               // inside a closed cell: start from the nearest open one
      let bd = Infinity, bc = -1;
      for (let c = 0; c < 1024; c++) if (open(c)) { const d = ((c >> 5) - r0) ** 2 + ((c & 31) - q0) ** 2; if (d < bd) { bd = d; bc = c; } }
      if (bc < 0) return [r0, q0 + 4];
      r0 = bc >> 5; q0 = bc & 31;
    }
    const best = new Map([[r0 * 32 + q0, 0]]), order = [r0 * 32 + q0];
    let frontier = [r0 * 32 + q0];
    while (frontier.length) {
      const nxt = [];
      for (const c of frontier) {
        const r = c >> 5, q = c & 31;
        for (let dr = -1; dr <= 1; dr++) for (let dq = -1; dq <= 1; dq++) {
          if (!dr && !dq) continue;
          const a = r + dr, b = q + dq, key = a * 32 + b;
          if (a < 0 || b < 0 || a >= 32 || b >= 32 || !open(key)) continue;
          const L = best.get(c) + 6 * (dr && dq ? 1.414 : 1), old = best.get(key);
          if (L <= 25 && L < (old === undefined ? 1e9 : old)) { if (old === undefined) order.push(key); best.set(key, L); nxt.push(key); }
        }
      }
      frontier = nxt;
    }
    let bc = order[0]; for (const c of order) if (route[c] < route[bc]) bc = c;
    return [bc >> 5, bc & 31];
  }
  function ruleDrive(k) {
    Sim.buildObs3(k, S, obsWalls, W, H, fields[car[k].t], gw, gh, prevA, railObs(), rpic, rvec);
    const cars = rpic.subarray(1024, 2048), route = rpic.subarray(4096, 5120);
    const [rr, qq] = routeTarget(route), tf = EF[qq], tl = EL[rr], ld = Math.max(15, Math.hypot(tf, tl)), err = Math.atan2(tl, tf);
    const d = Math.max(-MAXST, Math.min(MAXST, Math.atan(2 * Sim.WB * Math.sin(err) / ld)));
    let vd = Math.max(22, VMAX / (1 + 2.5 * Math.abs(d)));
    let gi = -1, gv = 9; for (let c = 0; c < 1024; c++) if (route[c] < 0.999 && route[c] < gv) { gv = route[c]; gi = c; }   // the goal: the lowest route value in the picture
    if (gi >= 0) {
      const gr = gi >> 5, gq = gi & 31;
      if (gr > 0 && gr < 31 && gq > 0 && gq < 31) { const tg = Math.hypot(EF[gq], EL[gr]); if (tg < 50) vd = Math.min(vd, 12 + (VMAX - 12) * tg / 50); }
    }
    let fedge = Infinity;                                                                    // car pixels in its lane (rows 3 px either side), 0..50 px ahead
    for (let r = 0; r < 32; r++) if (Math.abs(EL[r]) <= 3) for (let q = 0; q < 32; q++) if (EF[q] > 0 && EF[q] < 50 && cars[r * 32 + q] > 0.5) fedge = Math.min(fedge, EF[q]);
    if (fedge < Infinity) vd = Math.min(vd, Math.max(0, fedge - 16) * 1.5);
    if (vd < 5 && Math.random() < COIN_P) vd = Math.max(22, VMAX / (1 + 2.5 * Math.abs(d)));
    vt[k] = vd; dl[k] = d; prevA[k][0] = Math.max(-1, Math.min(1, (vd - 22.5) / 52.5)); prevA[k][1] = d / MAXST;
  }
  let viz = null, focus = -1, allT = 0, forceAll = false;
  const allViz = {};                                                                         // car -> its plan with every denoising step, for the denoising view
  const driving = k => S.state[k] === 1;
  async function decide() {                                                                  // every 0.25 s: D cars whose plan is used up denoise a new one, all in one batch
    vt.fill(0); dl.fill(0);
    const rd = ready();
    for (let k = 0; k < N; k++) if (driving(k)) {
      const c = car[k];
      if (rd && cost(fields[c.t], S.X[k], S.Y[k]) >= 1e8) {                                  // a moved box cut its tunnel off: switch to the nearest one it can reach
        let best = 1e8; for (const t of reachable(S.X[k], S.Y[k])) { const v = cost(fields[t], S.X[k], S.Y[k]); if (v < best) { best = v; c.t = t; c.best = 1e9; c.ptr = 99; } }
      }
      goals[k] = tun[c.t].goal;
      c.stranded = rd && cost(fields[c.t], S.X[k], S.Y[k]) >= 1e8;                           // no way to any tunnel from here: it waits (brakes) until one opens
      if (!c.d && fields[c.t] && !c.stranded) ruleDrive(k);
    }
    const need = [];
    for (let k = 0; k < N; k++) if (driving(k) && car[k].d && !car[k].stranded && car[k].ptr >= (car[k].exec || EXEC) && fields[car[k].t] && !(held && held.k === k)) need.push(k);
    if (focus < 0 || !driving(focus) || !car[focus].d) focus = S.state.findIndex((s, k) => s === 1 && car[k].d);
    const now = performance.now();
    const vizNow = showViz && !allMode && !inspecting && focus >= 0 && need.includes(focus) && (!viz || live || now - viz.t0 > 2500);
    if (need.length) {
      const refreshAll = showViz && allMode && (forceAll || live || !allT || now - allT > 2500);   // one shared cycle: all cars denoise together
      if (refreshAll) { allT = performance.now(); for (const k in allViz) delete allViz[k]; }
      for (let q = 1; q < DRIVERS.length; q++) {                                             // one batch per learned driver on the ground
        const net = nets[q], mine = need.filter(k => car[k].q === q);
        if (!net || !mine.length) continue;
        const rows = mine.map(k => ({ k, copies: vizNow && k === focus ? K : 1 }));
        const n = rows.reduce((a, r) => a + r.copies, 0), P = 5 * 1024;
        const crop = new Float32Array(n * P), vec = new Float32Array(n * 3), noise = new Float32Array(n * 16);
        let b = 0; const first = {};
        for (const { k, copies } of rows) {
          Sim.buildObs3(k, S, obsWalls, W, H, fields[car[k].t], gw, gh, prevA, railObs(), crop.subarray(b * P, (b + 1) * P), vec.subarray(b * 3, b * 3 + 3));
          first[k] = b;
          for (let c = 1; c < copies; c++) { crop.copyWithin((b + c) * P, b * P, (b + 1) * P); vec.copyWithin((b + c) * 3, b * 3, (b + 1) * 3); }
          b += copies;
        }
        for (let i = 0; i < noise.length; i++) noise[i] = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(6.283185 * Math.random());
        const out = net.run(crop, vec, noise, n);
        for (const { k } of rows) {
          const c = car[k], f = first[k];
          c.chunk.set(out.chunk.subarray(f * 16, f * 16 + 16)); c.ptr = 0; c.exec = EXEC;
        }
        const grab = (k, copies) => { const f = first[k];
          return { car: k, F: [S.X[k], S.Y[k], S.TH[k], S.V[k]], t0: performance.now(), memo: new Map(),
            xs: out.xs.slice(f * (STEPS + 1) * 16, (f + copies) * (STEPS + 1) * 16), xhat: out.xhat.slice(f * STEPS * 16, (f + copies) * STEPS * 16) }; };
        if (refreshAll) for (const { k } of rows) allViz[k] = grab(k, 1);
        if (vizNow && first[focus] !== undefined) viz = grab(focus, K);
      }
    }
    for (let k = 0; k < N; k++) if (driving(k) && car[k].d && !car[k].stranded && car[k].ptr < STEPS) {
      const c = car[k];
      vt[k] = c.chunk[c.ptr * 2] * 52.5 + 22.5; dl[k] = c.chunk[c.ptr * 2 + 1] * MAXST; c.ptr++;
      c.prevA[0] = Math.max(-1, Math.min(1, (vt[k] - 22.5) / 52.5)); c.prevA[1] = dl[k] / MAXST;
    }
  }
  function afterDecision() {                                                                 // every 0.25 s: progress, contacts
    for (let k = 0; k < N; k++) if (driving(k)) {
      const c = car[k], fv = cost(fields[c.t], S.X[k], S.Y[k]);
      if (fv < c.best - 15) { c.best = fv; c.since = 0; }
      else if ((c.since += DTC) > STUCK) {                                                  // stuck: try another tunnel it can reach (a new plan often frees it)
        const ok = reachable(S.X[k], S.Y[k]).filter(t => t !== c.t);
        if (ok.length) c.t = ok[(Math.random() * ok.length) | 0];
        c.since = 0; c.best = 1e9; c.ptr = 99; st(k).rerouted++;
      }
      if ((held && held.k === k) || c.ax === undefined || Math.hypot(S.X[k] - c.ax, S.Y[k] - c.ay) > 20) { c.ax = S.X[k]; c.ay = S.Y[k]; c.at = 0; } else c.at += DTC;   // red ring after 10 s in one spot
    }
    const liveIdx = []; for (let i = 0; i < N; i++) if (driving(i)) liveIdx.push(i);
    for (let u = 0; u < liveIdx.length; u++) {
      const i = liveIdx[u];
      const inBay = bays.some(b => S.X[i] > b[0] && S.X[i] < b[2] && S.Y[i] > b[1] && S.Y[i] < b[3]);   // inside a bay a car passes the walls, so that is not a contact
      const hitc = !inBay && Sim.cardHit(S.X[i], S.Y[i], S.TH[i], walls, W, H, 0);
      if (hitc && !contact.cd.has(i)) { st(i).cd++; contact.cd.add(i); } else if (!hitc) contact.cd.delete(i);
      for (let v = u + 1; v < liveIdx.length; v++) {
        const j = liveIdx[v];
        if (S.grp[i] !== S.grp[j]) continue;
        const key = i * 1000 + j, hit = Math.abs(S.X[i] - S.X[j]) < 40 && Math.abs(S.Y[i] - S.Y[j]) < 40 && Sim.carDist(S.X[i], S.Y[i], S.TH[i], S.X[j], S.Y[j], S.TH[j]) < 12;
        if (hit && !contact.cc.has(key)) { st(i).cc++; if (car[i].q !== car[j].q) st(j).cc++; contact.cc.add(key); } else if (!hit) contact.cc.delete(key);
      }
    }
  }
  function arrivals() {
    for (let k = 0; k < N; k++) if (driving(k) && !(held && held.k === k) && Math.hypot(S.X[k] - goals[k][0], S.Y[k] - goals[k][1]) < ARR) {
      const c = car[k], t = tun[c.t], g = t.goal; S.state[k] = 2; c.tun = 0;
      st(k).arrived++; st(k).tripSum += simT - c.t0;
      // Inside the bay every car drives straight up through the back wall, keeping its own x, at one speed (top speed): an arrived
      // car never slows down in front of the next one and never crosses another's path, so they cannot overlap.
      c.tv = Sim.VMAX; c.px = S.X[k] - g[0]; c.py = S.Y[k] - g[1]; c.pth = S.TH[k];
      c.park = (S.Y[k] - (t.bay[1] - HALF_L - 2)) / c.tv; S.V[k] = 0;
    }
  }

  // ---------------------------------------------------------------- drawing
  const cv = document.createElement("canvas"), ctx = cv.getContext("2d");
  cv.className = "cars"; cv.setAttribute("aria-hidden", "true"); document.body.append(cv);
  // The canvas covers the whole page and the browser recomposites all of it every frame. On a very large screen with a weak
  // graphics chip that alone costs frames (measured: 45 fps with the cars shown, 60 hidden, on a scaled 3440 px screen with
  // integrated graphics). So the canvas keeps to a pixel budget, and drops its resolution further, a step at a time,
  // while the page runs below 56 frames/s (a 60 Hz screen that is keeping up shows 59-61).
  const PIXBUDGET = 4e6;
  let dpr = 1, quality = 1;
  function sizeCanvas() {
    const pw = W * SCL, ph = H * SCL;                                                        // page px
    dpr = Math.min(1.5, devicePixelRatio || 1, Math.sqrt(PIXBUDGET / (pw * ph))) * quality;   // above 1.5 the cars look the same but the canvas costs far more to repaint
    dpr = Math.max(0.4, Math.round(dpr * 20) / 20);                                          // in steps, so a page that grows by a pixel does not resize the canvas
    if (cv.width !== Math.round(pw * dpr) || cv.height !== Math.round(ph * dpr)) { cv.width = Math.round(pw * dpr); cv.height = Math.round(ph * dpr); cv.style.width = pw + "px"; cv.style.height = ph + "px"; }
    cv.style.top = "0px";
  }
  function body(x, y, th, color, alpha, rev, g = ctx) {                                               // a car: the 22 x 11 px drawing scaled to the physics body (3 circles of radius 6 at -7, 0, +7: 26 x 12 px)
    g.save(); g.translate(x, y); g.rotate(th); g.scale(26 / 22, 12 / 11);
    g.globalAlpha = 0.22 * alpha; g.fillStyle = "#000"; g.beginPath(); g.roundRect(-9.5, -3.5, 22, 11, 3); g.fill();
    g.globalAlpha = alpha; g.fillStyle = color; g.beginPath(); g.roundRect(-11, -5.5, 22, 11, 3); g.fill();
    g.fillStyle = "rgba(20,30,40,.65)"; g.fillRect(2, -4, 4.5, 8); g.fillRect(-6.5, -3.7, 2.5, 7.4);
    g.fillStyle = "#fff6b0"; g.fillRect(9.2, -4.5, 1.8, 2); g.fillRect(9.2, 2.5, 1.8, 2);
    g.fillStyle = rev ? "#fff" : "#7a1a1a"; g.fillRect(-11, -4.5, 1.8, 2); g.fillRect(-11, 2.5, 1.8, 2);
    g.restore();
  }
  function rollout(F, chunk) {                                                               // where a plan takes the car (the simulator's own car model)
    let [x, y, th, v] = F; const pts = [[x, y]];
    for (let h = 0; h < HC; h++) {
      const t = Math.max(-1, Math.min(1, chunk[2 * h])) * 52.5 + 22.5, d = Math.max(-1, Math.min(1, chunk[2 * h + 1])) * MAXST;
      for (let k = 0; k < SUB; k++) { [x, y, th, v] = Sim.dyn(x, y, th, v, t, d); if (k === 2 || k === SUB - 1) pts.push([x, y]); }
    }
    return pts;
  }
  function path(v, which, i) {                                                               // the points of one plan, worked out once and kept with it
    const key = which + i;
    let pts = v.memo.get(key);
    if (!pts) { pts = rollout(v.F, (which === "x" ? v.xs : v.xhat).subarray(i * 16, i * 16 + 16)); v.memo.set(key, pts); }
    return pts;
  }
  const line = pts => { ctx.beginPath(); pts.forEach(([px, py], i) => i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)); ctx.stroke(); };
  let hudText = "";
  function drawOne(v, n, single) {                                                           // the plans of one denoising, animated step by step
    const el = performance.now() - v.t0, step = Math.min(STEPS, Math.floor(el / (live ? 20 : 95)));
    const fade = inspecting || live ? 1 : el > 2100 ? Math.max(0, 1 - (el - 2100) / 400) : 1;
    if (fade <= 0 || !driving(v.car)) return false;
    const tc = tun[car[v.car].t], col = tc.color, ex = car[v.car].exec || EXEC;
    if (single) hudText = (step < STEPS ? `denoising step ${step}/${STEPS}   ${n} samples` : "plan chosen: sample #1") + (live ? `   bold = the first ${ex} step${ex > 1 ? "s" : ""} it drives` : "");
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (n === 1) for (let s = 0; s < step; s++) {                                             // all-cars view: the earlier denoising steps of the plan as a fading trail
      ctx.strokeStyle = col; ctx.globalAlpha = (0.12 + 0.3 * s / STEPS) * fade; ctx.lineWidth = 1.2; line(path(v, "x", s));
    }
    for (let c = n - 1; c >= 0; c--) {
      const pts = path(v, "x", c * (STEPS + 1) + step), main0 = c === 0 && step === STEPS;
      ctx.strokeStyle = col;
      ctx.globalAlpha = (live ? (c === 0 ? 0.3 : 0.18 + 0.12 * step / STEPS) : (main0 ? 0.55 : n === 1 ? 0.6 : 0.18 + 0.12 * step / STEPS)) * fade;
      ctx.lineWidth = live ? (c === 0 ? 1 : 1.2) : (main0 ? 1.6 : n === 1 ? 1.4 : 1.2);
      ctx.setLineDash(live ? [] : [6, 5]); line(pts); ctx.setLineDash([]);                    // slow mode: the part of the plan that is not driven is dashed
      if (c === 0) { ctx.strokeStyle = tc.dark; ctx.globalAlpha = fade; ctx.lineWidth = 7; line(pts.slice(0, 2 * ex + 1)); }   // bold: the steps the car drives
    }
    if (single && step < STEPS) {                                                            // the model's current guess of the clean plan for sample #1
      ctx.setLineDash([5, 4]); ctx.strokeStyle = ink; ctx.globalAlpha = 0.8 * fade; ctx.lineWidth = 2;
      line(path(v, "h", step)); ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
    return true;
  }
  function drawViz() {
    hudText = "";
    if (!showViz || !ground.some(q => q > 0)) return;
    if (allMode) {
      let any = false; for (const k in allViz) if (driving(+k)) any = drawOne(allViz[k], 1, false) || any;
      if (any) hudText = live ? `every car, new plan every ${EXEC * 0.25} s: bold = the first ${EXEC} step${EXEC > 1 ? "s" : ""} it drives, faint = forecast that is replaced at the next plan`
        : `every car, plan refreshed every 2.5 s: bold = the first ${EXEC} step${EXEC > 1 ? "s" : ""} it drives, dashed = the rest, not executed`;
    } else if (viz) drawOne(viz, K, true);
  }
  function drawTrails() {                                                                    // where each car really went in the last 2 s
    if (!live || !showViz) return;
    ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let k = 0; k < N; k++) {
      if (!driving(k)) continue;
      const tr = car[k].trail; if (tr.length < 4) continue;
      ctx.strokeStyle = tun[car[k].t].color;
      for (let i = 2; i < tr.length; i += 2) { ctx.globalAlpha = 0.6 * (i / tr.length); ctx.beginPath(); ctx.moveTo(tr[i - 2], tr[i - 1]); ctx.lineTo(tr[i], tr[i + 1]); ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
  }
  let ink = "", inkT = 0;
  function draw() {
    if (!ink || (inkT += 1) >= 60) { ink = getComputedStyle(main).color; inkT = 0; }   // the text colour only changes with the theme
    ctx.setTransform(dpr * SCL, 0, 0, dpr * SCL, 0, 0); ctx.clearRect(0, 0, W, H);       // drawn in simulator units
    drawTrails(); drawViz();
    const badges = ground.length > 1;
    for (let k = 0; k < N; k++) {
      if (!S.state[k]) continue;
      const c = car[k], t = tun[c.t]; if (!t) continue;
      c.alpha = Math.min(1, c.alpha + 0.08);
      if (S.state[k] === 2) {                                                                // inside the bay it drives on through the back, left or right wall: only the part inside the bay is visible
        const g = t.goal, px = g[0] + c.px, py = g[1] + c.py - c.tv * c.tun;             // straight up at top speed
        let da = -Math.PI / 2 - c.pth; da = Math.atan2(Math.sin(da), Math.cos(da));
        ctx.save(); ctx.beginPath(); ctx.rect(t.bay[0], t.bay[1], t.bay[2] - t.bay[0], t.bay[3] - t.bay[1]); ctx.rect(0, t.bay[3], W, H); ctx.clip();   // the bay, and the open page below it (the rear may still stick out)
        body(px, py, c.pth + da * Math.min(1, c.tun / 0.25), t.color, 1, false); ctx.restore();
        continue;
      }
      if (S.state[k] === 3) {                                                                // under the arch: only the part below the arch top and within the opening is visible
        const g = gateNow.find(q => q.id === c.gate); if (!g) continue;
        ctx.save(); ctx.beginPath(); ctx.rect(g.x - g.hw, g.top, 2 * g.hw, g.bottom - g.top); ctx.rect(0, g.bottom, W, H); ctx.clip();
        body(S.X[k], S.Y[k], S.TH[k], t.color, 1, false); ctx.restore();
        continue;
      }
      body(S.X[k], S.Y[k], S.TH[k], t.color, c.alpha, S.V[k] < -1);
      if (badges) {                                                                          // the driver's tag: R or L
        const lx = S.X[k], ly = S.Y[k] - 17, D = DRIVERS[c.q], hw = D.tag.length > 1 ? 9 : 6;
        ctx.globalAlpha = c.alpha; ctx.fillStyle = D.color; ctx.beginPath(); ctx.roundRect(lx - hw, ly - 6, 2 * hw, 12, 3); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = "bold 8.5px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(D.tag, lx, ly + 0.5); ctx.globalAlpha = 1;
      }
      if (c.at > 10) { ctx.strokeStyle = "#e03131"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(S.X[k], S.Y[k], 15, 0, 6.283); ctx.stroke(); }   // red: has not moved for 10 s
    }
    if (held && S.state[held.k] === 1) { ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 2.5; ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.arc(S.X[held.k], S.Y[held.k], 17, 0, 6.283); ctx.stroke(); }
    for (let i = sparks.length - 1; i >= 0; i--) {                                           // a ring that grows and fades at every impact, bigger for harder hits
      const sp = sparks[i]; sp[3] += 1 / 60;
      if (sp[3] > 0.4) { sparks.splice(i, 1); continue; }
      ctx.globalAlpha = (1 - sp[3] / 0.4) * 0.9; ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sp[0], sp[1], 4 + (10 + 14 * sp[2]) * sp[3] / 0.4, 0, 6.283); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- fast drawing (the default view)
  // The canvas above covers the whole page, and the browser puts all of it on screen again every frame: on a large screen with
  // weak graphics that alone cost 15 frames/s (45 with the cars shown, 60 hidden). In the default view nothing needs that:
  // each car is a small element moved with a transform (the browser moves it without repainting anything), and each car's
  // plan is drawn on a canvas just big enough for it, redrawn only when the denoising moves on a step (about 10 times a second
  // while it runs, then not at all). The other views (live plans, one car, inspecting, several drivers) use the canvas.
  const fastOK = () => allMode && !live && !inspecting && ground.length <= 1;
  let fastNow = false;
  const layer = document.createElement("div"), planL = document.createElement("div"), carL = document.createElement("div");
  layer.className = "carsl"; layer.setAttribute("aria-hidden", "true"); layer.style.display = "none";
  layer.append(planL, carL); document.body.append(layer);
  const SPW = 34, SPH = 22, SPX = 17, SPY = 11;                                              // the box around a car, in simulator units, and the car's middle in it
  const sprites = new Map(); let spRes = 0;
  function sprite(color, rev) {                                                              // the car's picture, drawn once per colour (and reversing lights)
    const res = Math.round(SCL * Math.min(2, devicePixelRatio || 1) * 4) / 4;
    if (res !== spRes) { sprites.clear(); spRes = res; }
    const key = color + (rev ? "r" : ""); let url = sprites.get(key);
    if (!url) {
      const c = document.createElement("canvas"); c.width = Math.ceil(SPW * res); c.height = Math.ceil(SPH * res);
      const g = c.getContext("2d"); g.scale(res, res); body(SPX, SPY, 0, color, 1, rev, g);
      url = `url(${c.toDataURL()})`; sprites.set(key, url);
    }
    return url;
  }
  const set = (el, prop, v) => { if (el["_" + prop] !== v) { el["_" + prop] = v; el.style[prop] = v; } };
  const mkEl = (cls, parent) => { const e = document.createElement("div"); e.className = cls; parent.append(e); return e; };
  const fe = [];                                                                             // per car: its elements, made the first time the car is out
  function slot(k) {
    let r = fe[k]; if (r) return r;
    r = fe[k] = { free: mkEl("cs", carL), a: mkEl("cw", carL), b: mkEl("cw", carL), ring: mkEl("rg", carL), plan: document.createElement("canvas"), pv: null, ps: -1 };
    r.ae = mkEl("cs", r.a); r.be = mkEl("cs", r.b); planL.append(r.plan);
    for (const e of [r.free, r.a, r.b, r.ring, r.plan]) set(e, "display", "none");
    return r;
  }
  const show = (e, on) => set(e, "display", on ? "" : "none");
  const carT = (x, y, th) => `translate(${(x - SPX).toFixed(2)}px,${(y - SPY).toFixed(2)}px) rotate(${th.toFixed(4)}rad)`;
  function clipped(r, x, y, th, color, rect, below) {                                       // only the part inside rect, or below the line y = below, shows
    show(r.free, false); show(r.a, true); show(r.b, true);
    const [x0, y0, w, h] = rect;
    set(r.a, "transform", `translate(${x0}px,${y0}px)`); set(r.a, "width", w + "px"); set(r.a, "height", h + "px");
    set(r.b, "transform", `translate(0px,${below}px)`); set(r.b, "width", W + "px"); set(r.b, "height", Math.max(0, H - below) + "px");
    const url = sprite(color, false);
    set(r.ae, "backgroundImage", url); set(r.ae, "transform", carT(x - x0, y - y0, th)); set(r.ae, "opacity", "1");
    set(r.be, "backgroundImage", url); set(r.be, "transform", carT(x, y - below, th)); set(r.be, "opacity", "1");
  }
  function carRing(r, x, y, rad, color) {
    if (!rad) { show(r.ring, false); return; }
    show(r.ring, true); set(r.ring, "width", 2 * rad + 2.5 + "px"); set(r.ring, "height", 2 * rad + 2.5 + "px"); set(r.ring, "borderColor", color);
    set(r.ring, "transform", `translate(${(x - rad - 1.25).toFixed(1)}px,${(y - rad - 1.25).toFixed(1)}px)`);
  }
  function plan(r, k) {                                                                      // as drawOne(v, 1, false) in the slow-denoising view
    const v = allViz[k];
    if (!showViz || !v || !driving(k) || !ground.some(q => q > 0)) { show(r.plan, false); r.pv = null; return false; }
    const el = performance.now() - v.t0, step = Math.min(STEPS, Math.floor(el / 95));
    const fade = el > 2100 ? Math.max(0, 1 - (el - 2100) / 400) : 1;
    if (fade <= 0) { show(r.plan, false); return false; }
    const pr = Math.min(1.5, devicePixelRatio || 1) * SCL, cv2 = r.plan;
    if (r.pv !== v || r.ps !== step || r.pr !== pr) {
      if (!v.bb) {                                                                           // the box around every step of this plan
        let a = 1e9, b = 1e9, c = -1e9, d = -1e9;
        for (let s = 0; s <= STEPS; s++) for (const [px, py] of path(v, "x", s)) { a = Math.min(a, px); b = Math.min(b, py); c = Math.max(c, px); d = Math.max(d, py); }
        v.bb = [Math.floor(a - 5), Math.floor(b - 5), Math.ceil(c + 5), Math.ceil(d + 5)];
      }
      const [x0, y0, x1, y1] = v.bb, w = x1 - x0, h = y1 - y0;
      if (r.pv !== v || r.pr !== pr) {
        cv2.width = Math.max(1, Math.round(w * pr)); cv2.height = Math.max(1, Math.round(h * pr));
        set(cv2, "width", w + "px"); set(cv2, "height", h + "px"); set(cv2, "transform", `translate(${x0}px,${y0}px)`);
      }
      r.pv = v; r.ps = step; r.pr = pr;
      const g = cv2.getContext("2d"), tc = tun[car[k].t], col = tc.color, ex = car[k].exec || EXEC;
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv2.width, cv2.height); g.setTransform(pr, 0, 0, pr, -x0 * pr, -y0 * pr);
      g.lineCap = "round"; g.lineJoin = "round";
      const ln = pts => { g.beginPath(); pts.forEach(([px, py], i) => i ? g.lineTo(px, py) : g.moveTo(px, py)); g.stroke(); };
      g.strokeStyle = col; g.lineWidth = 1.2;
      for (let s = 0; s < step; s++) { g.globalAlpha = 0.12 + 0.3 * s / STEPS; ln(path(v, "x", s)); }
      const pts = path(v, "x", step), main0 = step === STEPS;
      g.globalAlpha = main0 ? 0.55 : 0.6; g.lineWidth = main0 ? 1.6 : 1.4; g.setLineDash([6, 5]); ln(pts); g.setLineDash([]);
      g.strokeStyle = tc.dark; g.globalAlpha = 1; g.lineWidth = 7; ln(pts.slice(0, 2 * ex + 1));
    }
    show(cv2, true); set(cv2, "opacity", fade.toFixed(2));
    return true;
  }
  function spark(x, y, st) {                                                                 // a ring that grows and fades, bigger for harder hits
    if (layer.childElementCount > 2 + 40) return;
    const R = 14 + 14 * st, e = mkEl("sp", layer);
    e.style.width = e.style.height = 2 * R + "px"; e.style.left = x - R + "px"; e.style.top = y - R + "px"; e.style.setProperty("--s", (4 / R).toFixed(3));
    e.addEventListener("animationend", () => e.remove(), { once: true });
  }
  function drawFast() {
    set(layer, "width", W + "px"); set(layer, "height", H + "px"); set(layer, "transform", `scale(${SCL})`);
    let any = false;
    for (let k = 0; k < N; k++) {
      const st = S.state[k], c = car[k], t = st ? tun[c.t] : null;
      if (!t) { const r = fe[k]; if (r) { show(r.free, false); show(r.a, false); show(r.b, false); show(r.ring, false); show(r.plan, false); r.pv = null; } continue; }
      const r = slot(k);
      c.alpha = Math.min(1, c.alpha + 0.08);
      any = plan(r, k) || any;
      if (st === 2) {
        const g = t.goal, px = g[0] + c.px, py = g[1] + c.py - c.tv * c.tun;
        let da = -Math.PI / 2 - c.pth; da = Math.atan2(Math.sin(da), Math.cos(da));
        clipped(r, px, py, c.pth + da * Math.min(1, c.tun / 0.25), t.color, [t.bay[0], t.bay[1], t.bay[2] - t.bay[0], t.bay[3] - t.bay[1]], t.bay[3]);
        carRing(r, 0, 0, 0); continue;
      }
      if (st === 3) {
        const g = gateNow.find(q => q.id === c.gate);
        if (!g) { show(r.free, false); show(r.a, false); show(r.b, false); carRing(r, 0, 0, 0); continue; }
        clipped(r, S.X[k], S.Y[k], S.TH[k], t.color, [g.x - g.hw, g.top, 2 * g.hw, g.bottom - g.top], g.bottom);
        carRing(r, 0, 0, 0); continue;
      }
      show(r.a, false); show(r.b, false); show(r.free, true);
      set(r.free, "backgroundImage", sprite(t.color, S.V[k] < -1));
      set(r.free, "transform", carT(S.X[k], S.Y[k], S.TH[k])); set(r.free, "opacity", c.alpha >= 1 ? "1" : c.alpha.toFixed(2));
      if (held && held.k === k) carRing(r, S.X[k], S.Y[k], 17, "rgba(232,89,12,.9)");
      else if (c.at > 10) carRing(r, S.X[k], S.Y[k], 15, "#e03131");
      else carRing(r, 0, 0, 0);
    }
    for (const sp of sparks) spark(sp[0], sp[1], sp[2]);
    sparks.length = 0;
    hudText = any ? `every car, plan refreshed every 2.5 s: bold = the first ${EXEC} step${EXEC > 1 ? "s" : ""} it drives, dashed = the rest, not executed` : "";
  }
  function render() {
    const f = fastOK();
    if (f !== fastNow) {
      fastNow = f; layer.style.display = f ? "" : "none"; cv.style.display = f ? "none" : "";
      if (f) { cv.width = cv.height = 0; } else for (const r of fe) if (r) { r.pv = null; }
    }
    if (f) drawFast(); else { sizeCanvas(); draw(); }
  }

  // ---------------------------------------------------------------- pointer: the spawn point, and grab and throw a car
  const THROWCAP = 420;
  const at = e => [(e.clientX - OX) / SCL, (e.clientY - OY) / SCL];                         // pointer in simulator units
  addEventListener("pointermove", e => { [mouse.x, mouse.y] = at(e); mouse.in = mouse.x > 0 && mouse.y > 0 && mouse.x < W && mouse.y < H; }, { passive: true });
  document.addEventListener("pointerleave", () => { mouse.in = false; });
  document.addEventListener("pointerdown", e => {                                            // before the box dragging sees it: a press on a car picks the car up
    if (e.button || e.target.closest(".carui")) return;
    const [x, y] = at(e); let best = -1, bd = 16;
    for (let k = 0; k < N; k++) if (S.state[k] === 1) { const d = Math.hypot(S.X[k] - x, S.Y[k] - y); if (d < bd) { bd = d; best = k; } }
    if (best < 0) return;
    e.stopPropagation(); e.preventDefault();
    try { main.setPointerCapture(e.pointerId); } catch (_) {}
    held = { k: best, id: e.pointerId, samples: [[performance.now(), x, y]] }; B.bx[best] = B.by[best] = B.om[best] = 0; S.V[best] = 0; main.style.cursor = "grabbing";
  }, true);
  document.addEventListener("pointermove", e => {
    if (!held || e.pointerId !== held.id) return;
    const [x, y] = at(e), t = performance.now(); held.samples.push([t, x, y]);
    while (held.samples.length > 2 && t - held.samples[0][0] > 120) held.samples.shift();
    e.stopPropagation();
  }, true);
  const release = e => {                                                                     // thrown: it flies off with the speed of the pointer, then its driver takes over again
    if (!held || e.pointerId !== held.id) return;
    e.stopPropagation();
    const sm = held.samples, a = sm[0], b = sm[sm.length - 1], dt = (b[0] - a[0]) / 1000, k = held.k;
    let vx = 0, vy = 0; if (dt > 0.01 && performance.now() - b[0] < 100) { vx = (b[1] - a[1]) / dt; vy = (b[2] - a[2]) / dt; }
    const sp = Math.hypot(vx, vy); if (sp > THROWCAP) { vx *= THROWCAP / sp; vy *= THROWCAP / sp; }
    B.bx[k] = vx; B.by[k] = vy; S.V[k] = 0; car[k].ptr = 99;
    held = null; main.style.cursor = "";
    const eat = ev => { ev.preventDefault(); ev.stopPropagation(); };                       // letting go over a link does not follow it
    addEventListener("click", eat, { capture: true, once: true }); setTimeout(() => removeEventListener("click", eat, true), 50);
  };
  document.addEventListener("pointerup", release, true); document.addEventListener("pointercancel", release, true);

  // ---------------------------------------------------------------- the options menu
  const ui = document.createElement("div");
  ui.className = "carui";
  let menu = document.querySelector(".carbtn");                                              // the gear and Stop in the caption box under the projects
  const stopBtn = document.querySelector(".carstop"), how = menu && menu.dataset.how;                    // a link to the cars' project page, when it is published (data-how on the gear)
  ui.innerHTML = `<div class="carpanel" id="carpanel" role="dialog" aria-label="How the cars drive" hidden><p class="cphead"><b>How the cars drive.</b> Every car is driven by a diffusion policy, a 16k-parameter network running in your browser. Four times a second it turns random noise into a 2-second plan: the faint lines are those plans, and the bold stretch is what the car drives. It learned by copying a planner, then by correcting its own mistakes, then by reinforcement learning.${how ? ` <a href="${how}">How it works</a>` : ""}</p>
<p class="cpsum"></p>
<details class="cpmore"><summary>Experiments</summary>
<p class="cphead">Two drivers can share the road: <b>L</b>, the learned diffusion policy, and <b>R</b>, hand-written rules that see the same picture of the world. Choose who drives. Orange rings are impacts; a red ring means a car has not moved for 10 s.</p>
<div class="cpdrv" role="group" aria-label="Who drives"></div>
<div class="cpbtns"></div><label class="cprate">generate cars / s <input type="range" min="0" max="1.5" step="0.25"><b></b></label>
<p class="cphud"></p><pre class="cpstats"></pre></details></div>`;
  const panel = ui.querySelector(".carpanel");
  if (!menu) {                                                                               // a page without the caption box: the old button in the corner
    menu = document.createElement("button"); menu.type = "button"; menu.className = "carbtn carfix"; menu.textContent = "cars";
    menu.setAttribute("aria-expanded", "false"); menu.setAttribute("aria-controls", "carpanel"); ui.append(menu); panel.classList.add("fix");
  }
  document.body.append(ui);
  const btns = ui.querySelector(".cpbtns"), hud = ui.querySelector(".cphud"), statsEl = ui.querySelector(".cpstats"), sumEl = ui.querySelector(".cpsum");
  const rateIn = ui.querySelector(".cprate input"), rateV = ui.querySelector(".cprate b");
  function placePanel() {                                                                    // pinned to the screen next to the gear, above it when it fits; never longer than the room it has
    if (panel.hidden || panel.classList.contains("fix")) return;
    const r = menu.getBoundingClientRect(), vw = document.documentElement.clientWidth, vh = innerHeight, M = 8, G = 6;
    panel.style.maxHeight = "none";
    const pw = panel.offsetWidth, natural = panel.offsetHeight;
    const above = Math.max(0, r.top - G - M), below = Math.max(0, vh - r.bottom - G - M);
    const up = natural <= above || above >= below, h = Math.min(natural, Math.max(120, up ? above : below));
    panel.style.maxHeight = h + "px";
    panel.style.left = Math.max(M, Math.min(r.left, vw - pw - M)) + "px";
    panel.style.top = Math.max(M, Math.min(vh - h - M, up ? r.top - G - h : r.bottom + G)) + "px";
  }
  function setOpen(open) { panel.hidden = !open; menu.setAttribute("aria-expanded", String(open)); if (open) { showStats(); placePanel(); } }
  menu.addEventListener("click", () => setOpen(panel.hidden));
  addEventListener("resize", placePanel); addEventListener("scroll", placePanel, { passive: true });
  ui.querySelector(".cpmore").addEventListener("toggle", placePanel);
  document.addEventListener("pointerdown", e => { if (!panel.hidden && !panel.contains(e.target) && !menu.contains(e.target)) setOpen(false); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !panel.hidden) { setOpen(false); menu.focus(); } });
  function clearRoad() {                                                                     // Stop takes every car off; Restart starts from an empty page
    for (let k = 0; k < N; k++) { S.state[k] = 0; car[k].trail.length = 0; }
    held = null; sparks.length = 0; layer.querySelectorAll(".sp").forEach(e => e.remove());
    clearViz(); contact.cc.clear(); contact.cd.clear(); stats.by = DRIVERS.map(mk); stats.t0 = simT; turn = 0; spawnT = 0;
    render();
  }
  if (stopBtn) stopBtn.addEventListener("click", () => { clearRoad(); paused = !paused; refresh(); });
  const clearViz = () => { viz = null; for (const k in allViz) delete allViz[k]; allT = 0; };
  const OPTS = [
    [() => "Cars come from: " + ["the h and n of my name", "the roundabout", "the mouse", "rush hour"][source], () => { source = rb ? (source + 1) % 4 : source === 0 ? 2 : 0; }],
    [() => "Denoising view: " + (showViz ? "on" : "off"), () => { showViz = !showViz; clearViz(); }],
    [() => live ? "Plans: live, every 0.25 s" : "Plans: slow denoising, every 2.5 s", () => { live = !live; clearViz(); }],
    [() => "Drive per plan: " + EXEC + (EXEC === 1 ? " step" : " steps"), () => { EXEC = EXEC % 4 + 1; for (const c of car) c.ptr = 99; clearViz(); }],
  ];
  const buttons = OPTS.map(([, act]) => {
    const b = document.createElement("button"); b.type = "button";
    b.onclick = async () => { await act(); refresh(); };
    btns.append(b); return b;
  });
  function refresh() {
    OPTS.forEach(([label], i) => { buttons[i].textContent = label(); }); rateV.textContent = rate; showStats();
    if (stopBtn) { stopBtn.textContent = paused ? "Restart" : "Stop"; stopBtn.setAttribute("aria-pressed", String(paused)); }
  }
  rateIn.value = rate; rateIn.oninput = () => { rate = +rateIn.value; refresh(); };
  // ---------------------------------------------------------------- who drives: the learned policy, the rules, or both
  const CHOICES = [["Learned", [1]], ["Rule-based", [0]], ["Learned and Rule-based", [0, 1]]];
  const drvEl = ui.querySelector(".cpdrv");
  const choiceBtns = CHOICES.map(([label, g]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = label;
    b.onclick = () => setGround(g); drvEl.append(b); return b;
  });
  function showChoice(loading) { CHOICES.forEach(([, g], i) => { const on = g.length === ground.length && g.every(q => ground.includes(q)); choiceBtns[i].setAttribute("aria-pressed", String(on)); choiceBtns[i].disabled = !!loading; }); }
  async function setGround(next) {                                                           // drivers put on or taken off: their cars come or go, and every counter starts over
    next = [...new Set(next)].sort((a, b) => a - b);
    for (const q of next) if (DRIVERS[q].file && !nets[q]) {
      showChoice(true);
      try { await loadDriver(q); } catch (_) { next = next.filter(x => x !== q); }
      showChoice(false);
    }
    ground = next; turn = 0;
    for (let k = 0; k < N; k++) if (S.state[k] && !ground.includes(car[k].q)) { S.state[k] = 0; if (held && held.k === k) held = null; }
    clearViz(); contact.cc.clear(); contact.cd.clear(); stats.by = DRIVERS.map(mk); stats.t0 = simT;
    showChoice(false); refresh();
  }
  showChoice(false);
  function showStats() {
    if (panel.hidden) return;
    const mins = (simT - stats.t0) / 60;
    const row = q => {
      const x = stats.by[q]; let n = 0, stuck = 0; for (let k = 0; k < N; k++) if (S.state[k] === 1 && car[k].q === q) { n++; if (car[k].at > 10) stuck++; }
      return `${(DRIVERS[q].tag + " " + DRIVERS[q].name).padEnd(21)} arrived ${x.arrived} (${mins > 0.05 ? (x.arrived / mins).toFixed(1) : "-"}/min)   avg trip ${x.arrived ? (x.tripSum / x.arrived).toFixed(1) : "-"} s   car-car contacts ${x.cc}   car-box contacts ${x.cd}   driving ${n}   stuck now ${stuck}   rerouted ${x.rerouted}`;
    };
    let arrived = 0; for (const q of ground) arrived += stats.by[q].arrived;
    sumEl.textContent = `${active()} cars on the road, ${arrived} arrived so far, ${stats.fps || "-"} frames/s`;
    statsEl.textContent = (ground.length ? ground.map(row).join("\n") : "no driver on the ground") + `\nlimit for this machine: ${stats.cap} cars`;
    hud.textContent = hudText;
  }
  refresh();

  // ---------------------------------------------------------------- load and main loop
  // The load: the share of the main thread the cars take, measured each second. The limit on cars goes up while the cars use
  // little of it and comes down when they use too much, so a fast machine gets up to MAXCARS and a slow one only a few.
  // Cars already out are never taken off; while over the limit, no new ones come.
  // The script's own time misses what the browser spends drawing the canvas, so the frame rate is watched too: when frames come
  // slower than the best rate seen on this screen, the browser is overloaded and the limit comes down.
  const LOAD_HI = 0.25, LOAD_LO = 0.15;
  let workMs = 0, winT = performance.now(), frames = 0, bestFps = 0, lowFps = 0;
  function adapt(now) {
    frames++;
    if (now - winT < 1000) return;
    const span = now - winT, load = workMs / span, fps = frames * 1000 / span;
    stats.load = +load.toFixed(3); stats.fps = Math.round(fps); workMs = 0; winT = now; frames = 0;
    showStats();
    if (paused || document.hidden || span > 1500) return;                                    // a hidden tab or a stalled window says nothing about the cars
    lowFps = fps < 56 ? lowFps + 1 : 0;                                                      // two slow seconds in a row: a lighter canvas
    if (lowFps >= 2 && quality > 0.5) { quality = Math.max(0.5, quality * 0.8); lowFps = 0; }
    stats.res = +dpr.toFixed(2);
    bestFps = Math.max(bestFps * 0.995, fps);                                                // the screen's rate, slowly forgotten in case it changes
    const n = active(), slow = fps < 0.8 * bestFps;
    if (load > LOAD_HI || slow) stats.cap = Math.max(2, Math.min(stats.cap - 1, Math.floor(n * 0.85)));
    else if (load < LOAD_LO && fps >= 0.92 * bestFps && n >= stats.cap - 2) stats.cap = Math.min(MAXCARS, stats.cap + 2);
  }

  let last = performance.now(), acc = 0, sub = 0, busy = false;
  async function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (paused && fastNow && !inspecting && !held) { requestAnimationFrame(frame); return; }  // stopped: the cars stay where they are and the page does no work
    if (!busy) {
      busy = true;
      const t0 = performance.now();
      try {
        geometry(); readPage(dt); buildRoutes(); buildFields(); gateNow = gates();
        if (!paused) {
          acc = Math.min(acc + dt, 0.15);
          while (acc >= DT) {
            if (sub === 0) await decide();
            spawnT -= DT; if (spawnT <= 0 && rate) { const n0 = active(); spawn(); spawnT = active() > n0 ? 1 / rate : 0; }   // a car that could not come yet tries again next step
            if (held) { const sm = held.samples[held.samples.length - 1], k = held.k; if (S.state[k] !== 1) held = null; else { vt[k] = 0; dl[k] = 0; S.X[k] = sm[1]; S.Y[k] = sm[2]; S.V[k] = 0; B.bx[k] = B.by[k] = B.om[k] = 0; } }
            for (let k = 0; k < N; k++) if (S.state[k] === 2 && (car[k].tun += DT) > car[k].park) S.state[k] = 0;
            emerge();
            simT += DT;
            Sim.substep(S, vt, dl);
            for (const e of Sim.physics(S, B, walls, W, H, bays, owners, wvel)) sparks.push([e[0], e[1], Math.min(1, e[2] / 60), 0]);
            arrivals();
            if (live && showViz) for (let k = 0; k < N; k++) if (driving(k)) { const tr = car[k].trail; tr.push(S.X[k], S.Y[k]); if (tr.length > 80) tr.splice(0, 2); }
            acc -= DT; if (++sub === SUB) { sub = 0; afterDecision(); }
          }
        }
        render();
      } catch (e) { console.error(e); }
      const t1 = performance.now(); workMs += t1 - t0; adapt(t1);
      busy = false;
    }
    requestAnimationFrame(frame);
  }
  window.__cars = { S, car, stats, tun: () => tun, obs: k => { const c = new Float32Array(5120), v = new Float32Array(3); Sim.buildObs3(k, S, obsWalls, W, H, fields[car[k].t], gw, gh, prevA, railObs(), c, v); return [Array.from(c), Array.from(v), [S.X[k], S.Y[k], S.TH[k], S.V[k], car[k].t, W, H]]; }, setSource: v => { source = v; refresh(); }, ground: () => ground.slice(), setGround, gates: () => ({ g: gateNow, m: glyphs, r: gateEls.map(e => { const r = e.getBoundingClientRect(); return [r.left, r.right]; }) }), rails: () => { let inRail = 0; for (let k = 0; k < N; k++) if (S.state[k] === 1) { const i = Math.floor((S.X[k] - rx0) / RG), j = Math.floor((S.Y[k] - ry0) / RG); if (i >= 0 && j >= 0 && i < rnx && j < rny && railMask[j * rnx + i]) inRail++; } return { rects: rails.length, inRail }; } };   // test hook: a view of the cars, and setGround to pick the drivers
  requestAnimationFrame(frame);
})();
