/* Free-floating boxes on the landing page, as the cards of the car-swarm demo: drag a box anywhere, over other boxes and over the
   roundabout too. The box last touched sits on top. Loaded after the page has rendered.
   With a mouse a drag starts at once. On a touch screen a quick swipe scrolls the page as usual; a finger held still on a box
   for HOLD ms picks it up (it lifts, and the phone gives a short buzz), and only then does the finger move it. */
(() => {
  const main = document.querySelector("main.home");
  if (!main) return;
  let els = [], x = [], y = [], w = [], h = [], x0 = [], y0 = [], dirty = false, lastW = 0, H0 = 0, drag = null, pending = null, raf = 0, ztop = 1;

  function init(pad) {
    main.classList.remove("free");
    main.classList.toggle("nopad", !pad);   // with the script running, boxes fill the space beside the roundabout instead of the page reserving a strip for it
    for (const e of els) for (const p of ["left", "top", "width", "height", "transform", "zIndex"]) e.style[p] = "";   // keep the inline --c tunnel colour
    main.style.height = "";
    const keep = [...main.querySelectorAll(".keep")];   // layout a: the text column and the yard keep their height once their boxes float free
    for (const k of keep) k.style.minHeight = "";
    const m = main.getBoundingClientRect();
    els = [...main.querySelectorAll(".home .bx")];
    const r = els.map(e => e.getBoundingClientRect());
    for (const k of keep) k.style.minHeight = k.getBoundingClientRect().height + "px";
    x = r.map(b => b.left - m.left); y = r.map(b => b.top - m.top); w = r.map(b => b.width); h = r.map(b => b.height);
    if (!pad && !clearRoundabout(m)) return init(true);   // could not make room: fall back to the reserved strip
    H0 = Math.max(m.height, innerHeight - m.top - 90);
    lastW = m.width; dirty = false; ztop = 1;
    x0 = x.slice(); y0 = y.slice();
    els.forEach((e, i) => { const s = e.style; s.left = x0[i] + "px"; s.top = y0[i] + "px"; s.width = w[i] + "px"; s.height = h[i] + "px"; });
    main.classList.add("free");
    fit();
  }
  // The starting layout leaves the roundabout free: everything from the first box under it downward moves down together.
  function clearRoundabout(m) {
    const rc = main.querySelector(".rbc");
    if (!rc) return true;
    const q = rc.getBoundingClientRect(), qx = q.left - m.left, qy = q.top - m.top, GAP = 16;
    const hits = i => x[i] < qx + q.width + GAP && qx < x[i] + w[i] + GAP && y[i] < qy + q.height + GAP && qy < y[i] + h[i] + GAP;
    let top = Infinity, dy = 0;
    els.forEach((_, i) => { if (hits(i)) { top = Math.min(top, y[i]); dy = Math.max(dy, qy + q.height + GAP - y[i]); } });
    if (top === Infinity) return true;
    els.forEach((_, i) => { if (y[i] >= top - 0.5) y[i] += dy; });
    return !els.some((_, i) => hits(i));
  }

  function place(i) { els[i].style.transform = `translate(${Math.round(x[i] - x0[i])}px,${Math.round(y[i] - y0[i])}px)`; }
  function fit() {                 // the page grows to hold the lowest box
    let low = H0;
    for (let i = 0; i < els.length; i++) low = Math.max(low, y[i] + h[i] + 24);
    main.style.height = low + "px";
  }

  function frame() {
    raf = 0;
    if (!drag || !pending) return;
    const dx = pending.x - drag.px, dy = pending.y - drag.py;
    pending = null;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true; dirty = true;
    const i = drag.i, m = main.getBoundingClientRect(), W = document.documentElement.clientWidth;
    x[i] = Math.max(-m.left, Math.min(W - m.left - w[i], drag.bx + dx)); y[i] = Math.max(-(m.top + scrollY), drag.by + dy);   // anywhere on the page, up to the screen's edges
    place(i); fit();
  }

  const HOLD = 320, SLOP = 10;      // ms a finger rests on a box to pick it up; px it may wander meanwhile before it counts as a scroll
  let hold = null;
  function pickUp(i, px, py, id, touch) {
    hold = null;
    drag = { i, px, py, bx: x[i], by: y[i], moved: false, touch };
    els[i].style.zIndex = ++ztop;   // the box last touched sits on top
    try { els[i].setPointerCapture(id); } catch (_) {}
    els[i].classList.add("grab");
    if (touch) { els[i].classList.add("lift"); try { navigator.vibrate && navigator.vibrate(12); } catch (_) {} }
  }
  const dropHold = () => { if (hold) { clearTimeout(hold.t); hold = null; } };
  main.addEventListener("pointerdown", e => {
    if (e.button) return;
    const i = els.findIndex(el => el === e.target || el.contains(e.target));
    if (i < 0) return;
    const ia = e.target.closest && e.target.closest("a,button,input,select,textarea,summary,label");
    if (ia && ia !== els[i]) return;                                  // a link or button inside a box is for clicking; the box is grabbed anywhere else
    dropHold();
    if (e.pointerType !== "touch") return pickUp(i, e.clientX, e.clientY, e.pointerId, false);
    const h = { i, px: e.clientX, py: e.clientY, id: e.pointerId };
    h.t = setTimeout(() => { if (hold === h) pickUp(h.i, h.px, h.py, h.id, true); }, HOLD);
    hold = h;
  });
  main.addEventListener("pointermove", e => {
    if (hold && Math.hypot(e.clientX - hold.px, e.clientY - hold.py) > SLOP) dropHold();   // the finger moved first: it is a scroll
    if (!drag) return;
    pending = { x: e.clientX, y: e.clientY };
    if (!raf) raf = requestAnimationFrame(frame);
  });
  const end = () => {
    dropHold();
    if (!drag) return;
    if (pending) frame();
    els[drag.i].classList.remove("grab", "lift");
    if (drag.moved || drag.touch) main.addEventListener("click", ev => { ev.preventDefault(); ev.stopPropagation(); }, { capture: true, once: true });   // a box picked up is not also a tap on its link
    drag = null;
  };
  main.addEventListener("pointerup", end);
  main.addEventListener("pointercancel", end);
  main.addEventListener("dragstart", e => e.preventDefault());
  main.addEventListener("contextmenu", e => { if (drag || hold) e.preventDefault(); });   // a long press picks a box up, it does not open the link menu
  document.addEventListener("touchmove", e => { if (drag && drag.touch) e.preventDefault(); }, { passive: false });   // while a box is held the page does not scroll


  // Resize: an untouched layout is rebuilt for the new width; boxes the user placed stay where they are, pulled in only if they
  // would stick out past the screen's edges.
  let lastFs = getComputedStyle(document.documentElement).fontSize;
  function resized() {
    const W = main.getBoundingClientRect().width, fs = getComputedStyle(document.documentElement).fontSize;
    if (fs !== lastFs) { lastFs = fs; return init(); }   // text size changed: the frozen box sizes no longer fit their text
    if (W === lastW) return;
    lastW = W;
    if (!dirty || w.some(v => v > W)) return init();
    const m = main.getBoundingClientRect(), sw = document.documentElement.clientWidth;
    for (let i = 0; i < els.length; i++) { const nx = Math.max(-m.left, Math.min(sw - m.left - w[i], x[i])); if (nx !== x[i]) { x[i] = nx; place(i); } }
  }
  addEventListener("resize", resized);
  if (window.ResizeObserver) new ResizeObserver(resized).observe(main);   // also catches width changes that fire no resize event (scrollbar, zoom)
  init();
})();
