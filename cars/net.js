/* The v3 driver network in plain JavaScript (Nov 1): the website runs it without the 11 MB ONNX runtime.
   Same maths as model.py (Encoder with 4 conv blocks, TimeEmb, FiLM blocks) and the same outputs as export3.py's ONNX graph:
   chunk (n, 8, 2), xs (n, steps + 1, 16) the noisy chunk at every step, xhat (n, steps, 16) the guess of the clean chunk at
   every step. Weights come from export_js.py (OUT.json). Works in the browser (global PolicyNet) and in node (require).
   usage: const net = PolicyNet.create(json); const out = net.run(crop, vec, noise, n)   // crop = picture / 100, (n, 5, 32, 32) */
(function (root) {
  'use strict';
  const HC = 8, AD = 2, A = HC * AD;

  function decode(b64) {                                   // base64 little-endian float32 -> Float32Array
    let bytes;
    if (typeof atob === 'function') {
      const s = atob(b64); bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else bytes = Uint8Array.from(Buffer.from(b64, 'base64'));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  const silu = x => x / (1 + Math.exp(-x));

  function linear(W, b, nin, nout, x, xo, y, yo, act) {     // y[yo..] = W x[xo..] + b, W is (nout, nin) row-major
    for (let o = 0; o < nout; o++) {
      let s = b[o]; const r = o * nin;
      for (let i = 0; i < nin; i++) s += W[r + i] * x[xo + i];
      y[yo + o] = act ? silu(s) : s;
    }
  }

  // 3x3 convolution, padding 1, stride s, on one picture (C, H, W) -> (O, Ho, Wo); then GroupNorm(groups) and SiLU in place.
  // Loop order: each output pixel gathers its 3x3 window for four output channels at a time, kept in local variables
  // (weights stored as [c][tap][o]); zero inputs are skipped, which saves most of the work on the sparse car layers.
  function convBlock(L, inp, C, H, W, out) {
    const { wt, b, gw, gb, O, s, groups } = L;
    const Ho = Math.floor((H - 1) / s) + 1, Wo = Math.floor((W - 1) / s) + 1, P = Ho * Wo, HW = H * W;
    for (let og = 0; og < O; og += 4) {                   // four output channels at a time in local variables (O is a multiple of 4)
      const b0 = b[og], b1 = b[og + 1], b2 = b[og + 2], b3 = b[og + 3];
      for (let y = 0; y < Ho; y++) {
        const iy0 = y * s - 1, ky0 = iy0 < 0 ? 1 : 0, ky1 = iy0 + 2 >= H ? H - 1 - iy0 : 2;
        for (let x = 0; x < Wo; x++) {
          const ix0 = x * s - 1, kx0 = ix0 < 0 ? 1 : 0, kx1 = ix0 + 2 >= W ? W - 1 - ix0 : 2;
          let a0 = b0, a1 = b1, a2 = b2, a3 = b3;
          for (let ky = ky0; ky <= ky1; ky++) {
            const row = (iy0 + ky) * W + ix0;
            for (let kx = kx0; kx <= kx1; kx++) {
              const ip = row + kx, tb = (ky * 3 + kx) * O + og;
              for (let c = 0, q = ip, wb = tb; c < C; c++, q += HW, wb += 9 * O) {
                const val = inp[q]; if (val === 0) continue;
                a0 += wt[wb] * val; a1 += wt[wb + 1] * val; a2 += wt[wb + 2] * val; a3 += wt[wb + 3] * val;
              }
            }
          }
          const p = y * Wo + x;
          out[og * P + p] = a0; out[(og + 1) * P + p] = a1; out[(og + 2) * P + p] = a2; out[(og + 3) * P + p] = a3;
        }
      }
    }
    const cg = O / groups, n = cg * P;                    // GroupNorm: mean and (biased) variance over each group's channels and pixels
    for (let g = 0; g < groups; g++) {
      const base = g * n;
      let m = 0; for (let i = 0; i < n; i++) m += out[base + i]; m /= n;
      let v = 0; for (let i = 0; i < n; i++) { const d = out[base + i] - m; v += d * d; } v /= n;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let c = 0; c < cg; c++) {
        const ch = g * cg + c, a = gw[ch] * inv, bb = gb[ch] - m * a, cb = ch * P;
        for (let p = 0; p < P; p++) out[cb + p] = silu(out[cb + p] * a + bb);
      }
    }
    return [O, Ho, Wo];
  }

  function layerNorm(x, n, gw, gb) {                      // in place; gw/gb null = no affine
    let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < n; i++) x[i] = gw ? (x[i] - m) * inv * gw[i] + gb[i] : (x[i] - m) * inv;
  }

  function create(json, steps) {
    steps = steps || 8;
    const T = {}, S = {};
    for (const k in json.tensors) { S[k] = json.tensors[k][0]; T[k] = decode(json.tensors[k][1]); }
    const groups = json.groups, cin = json.cfg.cin || 5;
    const conv = [0, 1, 2, 3].map(i => ({ w: T[`enc.cnn.${i}.0.weight`], b: T[`enc.cnn.${i}.0.bias`], gw: T[`enc.cnn.${i}.1.weight`], gb: T[`enc.cnn.${i}.1.bias`],
                                          O: S[`enc.cnn.${i}.0.weight`][0], s: i === 0 ? 1 : 2, groups }));
    for (const L of conv) {                               // weights (O, C, 3, 3) -> [c][tap][o]
      const O = L.O, C = L.w.length / (O * 9); if (O % 4) throw new Error('conv channels must be a multiple of 4'); L.wt = new Float32Array(L.w.length);
      for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let k = 0; k < 9; k++) L.wt[(c * 9 + k) * O + o] = L.w[(o * C + c) * 9 + k];
    }
    const lin = k => ({ W: T[k + '.weight'], b: T[k + '.bias'], nout: S[k + '.weight'][0], nin: S[k + '.weight'][1] });
    const fa = lin('enc.cnn.5'), v0 = lin('enc.vec.0'), v2 = lin('enc.vec.2'), m0 = lin('enc.mix.0'), m2 = lin('enc.mix.2');
    const t0 = lin('temb.mlp.0'), t2 = lin('temb.mlp.2'), inp = lin('inp'), cproj = lin('cproj'), out1 = lin('out.1');
    const blocks = [];
    for (let i = 0; T[`blocks.${i}.film.weight`]; i++) blocks.push({ film: lin(`blocks.${i}.film`), f0: lin(`blocks.${i}.f.0`), f2: lin(`blocks.${i}.f.2`) });
    const lnW = T['out.0.weight'], lnB = T['out.0.bias'];
    const d = m2.nout, w = inp.nout, td = t2.nout, te = json.te;

    // the time embedding depends only on the step: compute it once for t = k / steps
    const temb = [];
    for (let k = 0; k < steps; k++) {
      const t = k / steps, half = te / 2, e = new Float64Array(te), h = new Float64Array(t0.nout), o = new Float64Array(td);
      for (let i = 0; i < half; i++) { const a = t * 1000 * Math.exp(-Math.log(10000) * i / half); e[i] = Math.sin(a); e[half + i] = Math.cos(a); }
      linear(t0.W, t0.b, t0.nin, t0.nout, e, 0, h, 0, true); linear(t2.W, t2.b, t2.nin, t2.nout, h, 0, o, 0, false);
      temb.push(o);
    }

    // scratch buffers, sized for the largest layer
    const bufA = new Float32Array(conv[0].O * 32 * 32), bufB = new Float32Array(conv[0].O * 32 * 32);
    const feat = new Float64Array(fa.nout + v2.nout), hv = new Float64Array(v0.nout), hm = new Float64Array(m0.nout), e = new Float64Array(d);
    const ce = new Float64Array(d + td), c = new Float64Array(d), x = new Float64Array(A), v = new Float64Array(A);
    const h = new Float64Array(w), n = new Float64Array(w), gb = new Float64Array(2 * w), f = new Float64Array(w), f2 = new Float64Array(w);

    function velocity(x, c, v) {                          // Policy.net: the predicted velocity of the noisy chunk x given condition c
      linear(inp.W, inp.b, inp.nin, inp.nout, x, 0, h, 0, false);
      for (const B of blocks) {
        linear(B.film.W, B.film.b, B.film.nin, B.film.nout, c, 0, gb, 0, false);
        for (let i = 0; i < w; i++) n[i] = h[i];
        layerNorm(n, w, null, null);
        for (let i = 0; i < w; i++) n[i] = n[i] * (1 + gb[i]) + gb[w + i];
        linear(B.f0.W, B.f0.b, w, w, n, 0, f, 0, true); linear(B.f2.W, B.f2.b, w, w, f, 0, f2, 0, false);
        for (let i = 0; i < w; i++) h[i] += f2[i];
      }
      for (let i = 0; i < w; i++) n[i] = h[i];
      layerNorm(n, w, lnW, lnB);
      linear(out1.W, out1.b, w, A, n, 0, v, 0, false);
    }

    function run(crop, vec, noise, num) {
      const chunk = new Float32Array(num * A), xs = new Float32Array(num * (steps + 1) * A), xhat = new Float32Array(num * steps * A);
      const P = cin * 1024, nv = v0.nin;
      for (let r = 0; r < num; r++) {
        // encoder: four conv blocks (32 -> 32 -> 16 -> 8 -> 4), flatten, linear; the 3 numbers through their own MLP; mix
        let src = crop.subarray(r * P, (r + 1) * P), dims = [cin, 32, 32], a = bufA, bsw = bufB;
        for (const L of conv) { dims = convBlock(L, src, dims[0], dims[1], dims[2], a); src = a; [a, bsw] = [bsw, a]; }
        linear(fa.W, fa.b, fa.nin, fa.nout, src, 0, feat, 0, true);
        linear(v0.W, v0.b, nv, v0.nout, vec, r * nv, hv, 0, true);
        linear(v2.W, v2.b, v2.nin, v2.nout, hv, 0, feat, fa.nout, true);
        linear(m0.W, m0.b, m0.nin, m0.nout, feat, 0, hm, 0, true);
        linear(m2.W, m2.b, m2.nin, d, hm, 0, e, 0, false);
        // denoising: Euler steps of the flow from noise to an action chunk
        for (let i = 0; i < A; i++) { x[i] = noise[r * A + i]; xs[r * (steps + 1) * A + i] = x[i]; }
        for (let k = 0; k < steps; k++) {
          for (let i = 0; i < d; i++) ce[i] = e[i];
          for (let i = 0; i < td; i++) ce[d + i] = temb[k][i];
          linear(cproj.W, cproj.b, cproj.nin, d, ce, 0, c, 0, false);
          velocity(x, c, v);
          const left = 1 - k / steps, xh = (r * steps + k) * A, xo = (r * (steps + 1) + k + 1) * A;
          for (let i = 0; i < A; i++) {
            xhat[xh + i] = Math.max(-1, Math.min(1, x[i] + left * v[i]));
            x[i] += v[i] / steps; xs[xo + i] = x[i];
          }
        }
        for (let i = 0; i < A; i++) chunk[r * A + i] = Math.max(-1, Math.min(1, x[i]));
      }
      return { chunk, xs, xhat };
    }
    return { run, steps, params: Object.values(T).reduce((s, t) => s + t.length, 0) };
  }

  const api = { create };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.PolicyNet = api;
})(this);
