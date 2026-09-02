// A finished programme: the bundle, the clock, and the witness geometry.
//
// ONE FETCH FOR THE STRUCTURE AND ONE FOR THE NUMBERS.  `bundle.json` is a few
// tens of kilobytes of typed records whose array fields are {offset, length,
// dtype, shape} into `bundle.bin`; the binary is mapped straight into typed
// arrays with no copy and no parse (see `program_schema.BufferTable` for why
// the alignment matters).  A 200-second, six-arm programme is about 1.3 MB and
// loads in one frame, which is what makes the scrubber feel like a video
// player rather than a query.

import {api, fetchBin, view} from "./api.js";
import * as fk from "./fk.js";

export class Program {
  constructor(doc, buf) {
    this.doc = doc;
    this.buf = buf;
    this.meta = doc.meta;
    this.F = doc.meta.n_frames;
    this.fps = doc.meta.fps;

    this.arms = doc.arms.map(a => ({
      ...a,
      q: view(buf, a.q), seg: view(buf, a.seg), u: view(buf, a.u),
      drawnXY: view(buf, a.drawn_xy), drawnOff: view(buf, a.drawn_off),
    }));
    this.armById = new Map(this.arms.map(a => [a.arm, a]));

    this.strokes = doc.strokes.map(s => ({
      ...s, pts: pairs(view(buf, s.pts))}));
    this.strokeById = new Map(this.strokes.map(s => [s.id, s]));

    this.segments = doc.segments.map(s => ({...s, pts: pairs(view(buf, s.pts))}));
    this.segByArm = new Map();
    for (const s of this.segments) {
      if (!this.segByArm.has(s.arm)) this.segByArm.set(s.arm, new Map());
      this.segByArm.get(s.arm).set(s.index, s);
    }

    this.dropped = doc.dropped.map(d => ({...d, pts: pairs(view(buf, d.pts))}));

    // ink chunks, CSR
    const it = view(buf, doc.ink.t_s), ia = view(buf, doc.ink.arm),
          io = view(buf, doc.ink.off), ix = view(buf, doc.ink.xyz);
    this.ink = [];
    for (let c = 0; c + 1 < io.length; c++) {
      this.ink.push({t: it[c], arm: ia[c], hex: doc.ink.hex[c],
                     xyz: ix.subarray(io[c] * 3, io[c + 1] * 3)});
    }

    this.clearance = doc.clearance.pairs.map(p => ({
      a: p.a, b: p.b, d: view(buf, p.d)}));
    this.selfD = Object.fromEntries(
      Object.entries(doc.clearance.self_d).map(([k, v]) => [k, view(buf, v)]));
    this.margin = doc.clearance.margin_m;
    this.selfMargin = doc.clearance.self_margin_m;
    // The clearance series may be strided on a long programme; the ratio is
    // how a sparkline index becomes a frame index and back.
    this.clearStride = this.clearance.length
      ? this.F / this.clearance[0].d.length : 1;
  }

  static async load(jobId) {
    const doc = await api.bundle(jobId);
    const buf = await fetchBin(`/api/jobs/${jobId}/bundle.bin`);
    return new Program(doc, buf);
  }

  qAt(armId, frame) {
    const a = this.armById.get(armId);
    if (!a) return null;
    const i = Math.max(0, Math.min(this.F - 1, frame | 0)) * 7;
    return a.q.subarray(i, i + 7);
  }

  segAt(armId, frame) {
    const a = this.armById.get(armId);
    if (!a) return -1;
    return a.seg[Math.max(0, Math.min(this.F - 1, frame | 0))];
  }

  timeAt(frame) { return frame / this.fps; }

  // The phase a frame belongs to, for the lane background.
  phaseAt(frame) {
    const t = this.timeAt(frame);
    let last = null;
    for (const p of this.doc.phases) if (p.start_s <= t) last = p;
    return last;
  }
}

function pairs(flat) {
  if (!flat) return [];
  const out = new Array(flat.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = [flat[i*2], flat[i*2+1]];
  return out;
}

// --------------------------------------------------------------------------
// The witness for a clearance dip.
//
// `scene_check.pair_clearance` is a minimum over every pair of capsules from
// the two arms' chains; the DIP is the number, and the witness is which two
// capsules produced it and where on them.  Re-deriving that here — from the
// same chain points and the same radii table the scene carries — is what turns
// "arm 13 and arm 31 came within 82 mm somewhere around t = 41 s" into a red
// line the eye lands on.
// The 10 (inline) or 11 (lateral) world chain points of one configuration, in
// `scene_check._chain`'s order: base, J1..J7, TCP, [bracket corner,] tip.  The
// order is load-bearing — the radii table indexes into it — so it is written
// here in exactly the words the Python docstring uses.
export function chainPoints(base, q, penExt, penLat) {
  const F = fk.linkFrames(q);
  const P = [];
  for (let i = 0; i < 8; i++) {
    const W = fk.mul(base, F[i]);
    P.push([W[3], W[7], W[11]]);
  }
  const Ft = fk.identity(); Ft[11] = fk.D_HAND_TCP;
  const T = fk.mul(fk.mul(base, F[9]), Ft);
  const at = (x, y, z) => [T[3] + T[0]*x + T[1]*y + T[2]*z,
                           T[7] + T[4]*x + T[5]*y + T[6]*z,
                           T[11] + T[8]*x + T[9]*y + T[10]*z];
  P[8] = at(0, 0, 0);                    // the hand TCP
  P[9] = at(penLat, 0, penExt);          // the pen tip — ALWAYS index 9
  if (Math.abs(penLat) > 1e-9) {
    P[10] = at(penLat, 0, 0);            // the bracket elbow, lateral only
  }
  return P;
}

// (i, j, r) or (i, j, r, t0, t1) over chain indices -> the two capsule ends.
function capsuleEnds(P, row) {
  const i = row[0], j = row[1];
  let a = P[i], b = P[j];
  if (row.length >= 5) {
    const t0 = row[3], t1 = row[4];
    const d = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    a = [P[i][0] + t0*d[0], P[i][1] + t0*d[1], P[i][2] + t0*d[2]];
    b = [P[i][0] + t1*d[0], P[i][1] + t1*d[1], P[i][2] + t1*d[2]];
  }
  return [a, b, row[2]];
}

// The closest points of two segments, and their distance.  A direct transcript
// of the standard clamped parametric solve; `scene_check.segment_distance` is
// the independent Python derivation the verdict uses, and the two agree to
// float on everything they have been compared on.
export function segClosest(p0, p1, q0, q1) {
  const u = sub(p1, p0), v = sub(q1, q0), w = sub(p0, q0);
  const a = dot(u,u), b = dot(u,v), c = dot(v,v), d = dot(u,w), e = dot(v,w);
  const D = a*c - b*b;
  let sN, sD = D, tN, tD = D;
  if (D < 1e-12) { sN = 0; sD = 1; tN = e; tD = c; }
  else {
    sN = b*e - c*d; tN = a*e - b*d;
    if (sN < 0) { sN = 0; tN = e; tD = c; }
    else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
  }
  if (tN < 0) {
    tN = 0;
    if (-d < 0) sN = 0; else if (-d > a) sN = sD; else { sN = -d; sD = a; }
  } else if (tN > tD) {
    tN = tD;
    if (-d + b < 0) sN = 0; else if (-d + b > a) sN = sD;
    else { sN = -d + b; sD = a; }
  }
  const sc = Math.abs(sD) < 1e-12 ? 0 : sN / sD;
  const tc = Math.abs(tD) < 1e-12 ? 0 : tN / tD;
  const P = [p0[0]+sc*u[0], p0[1]+sc*u[1], p0[2]+sc*u[2]];
  const Q = [q0[0]+tc*v[0], q0[1]+tc*v[1], q0[2]+tc*v[2]];
  return {P, Q, d: Math.hypot(P[0]-Q[0], P[1]-Q[1], P[2]-Q[2])};
}

// -> {d, P, Q, i, j} the worst capsule pair between two chains.
export function witnessBetween(PA, PB, radii) {
  let best = null;
  for (const ra of radii) {
    if (ra[0] >= PA.length || ra[1] >= PA.length) continue;
    const [a0, a1, rA] = capsuleEnds(PA, ra);
    for (const rb of radii) {
      if (rb[0] >= PB.length || rb[1] >= PB.length) continue;
      const [b0, b1, rB] = capsuleEnds(PB, rb);
      const s = segClosest(a0, a1, b0, b1);
      const gap = s.d - rA - rB;
      if (!best || gap < best.d) best = {d: gap, P: s.P, Q: s.Q,
                                         i: ra, j: rb, rA, rB};
    }
  }
  return best;
}

const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
