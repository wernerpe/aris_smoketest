// Forward kinematics for the FR3, ported line for line from aris_sixarm/frames.py.
//
// WHY A SECOND IMPLEMENTATION EXISTS AT ALL.  A conducted programme is 4760
// frames x 6 arms; shipping the eleven 4x4 link poses of each would be fifteen
// megabytes, shipping seven joint angles is four hundred kilobytes, and the
// scrubber has to be able to jump anywhere instantly.  So the browser does the
// FK.  That makes this the one place in the project where a kinematic chain is
// written down twice, which is exactly the situation that produced the
// lean/tilt bug — so it is guarded: `/api/scene` carries six joint vectors
// together with the link poses PYTHON computed for them, `checkAgainst()` below
// re-derives them here, and `app.js` refuses to draw (with a red banner) if the
// two disagree by more than a micrometre.  A drift in either implementation is
// then a visible failure and not a slightly wrong picture.

// A 4x4 as a length-16 Float64Array, ROW MAJOR — the same order
// `T.reshape(-1)` produces in numpy, so a matrix from the server needs no
// transposition on the way in.  three.js's Matrix4.set() also takes row-major
// arguments, which is why this order and not the other.

export function identity() {
  return new Float64Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

export function mul(a, b, out) {
  const o = out || new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    const r0 = a[r*4], r1 = a[r*4+1], r2 = a[r*4+2], r3 = a[r*4+3];
    for (let c = 0; c < 4; c++) {
      o[r*4+c] = r0*b[c] + r1*b[4+c] + r2*b[8+c] + r3*b[12+c];
    }
  }
  return o;
}

// modified (Craig) DH, (alpha_{i-1}, a_{i-1}, d_i) — frames.DH
export const DH = [
  [0, 0, 0.333], [-Math.PI/2, 0, 0], [Math.PI/2, 0, 0.316],
  [Math.PI/2, 0.0825, 0], [-Math.PI/2, -0.0825, 0.384],
  [Math.PI/2, 0, 0], [Math.PI/2, 0.088, 0],
];

export const D_FLANGE = 0.107;
export const D_HAND_TCP = 0.1034;
export const TCP_D = D_FLANGE + D_HAND_TCP;

// The link-frame chain, in `frames.LINK_FRAMES` order:
//   0..7  link0..link7   8 link8 (flange)   9 hand
// Returns an array of ten row-major 4x4s, in the arm's OWN base frame.
export function linkFrames(q, tcp = TCP_D, out = null) {
  const F = out || Array.from({length: 10}, () => new Float64Array(16));
  let T = identity();
  F[0].set(T);
  const A = new Float64Array(16);
  for (let i = 0; i < 7; i++) {
    const al = DH[i][0], a = DH[i][1], d = DH[i][2];
    const ca = Math.cos(al), sa = Math.sin(al);
    const ct = Math.cos(q[i]), st = Math.sin(q[i]);
    A.fill(0);
    A[0] = ct;      A[1] = -st;     A[2] = 0;    A[3] = a;
    A[4] = st*ca;   A[5] = ct*ca;   A[6] = -sa;  A[7] = -sa*d;
    A[8] = st*sa;   A[9] = ct*sa;   A[10] = ca;  A[11] = ca*d;
    A[15] = 1;
    T = mul(T, A);
    F[i+1].set(T);
  }
  // link7 -> flange (link8): a pure translation of tcp - D_HAND_TCP along z
  const Ff = identity(); Ff[11] = tcp - D_HAND_TCP;
  mul(F[7], Ff, F[8]);
  // the flange twist -> the hand frame
  const c = Math.cos(-Math.PI/4), s = Math.sin(-Math.PI/4);
  const R = identity(); R[0] = c; R[1] = -s; R[4] = s; R[5] = c;
  mul(F[8], R, F[9]);
  return F;
}

// The hand-TCP pose, which is what the pen hangs off (frames.fk's `T`).
export function tcpPose(q, tcp = TCP_D) {
  const F = linkFrames(q, tcp);
  const Ft = identity(); Ft[11] = D_HAND_TCP;
  return mul(F[9], Ft);
}

// The pen tip in the arm's base frame: TCP + R @ (pen_lat, 0, pen_ext).
export function tipPos(q, penExt, penLat, tcp = TCP_D) {
  const T = tcpPose(q, tcp);
  return [
    T[3]  + T[0]*penLat + T[2]*penExt,
    T[7]  + T[4]*penLat + T[6]*penExt,
    T[11] + T[8]*penLat + T[10]*penExt,
  ];
}

// The bracket elbow of the lateral holder: TCP + R @ (pen_lat, 0, 0).
export function toolCorner(q, penLat, tcp = TCP_D) {
  const T = tcpPose(q, tcp);
  return [T[3] + T[0]*penLat, T[7] + T[4]*penLat, T[11] + T[8]*penLat];
}

// --------------------------------------------------------------------------
// the golden check
// --------------------------------------------------------------------------
// `scene.fk_check` is {q: [[7]...], T: Float64Array (N,10,4,4) flattened}.
// -> {ok, worst, n}.  `worst` is metres/radians of the largest disagreement.
export function checkAgainst(qs, flatT) {
  let worst = 0;
  for (let n = 0; n < qs.length; n++) {
    const F = linkFrames(qs[n]);
    for (let k = 0; k < 10; k++) {
      for (let e = 0; e < 16; e++) {
        const want = flatT[(n*10 + k)*16 + e];
        const got = F[k][e];
        const d = Math.abs(want - got);
        if (d > worst) worst = d;
      }
    }
  }
  return {ok: worst <= 1e-6, worst, n: qs.length};
}
