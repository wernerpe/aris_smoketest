// The 3D viewport: six arms, the paper, the table and the steel.
//
// THE SCENE IS THE RIG'S OWN GEOMETRY, NOT A DRAWING OF IT.  Every mesh comes
// from `robot_model.load_model()` (the vendored Franka glTF, welded and
// decimated exactly as the meshcat viewer does it) and every box from
// `system_model.bodies()` (the surveyed cage, in millimetres, converted once
// here).  Nothing is modelled twice: if the steel moves in `system_model.py`
// it moves here, and if it does not exist for a rig it is not drawn for that
// rig — see `program_schema._static_bodies` for why the `proposed` rig is the
// only one whose cage is real.
//
// EVERYTHING IS Z-UP AND IN THE CANVAS FRAME, because that is the frame every
// number in this project is already in: `fleet.SHEET` spans x in [0, 1.8034]
// and y in [0, 3.63064] with z = 0 the top of the paper.  Re-basing to
// three.js's y-up default would mean converting on every trajectory sample and
// getting it wrong once.

import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import * as fk from "./fk.js";

const LAYERS = ["arms", "pens", "paper", "cage", "mounts", "strokes", "ink",
                "bases", "chains"];

// +90 deg about x: a URDF cylinder stands on its own z, a three.js one on its
// y, and both are centred on their own origin.  Row major, as everything here.
const RX90 = new Float64Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]);

export class Scene3D {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x0f1114);
    el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 60);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-2.4, -1.8, 2.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.9, 1.8, 0.3);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xdfe6f0, 0x2a2620, 1.25));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(-2, -3, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xbfd0e0, 0.5);
    fill.position.set(3, 2, 1);
    this.scene.add(fill);

    this.groups = {};
    for (const k of LAYERS) {
      const g = new THREE.Group();
      g.name = k;
      this.groups[k] = g;
      this.scene.add(g);
    }
    this.groups.chains.visible = false;

    this.arms = new Map();       // arm id -> {group, meshes:[{mesh,idx}], pen}
    this.strokeLines = new Map();  // stroke id -> Line
    this.spanLines = new Map();    // "stroke:arm:s0" -> Line
    this.inkLines = [];
    this.witness = null;
    this._frames = Array.from({length: 10}, () => new Float64Array(16));

    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(el);
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  _resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setLayer(name, on) {
    if (this.groups[name]) this.groups[name].visible = !!on;
    // Turning the tool model off puts the STOCK fingers back: the hand is
    // then the manufacturer's, which is the honest picture of "no tool", and
    // the two are never both drawn.
    if (name === "pens" && this.tool) {
      for (const A of this.arms.values()) {
        for (const m of A.stock) m.visible = !on;
      }
    }
  }

  // ---- the static installation ---------------------------------------
  loadScene(doc, buf) {
    this.doc = doc;
    this.penExt = doc.pen_ext_m;
    this.penLat = doc.pen_lat_m;
    this.tool = doc.tool_model || null;
    for (const k of ["arms", "pens", "cage", "mounts", "paper", "bases",
                     "chains"]) {
      this.groups[k].clear();
    }
    this.arms.clear();

    // --- meshes, built once and shared by all six arms ----------------
    const geo = {};
    for (const [name, m] of Object.entries(doc.meshes)) {
      const v = viewOf(buf, m.verts), f = viewOf(buf, m.faces);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(v, 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(f), 1));
      g.computeVertexNormals();
      geo[name] = g;
    }

    // --- the tool, built once and shared by all six hands --------------
    // THE HOLDER IS THE THING ON THE ARM, not the planner's two numbers.
    // `scene.tool_model` is `assets/system_model/installation_fatfingers.urdf`
    // read out in the HAND frame (see `program_schema._tool_model`): the Fat
    // Franka Finger blades, the printed housing and cap, the spring, sleeve,
    // clutch and graphite inside the bore, and the 20 mm of lead standing out
    // of the cap at the 23 deg the photo fixed.  Nothing here is derived —
    // each part is one URDF <visual> with its own origin already composed
    // down to the hand.
    this.toolParts = [];
    if (this.tool) {
      for (const p of this.tool.parts) {
        let g;
        if (p.kind === "mesh") {
          const m = this.tool.meshes[p.mesh];
          if (!m) continue;
          const v = viewOf(buf, m.verts), f = viewOf(buf, m.faces);
          g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(v, 3));
          g.setIndex(new THREE.BufferAttribute(new Uint32Array(f), 1));
          g.computeVertexNormals();
        } else if (p.kind === "cylinder") {
          g = new THREE.CylinderGeometry(p.r, p.r, p.len, 20);
        } else if (p.kind === "sphere") {
          g = new THREE.SphereGeometry(p.r, 18, 12);
        } else {
          continue;
        }
        // A URDF CYLINDER STANDS ON ITS OWN z AND THREE.JS'S ON ITS y, and
        // both are centred — so a primitive's part matrix carries an extra
        // +90 deg about x.  Meshes are already vertex data in the link frame
        // and get the matrix unchanged.
        const rel = p.kind === "cylinder"
          ? fk.mul(Float64Array.from(p.T), RX90) : Float64Array.from(p.T);
        this.toolParts.push({geo: g, color: p.color, rel, name: p.name});
      }
    }

    // --- the static bodies --------------------------------------------
    for (const b of doc.bodies) {
      const sx = Math.max(b.hi[0] - b.lo[0], 1e-4);
      const sy = Math.max(b.hi[1] - b.lo[1], 1e-4);
      const sz = Math.max(b.hi[2] - b.lo[2], 1e-4);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(b.rgba[0], b.rgba[1], b.rgba[2]),
        transparent: b.rgba[3] < 1 || b.kind === "cage",
        opacity: b.kind === "cage" ? 0.42 : b.rgba[3],
        roughness: b.kind === "canvas" ? 0.95 : 0.6,
        metalness: b.kind === "cage" || b.kind === "mount" ? 0.35 : 0.0,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
      mesh.position.set((b.lo[0] + b.hi[0]) / 2, (b.lo[1] + b.hi[1]) / 2,
                        (b.lo[2] + b.hi[2]) / 2);
      mesh.name = b.name;
      const lay = b.kind === "canvas" || b.kind === "table" ? "paper"
                : b.kind === "mount" ? "mounts" : "cage";
      this.groups[lay].add(mesh);
    }

    // --- the arms ------------------------------------------------------
    const dark = new Set(["panda_hand", "panda_leftfinger", "panda_rightfinger",
                          "panda_link7"]);
    for (const a of doc.arms) {
      const g = new THREE.Group();
      const col = new THREE.Color(a.color_hex);
      const parts = [];
      for (const [name, idx] of Object.entries(doc.link_index)) {
        if (!geo[name]) continue;         // panda_link8 is a frame, not a body
        const mat = new THREE.MeshStandardMaterial({
          color: dark.has(name) ? 0x33363c : 0xeceff1,
          roughness: 0.45, metalness: 0.15});
        const m = new THREE.Mesh(geo[name], mat);
        m.matrixAutoUpdate = false;
        g.add(m);
        parts.push({mesh: m, idx});
      }
      // the fingers ride on the hand frame with a fixed offset
      const stock = [];
      for (const [name, T] of Object.entries(doc.finger_T || {})) {
        if (!geo[name]) continue;
        const mat = new THREE.MeshStandardMaterial({color: 0x33363c,
                                                    roughness: 0.5});
        const m = new THREE.Mesh(geo[name], mat);
        m.matrixAutoUpdate = false;
        g.add(m);
        parts.push({mesh: m, idx: 9, rel: Float64Array.from(T)});
        // The fat blades REPLACE these (docs/SYSTEM_MODEL.md 7d) — they bolt
        // to the carriages in the stock finger's place — so with the tool
        // model showing the stock pair is hidden rather than drawn inside it.
        if ((this.tool ? this.tool.hides : []).includes(name)) stock.push(m);
      }
      this.groups.arms.add(g);

      // the pen and (for the lateral holder) its bracket
      const pg = new THREE.Group();
      const penMat = new THREE.MeshStandardMaterial({color: 0x191b1f,
                                                     roughness: 0.35});
      const brMat = new THREE.MeshStandardMaterial({color: col,
                                                    roughness: 0.5});
      const pen = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0045, 0.0045, a.pen_ext_m + 0.05, 12),
        penMat);
      const bracket = new THREE.Mesh(
        new THREE.CylinderGeometry(0.008, 0.008, Math.abs(a.pen_lat_m) + 0.02,
                                   10), brMat);
      pen.matrixAutoUpdate = false;
      bracket.matrixAutoUpdate = false;
      // WITH A REAL MODEL THE SKETCH IS NOISE.  The two cylinders stay for the
      // inline pen and for any checkout without `assets/system_model/`, where
      // they are the only picture of the tool there is.
      pen.visible = !this.tool;
      bracket.visible = !this.tool && Math.abs(a.pen_lat_m) > 1e-6;
      pg.add(pen); pg.add(bracket);

      const tparts = [];
      let tip = null;
      for (const p of this.toolParts) {
        const m = new THREE.Mesh(p.geo, new THREE.MeshStandardMaterial(
          {color: new THREE.Color(p.color).getHex(),
           roughness: 0.45, metalness: 0.12}));
        m.matrixAutoUpdate = false;
        m.name = p.name;
        pg.add(m);
        tparts.push({mesh: m, rel: p.rel});
      }
      if (this.tool) {
        // THE TIP BALL IS PLACED FROM THIS ARM'S OWN (pen_lat, pen_ext) —
        // the same expression `tipOf` evaluates and the planner plans with —
        // so it cannot sit anywhere but on the planned tip.  The model's own
        // weld agrees with it to `tool.tip_urdf_err_m`, asserted in the tests.
        tip = new THREE.Mesh(
          new THREE.SphereGeometry(this.tool.tip_r, 20, 14),
          new THREE.MeshStandardMaterial({
            color: new THREE.Color(this.tool.tip_color).getHex(),
            emissive: new THREE.Color(this.tool.tip_color).getHex(),
            emissiveIntensity: 0.45, roughness: 0.5}));
        tip.matrixAutoUpdate = false;
        tip.name = "pen_tip";
        pg.add(tip);
      }
      this.groups.pens.add(pg);

      // a marker at the base, in the arm's own colour, so the timeline's
      // colours and the picture's are the same six
      const bs = new THREE.Mesh(new THREE.SphereGeometry(0.045, 18, 12),
                                new THREE.MeshStandardMaterial({color: col}));
      const T = a.T_world_base;
      bs.position.set(T[3], T[7], T[11]);
      this.groups.bases.add(bs);

      // the capsule chain, for reading a clearance dip
      const chain = new THREE.Line(
        new THREE.BufferGeometry().setAttribute(
          "position", new THREE.BufferAttribute(new Float32Array(11 * 3), 3)),
        new THREE.LineBasicMaterial({color: col}));
      this.groups.chains.add(chain);

      this.arms.set(a.arm, {
        group: g, parts, pen, bracket, chain, tparts, tip, stock,
        base: Float64Array.from(a.T_world_base),
        penExt: a.pen_ext_m, penLat: a.pen_lat_m, color: a.color_hex,
        q: Float64Array.from(a.q_seed)});
      this.setJoints(a.arm, a.q_seed);
    }
    this.setLayer("pens", this.groups.pens.visible);

    const s = doc.sheet_m;
    this.controls.target.set(s[0] / 2, s[1] / 2, 0.25);
    this.frameSheet(s);
    return this;
  }

  frameSheet(s) {
    this.camera.position.set(-1.6, -1.2, 2.4);
    this.controls.target.set(s[0] / 2, s[1] / 2, 0.3);
    this.controls.update();
  }

  // ---- posing ----------------------------------------------------------
  setJoints(armId, q) {
    const A = this.arms.get(armId);
    if (!A) return;
    A.q.set(q);
    const F = fk.linkFrames(q, fk.TCP_D, this._frames);
    for (const p of A.parts) {
      let W = fk.mul(A.base, F[p.idx]);
      if (p.rel) W = fk.mul(W, p.rel);
      setMatrix(p.mesh, W);
    }
    // the tool: a cylinder from the TCP down the pen axis, and for the
    // lateral holder a bracket from the TCP out along hand x
    const Thand = fk.mul(A.base, F[9]);
    const Ft = fk.identity(); Ft[11] = fk.D_HAND_TCP;
    const Ttcp = fk.mul(Thand, Ft);
    if (A.pen.visible) {
      setMatrix(A.pen, cylinderBetween(
        pt(Ttcp, A.penLat, 0, 0), pt(Ttcp, A.penLat, 0, A.penExt + 0.03)));
    }
    if (A.bracket.visible) {
      setMatrix(A.bracket, cylinderBetween(pt(Ttcp, 0, 0, 0),
                                           pt(Ttcp, A.penLat, 0, 0)));
    }
    // the modelled tool: every part is a fixed pose in the HAND frame
    for (const p of A.tparts) setMatrix(p.mesh, fk.mul(Thand, p.rel));
    if (A.tip) {
      const t = pt(Ttcp, A.penLat, 0, A.penExt);
      const M = fk.identity();
      M[3] = t[0]; M[7] = t[1]; M[11] = t[2];
      setMatrix(A.tip, M);
    }
    if (this.groups.chains.visible) this._updateChain(A, F, Ttcp);
  }

  _updateChain(A, F, Ttcp) {
    // The ten (inline) or eleven (lateral) points `scene_check` measures on:
    // base, J1..J7, TCP, [bracket corner,] tip — in that order, so a witness
    // index read off a clearance dip names the same body here as there.
    const pos = A.chain.geometry.attributes.position.array;
    let k = 0;
    for (let i = 0; i < 8; i++) {
      const W = fk.mul(A.base, F[i]);
      pos[k++] = W[3]; pos[k++] = W[7]; pos[k++] = W[11];
    }
    const tcp = pt(Ttcp, 0, 0, 0);
    pos[k++] = tcp[0]; pos[k++] = tcp[1]; pos[k++] = tcp[2];
    if (Math.abs(A.penLat) > 1e-6) {
      const c = pt(Ttcp, A.penLat, 0, 0);
      pos[k++] = c[0]; pos[k++] = c[1]; pos[k++] = c[2];
    }
    const tip = pt(Ttcp, A.penLat, 0, A.penExt);
    pos[k++] = tip[0]; pos[k++] = tip[1]; pos[k++] = tip[2];
    A.chain.geometry.setDrawRange(0, k / 3);
    A.chain.geometry.attributes.position.needsUpdate = true;
  }

  // The hand's world pose, row major.  A close-up of the tool has to be aimed
  // in the HAND's frame — "down the jaw axis at the barrel" is a sentence
  // about hand y and hand x, and in world coordinates it is six numbers that
  // are different for every arm and every pose.
  handPose(armId) {
    const A = this.arms.get(armId);
    if (!A) return null;
    return fk.mul(A.base, fk.linkFrames(A.q, fk.TCP_D)[9]);
  }

  tipOf(armId) {
    const A = this.arms.get(armId);
    if (!A) return null;
    const F = fk.linkFrames(A.q, fk.TCP_D, this._frames);
    const Ft = fk.identity(); Ft[11] = fk.D_HAND_TCP;
    const T = fk.mul(fk.mul(A.base, F[9]), Ft);
    return pt(T, A.penLat, 0, A.penExt);
  }

  // ---- the drawing on the paper ---------------------------------------
  clearStrokes() {
    this.groups.strokes.clear();
    this.strokeLines.clear();
    this.spanLines.clear();
  }

  // The traced target, drawn faint.  Everything a stage later says is said
  // ABOUT these, by (stroke id, s0, s1), so they are uploaded once.
  setStrokes(strokes) {
    this.clearStrokes();
    for (const s of strokes) {
      const line = polyline(s.pts, 0.0015, 0x4a4f58);
      line.userData = {stroke: s.id, cum: cumulative(s.pts), pts: s.pts};
      this.groups.strokes.add(line);
      this.strokeLines.set(s.id, line);
    }
  }

  // A certified span: the slice of its stroke, in the arm's colour, floating a
  // millimetre above the target so both are readable at once.
  addSpan(span, colorHex) {
    const host = this.strokeLines.get(span.stroke);
    if (!host) return;
    const key = `${span.stroke}:${span.arm}:${span.s0.toFixed(4)}`;
    if (this.spanLines.has(key)) return;
    const pts = sliceByS(host.userData.pts, host.userData.cum, span.s0, span.s1);
    if (pts.length < 2) return;
    const line = polyline(pts, 0.004, new THREE.Color(colorHex).getHex(), 2);
    this.groups.strokes.add(line);
    this.spanLines.set(key, line);
  }

  // WHERE THE PEN ACTUALLY GOES, which is not the same curve as the span it
  // was given.  `addSpan` slices the TARGET stroke by the span's s-range —
  // that is all the live event stream can say, because a running allocation
  // reports (stroke, s0, s1) and no geometry.  A finished programme carries
  // the plan's own dense tip path per segment (`segpts_<arm>` in the npz), and
  // that is the line to draw once it exists: it shows the splice overlaps, the
  // rounded corners and the ends `fly_shrink` gave back, none of which a slice
  // of the target can show.
  addPlanned(arm, xy, off, colorHex) {
    const col = new THREE.Color(colorHex).getHex();
    for (let s = 0; s + 1 < off.length; s++) {
      const n = off[s + 1] - off[s];
      if (n < 2) continue;
      const a = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        a[i*3] = xy[(off[s] + i) * 2];
        a[i*3+1] = xy[(off[s] + i) * 2 + 1];
        a[i*3+2] = 0.004;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(a, 3));
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({color: col}));
      line.userData = {arm, seg: s};
      this.groups.strokes.add(line);
    }
  }

  // The paper nobody certified, so a coverage number has a picture: the
  // residual IS the difference between the faint target line and the coloured
  // drawn one, and this names it.
  //
  // MAGENTA, NOT RED.  Arm 31's fleet colour is #d62629 and the obvious colour
  // for "not drawn" is a red — on the CSAIL mark at 0.30 m the two were
  // indistinguishable, and a viewer that draws a hole in the same colour as an
  // arm is worse than one that draws no holes.  No arm in any rig registry is
  // magenta.
  addDropped(list) {
    for (const d of list) {
      if (!d.pts || d.pts.length < 2) continue;
      const line = polyline(d.pts, 0.0055, 0xff2fd0, 2);
      line.userData = {dropped: d};
      this.groups.strokes.add(line);
    }
  }

  // ---- ink laid down during playback ----------------------------------
  setInk(chunks) {
    this.groups.ink.clear();
    this.inkLines = [];
    for (const c of chunks) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(c.xyz, 3));
      const l = new THREE.Line(g, new THREE.LineBasicMaterial(
        {color: new THREE.Color(c.hex).getHex()}));
      l.visible = false;
      l.userData = {t: c.t};
      this.groups.ink.add(l);
      this.inkLines.push(l);
    }
  }

  setInkTime(t) {
    for (const l of this.inkLines) l.visible = l.userData.t <= t;
  }

  // ---- the witness line for a clearance dip ---------------------------
  showWitness(p, q, text) {
    if (this.witness) { this.scene.remove(this.witness); this.witness = null; }
    if (!p || !q) return;
    const g = new THREE.BufferGeometry().setAttribute(
      "position", new THREE.BufferAttribute(
        Float32Array.from([p[0], p[1], p[2], q[0], q[1], q[2]]), 3));
    this.witness = new THREE.Line(
      g, new THREE.LineBasicMaterial({color: 0xd2544a}));
    this.witness.renderOrder = 99;
    this.witness.material.depthTest = false;
    this.scene.add(this.witness);
    this.witnessText = text;
  }
}

// --------------------------------------------------------------------------
function viewOf(buf, ref) {
  const n = ref.length / ({f4: 4, f8: 8, i4: 4}[ref.dtype]);
  if (ref.dtype === "f4") return new Float32Array(buf, ref.offset, n);
  if (ref.dtype === "f8") return new Float64Array(buf, ref.offset, n);
  return new Int32Array(buf, ref.offset, n);
}

function setMatrix(obj, m) {
  obj.matrix.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                 m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
  obj.matrixWorldNeedsUpdate = true;
}

// A point in world, given a row-major frame and a local offset.
function pt(T, x, y, z) {
  return [T[3] + T[0]*x + T[1]*y + T[2]*z,
          T[7] + T[4]*x + T[5]*y + T[6]*z,
          T[11] + T[8]*x + T[9]*y + T[10]*z];
}

// three.js cylinders are y-axis aligned and centred; this returns the
// row-major world matrix that puts one between two points.
const _u = new THREE.Vector3(), _v = new THREE.Vector3(),
      _q = new THREE.Quaternion(), _m = new THREE.Matrix4();
function cylinderBetween(a, b) {
  _u.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const len = _u.length() || 1e-6;
  _v.copy(_u).normalize();
  _q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), _v);
  _m.compose(new THREE.Vector3((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2),
             _q, new THREE.Vector3(1, len, 1));
  const e = _m.elements;                     // three.js stores COLUMN major
  return new Float64Array([e[0], e[4], e[8], e[12],
                           e[1], e[5], e[9], e[13],
                           e[2], e[6], e[10], e[14],
                           e[3], e[7], e[11], e[15]]);
}

export function polyline(pts, z, color, width = 1) {
  const a = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    a[i*3] = pts[i][0]; a[i*3+1] = pts[i][1]; a[i*3+2] = z;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(a, 3));
  return new THREE.Line(g, new THREE.LineBasicMaterial({color, linewidth: width}));
}

export function cumulative(pts) {
  const c = new Float64Array(pts.length);
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i-1][0], dy = pts[i][1] - pts[i-1][1];
    c[i] = c[i-1] + Math.hypot(dx, dy);
  }
  return c;
}

// The sub-polyline between two NORMALISED arc lengths, ends interpolated —
// the same operation `stroke_api.truncate_polyline` performs, so a span drawn
// here covers the paper the planner said it covers.
export function sliceByS(pts, cum, s0, s1) {
  const L = cum[cum.length - 1];
  if (!(L > 0)) return [];
  const a = Math.max(0, Math.min(1, s0)) * L, b = Math.max(0, Math.min(1, s1)) * L;
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const out = [interpAt(pts, cum, lo)];
  for (let i = 0; i < pts.length; i++)
    if (cum[i] > lo && cum[i] < hi) out.push(pts[i]);
  out.push(interpAt(pts, cum, hi));
  return out;
}

function interpAt(pts, cum, t) {
  let i = 1;
  while (i < cum.length - 1 && cum[i] < t) i++;
  const d = cum[i] - cum[i-1];
  const f = d > 0 ? (t - cum[i-1]) / d : 0;
  return [pts[i-1][0] + f * (pts[i][0] - pts[i-1][0]),
          pts[i-1][1] + f * (pts[i][1] - pts[i-1][1])];
}
