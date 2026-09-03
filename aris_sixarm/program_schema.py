"""The typed programme record, and the one file the browser viewer reads.

WHY THIS EXISTS, IN ONE SENTENCE: because `writing.densify` read `plan["tilt"]`
while `stroke_api` wrote `lean_vec`, every lean-rescued stroke executed
pen-upright, the JSON record said 0.0 degrees for all forty-seven segments, and
nothing anywhere raised — the two sides of an untyped dict simply never met.
A schema whose fields are declared, whose unknown keys are an error, and in
which a pen's lean has EXACTLY ONE NAME cannot fail that way silently: the
mismatch becomes a KeyError at the boundary instead of a wrong robot.

THE THREE RULES THIS FILE KEEPS
  1. UNKNOWN KEYS RAISE.  `from_dict` on every record checks the key set
     against the dataclass's fields and refuses anything it does not know.  A
     bundle written by a newer exporter and read by an older viewer is an
     error, loudly, at load; a field quietly ignored is the bug above.
  2. ONE REPRESENTATION PER QUANTITY.  The pen's lean is `lean_deg` — degrees,
     the largest lean the plan actually COMMANDED.  The permission it was
     planned under is `cone_deg`.  There is no `tilt`, no `lean_vec`, no
     `max_lean_deg` and no `tilt_max_deg` in this schema; the exporter is the
     single place those older spellings are resolved, and it resolves them the
     way `scripts/csail_allocate._phase_json` does, which is the spelling
     `scene_check` re-derives against.
  3. THE BULK IS BINARY AND THE STRUCTURE IS JSON.  A joint trajectory is
     4760 x 7 floats per arm; as JSON text that is megabytes of decimal digits
     that a browser must then re-parse into typed arrays anyway.  Every array
     lives in a side-car `.bin` and the JSON carries `{offset, length, dtype,
     shape}`, so the viewer does one fetch, one `ArrayBuffer`, and zero
     parsing.

WHAT THE BUNDLE IS MADE OF.  `out/<stem>_schedule.npz` (the conducted
timeline: joint trajectories, segment index and arc parameter per frame, the
ink chunks) and `out/<stem>_program.json` (the allocation: which arm drew which
span of which stroke, and every plan metric).  The two are written by the same
run and are the only artifacts that carry, between them, everything a viewer
needs.  Nothing here re-plans, re-checks or re-derives a gate — the clearance
series is the one computed quantity, and it is computed with `scene_check`'s
own primitives so it cannot disagree with the verdict that shipped.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """A record did not match the schema.  Always names the offending key."""


# --------------------------------------------------------------------------
# strictness
# --------------------------------------------------------------------------
def _from_dict(cls, d, where=""):
    """Build `cls` from a dict, refusing unknown and missing keys.

    THE REFUSAL IS THE WHOLE POINT.  `dataclass(**d)` already raises on an
    unknown key, but it raises `TypeError: __init__() got an unexpected
    keyword argument` with no path, which is unreadable four levels into a
    bundle.  This says which record and which key.
    """
    if not isinstance(d, dict):
        raise SchemaError(f"{where or cls.__name__}: expected an object, got "
                          f"{type(d).__name__}")
    names = {f.name for f in fields(cls)}
    extra = sorted(set(d) - names)
    if extra:
        raise SchemaError(f"{where or cls.__name__}: unknown key(s) {extra}; "
                          f"known keys are {sorted(names)}")
    required = {f.name for f in fields(cls)
                if f.default is _MISSING and f.default_factory is _MISSING}
    missing = sorted(required - set(d))
    if missing:
        raise SchemaError(f"{where or cls.__name__}: missing key(s) {missing}")
    return cls(**d)


class _Missing:
    pass


try:                                        # dataclasses' own sentinel
    from dataclasses import MISSING as _MISSING
except ImportError:                         # pragma: no cover
    _MISSING = _Missing


def _asdict(obj):
    """dataclass -> plain JSON types, recursively, numpy included."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------
# the buffer table — where every array actually lives
# --------------------------------------------------------------------------
@dataclass
class BufferRef:
    """One typed array in the side-car binary."""
    offset: int
    length: int          # bytes
    dtype: str           # "f4" | "f8" | "i4"
    shape: list


class BufferTable:
    """Collects arrays, hands out `BufferRef`s, and emits one bytes blob.

    Arrays are appended 8-byte aligned so a browser can build a typed array
    over the ArrayBuffer WITHOUT copying: `new Float32Array(buf, offset, n)`
    throws on a misaligned offset, and the copy-free path is the difference
    between a scrubber that is smooth and one that is not.
    """

    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr, dtype="f4"):
        a = np.ascontiguousarray(np.asarray(arr), dtype=np.dtype(dtype))
        pad = (-self.size) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        ref = BufferRef(offset=self.size, length=a.nbytes, dtype=dtype,
                        shape=list(a.shape))
        self.parts.append(a.tobytes())
        self.size += a.nbytes
        return ref

    def blob(self):
        return b"".join(self.parts)


# --------------------------------------------------------------------------
# the records
# --------------------------------------------------------------------------
@dataclass
class Meta:
    schema_version: int
    name: str
    source: str | None
    rig: str
    tool: str
    sheet_m: list              # [w, h] of the paper
    h_inv: float
    fps: float
    dt: float
    stride: int
    n_frames: int
    duration_s: float
    n_phases: int
    pause_s: float
    margin_m: float
    min_clearance_m: float
    makespan_s: float
    inks: list
    palette: dict              # ink name -> "#rrggbb"
    pens_mm: dict              # arm id (str) -> mm
    traced_m: float
    drawn_m: float
    dropped_m: float
    coverage_pct: float
    n_strokes: int
    n_segments: int
    logo: dict                 # the placement: scale, size, centre, rotation
    generated_by: str = "aris_sixarm.program_schema.export_bundle"


@dataclass
class ArmTrack:
    """One arm's whole timeline, as offsets into the binary."""
    arm: int
    name: str
    mount: str
    color_hex: str
    pen_ext_m: float
    pen_lat_m: float
    T_world_base: list         # 16 numbers, row-major 4x4
    q: BufferRef               # (F, 7) f4  — joint trajectory
    seg: BufferRef             # (F,)  i4   — global segment index, -1 = pen up
    u: BufferRef               # (F,)  f4   — arc parameter inside the segment
    drawn_xy: BufferRef        # (K, 2) f4  — the planned tip path, all segments
    drawn_off: BufferRef       # (nseg+1,) i4 — CSR offsets into drawn_xy
    drawing: bool


@dataclass
class Segment:
    """One certified span of one stroke, drawn by one arm."""
    arm: int
    index: int                 # global segment index; matches ArmTrack.seg
    phase: int
    stroke_id: int
    s_range: list              # [s0, s1] of the parent stroke
    direction: int
    flipped: bool
    length_m: float
    color: str
    kind: str
    min_sigma: float
    min_margin: float
    tip_err_m: float
    lean_deg: float            # THE lean the plan commanded.  One name.
    cone_deg: float            # the permission it was planned under
    draw_time_s: float
    plan_ok: bool
    validated: bool
    home_before: bool
    n_dense: int
    n_knots: int
    pts: BufferRef             # (N, 2) f4 — the TARGET span on the paper


@dataclass
class Stroke:
    """One traced stroke as the tracer put it on the paper."""
    id: int
    color: str
    kind: str
    length_m: float
    pts: BufferRef | None      # (N, 2) f4, None when the run kept no polyline


@dataclass
class Dropped:
    """A span nobody certified.  The honest half of the coverage number."""
    stroke_id: int
    color: str
    kind: str
    s_range: list
    length_m: float
    at: list
    pts: BufferRef


@dataclass
class InkChunks:
    """Every laid-down ink chunk, in one CSR block."""
    t_s: BufferRef             # (C,) f4 — when the chunk becomes visible
    arm: BufferRef             # (C,) i4
    off: BufferRef             # (C+1,) i4 into xyz
    xyz: BufferRef             # (P, 3) f4
    hex: list                  # C colour strings


@dataclass
class Phase:
    index: int
    name: str
    ink: str | None
    start_s: float
    duration_s: float
    floor_s: float
    scene_check_ok: bool
    min_clearance_m: float
    per_pair_m: dict
    arm_metres: dict
    arm_segments: dict
    arm_draw_s: dict
    arm_transit_s: dict
    paper_failed: list
    frame_failed: list
    column_failed: list


@dataclass
class Clearance:
    """Per-frame minimum distance, per body pair.  The killer feature's data.

    `pairs` is one entry per unordered arm pair; `d` is metres at every frame
    of the same clock the joint trajectories use, so a dip's frame index is a
    scrubber position without any conversion.  `self_d` is the same thing for
    one arm against its own metal.  Both come from `scene_check`'s own
    primitives (`pair_clearance`, `self_clearance`), so a dip drawn here is
    the number the verdict was made on and not a second opinion.
    """
    pairs: list                # [{"a": int, "b": int, "d": BufferRef}]
    self_d: dict               # arm (str) -> BufferRef (F,) f4
    margin_m: float
    self_margin_m: float


@dataclass
class Timings:
    """Where the wall clock went.  The reason the GUI exists."""
    stages: dict               # stage -> seconds (summed over repeats)
    substages: dict            # stage -> {sub -> seconds}
    total_s: float


@dataclass
class Bundle:
    meta: Meta
    arms: list                 # [ArmTrack]
    strokes: list              # [Stroke]
    segments: list             # [Segment]
    dropped: list              # [Dropped]
    ink: InkChunks
    phases: list               # [Phase]
    clearance: Clearance
    timings: Timings

    # -- json round trip -------------------------------------------------
    def to_json(self):
        return json.dumps(_asdict(self))

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d):
        extra = sorted(set(d) - {f.name for f in fields(cls)})
        if extra:
            raise SchemaError(f"Bundle: unknown key(s) {extra}")
        return cls(
            meta=_from_dict(Meta, d["meta"], "Bundle.meta"),
            arms=[_from_dict(ArmTrack, _refs(x, ("q", "seg", "u", "drawn_xy",
                                                 "drawn_off")),
                             f"Bundle.arms[{i}]")
                  for i, x in enumerate(d["arms"])],
            strokes=[_from_dict(Stroke, _refs(x, ("pts",)),
                                f"Bundle.strokes[{i}]")
                     for i, x in enumerate(d["strokes"])],
            segments=[_from_dict(Segment, _refs(x, ("pts",)),
                                 f"Bundle.segments[{i}]")
                      for i, x in enumerate(d["segments"])],
            dropped=[_from_dict(Dropped, _refs(x, ("pts",)),
                                f"Bundle.dropped[{i}]")
                     for i, x in enumerate(d["dropped"])],
            ink=_from_dict(InkChunks, _refs(d["ink"],
                                            ("t_s", "arm", "off", "xyz")),
                           "Bundle.ink"),
            phases=[_from_dict(Phase, x, f"Bundle.phases[{i}]")
                    for i, x in enumerate(d["phases"])],
            clearance=_clearance_from(d["clearance"]),
            timings=_from_dict(Timings, d["timings"], "Bundle.timings"))


def _refs(d, keys):
    """Turn the named sub-dicts into `BufferRef`s, in a copy."""
    out = dict(d)
    for k in keys:
        if out.get(k) is not None:
            out[k] = _from_dict(BufferRef, out[k], f"BufferRef({k})")
    return out


def _clearance_from(d):
    c = _from_dict(Clearance, dict(d), "Bundle.clearance")
    c.pairs = [dict(p, d=_from_dict(BufferRef, p["d"], "clearance.pairs[].d"))
               for p in c.pairs]
    c.self_d = {k: _from_dict(BufferRef, v, "clearance.self_d[]")
                for k, v in c.self_d.items()}
    return c


# --------------------------------------------------------------------------
# the exporter
# --------------------------------------------------------------------------
def export_bundle(npz_path, program_path, out_path, summary=None,
                  clearance=True, max_clearance_frames=4000):
    """Schedule npz + programme json -> one bundle (json + .bin). -> Bundle.

    `out_path` names the JSON; the binary is written beside it with the same
    stem and a `.bin` suffix, because the viewer fetches them as a pair and a
    bundle whose halves can be separated is a bundle that will be.
    """
    npz_path, program_path = Path(npz_path), Path(program_path)
    out_path = Path(out_path)
    z = np.load(npz_path, allow_pickle=False)
    prog = json.loads(program_path.read_text())
    summ = json.loads(Path(summary).read_text()) if summary and \
        Path(summary).exists() else {}
    buf = BufferTable()

    arm_ids = [int(x) for x in z["arms"]]
    drawing = {int(x) for x in z["drawing_arms"]}
    F = int(z["n_frames"])

    from . import fleet as fleet_mod
    from . import frames as frames_mod
    FL = fleet_mod.FLEET

    # ---- arms ----------------------------------------------------------
    pen_ext = {a: float(p) for a, p in zip(sorted(FL), np.asarray(z["pen_ext"]))}
    arms = []
    for a in arm_ids:
        spec = FL.get(a)
        T = spec.T_world_base(float(prog.get("h_inv", 0.94))) if spec is not None \
            else np.eye(4)
        arms.append(ArmTrack(
            arm=a,
            name=getattr(spec, "name", str(a)),
            mount=getattr(spec, "mount", "inv"),
            color_hex=_hex(getattr(spec, "color", (0.5, 0.5, 0.5))),
            pen_ext_m=pen_ext.get(a, 0.110),
            pen_lat_m=float(frames_mod.PEN_LAT),
            T_world_base=[float(x) for x in np.asarray(T, float).reshape(-1)],
            q=buf.add(z[f"q_{a}"], "f4"),
            seg=buf.add(z[f"seg_{a}"], "i4"),
            u=buf.add(z[f"u_{a}"], "f4"),
            drawn_xy=buf.add(z[f"segpts_{a}"], "f4"),
            drawn_off=buf.add(z[f"segoff_{a}"], "i4"),
            drawing=a in drawing))

    # ---- strokes, segments, dropped -------------------------------------
    stroke_pts = _stroke_polylines(program_path, prog)
    strokes = [Stroke(id=int(s["id"]), color=s["color"], kind=s["kind"],
                      length_m=float(s["length"]),
                      pts=(buf.add(stroke_pts[int(s["id"])], "f4")
                           if int(s["id"]) in stroke_pts else None))
               for s in prog.get("strokes", [])]

    segments, dropped = [], []
    # THE GLOBAL SEGMENT INDEX IS THE npz's, NOT THE JSON's.  `payload` numbers
    # an arm's segments consecutively ACROSS phases (`base += len(segs)`), and
    # `seg_<arm>` in the timeline refers to that numbering.  Rebuilding it here
    # the same way is what lets a click on a frame find the segment that frame
    # is drawing; numbering per phase would silently point at the wrong span
    # in every phase after the first.
    base = {a: 0 for a in arm_ids}
    for pi, ph in enumerate(prog.get("phases", [])):
        for astr, segs in (ph.get("arms") or {}).items():
            a = int(astr)
            for s in segs:
                segments.append(Segment(
                    arm=a, index=base.get(a, 0) + int(s["seg"]), phase=pi,
                    stroke_id=int(s["stroke_id"]),
                    s_range=[float(x) for x in s["s_range"]],
                    direction=int(s["direction"]),
                    flipped=bool(s.get("flipped", False)),
                    length_m=float(s["length_m"]),
                    color=s["color"], kind=s["kind"],
                    min_sigma=float(s["min_sigma"]),
                    min_margin=float(s["min_margin"]),
                    tip_err_m=float(s["tip_err_m"]),
                    # ONE NAME FOR THE LEAN, RESOLVED HERE AND NOWHERE ELSE.
                    lean_deg=float(s.get("max_lean_deg", 0.0) or 0.0),
                    cone_deg=float(s.get("tilt_cone_deg", 0.0) or 0.0),
                    draw_time_s=float(s["draw_time_s"]),
                    plan_ok=bool(s["plan_ok"]),
                    validated=bool(s["validated"]),
                    home_before=bool(s.get("home_before", False)),
                    n_dense=int(s.get("n_dense", 0)),
                    n_knots=int(s.get("n_knots", 0)),
                    pts=buf.add(np.asarray(s["pts"], float), "f4")))
        for astr, segs in (ph.get("arms") or {}).items():
            base[int(astr)] = base.get(int(astr), 0) + len(segs)
        for d in ph.get("dropped", []):
            dropped.append(Dropped(
                stroke_id=int(d["stroke_id"]), color=d["color"], kind=d["kind"],
                s_range=[float(x) for x in d["s_range"]],
                length_m=float(d["length_m"]),
                at=[float(x) for x in d["at"]],
                pts=buf.add(np.asarray(d["pts"], float), "f4")))

    # ---- ink -------------------------------------------------------------
    ink = InkChunks(t_s=buf.add(z["ink_t"], "f4"),
                    arm=buf.add(z["ink_arm"], "i4"),
                    off=buf.add(z["ink_off"], "i4"),
                    xyz=buf.add(z["ink_xyz"], "f4"),
                    hex=[str(x) for x in z["ink_hex"]])

    # ---- phases ----------------------------------------------------------
    starts = [float(x) for x in np.atleast_1d(z["phase_start_s"])]
    phases = []
    for i, ph in enumerate(summ.get("phases", [])):
        phases.append(Phase(
            index=i, name=ph.get("name", f"phase {i + 1}"), ink=ph.get("ink"),
            start_s=starts[i] if i < len(starts) else 0.0,
            duration_s=float(ph.get("duration_s", 0.0)),
            floor_s=float(ph.get("floor_s", 0.0)),
            scene_check_ok=bool(ph.get("scene_check_ok", False)),
            min_clearance_m=float(ph.get("min_clearance", 0.0)),
            per_pair_m={str(k): float(v)
                        for k, v in (ph.get("per_pair") or {}).items()},
            arm_metres=_fdict(ph.get("arm_metres")),
            arm_segments=_fdict(ph.get("arm_segments")),
            arm_draw_s=_fdict(ph.get("arm_draw_s")),
            arm_transit_s=_fdict(ph.get("arm_transit_s")),
            paper_failed=list(ph.get("paper_failed") or []),
            frame_failed=list(ph.get("frame_failed") or []),
            column_failed=list(ph.get("column_failed") or [])))

    # ---- clearance -------------------------------------------------------
    clr = _clearance_series(z, arm_ids, buf, float(z["margin"]),
                            enabled=clearance,
                            max_frames=max_clearance_frames)

    # ---- timings ---------------------------------------------------------
    tim = _timings(prog, summ)

    meta = Meta(
        schema_version=SCHEMA_VERSION,
        name=prog.get("name", npz_path.stem), source=prog.get("source"),
        rig=str(prog.get("rig", fleet_mod.ACTIVE_RIG)),
        tool=("lateral" if float(frames_mod.PEN_LAT) else "inline"),
        sheet_m=[float(x) for x in np.asarray(z["sheet"])],
        h_inv=float(prog.get("h_inv", 0.94)),
        fps=float(z["fps"]), dt=float(z["dt"]), stride=int(z["stride"]),
        n_frames=F, duration_s=float(z["duration"]),
        n_phases=int(z["n_phases"]), pause_s=float(z["pause_s"]),
        margin_m=float(z["margin"]), min_clearance_m=float(z["min_clearance"]),
        makespan_s=float(summ.get("makespan_s", z["duration"])),
        inks=[str(x) for x in prog.get("inks", [])],
        palette=dict(prog.get("palette", {})),
        pens_mm={str(k): float(v) for k, v in (prog.get("pens_mm") or {}).items()},
        traced_m=float(prog.get("totals", {}).get("traced_m", 0.0)),
        drawn_m=float(prog.get("totals", {}).get("drawn_m", 0.0)),
        dropped_m=float(prog.get("totals", {}).get("dropped_m", 0.0)),
        coverage_pct=float(prog.get("totals", {}).get("coverage_pct", 0.0)),
        n_strokes=int(prog.get("totals", {}).get("n_strokes", len(strokes))),
        n_segments=int(prog.get("totals", {}).get("n_segments", len(segments))),
        logo=dict(prog.get("logo", {})))

    b = Bundle(meta=meta, arms=arms, strokes=strokes, segments=segments,
               dropped=dropped, ink=ink, phases=phases, clearance=clr,
               timings=tim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(b.to_json())
    out_path.with_suffix(".bin").write_bytes(buf.blob())
    return b


def _fdict(d):
    return {str(k): float(v) for k, v in (d or {}).items()}


def _hex(rgb):
    r, g, bl = (int(round(255 * float(c))) for c in tuple(rgb)[:3])
    return f"#{r:02x}{g:02x}{bl:02x}"


def _stroke_polylines(program_path, prog):
    """The traced strokes' own points. -> {stroke_id: (N,2)}.

    `program_json` records a stroke's id, colour, kind and length but NOT its
    polyline — the geometry it keeps is per SEGMENT.  `scripts/draw.py` writes
    the polylines separately as `<stem>_strokes.json`, so the target curve (as
    opposed to the drawn one) is available whenever that run wrote it, and the
    overlay degrades to "drawn only" when it did not, instead of failing.
    """
    p = Path(program_path)
    # `draw.py` names it `<stem>_strokes.json`; `csail_schedule.py` writes
    # `csail_program<tag>.json` and no stroke file at all.  Both spellings are
    # tried and neither is required.
    cands = [p.with_name(p.name.replace("_program.json", "_strokes.json")),
             p.with_name(p.name.replace("program", "strokes", 1))]
    for cand in cands:
        if cand == p or not cand.exists():
            continue
        try:
            doc = json.loads(cand.read_text())
        except Exception:
            continue
        return {int(s["id"]): np.asarray(s["pts"], float)
                for s in doc.get("strokes", []) if len(s.get("pts", [])) > 1}
    return {}


def _clearance_series(z, arm_ids, buf, margin, enabled=True, max_frames=4000):
    """Per-frame pair and self clearance. -> Clearance.

    Computed with `scene_check.pair_clearance` and `scene_check.self_clearance`
    over the SAME chain points the verdict used, so the curve the viewer draws
    and the number the programme shipped with are one derivation.  On a long
    programme the frames are strided so the series stays a few hundred kB; the
    stride is reported through the shape, and a dip is never hidden by it
    because the minimum over the whole timeline is in `Phase.per_pair_m`
    already — this series is for FINDING the dip, not for certifying it.
    """
    from . import scene_check
    from . import fleet as fleet_mod
    FL = fleet_mod.FLEET
    pairs, self_d = [], {}
    if not enabled or len(arm_ids) < 1:
        return Clearance(pairs=[], self_d={}, margin_m=float(margin),
                         self_margin_m=float(scene_check.SELF_MARGIN))
    F = int(z["n_frames"])
    step = max(1, int(np.ceil(F / max(1, max_frames))))
    idx = np.arange(0, F, step)
    pen_ext = {a: float(p) for a, p in zip(sorted(FL), np.asarray(z["pen_ext"]))}
    h_inv = None
    P = {}
    for a in arm_ids:
        spec = FL.get(a)
        if spec is None:
            continue
        Q = np.asarray(z[f"q_{a}"], float)[idx]
        try:
            P[a] = np.array([scene_check._chain(q, spec, h_inv,
                                                pen_ext.get(a, 0.110))
                             for q in Q])
            self_d[str(a)] = buf.add(
                scene_check.self_clearance(Q, pen_ext=pen_ext.get(a, 0.110)),
                "f4")
        except Exception:
            continue
    present = sorted(P)
    rr = None
    try:
        rr = scene_check._radii_for(FL, present)
    except Exception:
        rr = scene_check.RADII
    for i, ai in enumerate(present):
        for aj in present[i + 1:]:
            try:
                d = scene_check.pair_clearance(P[ai], P[aj], rr)
            except Exception:
                continue
            pairs.append({"a": int(ai), "b": int(aj),
                          "d": buf.add(np.asarray(d, float), "f4")})
    return Clearance(pairs=pairs, self_d=self_d, margin_m=float(margin),
                     self_margin_m=float(scene_check.SELF_MARGIN))


def _timings(prog, summ):
    """The stage clock, from what the run already recorded.

    `program_json` carries the allocation's own wall time per phase and
    `summary_json` the conduct's; the GUI's live event stream carries the rest.
    A bundle exported from an old run therefore still has a timing panel, and
    a bundle exported from a GUI run has the same one with the live numbers
    written over it (see `docs/VIEWER.md`).
    """
    alloc = float(prog.get("totals", {}).get("wall_s", 0.0))
    conduct = float(sum(float(p.get("duration_s", 0.0))
                        for p in summ.get("phases", [])))
    stages = {"allocation": alloc}
    subs = {}
    for i, ph in enumerate(prog.get("phases", [])):
        subs.setdefault("allocation", {})[ph.get("name", f"phase {i + 1}")] = \
            float(ph.get("wall_s", 0.0))
    if conduct:
        stages["conduction"] = conduct
    return Timings(stages=stages, substages=subs,
                   total_s=float(sum(stages.values())))


# --------------------------------------------------------------------------
# the scene: what the viewer draws once and never again
# --------------------------------------------------------------------------
def export_scene(out_path, h_inv=None):
    """Meshes, arm bases and the static installation. -> dict (and a .bin).

    THE RIG IS WHATEVER THIS PROCESS ACTIVATED.  This function reads
    `fleet.FLEET` and `frames.PEN_LAT`, which are process globals chosen at
    import time by ARIS_RIG / ARIS_TOOL, so it must be called from a process
    started for that rig — which is what `aris_sixarm.gui.server` does, one
    subprocess per (rig, tool), cached.  Calling it and hoping is how a viewer
    ends up drawing four arms of one rig at the base positions of another.

    THE FK GOLDEN CHECK.  The viewer does its own forward kinematics, because
    shipping 4760 frames x 6 arms x 11 link poses is fifteen megabytes and
    shipping seven joint angles is nothing.  A second implementation of the FK
    is a second thing that can drift from `frames.DH`, so the scene carries a
    handful of joint vectors together with the link poses THIS module computed
    for them, and the viewer refuses to draw (visibly, with a banner) if its
    own answer differs by more than a micrometre.
    """
    from . import fleet as fleet_mod
    from . import frames as frames_mod
    from .viz import robot_model

    out_path = Path(out_path)
    buf = BufferTable()
    FL = fleet_mod.FLEET
    h = fleet_mod.H_INV_DEFAULT if h_inv is None else float(h_inv)

    links, joints = robot_model.load_model()
    meshes = {}
    for name, (v, f) in links.items():
        meshes[name] = {
            "verts": _asdict(buf.add(np.asarray(v, float), "f4")),
            "faces": _asdict(buf.add(np.asarray(f, int), "i4"))}

    # THE MESH ORDER IS THE FK ORDER, DECLARED.  `frames.LINK_FRAMES` is
    # link0..link8 then hand; `robot_model`'s meshes are named panda_*.  The
    # map is written down here rather than inferred in JS from a name prefix,
    # because "panda_link8 has no mesh" and "panda_hand is FK index 9" are two
    # facts a string match gets wrong in opposite directions.
    link_index = {f"panda_link{i}": i for i in range(9)}
    link_index["panda_hand"] = 9
    # The fingers do not move (the holder is bolted, not gripped): their pose
    # is the hand's, times a fixed offset taken from the URDF once.
    finger_T = {}
    try:
        poses = robot_model.link_poses(joints, np.zeros(7))
        Th = np.asarray(poses["panda_hand"], float)
        for nm in ("panda_leftfinger", "panda_rightfinger"):
            if nm in poses:
                finger_T[nm] = [float(x) for x in
                                (np.linalg.inv(Th) @ np.asarray(poses[nm],
                                                                float)).reshape(-1)]
    except Exception:
        finger_T = {}

    arms = []
    for aid in sorted(FL):
        spec = FL[aid]
        arms.append(dict(
            arm=int(aid), name=spec.name, mount=spec.mount,
            active=bool(spec.active), color_hex=_hex(spec.color),
            xy=[float(x) for x in spec.xy],
            pen_ext_m=float(spec.pen or frames_mod.ext_of()),
            pen_lat_m=float(frames_mod.PEN_LAT),
            q_seed=[float(x) for x in np.asarray(spec.q_seed, float)],
            T_world_base=[float(x) for x in
                          np.asarray(spec.T_world_base(h), float).reshape(-1)]))

    bodies = _static_bodies(FL, fleet_mod)

    # THE CAPSULE TABLE TRAVELS WITH THE SCENE, so the witness line the viewer
    # draws for a clearance dip is between the same two capsules
    # `scene_check.pair_clearance` measured — not between the nearest pair of
    # chain POINTS, which is a different and usually wrong answer once the
    # radii differ by 10 cm.  Entries are (i, j, r) or (i, j, r, t0, t1) over
    # the chain-point indices, exactly as `scene_check.RADII` has them.
    from . import scene_check
    try:
        radii = scene_check._radii_for(FL, sorted(FL))
    except Exception:
        radii = scene_check.RADII
    # (i, j) are CHAIN INDICES and stay integers; r and the optional
    # sub-segment parameters (t0, t1) are lengths.  Emitting the indices as
    # floats works in JS by accident — `a[0.0]` is `a["0"]` — and is the kind
    # of accident that stops working the first time something rounds.
    radii = [[int(row[0]), int(row[1])] + [float(x) for x in row[2:]]
             for row in radii]

    # the golden FK samples
    rng = np.random.default_rng(0)
    lo, hi = frames_mod.FR3_MIN, frames_mod.FR3_MAX
    qs = np.vstack([np.zeros(7), 0.5 * (lo + hi),
                    lo + (hi - lo) * rng.random((4, 7))])
    T = frames_mod.link_frames_many(qs)
    fk_check = dict(q=[[float(x) for x in row] for row in qs],
                    T=buf.add(np.asarray(T, float), "f8"),
                    link_frames=list(frames_mod.LINK_FRAMES))

    doc = dict(
        schema_version=SCHEMA_VERSION,
        rig=str(fleet_mod.ACTIVE_RIG),
        tool=("lateral" if float(frames_mod.PEN_LAT) else "inline"),
        h_inv=h,
        sheet_m=[float(x) for x in fleet_mod.SHEET],
        dh=[[float(x) for x in row] for row in frames_mod.DH],
        tcp_d=float(frames_mod.TCP_D),
        d_hand_tcp=float(frames_mod.D_HAND_TCP),
        pen_ext_m=float(frames_mod.ext_of()),
        pen_lat_m=float(frames_mod.PEN_LAT),
        fr3_min=[float(x) for x in frames_mod.FR3_MIN],
        fr3_max=[float(x) for x in frames_mod.FR3_MAX],
        arms=arms, bodies=bodies, meshes=meshes, radii=radii,
        self_margin_m=float(scene_check.SELF_MARGIN),
        link_index=link_index, finger_T=finger_T,
        fk_check=dict(fk_check, T=_asdict(fk_check["T"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc))
    out_path.with_suffix(".bin").write_bytes(buf.blob())
    return doc


def _static_bodies(FL, fleet_mod):
    """The paper, the table, the columns and the mount steel. -> [dict], metres.

    TWO SOURCES, AND THE RIG DECIDES WHICH.  `system_model.bodies()` is the
    surveyed installation — floor, table, paper, cage, per-arm drop posts,
    gussets, clamps, plates — and it is built from `layout.FLEET_PROPOSED`, so
    it is the truth for the `proposed` rig and a picture of a DIFFERENT room
    for any other.  Drawing it under `final6_opt` would put steel where there
    is none.  Every other rig falls back to what its own arms declare
    (`ArmSpec.static_obstacles`), plus the paper, which is always known.
    """
    out = []
    rig = str(fleet_mod.ACTIVE_RIG)
    if rig == "proposed":
        try:
            from . import system_model
            for b in system_model.bodies():
                out.append(dict(
                    name=b.name, kind=b.kind,
                    lo=[float(x) / 1000.0 for x in b.lo],
                    hi=[float(x) / 1000.0 for x in b.hi],
                    rgba=[float(x) for x in b.rgba],
                    collision=bool(b.collision),
                    provenance=str(b.provenance)))
            return out
        except Exception:
            out = []
    W, H = (float(fleet_mod.SHEET[0]), float(fleet_mod.SHEET[1]))
    out.append(dict(name="paper", kind="canvas", lo=[0.0, 0.0, -0.002],
                    hi=[W, H, 0.0], rgba=[0.97, 0.96, 0.93, 1.0],
                    collision=False, provenance="CODE (fleet.SHEET)"))
    out.append(dict(name="table", kind="table",
                    lo=[-0.225, -0.125, -0.054], hi=[W + 0.225, H + 0.125, -0.004],
                    rgba=[0.42, 0.34, 0.26, 1.0], collision=False,
                    provenance="CODE (viz.scene)"))
    seen = set()
    for aid in sorted(FL):
        for ob in (FL[aid].static_obstacles() or []):
            nm = str(ob.get("name", f"arm{aid}"))
            if nm in seen:
                continue
            seen.add(nm)
            out.append(dict(name=nm, kind="mount",
                            lo=[float(x) for x in ob["lo"]],
                            hi=[float(x) for x in ob["hi"]],
                            rgba=[0.66, 0.69, 0.73, 1.0], collision=True,
                            provenance=str(ob.get("source", "ArmSpec"))))
    return out


# --------------------------------------------------------------------------
def b64(blob):
    """For the rare consumer that wants one file after all (tests)."""
    return base64.b64encode(blob).decode("ascii")
