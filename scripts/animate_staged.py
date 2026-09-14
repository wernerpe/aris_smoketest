#!/usr/bin/env python3
"""Play a STAGED PROGRAMME (``staged.programme``) back on ONE fleet clock.

    ARIS_RIG=proposed ARIS_TOOL=lateral .venv/bin/python scripts/animate_staged.py \
        out/staged_csail_h097_program_v6.json --meshcat --port 7006 --loop \
        --html out/staged_v6_anim.html --gif out/staged_v6_topdown.gif

THE ONE THING THIS SCRIPT DECIDES IS THE CLOCK, and it decides it the way the
programme means it rather than the way the JSON is laid out:

  * a stage's arms all start at the stage's own start time, EXCEPT a residue
    arm (``residue: true``), whose bucket no order could fly concurrently and
    which `staged.py` therefore SERIALISES after the concurrent part;
  * the stage's duration is `max(concurrent durations) + sum(residue
    durations)`, which is exactly `duration_s`, and this script asserts it;
  * stage k+1 begins only when every arm of stage k is back at its park.  That
    is the BARRIER, and between the end of an arm's own trajectory and the
    barrier the arm HOLDS ITS PARK — it does not drift, and it does not start
    early.  An arm not in a stage's `arms` holds its park for the whole stage.

Nothing here re-plans, re-checks or re-plots anything: every joint vector comes
out of the JSON, every pose comes from the same FK (`frames`/`coordination`)
the certificate was computed with, and the only new numbers are the fleet-clock
offsets above and the clearances `--analyse` re-measures for the sanity read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GREY = (0.62, 0.62, 0.63)          # a parked arm
PARKED_ALPHA = 0.55
INK_CHUNK = 4                      # ink segments revealed as one meshcat object
MAX_INK_SEGMENTS = 5000            # hard cap; the sampler decimates to meet it


# ---------------------------------------------------------------------------
# 1.  THE FLEET CLOCK
# ---------------------------------------------------------------------------
class ArmTrack:
    """One (stage, arm) joint trajectory placed on the fleet clock."""

    def __init__(self, arm, d, t_start):
        self.arm = int(arm)
        self.q_park = np.asarray(d["q_park"], float).reshape(7)
        self.residue = bool(d["residue"])
        self.priority = d.get("priority")
        self.duration = float(d["duration_s"])
        self.ink_m = float(d["ink_m"])
        self.n_pieces = int(d["n_pieces"])
        self.t_start = float(t_start)
        tr = d.get("trajectory")
        if tr is None:
            self.t = np.zeros(1)
            self.q = self.q_park[None, :].copy()
            self.seg = np.full(1, -1)
        else:
            self.t = np.asarray(tr["t"], float)
            self.q = np.asarray(tr["q"], float)
            self.seg = np.asarray(tr["seg"], int)

    @property
    def t_end(self):
        return self.t_start + self.duration

    def sample(self, ts):
        """Fleet-clock times -> ((N,7) q, (N,) pen-down).  Outside: the park."""
        ts = np.asarray(ts, float)
        tau = np.clip(ts - self.t_start, self.t[0], self.t[-1])
        q = np.column_stack([np.interp(tau, self.t, self.q[:, j])
                             for j in range(7)])
        # pen-down on [t_i, t_{i+1}) iff BOTH ends of the sample interval draw:
        # `writing.arm_program` marks the touchdown sample with the stroke id
        # and every lifted sample with -1.
        i = np.clip(np.searchsorted(self.t, tau, "right") - 1, 0,
                    len(self.t) - 1)
        j = np.minimum(i + 1, len(self.t) - 1)
        down = (self.seg[i] >= 0) & (self.seg[j] >= 0)
        live = (ts >= self.t_start - 1e-9) & (ts <= self.t_end + 1e-9)
        return q, down & live

    def live(self, ts):
        ts = np.asarray(ts, float)
        return (ts >= self.t_start - 1e-9) & (ts <= self.t_end + 1e-9)


class Programme:
    """The staged JSON, laid on one clock."""

    def __init__(self, doc):
        self.doc = doc
        self.pattern = doc["pattern"]
        self.n_stages = int(doc["n_stages"])
        self.parks = {int(a): np.asarray(q, float).reshape(7)
                      for a, q in doc["parks"].items()}
        self.arms = sorted(self.parks)
        self.makespan = float(doc["makespan_s"])
        self.barriers = doc["barriers"]
        self.stages = []
        t = 0.0
        for s in doc["stages"]:
            tracks, conc = {}, 0.0
            for a, d in s["arms"].items():
                if not d["residue"]:
                    conc = max(conc, float(d["duration_s"]))
            cursor = conc
            # residues are serialised in the order the planner ranked them
            res = sorted((a for a, d in s["arms"].items() if d["residue"]),
                         key=lambda a: (s["arms"][a]["priority"] is None,
                                        s["arms"][a]["priority"]))
            for a, d in s["arms"].items():
                if not d["residue"]:
                    tracks[int(a)] = ArmTrack(a, d, t)
            for a in res:
                d = s["arms"][a]
                tracks[int(a)] = ArmTrack(a, d, t + cursor)
                cursor += float(d["duration_s"])
            assert abs(cursor - float(s["duration_s"])) < 1e-6, (
                f"stage {s['stage']}: max(concurrent)+sum(residue) = {cursor} "
                f"but duration_s = {s['duration_s']}")
            self.stages.append(dict(
                stage=int(s["stage"]), t0=t, t1=t + cursor,
                duration=float(s["duration_s"]),
                actives=[int(x) for x in s["actives"]],
                residues=[int(a) for a in res],
                concurrent=conc, n_pieces=int(s["n_pieces"]),
                ink_m=float(s["ink_m"]), checks=s.get("checks", {}),
                tracks=tracks, raw=s))
            t += cursor
        assert abs(t - self.makespan) < 1e-6, (t, self.makespan)

    def stage_at(self, ts):
        """-> (N,) stage index for each fleet-clock time."""
        edges = np.array([s["t0"] for s in self.stages] + [self.makespan])
        return np.clip(np.searchsorted(edges, np.asarray(ts, float), "right")
                       - 1, 0, len(self.stages) - 1)

    def sample(self, ts):
        """-> (Q {arm: (N,7)}, DOWN {arm: (N,)}, LIVE {arm: (N,)}, stage (N,))."""
        ts = np.asarray(ts, float)
        n = len(ts)
        Q = {a: np.repeat(self.parks[a][None, :], n, axis=0) for a in self.arms}
        DOWN = {a: np.zeros(n, bool) for a in self.arms}
        LIVE = {a: np.zeros(n, bool) for a in self.arms}
        for s in self.stages:
            win = (ts >= s["t0"] - 1e-9) & (ts < s["t1"] + 1e-9)
            if not win.any():
                continue
            for a, tr in s["tracks"].items():
                q, down = tr.sample(ts[win])
                lv = tr.live(ts[win])
                idx = np.nonzero(win)[0]
                Q[a][idx] = np.where(lv[:, None], q, self.parks[a][None, :])
                DOWN[a][idx] = down
                LIVE[a][idx] = lv
        return Q, DOWN, LIVE, self.stage_at(ts)

    def table(self):
        rows = []
        for s in self.stages:
            rows.append(dict(
                stage=s["stage"], t0=s["t0"], t1=s["t1"],
                duration=s["duration"], actives=s["actives"],
                residues=s["residues"],
                per_arm={a: (tr.t_start, tr.t_end, tr.duration, tr.residue)
                         for a, tr in sorted(s["tracks"].items())}))
        return rows


# ---------------------------------------------------------------------------
# 2.  GEOMETRY — the same FK the certificate saw
# ---------------------------------------------------------------------------
def chains_for(prog, Q, h_inv):
    """{arm: (N, 11, 3)} world chain points, via `coordination.chain_world`."""
    from aris_sixarm import coordination
    from aris_sixarm.fleet import FLEET
    return {a: coordination.chain_world(Q[a], FLEET[a], h_inv, FLEET[a].pen_ext)
            for a in prog.arms}


def capsule_table(n_points):
    from aris_sixarm import coordination
    return (coordination.CAPSULES_LAT if n_points >= 11
            else coordination.CAPSULES)


def ink_segments(prog, Q, DOWN, times, h_inv):
    """{arm: (M,2,3)} world pen-tip segments, only where the pen is down."""
    ch = chains_for(prog, Q, h_inv)
    out = {}
    for a in prog.arms:
        tip = ch[a][:, 9, :]                      # chain point 9 is the pen tip
        d = DOWN[a]
        keep = d[:-1] & d[1:]
        segs = np.stack([tip[:-1][keep], tip[1:][keep]], axis=1)
        out[a] = segs
    return out


# ---------------------------------------------------------------------------
# 3.  THE SANITY READ — clearances re-measured on the fleet clock
# ---------------------------------------------------------------------------
def pair_clearance(Pa, Pb, capsa=None, capsb=None):
    """(N,K,3) x (N,K,3) chains -> (N,) capsule-to-capsule clearance."""
    from aris_sixarm.coordination import cap_endpoints, seg_seg_dist
    ca = capsa or capsule_table(Pa.shape[1])
    cb = capsb or capsule_table(Pb.shape[1])
    A0, A1 = cap_endpoints(Pa, ca)                # (N, Ca, 3)
    B0, B1 = cap_endpoints(Pb, cb)
    ra = np.array([c[2] for c in ca])
    rb = np.array([c[2] for c in cb])
    d = seg_seg_dist(A0[:, :, None, :], A1[:, :, None, :],
                     B0[:, None, :, :], B1[:, None, :, :])
    return (d - ra[None, :, None] - rb[None, None, :]).min(axis=(1, 2))


def analyse(prog, h_inv, dt=0.05, verbose=True):
    """Re-measure every pair on the fleet clock. -> list of per-stage dicts."""
    from aris_sixarm.fleet import FLEET
    rows = []
    for s in prog.stages:
        ts = np.arange(s["t0"], s["t1"] + 1e-9, dt)
        Q, DOWN, LIVE, _ = prog.sample(ts)
        ch = chains_for(prog, Q, h_inv)
        worst = []
        for i, a in enumerate(prog.arms):
            for b in prog.arms[i + 1:]:
                if not (LIVE[a].any() or LIVE[b].any()):
                    continue                      # both parked all stage
                g = pair_clearance(ch[a], ch[b])
                k = int(np.argmin(g))
                worst.append(dict(pair=(a, b), min_m=float(g[k]),
                                  t=float(ts[k]),
                                  moving=(bool(LIVE[a][k]), bool(LIVE[b][k]))))
        worst.sort(key=lambda r: r["min_m"])
        # pen through paper?  the tip z at every DRAWING sample.
        pen = {}
        for a in prog.arms:
            if not DOWN[a].any():
                continue
            z = ch[a][DOWN[a], 9, 2]
            pen[a] = (float(z.min()), float(z.max()))
        # did anything move that should have been held?
        drift = {}
        for a in prog.arms:
            held = ~LIVE[a]
            if held.any():
                drift[a] = float(np.abs(Q[a][held] - prog.parks[a]).max())
        rows.append(dict(stage=s["stage"], t0=s["t0"], t1=s["t1"],
                         worst=worst[:4], pen_z=pen, park_drift=drift))
        if verbose:
            w = worst[0] if worst else None
            print(f"  stage {s['stage']}  [{s['t0']:7.2f}, {s['t1']:7.2f}] s  "
                  f"actives {s['actives']}"
                  + (f"   worst pair {w['pair']} = {1000 * w['min_m']:+7.1f} mm "
                     f"at t = {w['t']:.2f} s  moving={w['moving']}"
                     if w else ""))
            for r in worst[1:4]:
                print(f"        next  {r['pair']} = {1000 * r['min_m']:+7.1f} mm"
                      f" at t = {r['t']:.2f} s  moving={r['moving']}")
            for a, (lo, hi) in sorted(pen.items()):
                print(f"        arm {a} pen-down tip z in "
                      f"[{1000 * lo:+.2f}, {1000 * hi:+.2f}] mm")
            bad = {a: v for a, v in drift.items() if v > 1e-9}
            if bad:
                print(f"        PARK DRIFT (should be empty): {bad}")
    return rows


def ink_twice(prog):
    """Any (line, piece) planned in more than one place? -> list of dups."""
    seen, dup = {}, []
    for s in prog.stages:
        for a, d in s["raw"]["arms"].items():
            for p in d["pieces"]:
                k = (int(p["line"]), int(p["piece"]))
                if k in seen:
                    dup.append((k, seen[k], (s["stage"], int(a))))
                seen[k] = (s["stage"], int(a))
    return dup


# ---------------------------------------------------------------------------
# 4.  MESHCAT
# ---------------------------------------------------------------------------
def start_bridge(port, host="127.0.0.1"):
    """A meshcat zmq/web bridge in THIS process, bound to `port`. -> zmq_url.

    `meshcat-server` has no --port, and `Visualizer()` would pick 7000-7005 and
    land on somebody else's scene, so the bridge is constructed directly.  The
    tornado app listens on every interface (that is `app.listen`'s default), so
    the scene is reachable from outside; the zmq socket stays on localhost.
    """
    import asyncio
    from meshcat.servers.zmqserver import ZMQWebSocketBridge

    box, ready = {}, threading.Event()

    def run():
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            b = ZMQWebSocketBridge(host=host, port=int(port))
            box["url"] = b.zmq_url
            box["web"] = b.web_url
        except Exception as e:                            # pragma: no cover
            box["err"] = e
            ready.set()
            return
        ready.set()
        b.run()

    threading.Thread(target=run, daemon=True, name="meshcat-bridge").start()
    if not ready.wait(20):                                # pragma: no cover
        raise RuntimeError("meshcat bridge did not come up")
    if "err" in box:                                      # pragma: no cover
        raise box["err"]
    return box["url"], box["web"]


def _hex(rgb):
    return (int(round(255 * rgb[0])) << 16 | int(round(255 * rgb[1])) << 8
            | int(round(255 * rgb[2])))


def _mix(rgb, w, target=(1.0, 1.0, 1.0)):
    return tuple((1 - w) * c + w * t for c, t in zip(rgb, target))


def _caption_png(line1, line2, w=1040, h=168):
    """Two lines of text as PNG bytes — meshcat here has no text primitive."""
    import io
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (w, h), (250, 250, 246))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=(120, 120, 120), width=3)
    big = small = None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            big = ImageFont.truetype(path, 54)
            small = ImageFont.truetype(path, 34)
            break
    if big is None:                                       # pragma: no cover
        big = small = ImageFont.load_default()
    d.text((26, 18), line1, fill=(20, 20, 20), font=big)
    d.text((26, 92), line2, fill=(90, 90, 90), font=small)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _tool_transforms(poses, lat, pen_len):
    """`robot_model.add_robot`'s own pen/bracket placement, as transforms."""
    T_tcp = np.eye(4)
    T_tcp[2, 3] = 0.1034
    T_pen = poses["panda_hand"] @ T_tcp
    rx = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    out = {}
    if lat == 0.0:
        seg = np.eye(4)
        seg[2, 3] = pen_len / 2 - 0.025
        out["pen"] = T_pen @ seg @ rx
    else:
        rz = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, 1.0]])
        brk = np.eye(4)
        brk[0, 3] = lat / 2 - 0.01
        out["pen_bracket"] = T_pen @ brk @ rz
        seg = np.eye(4)
        seg[0, 3] = lat
        seg[2, 3] = pen_len / 2 - 0.01
        out["pen"] = T_pen @ seg @ rx
    return out


class MeshcatScene:
    """The scene, the ink chunks and the stage bar.  Built once, reused.

    THE INK IS PRE-BUILT AND REVEALED, not created as it is drawn, because that
    is the only encoding `meshcat.animation` can carry: an animation clip moves
    and toggles objects, it cannot create them.  So the live loop and the static
    html drive the SAME scene the SAME way — `visible` goes false -> true — and
    the html carries the ink build-up rather than only the arms.
    """

    def __init__(self, prog, zmq_url, h_inv, pen_seg_lw=0.0):
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
        from aris_sixarm import frames
        from aris_sixarm.fleet import FLEET, SHEET
        from aris_sixarm.viz import robot_model

        self.g, self.tf, self.prog, self.h_inv = g, tf, prog, h_inv
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.links, self.joints = robot_model.load_model()
        self.lat = frames.lat_of()
        self.robot_model = robot_model
        self.bases = {a: FLEET[a].T_world_base(h_inv) for a in prog.arms}
        self.pen_len = {a: frames.ext_of(FLEET[a].pen_ext) for a in prog.arms}

        self.vis["/Background"].set_property("top_color", [0.96, 0.96, 0.98])
        self.vis["/Background"].set_property("bottom_color", [0.84, 0.85, 0.90])
        self.vis["paper"].set_object(
            g.Box([SHEET[0], SHEET[1], 0.004]),
            g.MeshLambertMaterial(color=0xFAFAF2))
        self.vis["paper"].set_transform(
            tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.002]))
        self.vis["table"].set_object(
            g.Box([SHEET[0] + 0.40, SHEET[1] + 0.40, 0.05]),
            g.MeshLambertMaterial(color=0x8A7358))
        self.vis["table"].set_transform(
            tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.030]))

        for a in prog.arms:
            spec = FLEET[a]
            col = _hex(spec.color)
            T = self.bases[a]
            self.vis[f"bases/arm{a}"].set_object(
                g.Sphere(0.05), g.MeshLambertMaterial(color=col))
            self.vis[f"bases/arm{a}"].set_transform(T)
            robot_model.add_robot(
                self.vis, f"live/arm{a}", self.links, self.joints,
                prog.parks[a], T, pen_color=col, pen_len=self.pen_len[a],
                pen_lat=self.lat,
                body_color=_hex(_mix(spec.color, 0.55)),
                dark_color=_hex(_mix(spec.color, 0.10, (0, 0, 0))))
            robot_model.add_robot(
                self.vis, f"held/arm{a}", self.links, self.joints,
                prog.parks[a], T, pen_color=_hex(_mix(GREY, 0.3, (0, 0, 0))),
                pen_len=self.pen_len[a], pen_lat=self.lat,
                body_color=_hex(GREY), dark_color=_hex(_mix(GREY, 0.35,
                                                            (0, 0, 0))))
            self.vis[f"live/arm{a}"].set_property("visible", False)
        self.link_names = [n for n in self.links if n in
                           robot_model.link_poses(self.joints, prog.parks[
                               prog.arms[0]])]

        # -- the stage indicator: a lit/dim pair per stage, plus a title plane
        self.n_stages = prog.n_stages
        w, gap = 0.30, 0.05
        x0 = SHEET[0] / 2 - (self.n_stages * (w + gap) - gap) / 2
        self.slots = []
        for k in range(self.n_stages):
            Tk = tf.translation_matrix([x0 + k * (w + gap) + w / 2,
                                        SHEET[1] + 0.32, 0.25])
            for lit in (0, 1):
                p = f"stagebar/s{k}/{'lit' if lit else 'dim'}"
                self.vis[p].set_object(
                    g.Box([w, 0.10, 0.10 if lit else 0.04]),
                    g.MeshLambertMaterial(
                        color=0xE8552C if lit else 0xBBBBBB,
                        opacity=1.0 if lit else 0.5))
                self.vis[p].set_transform(Tk)
                self.vis[p].set_property("visible", bool(lit and k == 0))
            self.slots.append(k)
        self.text_ok = self._build_titles(SHEET)

    # -- stage captions --------------------------------------------------
    def _build_titles(self, SHEET):
        """A caption plate per stage, toggled by `visible`.

        This meshcat has neither `TextTexture` nor a `Plane` primitive, so the
        caption is a PNG rendered here and pasted on a thin box — which is the
        one text encoding an ANIMATION CLIP can also carry, since it toggles an
        object rather than creating one.
        """
        g, tf = self.g, self.tf
        try:
            for k, s in enumerate(self.prog.stages):
                act = " ".join(str(a) for a in s["actives"])
                res = (f"  residue {' '.join(map(str, s['residues']))}"
                       if s["residues"] else "")
                png = _caption_png(
                    f"STAGE {k} / {self.n_stages - 1}",
                    f"active {act}{res}   t {s['t0']:.0f}-{s['t1']:.0f} s"
                    f"   ink {s['ink_m']:.2f} m")
                self.vis[f"title/s{k}"].set_object(
                    g.Box([2.6, 0.012, 0.42]),
                    g.MeshBasicMaterial(map=g.ImageTexture(g.PngImage(png))))
                self.vis[f"title/s{k}"].set_transform(
                    tf.translation_matrix([SHEET[0] / 2, SHEET[1] + 0.60, 0.62]))
                self.vis[f"title/s{k}"].set_property("visible", k == 0)
            return True
        except Exception as e:                                # pragma: no cover
            print(f"  (no caption plate: {e}; the stage bar is the indicator)")
            return False

    # -- ink -------------------------------------------------------------
    def build_ink(self, segs):
        """{arm: (M,2,3)} -> chunked LineSegments, all hidden. -> chunk index."""
        from aris_sixarm.fleet import FLEET
        g = self.g
        self.chunks = []                       # [(path, first_frame_index)]
        self.chunk_of = {}
        for a in self.prog.arms:
            S, F = segs[a]["seg"], segs[a]["frame"]
            col = _hex(FLEET[a].color)
            for c0 in range(0, len(S), INK_CHUNK):
                blk = S[c0:c0 + INK_CHUNK]
                path = f"ink/arm{a}/c{c0 // INK_CHUNK}"
                v = blk.reshape(-1, 3).T.astype(np.float32)
                v = v + np.array([[0], [0], [0.0015]], np.float32)
                self.vis[path].set_object(
                    g.LineSegments(g.PointsGeometry(position=v),
                                   g.LineBasicMaterial(color=col,
                                                       linewidth=3.0)))
                self.vis[path].set_property("visible", False)
                self.chunks.append((path, int(F[c0:c0 + INK_CHUNK].max())))
        return self.chunks

    # -- per-frame -------------------------------------------------------
    def pose(self, a, q, show_live):
        rm, T = self.robot_model, self.bases[a]
        poses = rm.link_poses(self.joints, q)
        root = "live" if show_live else "held"
        for name in self.links:
            if name in poses:
                self.vis[f"{root}/arm{a}/{name}"].set_transform(T @ poses[name])
        for name, M in _tool_transforms(poses, self.lat,
                                        self.pen_len[a]).items():
            self.vis[f"{root}/arm{a}/{name}"].set_transform(T @ M)

    def set_active(self, a, live):
        self.vis[f"live/arm{a}"].set_property("visible", bool(live))
        self.vis[f"held/arm{a}"].set_property("visible", not bool(live))

    def set_stage(self, k):
        for j in range(self.n_stages):
            self.vis[f"stagebar/s{j}/lit"].set_property("visible", j == k)
            self.vis[f"stagebar/s{j}/dim"].set_property("visible", j != k)
            if self.text_ok:
                self.vis[f"title/s{j}"].set_property("visible", j == k)

    def reset_ink(self):
        for path, _ in self.chunks:
            self.vis[path].set_property("visible", False)


def sample_grid(prog, dt, stage=None):
    if stage is None:
        t0, t1 = 0.0, prog.makespan
    else:
        s = prog.stages[int(stage)]
        t0, t1 = s["t0"], s["t1"]
    n = int(np.floor((t1 - t0) / dt)) + 1
    return t0 + dt * np.arange(n)


def build_ink_frames(prog, Q, DOWN, times, h_inv):
    """{arm: dict(seg=(M,2,3), frame=(M,))} — each segment and when it lands."""
    ch = chains_for(prog, Q, h_inv)
    out, total = {}, 0
    for a in prog.arms:
        tip = ch[a][:, 9, :]
        d = DOWN[a]
        keep = np.nonzero(d[:-1] & d[1:])[0]
        seg = np.stack([tip[keep], tip[keep + 1]], axis=1)
        out[a] = dict(seg=seg, frame=keep + 1)
        total += len(seg)
    if total > MAX_INK_SEGMENTS:                      # pragma: no cover
        step = int(np.ceil(total / MAX_INK_SEGMENTS))
        for a in prog.arms:
            out[a] = dict(seg=out[a]["seg"][::step], frame=out[a]["frame"][::step])
        total = sum(len(v["seg"]) for v in out.values())
    return out, total


def meshcat_run(prog, args, h_inv):
    zmq_url, web = start_bridge(args.port)
    scene = MeshcatScene(prog, zmq_url, h_inv)
    dt = args.rate / args.fps
    ts = sample_grid(prog, dt, args.stage)
    Q, DOWN, LIVE, STAGE = prog.sample(ts)
    segs, n_ink = build_ink_frames(prog, Q, DOWN, ts, h_inv)
    scene.build_ink(segs)
    url = f"http://{args.hostname}:{args.port}/static/"
    print(f"meshcat scene: {url}   ({len(ts)} frames, dt = {dt:.3f} s of "
          f"programme, {args.rate:g}x wall clock, {n_ink} ink segments)")

    if args.html:
        write_html(scene, prog, ts, Q, LIVE, STAGE, args)

    # prime: everybody parked, stage 0
    for a in prog.arms:
        scene.pose(a, prog.parks[a], False)
        scene.set_active(a, False)
    scene.set_stage(0)

    reveal = {}
    for path, f in scene.chunks:
        reveal.setdefault(int(f), []).append(path)
    period = 1.0 / args.fps
    while True:
        scene.reset_ink()
        last_stage, last_live = -1, {a: None for a in prog.arms}
        t_wall = time.perf_counter()
        for i, t in enumerate(ts):
            if STAGE[i] != last_stage:
                scene.set_stage(int(STAGE[i]))
                last_stage = int(STAGE[i])
            for a in prog.arms:
                lv = bool(LIVE[a][i])
                if lv != last_live[a]:
                    scene.set_active(a, lv)
                    last_live[a] = lv
                if lv:
                    scene.pose(a, Q[a][i], True)
            for path in reveal.get(i, ()):
                scene.vis[path].set_property("visible", True)
            t_wall += period
            slack = t_wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                t_wall = time.perf_counter()
        if not args.loop:
            break
        time.sleep(args.pause)
    return url


def write_html(scene, prog, ts, Q, LIVE, STAGE, args):
    """The same scene as a self-contained html with the animation baked in."""
    from meshcat.animation import Animation
    anim = Animation(default_framerate=args.fps)
    reveal = {}
    for path, f in scene.chunks:
        reveal.setdefault(int(f), []).append(path)
    last_stage, last_live = -1, {a: None for a in prog.arms}
    for i in range(len(ts)):
        with anim.at_frame(scene.vis, i) as fr:
            if int(STAGE[i]) != last_stage:
                k = int(STAGE[i])
                for j in range(prog.n_stages):
                    fr[f"stagebar/s{j}/lit"].set_property(
                        "visible", "boolean", bool(j == k))
                    fr[f"stagebar/s{j}/dim"].set_property(
                        "visible", "boolean", bool(j != k))
                    if scene.text_ok:
                        fr[f"title/s{j}"].set_property(
                            "visible", "boolean", bool(j == k))
                last_stage = k
            for a in prog.arms:
                lv = bool(LIVE[a][i])
                if lv != last_live[a]:
                    fr[f"live/arm{a}"].set_property("visible", "boolean",
                                                    bool(lv))
                    fr[f"held/arm{a}"].set_property("visible", "boolean",
                                                    bool(not lv))
                    last_live[a] = lv
                if not lv:
                    continue
                poses = scene.robot_model.link_poses(scene.joints, Q[a][i])
                T = scene.bases[a]
                for name in scene.links:
                    if name in poses:
                        fr[f"live/arm{a}/{name}"].set_transform(T @ poses[name])
                for name, M in _tool_transforms(poses, scene.lat,
                                                scene.pen_len[a]).items():
                    fr[f"live/arm{a}/{name}"].set_transform(T @ M)
            for path in reveal.get(i, ()):
                fr[path].set_property("visible", "boolean", True)
    scene.vis.set_animation(anim, play=True, repetitions=10 ** 6)
    html = scene.vis.static_html()
    Path(args.html).write_text(html)
    print(f"wrote {args.html}  ({len(html) / 1e6:.1f} MB, {len(ts)} frames "
          f"at {args.fps} fps = {args.rate:g}x)")


# ---------------------------------------------------------------------------
# 5.  THE OFFLINE GIF
# ---------------------------------------------------------------------------
def render_gif(prog, args, h_inv, path, view="top"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image
    from aris_sixarm.fleet import FLEET, SHEET

    dt = args.gif_rate / args.gif_fps
    ts = sample_grid(prog, dt, args.stage)
    Q, DOWN, LIVE, STAGE = prog.sample(ts)
    ch = chains_for(prog, Q, h_inv)
    caps = capsule_table(ch[prog.arms[0]].shape[1])
    # top: horizontal = paper y, vertical = paper x (the sheet is 1.80 x 3.63)
    if view == "top":
        ax_h, ax_v = 1, 0
        xlim = (-0.30, SHEET[1] + 0.30)
        ylim = (-0.30, SHEET[0] + 0.30)
        xlabel, ylabel = "paper y  [m]", "paper x  [m]"
    else:                                   # side: looking along +y
        ax_h, ax_v = 0, 2
        xlim = (-0.30, SHEET[0] + 0.30)
        ylim = (-0.12, 1.25)
        xlabel, ylabel = "paper x  [m]", "z  [m]"

    fig_w = 8.0
    fig_h = fig_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) + 1.25
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.gif_dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.065, 0.40 / fig_h, 0.925,
                       1 - (0.82 + 0.40) / fig_h])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f2f2f0")
    if view == "top":
        ax.add_patch(plt.Rectangle((0, 0), SHEET[1], SHEET[0], fc="#fbfbf4",
                                   ec="#9a9a90", lw=1.0, zorder=0))
    else:
        ax.plot([0, SHEET[0]], [0, 0], color="#9a9a90", lw=2.0, zorder=0)
        ax.axhline(0.970, color="#cbb", lw=0.8, ls=":", zorder=0)

    # points per metre, for capsule radii drawn as line widths
    pts_per_m = fig_w * 0.92 * 72.0 / (xlim[1] - xlim[0])
    lw = np.array([2 * c[2] * pts_per_m for c in caps])

    arm_lc, ink_lc, base_pt = {}, {}, {}
    for a in prog.arms:
        col = FLEET[a].color
        base = FLEET[a].T_world_base(h_inv)[:3, 3]
        base_pt[a] = ax.plot([base[ax_h]], [base[ax_v]], "o", ms=7, mfc=col,
                             mec="#333", mew=0.6, zorder=6)[0]
        ax.annotate(str(a), (base[ax_h], base[ax_v]), fontsize=7,
                    xytext=(6, 6), textcoords="offset points",
                    color="#333", zorder=7)
        ink_lc[a] = LineCollection([], colors=[col], linewidths=1.6, zorder=3)
        ax.add_collection(ink_lc[a])
        arm_lc[a] = LineCollection([], colors=[col], linewidths=lw,
                                   capstyle="round", alpha=0.85, zorder=5)
        ax.add_collection(arm_lc[a])

    banner = fig.text(0.5, 1 - 0.30 / fig_h, "", ha="center", va="top",
                      fontsize=11, family="monospace")
    sub = fig.text(0.5, 1 - 0.60 / fig_h, "", ha="center", va="top",
                   fontsize=8, color="#555", family="monospace")

    from aris_sixarm.coordination import cap_endpoints
    A = {a: cap_endpoints(ch[a], caps) for a in prog.arms}
    tips = {a: ch[a][:, 9, :] for a in prog.arms}
    ink_acc = {a: [] for a in prog.arms}

    frames_out = []
    for i in range(len(ts)):
        for a in prog.arms:
            live = bool(LIVE[a][i])
            col = FLEET[a].color if live else GREY
            A0, A1 = A[a][0][i], A[a][1][i]
            arm_lc[a].set_segments([[(A0[k, ax_h], A0[k, ax_v]),
                                     (A1[k, ax_h], A1[k, ax_v])]
                                    for k in range(len(caps))])
            arm_lc[a].set_color([col])
            arm_lc[a].set_alpha(0.9 if live else PARKED_ALPHA)
            arm_lc[a].set_zorder(5 if live else 4)
            base_pt[a].set_mfc(col)
            if i and DOWN[a][i] and DOWN[a][i - 1]:
                p, q = tips[a][i - 1], tips[a][i]
                ink_acc[a].append([(p[ax_h], p[ax_v]), (q[ax_h], q[ax_v])])
                ink_lc[a].set_segments(ink_acc[a])
        k = int(STAGE[i])
        s = prog.stages[k]
        # WHO IS MOVING AND WHO IS HELD, named, every frame.  Read off LIVE,
        # not off the stage's `actives`: an active arm that has finished its
        # bucket is HELD at its park until the barrier, and with more than
        # three actives in a stage that difference is the whole picture.
        mv = [a for a in prog.arms if LIVE[a][i]]
        pk = [a for a in prog.arms if not LIVE[a][i]]
        banner.set_text(f"stage {k}/{prog.n_stages - 1}   t = {ts[i]:6.1f} s"
                        f"   moving: {' '.join(map(str, mv)) or '(barrier)'}"
                        f"   |  parked: {' '.join(map(str, pk)) or '-'}")
        sub.set_text(f"{prog.pattern}   stage actives {s['actives']}"
                     f"{'  residue ' + str(s['residues']) if s['residues'] else ''}"
                     f"   makespan {prog.makespan:.1f} s   {args.gif_rate:g}x")
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                              fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        frames_out.append(im.convert("RGB").convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=64))
    plt.close(fig)

    frames_out[0].save(path, save_all=True, append_images=frames_out[1:],
                       duration=int(round(1000.0 / args.gif_fps)), loop=0,
                       optimize=True, disposal=2)
    mb = os.path.getsize(path) / 1e6
    print(f"wrote {path}  ({len(frames_out)} frames, {args.gif_fps} fps, "
          f"{args.gif_rate:g}x, {len(frames_out) / args.gif_fps:.1f} s, "
          f"{mb:.1f} MB)")
    if args.mp4 and _which("ffmpeg"):
        mp4 = str(Path(path).with_suffix(".mp4"))
        _mp4_from(frames_out, mp4, args.gif_fps)
    return path, mb, len(frames_out) / args.gif_fps


def _which(x):
    from shutil import which
    return which(x)


def _mp4_from(frames_out, mp4, fps):                      # pragma: no cover
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for i, im in enumerate(frames_out):
            im.convert("RGB").save(f"{d}/f{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                        str(fps), "-i", f"{d}/f%05d.png", "-pix_fmt",
                        "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        mp4], check=True)
    print(f"wrote {mp4}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("program", nargs="?",
                    default="out/staged_csail_h097_program_v6.json")
    ap.add_argument("--h-inv", type=float, default=0.970)
    ap.add_argument("--meshcat", action="store_true")
    ap.add_argument("--port", type=int, default=7006)
    ap.add_argument("--hostname", default="frankastation.drl.csail.mit.edu")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="wall seconds held at the end before the loop restarts")
    ap.add_argument("--rate", type=float, default=4.0,
                    help="programme seconds per wall second (default 4x)")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--stage", type=int, default=None,
                    help="animate ONE stage only")
    ap.add_argument("--slow", action="store_true",
                    help="--rate 0.5 --fps 24: the close-approach inspection")
    ap.add_argument("--html", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--side-gif", default=None)
    ap.add_argument("--gif-fps", type=float, default=15.0)
    ap.add_argument("--gif-rate", type=float, default=8.0)
    ap.add_argument("--gif-dpi", type=int, default=78)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--analyse", "--analyze", dest="analyse",
                    action="store_true", help="re-measure the clearances")
    ap.add_argument("--analyse-dt", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.slow:
        a.rate, a.fps = 0.5, 24.0

    if os.environ.get("ARIS_RIG") != "proposed":
        print("WARNING: ARIS_RIG is not 'proposed'; this programme was planned "
              "with ARIS_RIG=proposed ARIS_TOOL=lateral", file=sys.stderr)

    doc = json.loads(Path(a.program).read_text())
    prog = Programme(doc)
    print(f"{a.program}: schema {doc['schema']}, {prog.pattern}, "
          f"{prog.n_stages} stages, makespan {prog.makespan:.1f} s")
    print("  fleet clock (stage: [t0, t1) s  actives / residue):")
    for r in prog.table():
        rs = f"  residue {r['residues']}" if r["residues"] else ""
        print(f"    stage {r['stage']}: [{r['t0']:7.2f}, {r['t1']:7.2f})  "
              f"{r['duration']:6.2f} s  actives {r['actives']}{rs}")
        for arm, (t0, t1, d, res) in r["per_arm"].items():
            print(f"        arm {arm:>3}: [{t0:7.2f}, {t1:7.2f}]  {d:6.2f} s"
                  + ("   RESIDUE (serialised after the concurrent part)"
                     if res else ""))
    dup = ink_twice(prog)
    print(f"  pieces planned twice: {len(dup)}"
          + (f"  {dup[:5]}" if dup else "  (none)"))

    if a.analyse:
        print("  clearances re-measured on the fleet clock "
              f"(dt = {a.analyse_dt} s, gate = "
              f"{1000 * doc['pair_margin_m']:.0f} mm):")
        analyse(prog, a.h_inv, a.analyse_dt)

    if a.gif:
        render_gif(prog, a, a.h_inv, a.gif, "top")
    if a.side_gif:
        render_gif(prog, a, a.h_inv, a.side_gif, "side")
    if a.meshcat or a.html:
        meshcat_run(prog, a, a.h_inv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
