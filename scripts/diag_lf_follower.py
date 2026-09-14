"""Why stage A's follower keeps ZERO pieces: one bucket, measured.

    ARIS_RIG=proposed ARIS_TOOL=lateral .venv/bin/python scripts/diag_lf_follower.py \
        --lines out/csail_schedule_h097_v19_strokes.json --split 0.15 \
        --phase all --port 7007 --html out/lf_follower_diag.html

READ-ONLY on the planner.  Everything here rebuilds stage A of
`traces.leader_follower_pattern` exactly as `staged.run` does -- the same DP,
the same role order, the same `trajectory_room` -- stops at the follower with
the most offered ink, and asks five questions nobody had asked:

  1. is the follower's INK inside a leader's room, per leader, per sample?
  2. which pen-up LEG fails first, at what clearance, against which room and
     which (follower link x leader room) pair?
  3. H1..H5, each with a number (see `--phase measure`'s table);
  4. the scene, on meshcat 7007, so the geometry can be looked at;
  5. what the follower would keep if the leader's bag were capped by reach.

Phases are cached to a pickle (`--cache`) so the expensive rebuild is paid once.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from aris_sixarm import (allocate, coordination, frozen, paper, rig_final,
                         staged, stroke_api, writing)
from aris_sixarm import traces as traces_mod
from aris_sixarm.fleet import FLEET, H_INV_DEFAULT, SHEET

GATE = coordination.PAIR_MARGIN          # 0.050 m, the arm-to-arm gate
CAPS = rig_final.STATIC_CAPSULES_LAT     # the OBSERVER's capsule table
CAP_NAME = ("upper", "elbow", "forearm", "wrist", "hand", "bracket", "pen")


# ---------------------------------------------------------------------------
# 1.  REBUILD STAGE A, EXACTLY AS `staged.run` DOES
# ---------------------------------------------------------------------------
def build(lines_path, split_m, route_jobs, verbose=True):
    """The DP, then stage A's leaders in role order. -> dict of state."""
    from aris_sixarm import sequence as _sequence
    _sequence.ROUTE_JOBS = int(route_jobs)

    lines = traces_mod.load_lines(lines_path)
    pat = traces_mod.leader_follower_pattern(split_m=split_m)
    cov = traces_mod.coverage_from_atlas(staged.ATLAS_DEFAULT,
                                         arms=tuple(sorted(FLEET)))
    cap = traces_mod.capability(cov, pat)
    fl, h_inv = FLEET, H_INV_DEFAULT
    pens = {a: fl[a].pen for a in fl}
    parks = staged.shipped_parks(fl)
    opts = dict(tilt_max_deg=0.0)

    t0 = time.perf_counter()
    staged.plan_memo_clear()
    tplan, _masks, _rlog = staged.resolve_refusals(
        lines, cap, fl, pens, parks, h_inv, opts, None, [0, 1], pat, None,
        staged.REFUSAL_ROUNDS, True, None, verbose)
    buckets = staged.bucket(staged.pieces_of(tplan))
    if verbose:
        print(f"  DP: {time.perf_counter() - t0:.1f} s")

    roles = pat.roles(0)
    acts = staged.stage_actives(pat, 0)
    order = staged.role_order(roles, acts, buckets, 0)
    if verbose:
        print(f"  stage A order: {order}")

    # THE FOLLOWER UNDER TEST is the one with the most OFFERED ink.
    foll = max((a for a in order if roles[a] == "follower"),
               key=lambda a: sum(p.length_m for p in buckets.get((0, a), ())))

    fixed, arms = {}, {}
    for k, a in enumerate(order):
        if roles[a] == "follower":
            break                       # the leaders are all we need built
        st, drop = staged._fly_or_defer(
            0, a, list(buckets.get((0, a), [])), fl, pens, parks, h_inv, opts,
            True, None, None, (dict(fixed) if fixed else None), None,
            staged.LF_MAX_DROPS, writing.PARK_FREEZE, verbose)
        st.role, st.priority = roles[a], k
        arms[a] = st
        fixed[a] = staged.trajectory_room(st, fl, pens, h_inv, staged.CHECK_DT,
                                          staged.ENVELOPE_CLUSTER)
        if verbose:
            print(f"  leader {a}: {len(st.accepted)} pieces, {st.ink_m:.3f} m, "
                  f"{len(fixed[a][1])} spheres"
                  + ("" if st.timeline is not None else "  [NO TIMELINE]"))
    staged.thaw()
    return dict(split_m=split_m, order=order, roles=roles, follower=foll,
                rooms={a: (np.asarray(v[0]), np.asarray(v[1]), v[2])
                       for a, v in fixed.items()},
                leader_tl={a: (None if st.timeline is None else
                               dict(q=np.asarray(st.timeline["q"], float),
                                    t=np.asarray(st.timeline["t"], float),
                                    seg=np.asarray(st.timeline["seg"], int)))
                           for a, st in arms.items()},
                leader_ink={a: [np.asarray(p.plan["pts"], float)
                                for p in st.accepted] for a, st in arms.items()},
                pieces={a: [(p.line, p.k, p.length_m, np.asarray(p.pts, float))
                            for p in buckets.get((0, a), ())]
                        for a in order},
                parks={a: np.asarray(parks[a], float) for a in parks})


# ---------------------------------------------------------------------------
# 2.  THE MEASUREMENT PRIMITIVES
# ---------------------------------------------------------------------------
def install(observer, rooms, parks, fl=FLEET, pens=None, h_inv=H_INV_DEFAULT):
    """Freeze the room `observer` has to fly in. -> the frozen ids.

    `rooms` is {arm: (C, R)} for the arms modelled as a trajectory room; every
    other arm is its park's capsules.  Exactly `staged.freeze_stage`, with the
    leg store left alone so a probe cannot poison a run's cache.
    """
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    env = {int(a): (np.asarray(v[0], float), np.asarray(v[1], float))
           for a, v in (rooms or {}).items() if int(a) != int(observer)}
    sets = {int(a): np.asarray(parks[a], float).reshape(1, 7)
            for a in fl if int(a) != int(observer)}
    frozen.freeze_sets(sets, fl, pens, h_inv, clusters=env)
    frozen.observe(int(observer))
    paper.clear_cache()
    paper.disk_cache_close()
    return tuple(sorted(sets))


def only(room_arm, rooms, observer, parks, fl=FLEET, pens=None,
         h_inv=H_INV_DEFAULT):
    """Freeze ONE arm, as its room, and nobody else. -> None.

    The per-leader reading H4 needs: a clearance measured against this room
    alone rather than against the union the planner actually flew in.
    """
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    C, R = np.asarray(rooms[room_arm][0]), np.asarray(rooms[room_arm][1])
    frozen.freeze_sets({int(room_arm): np.asarray(parks[room_arm]).reshape(1, 7)},
                       fl, pens, h_inv, clusters={int(room_arm): (C, R)})
    frozen.observe(int(observer))
    paper.clear_cache()


def one_room(arm, C, R, observer, parks, fl=FLEET, h_inv=H_INV_DEFAULT):
    """Freeze a bare (C, R) as the only obstacle. -> None."""
    pens = {a: fl[a].pen for a in fl}
    frozen.freeze_sets({int(arm): np.asarray(parks[arm]).reshape(1, 7)},
                       fl, pens, h_inv,
                       clusters={int(arm): (np.asarray(C, float),
                                            np.asarray(R, float))})
    frozen.observe(int(observer))
    paper.clear_cache()


def chain_of(Q, arm, h_inv=H_INV_DEFAULT):
    return coordination.chain_world(np.asarray(Q, float).reshape(-1, 7),
                                    FLEET[arm], h_inv, float(FLEET[arm].pen))


def clearance(Q, arm):
    """Observer poses -> (N,) clearance to whatever is frozen."""
    return np.asarray(frozen.partner_clearance(chain_of(Q, arm)), float)


def binding_pair(Q, arm, C, R):
    """Which (observer capsule, room sphere) binds. -> dict.

    The room is spheres, so the "leader link" is named by which sphere and
    where that sphere sits; `provenance` turns that into a link and a time.
    """
    P = chain_of(Q, arm)
    C = np.asarray(C, float).reshape(-1, 3)
    R = np.asarray(R, float).reshape(-1)
    best = (np.inf, -1, -1, -1)
    for ci, (i, j, r) in enumerate(CAPS):
        d = coordination.seg_seg_dist(P[:, i][:, None, :], P[:, j][:, None, :],
                                      C[None, :, :], C[None, :, :]) \
            - (R[None, :] + r)
        k = int(np.argmin(d))
        n, s = divmod(k, d.shape[1])
        if d[n, s] < best[0]:
            best = (float(d[n, s]), ci, int(n), int(s))
    return dict(mm=1000 * best[0], link=CAP_NAME[best[1]], sample=best[2],
                sphere=best[3], sphere_xyz=[float(x) for x in C[best[3]]],
                sphere_r=float(R[best[3]]))


def room_of(Q, arm, dt=staged.CHECK_DT, cluster=staged.ENVELOPE_CLUSTER,
            h_inv=H_INV_DEFAULT, pad=None):
    """`staged.trajectory_room` from a bare (N,7) rather than an ArmStage."""
    spec = FLEET[arm]
    Q = np.asarray(Q, float).reshape(-1, 7)
    path = coordination.ArmPath(int(arm), Q, dt, h_inv, float(spec.pen), spec)
    keep = [k for k in range(len(path.r))
            if k not in coordination.FROZEN_SWEEP_BANDS]
    A3 = np.asarray(path.A, float)[:, keep]
    B3 = np.asarray(path.B, float)[:, keep]
    step = 0.0
    if len(A3) > 1:
        step = max(float(np.max(np.linalg.norm(np.diff(A3, axis=0), axis=2))),
                   float(np.max(np.linalg.norm(np.diff(B3, axis=0), axis=2))))
    R = np.tile(np.asarray(path.r, float)[keep], len(A3))
    pd = staged.SWEEP_FRAC * step if pad is None else float(pad)
    return staged.cluster_capsules(A3.reshape(-1, 3), B3.reshape(-1, 3), R,
                                   cluster, pd), pd


def exact_caps(Q, arm, h_inv=H_INV_DEFAULT):
    """The leader's UNREDUCED capsules over its timeline. -> (A, B, R)."""
    spec = FLEET[arm]
    Q = np.asarray(Q, float).reshape(-1, 7)
    path = coordination.ArmPath(int(arm), Q, staged.CHECK_DT, h_inv,
                                float(spec.pen), spec)
    keep = [k for k in range(len(path.r))
            if k not in coordination.FROZEN_SWEEP_BANDS]
    A = np.asarray(path.A, float)[:, keep].reshape(-1, 3)
    B = np.asarray(path.B, float)[:, keep].reshape(-1, 3)
    R = np.tile(np.asarray(path.r, float)[keep], len(path.A))
    return A, B, R


def cap_clearance(Q, arm, A, B, R):
    """Observer poses against a RAW capsule block (no sphere reduction)."""
    P = chain_of(Q, arm)
    out = np.full(len(P), np.inf)
    for (i, j, r) in CAPS:
        d = coordination.seg_seg_dist(P[:, i][:, None, :], P[:, j][:, None, :],
                                      A[None, :, :], B[None, :, :])
        out = np.minimum(out, (d - (R[None, :] + r)).min(axis=1))
    return out


# ---------------------------------------------------------------------------
# 3.  THE MEASUREMENTS
# ---------------------------------------------------------------------------
def measure(S, verbose=True):
    fl, h_inv = FLEET, H_INV_DEFAULT
    pens = {a: fl[a].pen for a in fl}
    parks = S["parks"]
    foll = S["follower"]
    rooms = S["rooms"]
    leaders = [a for a in S["order"] if S["roles"][a] == "leader"]
    same_row = next(a for a in leaders
                    if traces_mod.ROW_OF[a] == traces_mod.ROW_OF[foll])
    out = dict(follower=foll, leaders=leaders, same_row_leader=same_row,
               split_m=S["split_m"],
               offered=[dict(line=l, k=k, m=m) for l, k, m, _ in
                        S["pieces"][foll]],
               offered_m=float(sum(m for _, _, m, _ in S["pieces"][foll])))
    spec = fl[foll]

    # -- (1) THE INK, POSE BY POSE, AGAINST EACH ROOM SEPARATELY -----------
    # Plan the follower's pieces the way ROOM PASS 1 does: solo, against the
    # parked fleet.  `plan_stroke` never consults the static set, so this is
    # the same geometry the planner would carry into the room.
    staged.freeze_partners(foll, parks, fl, pens, h_inv, leg_cache=False)
    plans = []
    for (l, k, m, pts) in S["pieces"][foll]:
        r = stroke_api.plan_stroke(pts, spec, dict(h_inv=h_inv,
                                                   pen_ext=pens[foll],
                                                   tilt_max_deg=0.0))
        plans.append(dict(line=l, k=k, m=m, status=str(r.get("status")),
                          qs=(np.asarray(r["qs"], float)
                              if r.get("status") == "ok" else None),
                          pts=(np.asarray(r["pts"], float)
                               if r.get("status") == "ok" else None)))
    out["ink"] = []
    QI = np.concatenate([p["qs"] for p in plans if p["qs"] is not None])
    for who, rs in ([(a, {a: rooms[a]}) for a in leaders]
                    + [("union", {a: rooms[a] for a in leaders})]):
        if isinstance(who, str):
            install(foll, rs, parks)
        else:
            only(who, rooms, foll, parks)
        d = clearance(QI, foll)
        row = dict(room=str(who), n=int(len(d)), min_mm=float(1000 * d.min()),
                   p05_mm=float(1000 * np.percentile(d, 5)),
                   median_mm=float(1000 * np.median(d)),
                   frac_clear=float(np.mean(d >= GATE)))
        if not isinstance(who, str):
            row.update(bind=binding_pair(QI, foll, *rooms[who][:2]))
        out["ink"].append(row)
        if verbose:
            print(f"  ink vs room {who}: min {row['min_mm']:+.1f} mm, "
                  f"{100 * row['frac_clear']:.1f} % of {row['n']} samples "
                  f"clear the {1000 * GATE:.0f} mm gate")
    # per piece, against the same-row leader alone
    only(same_row, rooms, foll, parks)
    out["ink_per_piece"] = []
    for p in plans:
        if p["qs"] is None:
            continue
        d = clearance(p["qs"], foll)
        out["ink_per_piece"].append(
            dict(line=p["line"], k=p["k"], m=p["m"],
                 min_mm=float(1000 * d.min()),
                 frac_clear=float(np.mean(d >= GATE))))

    # -- (2) THE LEGS ------------------------------------------------------
    install(foll, rooms, parks)
    boxes = paper.static_boxes(spec)
    q0 = np.asarray(parks[foll], float).reshape(7)
    leg = dict(start_pose={}, hovers=[], legs=[])
    # the START pose, against everything and against each room
    leg["start_pose"]["vs_room_mm"] = float(1000 * clearance(q0[None], foll)[0])
    leg["start_pose"]["vs_static_mm"] = float(
        1000 * paper.chain_static(q0[None], spec, pens[foll], h_inv, boxes)[0])
    leg["start_pose"]["frame_floor_mm"] = float(1000 * paper.FRAME_FLOOR)
    for a in leaders:
        only(a, rooms, foll, parks)
        leg["start_pose"][f"vs_{a}_mm"] = float(
            1000 * clearance(q0[None], foll)[0])
    install(foll, rooms, parks)

    # the hovers the entry needs, and whether ANY rung of the ladder is legal
    gate = writing.static_gate(spec, pens[foll], h_inv)
    for p in plans:
        if p["qs"] is None:
            continue
        for end, (qr, xy) in (("start", (p["qs"][0], p["pts"][0])),
                              ("end", (p["qs"][-1], p["pts"][-1]))):
            row = dict(line=p["line"], k=p["k"], end=end)
            row["ink_pose_mm"] = float(1000 * clearance(qr[None], foll)[0])
            rung = []
            for z in writing.HOVER_LADDER:
                q = writing.hover_solve(spec, qr, xy, z=z, h_inv=h_inv,
                                        pen_ext=pens[foll], ok=gate)
                rung.append(dict(z=float(z), ok=q is not None,
                                 mm=(None if q is None else
                                     float(1000 * clearance(np.asarray(q)[None],
                                                            foll)[0]))))
            row["ladder"] = rung
            qh, zh = writing.lifted_or_lower(spec, qr, xy, h_inv=h_inv,
                                             pen_ext=pens[foll])
            row["chosen_z"] = float(zh)
            row["chosen_mm"] = float(1000 * clearance(np.asarray(qh)[None],
                                                      foll)[0])
            row["any_rung"] = bool(any(r["ok"] for r in rung))
            leg["hovers"].append(row)

    # the legs themselves, in the order `arm_program` would fly them
    good = [p for p in plans if p["qs"] is not None]
    hov = [writing.lifted_or_lower(spec, p["qs"][0], p["pts"][0], h_inv=h_inv,
                                   pen_ext=pens[foll]) for p in good]
    hox = [writing.lifted_or_lower(spec, p["qs"][-1], p["pts"][-1], h_inv=h_inv,
                                   pen_ext=pens[foll]) for p in good]
    q_home = np.asarray(spec.q_seed, float).reshape(7)

    def probe(name, a, b, floor, q_home_=None):
        t0 = time.perf_counter()
        r = paper.route(spec, a, b, pen_ext=pens[foll], h_inv=h_inv,
                        tip_floor=floor, q_home=q_home_)
        d = clearance(paper.line_samples(a, b, 65), foll)
        row = dict(leg=name, routed=r is not None,
                   mode=(None if r is None else str(r["mode"])),
                   tried=(None if r is None else int(r["tried"])),
                   straight_min_mm=float(1000 * d.min()),
                   effective_floor_mm=float(1000 * paper.effective_static_floor(
                       spec, a, b, pens[foll], h_inv, boxes=boxes)),
                   static_lb_mm=float(1000 * paper.leg_static_lb(
                       spec, a, b, pens[foll], h_inv, boxes)),
                   s=float(time.perf_counter() - t0))
        row["bind"] = binding_pair(paper.line_samples(a, b, 65), foll,
                                   *rooms[same_row][:2])
        for x in leaders:
            only(x, rooms, foll, parks)
            row[f"straight_vs_{x}_mm"] = float(
                1000 * clearance(paper.line_samples(a, b, 65), foll).min())
        install(foll, rooms, parks)
        return row

    leg["legs"].append(probe("entry.home", q0, hov[0][0],
                             paper.travel_floor(writing.LIFT_Z,
                                                writing.LIFT_Z)))
    leg["legs"].append(probe("entry.lower", hov[0][0], good[0]["qs"][0],
                             paper.CONTACT_FLOOR))
    for k in range(len(good) - 1):
        leg["legs"].append(probe(f"transit{k}.lift", good[k]["qs"][-1],
                                 hox[k][0], paper.CONTACT_FLOOR))
        leg["legs"].append(probe(
            f"transit{k}.travel", hox[k][0], hov[k + 1][0],
            paper.travel_floor(hox[k][1], hov[k + 1][1]), q_home))
        leg["legs"].append(probe(f"transit{k}.lower", hov[k + 1][0],
                                 good[k + 1]["qs"][0], paper.CONTACT_FLOOR))
    leg["legs"].append(probe("final.lift", good[-1]["qs"][-1], hox[-1][0],
                             paper.CONTACT_FLOOR))
    out["legs"] = leg
    if verbose:
        for r in leg["legs"]:
            print(f"  leg {r['leg']:18s} {'ROUTED' if r['routed'] else 'REFUSED'}"
                  f"  straight {r['straight_min_mm']:+8.1f} mm  "
                  f"floor {r['effective_floor_mm']:+6.1f}  "
                  f"lb {r['static_lb_mm']:+8.1f}  {r['mode']}")

    # -- (3) THE HYPOTHESES ------------------------------------------------
    H = {}
    base = np.asarray(fl[foll].T_world_base(h_inv), float)[:3, 3]
    C, R = np.asarray(rooms[same_row][0]), np.asarray(rooms[same_row][1])
    dxy = np.linalg.norm(C[:, :2] - base[None, :2], axis=1) - R
    H["H1"] = dict(
        name="the same-row leader's room reaches under the follower's base",
        follower_base_xy=[float(base[0]), float(base[1])],
        leader_base_xy=[float(fl[same_row].xy[0]), float(fl[same_row].xy[1])],
        room_min_xy_to_follower_base_mm=float(1000 * dxy.min()),
        room_spheres_within_300mm_of_base=int(np.sum(dxy < 0.30)),
        room_spheres=int(len(R)),
        room_x_span=[float(np.min(C[:, 0] - R)), float(np.max(C[:, 0] + R))],
        room_y_span=[float(np.min(C[:, 1] - R)), float(np.max(C[:, 1] + R))],
        room_z_span=[float(np.min(C[:, 2] - R)), float(np.max(C[:, 2] + R))],
        # the follower's own UPPER links, at every pose it is asked to hold
        upper_link_min_mm=None)
    only(same_row, rooms, foll, parks)
    QALL = np.concatenate([QI, q0[None]]
                          + [np.asarray(h[0])[None] for h in hov + hox])
    P = chain_of(QALL, foll)
    per = {}
    for ci, (i, j, r) in enumerate(CAPS):
        d = coordination.seg_seg_dist(P[:, i][:, None, :], P[:, j][:, None, :],
                                      C[None, :, :], C[None, :, :]) \
            - (R[None, :] + r)
        per[CAP_NAME[ci]] = dict(min_mm=float(1000 * d.min()),
                                 frac_clear=float(np.mean(d.min(axis=1) >= GATE)))
    H["H1"]["per_follower_link"] = per
    H["H1"]["upper_link_min_mm"] = per["upper"]["min_mm"]

    # H2: which HALF of the leader's timeline makes the offending spheres
    tl = S["leader_tl"][same_row]
    if tl is not None:
        Q = tl["q"]
        seg = tl["seg"]
        parts = {}
        for tag, m in (("pen_down", seg >= 0), ("pen_up", seg < 0)):
            if not m.any():
                continue
            (Cx, Rx), _ = room_of(Q[m], same_row)
            one_room(same_row, Cx, Rx, foll, parks)
            d = clearance(QALL, foll)
            parts[tag] = dict(spheres=int(len(Rx)),
                              min_mm=float(1000 * d.min()),
                              frac_clear=float(np.mean(d >= GATE)))
        # ...and the first/last tenth of the tour, which is the park-out leg
        n = len(Q)
        for tag, sl in (("first_10pct", slice(0, max(2, n // 10))),
                        ("last_10pct", slice(n - max(2, n // 10), n))):
            (Cx, Rx), _ = room_of(Q[sl], same_row)
            one_room(same_row, Cx, Rx, foll, parks)
            d = clearance(QALL, foll)
            parts[tag] = dict(spheres=int(len(Rx)),
                              min_mm=float(1000 * d.min()),
                              frac_clear=float(np.mean(d >= GATE)))
        H["H2"] = dict(name="the room is the leader's park-out leg / held pose",
                       parts=parts)
    install(foll, rooms, parks)

    # H3: can the router go AROUND, and is the room padded past the truth?
    A, B, Rc = exact_caps(S["leader_tl"][same_row]["q"], same_row)
    only(same_row, rooms, foll, parks)
    d_sph = clearance(QALL, foll)
    d_cap = cap_clearance(QALL, foll, A, B, Rc)
    H["H3"] = dict(
        name="the router cannot go around, or the room is padded past the truth",
        skirt_sees_only_boxes=True,      # paper._skirt(spec, xy0, xy1, boxes)
        n_static_boxes=int(len(boxes)),
        rrt_tier_on=bool(paper.RRT_SAFE),
        sphere_vs_capsule_median_mm=float(1000 * np.median(d_cap - d_sph)),
        sphere_vs_capsule_max_mm=float(1000 * np.max(d_cap - d_sph)),
        sphere_min_mm=float(1000 * d_sph.min()),
        capsule_min_mm=float(1000 * d_cap.min()),
        frac_clear_spheres=float(np.mean(d_sph >= GATE)),
        frac_clear_capsules=float(np.mean(d_cap >= GATE)),
        cluster_cell_m=float(staged.ENVELOPE_CLUSTER),
        median_sphere_r_mm=float(1000 * np.median(R)),
        max_sphere_r_mm=float(1000 * np.max(R)))
    install(foll, rooms, parks)

    # H4: each leader's room, alone -- does the bucket fly against ONE of them?
    H["H4"] = dict(name="it is the UNION of three rooms, not the same-row one",
                   per_leader={})
    for a in leaders:
        st = replan(foll, S, {a: rooms[a]}, verbose=False)
        H["H4"]["per_leader"][str(a)] = st
    H["H4"]["union"] = replan(foll, S, {a: rooms[a] for a in leaders},
                              verbose=False)
    H["H4"]["free"] = replan(foll, S, {}, verbose=False)

    # H5: is the park on the wrong side?
    install(foll, rooms, parks)
    tip = paper.tip_xy(q0, spec, pens[foll], h_inv)
    ink_xy = np.concatenate([p["pts"] for p in good])
    H["H5"] = dict(
        name="the follower's park is on the wrong side of the leader",
        park_tip_xy=[float(tip[0]), float(tip[1])],
        follower_base_xy=[float(base[0]), float(base[1])],
        leader_base_x=float(fl[same_row].xy[0]),
        offered_ink_x=[float(ink_xy[:, 0].min()), float(ink_xy[:, 0].max())],
        offered_ink_y=[float(ink_xy[:, 1].min()), float(ink_xy[:, 1].max())],
        park_is_outboard=bool((tip[0] - base[0]) *
                              (fl[same_row].xy[0] - base[0]) < 0))
    out["hypotheses"] = H
    staged.thaw()
    return out


def uniform_q(tl, dt=staged.CHECK_DT):
    """The leader's waypoint timeline on a uniform clock. -> (N,7), (N,) seg.

    `staged.trajectory_room` goes through `writing.uniform_samples`; a room
    built from the RAW waypoints charges a pad computed on jumps the timeline
    never makes, which is what made the first cut of H2 read tighter than the
    room it was a subset of.
    """
    t = np.asarray(tl["t"], float)
    Q = np.asarray(tl["q"], float)
    seg = np.asarray(tl["seg"], int)
    ts = np.arange(0.0, float(t[-1]) + 1e-9, float(dt))
    Qu = np.column_stack([np.interp(ts, t, Q[:, j]) for j in range(7)])
    su = seg[np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(seg) - 1)]
    return Qu, su


def rooms_probe(S, M):
    """H2 and H3, on the UNIFORM clock and with the pad held fixed.

    -> (H2, H3).  The only thing that changes between the whole room and a
    sub-room here is WHICH capsules are in the union: the cluster cell and the
    sweep pad are the whole trajectory's, so a subset cannot read tighter than
    the set that contains it.
    """
    fl, h_inv = FLEET, H_INV_DEFAULT
    foll, parks = S["follower"], S["parks"]
    same_row = M["same_row_leader"]
    Qu, su = uniform_q(S["leader_tl"][same_row])
    (C0, R0), pad = room_of(Qu, same_row)

    # the follower's own pose set: its park, and the ink it was offered
    staged.freeze_partners(foll, parks, fl, {a: fl[a].pen for a in fl}, h_inv,
                           leg_cache=False)
    QF = [np.asarray(parks[foll], float).reshape(1, 7)]
    for (l, k, m, pts) in S["pieces"][foll]:
        r = stroke_api.plan_stroke(pts, fl[foll], dict(h_inv=h_inv,
                                                       pen_ext=fl[foll].pen,
                                                       tilt_max_deg=0.0))
        if r.get("status") == "ok":
            QF.append(np.asarray(r["qs"], float))
    QF = np.concatenate(QF)
    staged.thaw()

    def vs(C, R):
        one_room(same_row, C, R, foll, parks)
        d = clearance(QF, foll)
        return dict(spheres=int(len(R)), min_mm=float(1000 * d.min()),
                    frac_clear=float(np.mean(d >= GATE)))

    n = len(Qu)
    parts = dict(whole=vs(C0, R0))
    for tag, m in (("pen_down", su >= 0), ("pen_up", su < 0),
                   ("first_half", np.arange(n) < n // 2),
                   ("second_half", np.arange(n) >= n // 2)):
        if not m.any():
            continue
        (Cx, Rx), _ = room_of(Qu[m], same_row, pad=pad)
        parts[tag] = vs(Cx, Rx)
    H2 = dict(name="the room is the leader's park-out leg / held pose",
              pad_mm=float(1000 * pad), n_uniform=int(n), parts=parts)

    # H3, decomposed: what a room sphere's radius is actually made of
    A, B, Rc = exact_caps(Qu, same_row)
    one_room(same_row, C0, R0, foll, parks)
    d_sph = clearance(QF, foll)
    d_cap = cap_clearance(QF, foll, A, B, Rc)
    cell = staged.ENVELOPE_CLUSTER
    H3 = dict(
        name="the router cannot go around, or the room is padded past the truth",
        skirt_sees_only_boxes=True,
        n_static_boxes=int(len(paper.static_boxes(fl[foll]))),
        rrt_tier_on=bool(paper.RRT_SAFE),
        cluster_cell_m=float(cell),
        sweep_pad_mm=float(1000 * pad),
        max_capsule_r_mm=float(1000 * np.max(Rc)),
        cell_half_diag_mm=float(1000 * 0.5 * np.sqrt(3) * cell),
        median_sphere_r_mm=float(1000 * np.median(R0)),
        max_sphere_r_mm=float(1000 * np.max(R0)),
        sphere_min_mm=float(1000 * d_sph.min()),
        capsule_min_mm=float(1000 * d_cap.min()),
        frac_clear_spheres=float(np.mean(d_sph >= GATE)),
        frac_clear_capsules=float(np.mean(d_cap >= GATE)),
        sphere_vs_capsule_median_mm=float(1000 * np.median(d_cap - d_sph)),
        sphere_vs_capsule_max_mm=float(1000 * np.max(d_cap - d_sph)))
    staged.thaw()
    print(f"  room pad {1000 * pad:.1f} mm, spheres r median "
          f"{H3['median_sphere_r_mm']:.0f} mm / max "
          f"{H3['max_sphere_r_mm']:.0f} mm")
    for tag, v in parts.items():
        print(f"  follower poses vs {tag:12s} ({v['spheres']:4d} spheres): "
              f"min {v['min_mm']:+8.1f} mm, {100 * v['frac_clear']:5.1f} % clear")
    print(f"  ...vs the leader's EXACT capsules: min "
          f"{H3['capsule_min_mm']:+.1f} mm, "
          f"{100 * H3['frac_clear_capsules']:.1f} % clear")
    return H2, H3


def replan(arm, S, rooms, verbose=False):
    """Re-plan the follower's whole bucket in a given room. -> dict.

    The same `staged._fly_or_defer` the stage uses, gate OFF, so the answer is
    about the LEGS and not about the per-piece ink gate.
    """
    fl, h_inv = FLEET, H_INV_DEFAULT
    pens = {a: fl[a].pen for a in fl}
    pieces = [staged.Piece(0, int(arm), int(l), int(k), pts, float(m))
              for (l, k, m, pts) in S["pieces"][arm]]
    st, drop = staged._fly_or_defer(
        0, int(arm), pieces, fl, pens, S["parks"], h_inv,
        dict(tilt_max_deg=0.0), False, None, None,
        (dict(rooms) if rooms else None), None, staged.LF_MAX_DROPS,
        writing.PARK_FREEZE, verbose)
    return dict(rooms=sorted(int(a) for a in rooms),
                pieces=len(st.accepted), ink_m=float(st.ink_m),
                flew=bool(st.timeline is not None),
                deferred_m=float(sum(p.length_m for p in drop)),
                note=str(st.note))


# ---------------------------------------------------------------------------
# 4.  THE REMEDY MEASUREMENT: cap the leader's bag by reach
# ---------------------------------------------------------------------------
def reach_sweep(S, radii=(0.55, 0.65, 0.75), verbose=True):
    """What the follower would keep if the leader's bag were capped. -> [dict]."""
    fl, h_inv = FLEET, H_INV_DEFAULT
    pens = {a: fl[a].pen for a in fl}
    foll, parks = S["follower"], S["parks"]
    same_row = next(a for a in S["order"] if S["roles"][a] == "leader"
                    and traces_mod.ROW_OF[a] == traces_mod.ROW_OF[foll])
    lbase = np.asarray(fl[same_row].xy, float)
    out = []
    for r in radii:
        keep = [(l, k, m, pts) for (l, k, m, pts) in S["pieces"][same_row]
                if float(np.max(np.linalg.norm(pts - lbase[None], axis=1))) <= r]
        pieces = [staged.Piece(0, int(same_row), int(l), int(k), pts, float(m))
                  for (l, k, m, pts) in keep]
        if not pieces:
            out.append(dict(r=r, leader_pieces=0, leader_m=0.0,
                            follower=dict(pieces=0, ink_m=0.0, flew=False)))
            continue
        st, _ = staged._fly_or_defer(
            0, int(same_row), pieces, fl, pens, parks, h_inv,
            dict(tilt_max_deg=0.0), False, None, None, None, None,
            staged.LF_MAX_DROPS, writing.PARK_FREEZE, False)
        if st.timeline is None:
            out.append(dict(r=r, leader_pieces=len(st.accepted),
                            leader_m=float(st.ink_m),
                            follower=dict(pieces=0, ink_m=0.0, flew=False,
                                          note="leader itself did not fly")))
            continue
        room = staged.trajectory_room(st, fl, pens, h_inv, staged.CHECK_DT,
                                      staged.ENVELOPE_CLUSTER)
        got = replan(foll, S, {int(same_row): room})
        row = dict(r=r, leader_pieces=len(st.accepted),
                   leader_m=float(st.ink_m), spheres=int(len(room[1])),
                   follower=got)
        out.append(row)
        if verbose:
            print(f"  leader reach <= {r:.2f} m: {len(st.accepted)} pieces, "
                  f"{st.ink_m:.3f} m, {len(room[1])} spheres -> follower "
                  f"{got['pieces']} pieces, {got['ink_m']:.3f} m, "
                  f"flew={got['flew']}")
    staged.thaw()
    return out


# ---------------------------------------------------------------------------
# 5.  THE SCENE
# ---------------------------------------------------------------------------
def scene(S, M, port, html=None, host="127.0.0.1"):
    """Paper, bases, the leader's room, the leader's tour, the follower's poses."""
    import asyncio
    import threading

    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from meshcat.servers.zmqserver import ZMQWebSocketBridge

    from aris_sixarm.viz import robot_model

    box, ready = {}, threading.Event()

    def run():
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            b = ZMQWebSocketBridge(host=host, port=int(port))
            box["url"], box["web"] = b.zmq_url, b.web_url
        except Exception as e:                            # pragma: no cover
            box["err"] = e
            ready.set()
            return
        ready.set()
        b.run()

    threading.Thread(target=run, daemon=True, name="meshcat-bridge").start()
    if not ready.wait(20):
        raise RuntimeError("meshcat bridge did not come up")
    if "err" in box:
        raise box["err"]
    vis = meshcat.Visualizer(zmq_url=box["url"])
    h_inv = H_INV_DEFAULT
    links, joints = robot_model.load_model()
    from aris_sixarm import frames as _frames
    lat = _frames.lat_of()

    foll, parks = S["follower"], S["parks"]
    same_row = M["same_row_leader"]
    vis["/Background"].set_property("top_color", [0.96, 0.96, 0.98])
    vis["/Background"].set_property("bottom_color", [0.84, 0.85, 0.90])
    vis["paper"].set_object(g.Box([SHEET[0], SHEET[1], 0.004]),
                            g.MeshLambertMaterial(color=0xFAFAF2))
    vis["paper"].set_transform(tf.translation_matrix(
        [SHEET[0] / 2, SHEET[1] / 2, -0.002]))
    vis["table"].set_object(g.Box([SHEET[0] + 0.40, SHEET[1] + 0.40, 0.05]),
                            g.MeshLambertMaterial(color=0x8A7358))
    vis["table"].set_transform(tf.translation_matrix(
        [SHEET[0] / 2, SHEET[1] / 2, -0.030]))

    def hexc(rgb):
        return (int(round(255 * rgb[0])) << 16 | int(round(255 * rgb[1])) << 8
                | int(round(255 * rgb[2])))

    # the six bases, and every arm's static column, at the park
    for a, spec in sorted(FLEET.items()):
        T = spec.T_world_base(h_inv)
        vis[f"bases/arm{a}"].set_object(
            g.Sphere(0.05), g.MeshLambertMaterial(color=hexc(spec.color)))
        vis[f"bases/arm{a}"].set_transform(T)
        if a in (foll, same_row):
            continue
        robot_model.add_robot(vis, f"parked/arm{a}", links, joints,
                              np.asarray(parks[a], float), T,
                              pen_color=0x999999, pen_len=_frames.ext_of(spec.pen_ext),
                              pen_lat=lat, body_color=0xBBBBBB,
                              dark_color=0x777777)
    for b in paper.static_boxes(FLEET[foll]):
        lo, hi = np.asarray(b["lo"], float), np.asarray(b["hi"], float)
        vis[f"static/{b.get('name', 'box')}"].set_object(
            g.Box(list(hi - lo)),
            g.MeshLambertMaterial(color=0x555566, opacity=0.25,
                                  transparent=True))
        vis[f"static/{b.get('name', 'box')}"].set_transform(
            tf.translation_matrix(list(0.5 * (lo + hi))))

    # THE ROOM, as translucent spheres
    C, R = np.asarray(S["rooms"][same_row][0]), np.asarray(S["rooms"][same_row][1])
    mat = g.MeshLambertMaterial(color=hexc(FLEET[same_row].color), opacity=0.16,
                                transparent=True)
    for i, (c, r) in enumerate(zip(C, R)):
        vis[f"room{same_row}/s{i}"].set_object(g.Sphere(float(r)), mat)
        vis[f"room{same_row}/s{i}"].set_transform(
            tf.translation_matrix([float(x) for x in c]))

    # ...AND as ghost poses of the leader along its tour
    tl = S["leader_tl"][same_row]
    if tl is not None:
        Q = np.asarray(tl["q"], float)
        idx = np.linspace(0, len(Q) - 1, 8).astype(int)
        for n, i in enumerate(idx):
            robot_model.add_robot(
                vis, f"ghost{same_row}/p{n}", links, joints, Q[i],
                FLEET[same_row].T_world_base(h_inv),
                pen_color=hexc(FLEET[same_row].color),
                pen_len=_frames.ext_of(FLEET[same_row].pen_ext), pen_lat=lat,
                body_color=hexc(FLEET[same_row].color), dark_color=0x333333)
    # the leader's ink
    for n, pts in enumerate(S["leader_ink"][same_row]):
        P = np.column_stack([np.asarray(pts, float),
                             np.full(len(pts), 0.004)])
        vis[f"ink{same_row}/{n}"].set_object(g.Line(
            g.PointsGeometry(P.T.astype(np.float32)),
            g.LineBasicMaterial(color=hexc(FLEET[same_row].color),
                                linewidth=4)))

    # THE FOLLOWER: park, first offered piece's start pose, the failing leg
    spec = FLEET[foll]
    Tf = spec.T_world_base(h_inv)
    poses = [("park", np.asarray(parks[foll], float), 0x1F77B4)]
    staged.freeze_partners(foll, parks, FLEET,
                           {a: FLEET[a].pen for a in FLEET}, h_inv,
                           leg_cache=False)
    first = None
    for (l, k, m, pts) in S["pieces"][foll]:
        r = stroke_api.plan_stroke(pts, spec, dict(h_inv=h_inv,
                                                   pen_ext=spec.pen,
                                                   tilt_max_deg=0.0))
        if r.get("status") == "ok":
            first = r
            break
    staged.thaw()
    if first is not None:
        poses.append(("first_stroke_start", np.asarray(first["qs"], float)[0],
                      0x2CA02C))
        install(foll, {a: S["rooms"][a] for a in S["rooms"]}, parks)
        qh, _z = writing.lifted_or_lower(spec, np.asarray(first["qs"])[0],
                                         np.asarray(first["pts"])[0],
                                         h_inv=h_inv, pen_ext=spec.pen)
        poses.append(("entry_hover_best_attempt", np.asarray(qh, float),
                      0xD62728))
        staged.thaw()
        # the follower's offered ink
        for n, (l, k, m, pts) in enumerate(S["pieces"][foll]):
            P = np.column_stack([np.asarray(pts, float),
                                 np.full(len(pts), 0.004)])
            vis[f"ink{foll}/{n}"].set_object(g.Line(
                g.PointsGeometry(P.T.astype(np.float32)),
                g.LineBasicMaterial(color=hexc(spec.color), linewidth=4)))
    for name, q, col in poses:
        robot_model.add_robot(vis, f"follower/{name}", links, joints, q, Tf,
                              pen_color=col, pen_len=_frames.ext_of(spec.pen_ext),
                              pen_lat=lat, body_color=col, dark_color=0x222222)

    # the binding link pair, in red
    bind = None
    for r in M["legs"]["legs"]:
        if not r["routed"]:
            bind = r
            break
    if bind is None and M["legs"]["legs"]:
        bind = min(M["legs"]["legs"], key=lambda r: r["straight_min_mm"])
    if bind is not None:
        b = bind["bind"]
        vis["bind/sphere"].set_object(
            g.Sphere(float(b["sphere_r"])),
            g.MeshLambertMaterial(color=0xFF0000, opacity=0.75,
                                  transparent=True))
        vis["bind/sphere"].set_transform(
            tf.translation_matrix([float(x) for x in b["sphere_xyz"]]))
    url = f"http://frankastation.drl.csail.mit.edu:{port}/static/"
    print(f"meshcat: {url}")
    if bind is not None:
        print(f"  binding leg {bind['leg']}: {bind['bind']['mm']:+.1f} mm, "
              f"follower {bind['bind']['link']} vs leader {same_row}'s room")
    if html:
        Path(html).write_text(vis.static_html())
        print(f"wrote {html}")
    return url


# ---------------------------------------------------------------------------
def _enc(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lines", default="out/csail_schedule_h097_v19_strokes.json")
    ap.add_argument("--split", type=float, default=0.15)
    ap.add_argument("--route-jobs", type=int, default=4)
    ap.add_argument("--cache", default="out/lf_follower_diag_state.pkl")
    ap.add_argument("--json", default="out/lf_follower_diag.json")
    ap.add_argument("--html", default="out/lf_follower_diag.html")
    ap.add_argument("--port", type=int, default=7007)
    ap.add_argument("--phase", default="all",
                    choices=("build", "measure", "reach", "rooms",
                             "viz", "all"))
    ap.add_argument("--serve", action="store_true",
                    help="hold the meshcat bridge open after the scene is up")
    a = ap.parse_args(argv)

    cache = Path(a.cache)
    if a.phase in ("build", "all") or not cache.exists():
        S = build(a.lines, a.split, a.route_jobs)
        cache.write_bytes(pickle.dumps(S))
        print(f"wrote {cache}")
    else:
        S = pickle.loads(cache.read_bytes())
    if a.phase == "build":
        return 0

    rep = Path(a.json)
    M = (json.loads(rep.read_text())
         if (rep.exists() and a.phase in ("viz", "reach", "rooms")) else None)
    if M is None:
        M = measure(S)
        rep.write_text(json.dumps(M, indent=1, default=_enc))
        print(f"wrote {rep}")
    if a.phase in ("rooms", "all"):
        M["hypotheses"]["H2"], M["hypotheses"]["H3"] = rooms_probe(S, M)
        rep.write_text(json.dumps(M, indent=1, default=_enc))
        print(f"wrote {rep}")
    if a.phase in ("reach", "all"):
        M["reach_sweep"] = reach_sweep(S)
        rep.write_text(json.dumps(M, indent=1, default=_enc))
        print(f"wrote {rep}")
    if a.phase in ("viz", "all"):
        scene(S, M, a.port, a.html)
        if a.serve:
            while True:
                time.sleep(3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
