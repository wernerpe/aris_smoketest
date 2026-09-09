"""Does re-clocking the bases 180 deg move the under-base purple cells?

The 14 purple (hover ok / no route) cells at h = 0.970 that sit under a
LEFT-column base.  Each is drawn by the ACROSS-BAY partner, never by the arm
whose base it is.  Four clockings: baseline, left column turned, right column
(the drawers) turned, all six turned.

READ THE CAVEAT BEFORE READING THE NUMBERS.  This is a STANDALONE
reconstruction of the three layers, not the pipeline, and it does NOT
reproduce the shipped map's baseline: it calls `atlas.solve_cell` directly at
the 63 mm gate, whereas the shipped atlas is a 50 mm sweep RE-GATED at 63 mm
by `scripts/regate_atlas.py`.  Re-solving finds drawing poses the re-gate never
had — it reports two drawers per cell where the atlas has one — and everything
downstream inherits that.  So the baseline here reads 13 of 14 cells feasible
where the shipped map reads 0 of 14.

What the script is therefore GOOD FOR is the comparison BETWEEN clockings under
one consistent method, and for the yaw-invariance check it prints on every
variant.  What it must NOT be used for is an absolute count.  Making it
authoritative means re-sweeping and re-gating an atlas per clocking, which is
the expensive thing this was written to avoid.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/home/franka/aris_project/aris_sixarm')
sys.path.insert(0, '/home/franka/aris_project/aris_sixarm/scripts')
import feasible_workspace as fw
from aris_sixarm import atlas, layout, mounts, paper, writing, rig_final, frames
from aris_sixarm.layout import StudySpec
from aris_sixarm.frames import roty, rotz

H = 0.970
LAT = frames.PEN_LAT_HOLDER
LEFT, RIGHT = (13, 31, 2), (17, 71, 97)
CELLS = [(0.60,0.56),(0.60,0.58),(0.62,0.58),(0.60,0.60),(0.60,0.62),
         (0.60,1.78),(0.62,1.78),(0.60,1.80),(0.60,1.82),(0.60,1.84),
         (0.60,3.00),(0.62,3.00),(0.60,3.02),(0.60,3.04)]
PARKS = {int(k): np.asarray(v['q'], float) for k, v in
         json.load(open('out/park_search_h0970_lat0860.json'))['best'].items()}
base0 = layout.build_fleet(layout.paired_grid(spacing=0.61, rows=3, h=H))


def fleet_yaw(turned):
    """The 0.970 fleet with `turned` arms clocked 180 deg about base z."""
    bare = {}
    for a, s in base0.items():
        yaw = np.pi if a in turned else 0.0
        R = roty(np.pi) @ rotz(yaw)
        bare[a] = StudySpec(a, s.name, s.mount, s.xy, yaw, True, s.color,
                            z=float(H), R=tuple(np.asarray(R, float).flatten()))
    fl = {a: StudySpec(s.arm_id, s.name, s.mount, s.xy, s.yaw, True, s.color,
                       z=s.z, R=s.R,
                       mount_boxes=tuple(mounts.obstacles_for(a, bare, H)))
          for a, s in bare.items()}
    return mounts.attach_body_columns(fl, h_inv=H)


GROUPS = atlas._gated_groups(15.0)      # cached once, reused for every solve
CANDS = atlas._candidates(15.0)


def draw_pose(spec, x, y):
    """-> q | None, using the SAME gates the atlas sweep uses."""
    Twb = spec.T_world_base(H)
    r = atlas.solve_cell(x, y, Twb, np.linalg.inv(Twb), spec, CANDS,
                         spec.pen, spec.static_obstacles(), pen_lat=LAT,
                         gate_groups=GROUPS,
                         static_margin=rig_final.STATIC_PLAN_MARGIN)
    return None if r is None else np.asarray(r[7], float)


def probe_for(fl):
    """The parked-fleet gate for one variant. -> ParkProbe."""
    from aris_sixarm import allocate, coordination
    return allocate.ParkProbe(PARKS, fl, {a: fl[a].pen for a in fl}, h_inv=H,
                              margin=float(coordination.SAFETY_M
                                           + coordination.CALIB_M))


def verdict(fl, x, y, budget, probe=None):
    """-> (drawers, feasible_arms, notes)."""
    drawers, feas, notes = [], [], []
    for a in sorted(fl):
        spec = fl[a]
        q = draw_pose(spec, x, y)
        if q is None:
            continue
        drawers.append(a)
        hov = list(_hovers(spec, q, (x, y)))
        if not hov:
            notes.append('%d:draw,no-hover' % a)
            continue
        tf = paper.travel_floor(writing.LIFT_Z, writing.LIFT_Z)
        nh = nl = both = 0
        for qh, _z in hov[:budget]:
            qh = np.asarray(qh, float)
            # BOTH LEGS OF THE SAME HOVER, which is what enter_beats needs;
            # counting them separately passes a cell no single hover can fly
            d1 = paper.route(spec, PARKS[a], qh, pen_ext=spec.pen, h_inv=H,
                             tip_floor=tf) is not None
            d2 = paper.route(spec, qh, q, pen_ext=spec.pen, h_inv=H,
                             tip_floor=paper.CONTACT_FLOOR) is not None
            nh += d1
            nl += d2
            if d1 and d2:
                # ...AND THE PARKED FLEET, which is the gate the map applies
                # after the legs plan (`feasible_workspace._enter_clear`).
                # Without it a cell the parked partners veto reads feasible.
                beats = writing.enter_beats(spec, PARKS[a], qh, q,
                                            pen_ext=spec.pen, h_inv=H)
                if beats is not None:
                    c = float(probe.clearance(
                        a, fw._dense(beats["steps"], PARKS[a])))
                    if c >= probe.margin:
                        both += 1
        if both:
            feas.append(a)
        notes.append('%d:%dhov,%ddepot,%ddesc,%dboth'
                     % (a, len(hov), nh, nl, both))
    return drawers, feas, notes


def _hovers(spec, q_draw, xy):
    # the PIPELINE's hover set: the ladder AND the fiber, which is where the
    # map's hovers actually come from (ladder alone finds none on these cells)
    return list(fw._certified_hovers(spec, q_draw, xy, H)) + \
        list(fw._fiber_hovers(spec, q_draw, xy, H, 48))


fw._init('out/atlas_proposed_h0970_lat0860_gated63', H, True, None, True,
         15.0, 48, 0.0, 300, 1, None, None, None, False)
paper.RRT_SAFE = False          # ladder only: this is a shape question
VARIANTS = [("baseline  (all yaw 0)", ()),
            ("LEFT column 180 (13,31,2)", LEFT),
            ("RIGHT column 180 (17,71,97) - the DRAWERS", RIGHT),
            ("ALL SIX 180", LEFT + RIGHT)]
for name, turned in VARIANTS:
    t0 = time.time()
    fl = fleet_yaw(turned)
    if turned:
        same = all(np.allclose(b['lo'], c['lo']) and np.allclose(b['hi'], c['hi'])
                   for b, c in zip(sorted(base0[71].static_obstacles(), key=lambda d: d['name']),
                                   sorted(fl[71].static_obstacles(), key=lambda d: d['name'])))
        print('--- %s   (obstacle set identical to baseline: %s) ---' % (name, same), flush=True)
    else:
        print('--- %s ---' % name, flush=True)
    pr = probe_for(fl)
    nfeas = ndraw = 0
    for (x, y) in CELLS:
        dr, fe, no = verdict(fl, x, y, 12, pr)
        ndraw += bool(dr); nfeas += bool(fe)
        print('   (%.2f,%.2f) draws=%s feasible=%s | %s'
              % (x, y, dr, fe, ' '.join(no)), flush=True)
    print('   => %d of %d cells have a drawer, %d FEASIBLE  (%.0f s)'
          % (ndraw, len(CELLS), nfeas, time.time() - t0), flush=True)
