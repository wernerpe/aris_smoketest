#!/usr/bin/env python3
"""Solve the pen-tip offset from touchdowns.  REPORT ONLY — writes no constant.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/touchdown_calibrate.py \
        out/touchdown_arm31.json --json out/touchdown_arm31_fit.json

    --self-test    solve a synthetic set with a known answer and stop

WHY THIS EXISTS.  `frames.PEN_LAT_HOLDER` / `PEN_EXT_HOLDER` =
0.0860369 / 0.0460262 are USER-SPECIFIED, from a photograph of the gripper and
two sentences about how far the graphite sticks out ("about 2 cm").  They have
never been measured against a robot.  The only tip number in this project that
HAS is the legacy inline pen's `PEN_EXT = 0.110`, from the arm-31 touchdown of
2026-07-12, and its derivation was one division:

    PEN_EXT = (paper_z - TCP_z_at_contact) / cos(tool tilt)

That works for a pen on the wrist axis and one touchdown.  The holder's pen is
NOT on the wrist axis — it is 86 mm across the hand — so a single touchdown
cannot separate the lateral offset from the axial one: any (lat, ext) pair that
puts the tip on the paper in THAT orientation fits.  Several touchdowns at
DIFFERENT tool orientations do separate them, and that is what this script is.

THE MATHEMATICS, WHICH IS SMALLER THAN THE PROBLEM LOOKS.  Let `p` be the tip
in the hand-TCP frame — the thing we want, three unknowns.  For a touchdown at
joints `q` on an arm whose base is `T_wb`, the tip's world height is

    z(p) = [ R_wb @ ( R_tcp(q) @ p + t_tcp(q) ) + t_wb ]_z

which is AFFINE in `p`.  Contact means `z(p) = z_paper`.  So N touchdowns are N
linear equations in three unknowns, and the fit is one `lstsq`:

    A p = b,    A_i = ( R_wb R_tcp(q_i) )[2, :],
                b_i = z_paper_i - [ R_wb t_tcp(q_i) + t_wb ]_z

No optimiser, no initial guess, and the condition number of `A` says outright
whether the poses that were taken can answer the question.

WHAT MAKES `A` ILL-CONDITIONED, AND THEREFORE WHAT TO MEASURE.  Row `i` is the
third row of the composed rotation — the direction the paper's normal points in
the HAND's frame at that pose.  Touchdowns that share a tool orientation share
a row, so twenty of them are one equation.  The rows have to SPAN, which means
varying the tool yaw about the approach axis (separates lat from the rest) and
the lean off vertical (separates lat from ext).  `--min-cond` refuses a fit
whose rows do not, rather than reporting three digits of noise.

WHAT COMES BACK, AND WHAT IT MUST BE CHECKED AGAINST.  `p_y` should be ~0: the
holder sits on the hand's x axis by construction (`frames.py`), so a `p_y` of
any size is the holder clocked out of the jaw plane and is a finding about the
BUILD, not about the pen.  And the fitted `p_z` is the number
`docs/DECISIONS.md` warns about — v18's paper-chain clearance is 20.5 mm
against a 20 mm gate, so a tip that comes back DEEPER than 0.0460262 eats that
0.5 mm first and the programme must be re-conducted before it is flown.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import frames                                  # noqa: E402

#: Refuse a fit whose design matrix is worse conditioned than this.  20 is
#: loose — it is "the three directions are distinguishable at all", not "the
#: fit is good"; the residual is the quality number.
MIN_COND = 20.0
#: The gate the fitted depth is checked against, from `docs/DECISIONS.md`:
#: v18 holds 20.5 mm of paper-chain clearance against a 20 mm floor.
V18_CHAIN_CLEARANCE_M = 0.0205
CHAIN_GATE_M = 0.020


def design(records, fleet):
    """The touchdown records -> (A, b, labels).  See the module docstring.

    Each record is `{"arm": int, "q": [7], "paper_z": float}`; `paper_z` is the
    world height of the paper AT THAT POINT (0.0 on a flat sheet in this
    project's canvas frame, or the probed plane's value if the table is not).
    """
    A, b, lab = [], [], []
    for i, r in enumerate(records):
        aid = int(r["arm"])
        q = np.asarray(r["q"], float).reshape(7)
        spec = fleet[aid]
        T_wb = spec.T_world_base()
        T_tcp, _ = frames.fk(q)
        M = T_wb[:3, :3] @ T_tcp[:3, :3]
        t = T_wb[:3, :3] @ T_tcp[:3, 3] + T_wb[:3, 3]
        A.append(M[2, :])
        b.append(float(r.get("paper_z", 0.0)) - t[2])
        lab.append(r.get("label", f"touchdown {i} (arm {aid})"))
    return np.asarray(A, float), np.asarray(b, float), lab


def solve(records, fleet, min_cond=MIN_COND):
    """-> a dict: the tip in the hand-TCP frame, plus everything to judge it."""
    A, b, lab = design(records, fleet)
    if len(A) < 3:
        raise SystemExit(f"{len(A)} touchdown(s): the tip has three unknowns "
                         "and needs at least three independent orientations")
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ p - b
    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    lat, ext = float(p[0]), float(p[2])
    out = dict(
        tip_hand_tcp=[float(x) for x in p],
        pen_lat_m=lat, pen_ext_m=ext, off_jaw_plane_m=float(p[1]),
        lean_deg=float(np.degrees(np.arctan2(np.hypot(p[0], p[1]), p[2]))),
        n=len(A), cond=cond, singular_values=[float(x) for x in s],
        resid_rms_m=float(np.sqrt(np.mean(resid ** 2))),
        resid_max_m=float(np.abs(resid).max()),
        residuals_m=[float(x) for x in resid], labels=lab,
        shipped=dict(pen_lat_m=frames.PEN_LAT_HOLDER,
                     pen_ext_m=frames.PEN_EXT_HOLDER,
                     lean_deg=float(np.degrees(frames.PEN_LEAN_HOLDER))),
        d_lat_m=lat - frames.PEN_LAT_HOLDER,
        d_ext_m=ext - frames.PEN_EXT_HOLDER,
        well_posed=bool(cond <= min_cond))
    # THE CONSEQUENCE, COMPUTED HERE SO NOBODY HAS TO REMEMBER IT.  A DEEPER
    # tip (larger axial depth) holds the wrist that much closer to the paper
    # for the same ink, which comes straight off the paper-chain clearance.
    eaten = out["d_ext_m"]
    out["chain_clearance_after_m"] = V18_CHAIN_CLEARANCE_M - eaten
    out["chain_gate_m"] = CHAIN_GATE_M
    out["chain_ok"] = bool(out["chain_clearance_after_m"] >= CHAIN_GATE_M)
    return out


def report(fit):
    L = ["TOUCHDOWN FIT — the tip in the hand-TCP frame",
         f"  {fit['n']} touchdowns, cond(A) = {fit['cond']:.1f}"
         + ("" if fit["well_posed"] else "   <-- ILL-POSED, see below"),
         f"  singular values {np.array2string(np.array(fit['singular_values']), precision=4)}",
         f"  tip = ({fit['tip_hand_tcp'][0]:+.6f}, {fit['tip_hand_tcp'][1]:+.6f},"
         f" {fit['tip_hand_tcp'][2]:+.6f}) m",
         f"  PEN_LAT  {fit['pen_lat_m']:.6f}  (shipped "
         f"{fit['shipped']['pen_lat_m']:.6f}, delta {1000 * fit['d_lat_m']:+.2f} mm)",
         f"  PEN_EXT  {fit['pen_ext_m']:.6f}  (shipped "
         f"{fit['shipped']['pen_ext_m']:.6f}, delta {1000 * fit['d_ext_m']:+.2f} mm)",
         f"  off the jaw plane {1000 * fit['off_jaw_plane_m']:+.2f} mm "
         "(should be ~0; anything else is the holder clocked out of the jaw "
         "plane, which is a build finding)",
         f"  TCP->tip ray {fit['lean_deg']:.2f} deg",
         f"  residual rms {1000 * fit['resid_rms_m']:.3f} mm, worst "
         f"{1000 * fit['resid_max_m']:.3f} mm"]
    for r, lb in zip(fit["residuals_m"], fit["labels"]):
        L.append(f"      {1000 * r:+7.3f} mm   {lb}")
    if not fit["well_posed"]:
        L += ["", "  ILL-POSED: the touchdown orientations do not span. Add "
                  "poses at different TOOL YAWS about the approach axis (that "
                  "is what separates the lateral offset) and at different "
                  "LEANS (that is what separates lateral from axial)."]
    L += ["",
          "  PAPER-CHAIN CONSEQUENCE (docs/DECISIONS.md, v18):",
          f"    v18 holds {1000 * V18_CHAIN_CLEARANCE_M:.1f} mm against a "
          f"{1000 * CHAIN_GATE_M:.0f} mm gate — {1000 * (V18_CHAIN_CLEARANCE_M - CHAIN_GATE_M):.1f} mm,",
          "    the thinnest margin in the programme.",
          f"    this tip is {1000 * fit['d_ext_m']:+.2f} mm deeper than the "
          f"shipped one -> {1000 * fit['chain_clearance_after_m']:.1f} mm",
          "    " + ("OK: the shipped programme still clears the gate."
                    if fit["chain_ok"] else
                    "*** THE GATE IS EATEN. The shipped v18 timeline must be "
                    "RE-CONDUCTED at this tool before it is flown. ***"),
          "",
          "  NOTHING WAS WRITTEN. `frames.PEN_LAT_HOLDER` / `PEN_EXT_HOLDER` "
          "are unchanged;",
          "  adopting this fit is an edit to frames.py plus a re-run of the "
          "atlas, the park",
          "  search and the programme, and it belongs with the decision."]
    return L


# ---------------------------------------------------------------------------
def synthetic(fleet, arm=31, tip=(0.0860369, 0.0, 0.0460262), n=8,
              noise_m=0.0, seed=0):
    """Touchdowns generated FROM a known tip, for the self-test and the tests.

    The poses are not planned: they are random configurations whose tip happens
    to land on z = 0 by construction — `paper_z` is set to whatever height the
    known tip reaches, which is the same equation the solver inverts, so a
    zero-noise round trip must return `tip` exactly.
    """
    rng = np.random.default_rng(seed)
    spec, recs = fleet[arm], []
    T_wb = spec.T_world_base()
    p = np.asarray(tip, float)
    lo = np.maximum(frames.FR3_MIN + 0.3, spec.q_seed - 1.2)
    hi = np.minimum(frames.FR3_MAX - 0.3, spec.q_seed + 1.2)
    for i in range(n):
        q = rng.uniform(lo, hi)
        T_tcp, _ = frames.fk(q)
        z = float((T_wb[:3, :3] @ (T_tcp[:3, :3] @ p + T_tcp[:3, 3])
                   + T_wb[:3, 3])[2])
        recs.append(dict(arm=int(arm), q=[float(x) for x in q],
                         paper_z=z + (noise_m * rng.standard_normal()
                                      if noise_m else 0.0),
                         label=f"synthetic {i}"))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("log", nargs="?", default=None,
                    help="the touchdown log: {\"touchdowns\": [{arm, q, "
                         "paper_z, label}, ...]}")
    ap.add_argument("--json", default=None, help="write the fit here")
    ap.add_argument("--min-cond", type=float, default=MIN_COND)
    ap.add_argument("--self-test", action="store_true",
                    help="solve a synthetic set with a known answer and stop")
    a = ap.parse_args(argv)

    from aris_sixarm.fleet import FLEET

    if a.self_test:
        for noise in (0.0, 0.0005):
            recs = synthetic(FLEET, noise_m=noise, n=10)
            fit = solve(recs, FLEET, a.min_cond)
            print(f"--- synthetic, {1000 * noise:.1f} mm of noise ---")
            print("\n".join(report(fit)[:9]))
        return 0

    if not a.log:
        ap.error("a touchdown log is required (or --self-test)")
    doc = json.loads(Path(a.log).read_text())
    recs = doc["touchdowns"] if isinstance(doc, dict) else doc
    fit = solve(recs, FLEET, a.min_cond)
    fit["source"] = str(a.log)
    print("\n".join(report(fit)))
    if a.json:
        Path(a.json).write_text(json.dumps(fit, indent=1))
        print(f"wrote {a.json}")
    return 0 if (fit["well_posed"] and fit["chain_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
