"""C1 smoothing of the PWL redundancy plan, by rounding its corners.

`pwl.plan_pwl` hands back a handful of knots — the rim arc is 2, the R bowl 6 —
and that polyline is already feasible: every segment was chased with
case-consistent IK before it was accepted.  What it is not is differentiable.
At each interior knot dq7/ds jumps, and since the joints track q7 through the
IK, so does dq/ds: the arm is asked to change its null-space drift rate
instantaneously.  Nothing about the *geometry* objects — the pen stays on the
line, the gates hold — but a velocity discontinuity is a step input to whatever
tracks it, and it is the one defect a 6-knot plan has that a 131-step schedule
does not.  This module removes it without giving up the compactness.

WHY THIS IS EASY, AND WHY THAT IS THE POINT.  The plan is a *graph over s*:
one q7 per arc length, by construction (planner.py fixes yaw so q7 carries the
whole 1-D self-motion, and the pen advances monotonically along the stroke).  A
graph cannot double back, so:

  * monotonicity in s needs no constraint, no check and no repair — it is a
    property of the representation, not of the solution.  Every smoother that
    fights to keep a path from reversing is paying for a representation that
    allowed reversal in the first place;
  * "smooth the plan" is the scalar problem of smoothing a 1-D function, not
    the 7-D problem of smoothing a trajectory through a constraint manifold.
    The joints are never smoothed at all.  They are RE-SOLVED from the smoothed
    q7(s), the same way `pwl.backout` re-solves them from the polyline, so the
    tip stays pinned to the paper to 1e-12 m no matter what the smoother does.

CORNER ROUNDING, NOT A GLOBAL FIT.  A spline through the knots would be C2 but
would leave the straight segments — the thing the RDP simplification worked to
find, and the thing that is cheap to certify, log and re-time — and it would
move q7 everywhere, so the whole stroke would need re-certifying.  Instead each
interior corner is replaced by a quadratic Bezier over a window in s, and the
segments between windows are untouched.  Only the windows can fail, so only the
windows have to be repaired, and the repair is obvious: shrink the window.  At
window zero the corner is the original sharp knot, which `plan_pwl` already
certified — the fallback is known-valid, so the bisection always terminates.

WHY A QUADRATIC BEZIER IS EXACTLY THE RIGHT PRIMITIVE HERE.  Take the window
symmetric, [s_k - w, s_k + w], with control points P0 = (s_k - w, f(s_k - w)),
P1 = the corner itself (s_k, q7_k), P2 = (s_k + w, f(s_k + w)).  The three
control abscissae are then equally spaced, so the Bezier's s-component
collapses to s(t) = s_k + w(2t - 1) — exactly linear.  That has three
consequences that together make the rest of the module trivial:

  1. q7 is an explicit quadratic polynomial IN s, so f(s) is a closed-form
     function, not a curve to be inverted or root-found;
  2. it is still a graph over s, so monotone-s survives rounding untouched;
  3. dq7/ds = (1-t) m_in + t m_out — the slope interpolates LINEARLY between
     the two segment slopes, so it matches them exactly at the window edges
     (C1) and never leaves the interval between them.  Rounding therefore
     cannot make |dq7/ds| worse anywhere; it can only take the step out of it.

The blend is C1 and not C2 (curvature jumps at the window edges); a cubic with
a doubled middle control point buys nothing here, because it loses the linear-s
property above and still is not C2 at the joins.  C2 would need the curvature
to vanish at the edges, which is a different (and much less local) primitive.
"""
import numpy as np

from . import planner, pwl
from .pwl import MARGIN_GATE, SIGMA_GATE

ROUND_FRAC = 0.5     # default window, as a fraction of the largest admissible
MAX_BISECT = 6       # halvings before a corner is given up and left sharp


def max_windows(knots):
    """Largest admissible half-window per INTERIOR knot -> (K-2,).

    Half the shorter adjacent segment.  That bound does two jobs at once: it
    keeps a blend inside the two segments that meet at its corner (so rounding
    never reaches past a neighbouring knot and changes a decision it was not
    asked about), and since two adjacent blends each claim at most half of the
    segment they share, they can touch but never overlap.  Non-overlap is what
    lets the evaluator write blends into the polyline one at a time with no
    bookkeeping, and what keeps the result single-valued.
    """
    s = np.asarray(knots, float)[:, 0]
    seg = np.diff(s)
    return 0.5 * np.minimum(seg[:-1], seg[1:])


class RoundedPWL:
    """C1 q7 = f(s): a knot polyline with quadratic-Bezier corners.

    Callable on a scalar or an array of normalised arc length; `deriv` gives
    dq7/ds in the same coordinate.  `windows` holds one half-window per
    INTERIOR knot (length K-2), in units of s — that is the per-knot rounding
    parameter, and a zero entry means "leave this corner sharp".  Windows are
    clipped to `max_windows` on construction, so an over-eager request is
    silently made legal rather than allowed to produce overlapping blends.
    """

    def __init__(self, knots, windows):
        self.knots = np.asarray(knots, float)
        if len(self.knots) < 2:
            raise ValueError("need at least two knots")
        self.s, self.q = self.knots[:, 0], self.knots[:, 1]
        seg = np.diff(self.s)
        if np.any(seg <= 0):
            raise ValueError("knots must be strictly increasing in s "
                             "— q7(s) is a graph over s")
        self.slope = np.diff(self.q) / seg
        self.wmax = max_windows(self.knots)
        w = np.broadcast_to(np.atleast_1d(np.asarray(windows, float)),
                            self.wmax.shape)
        self.windows = np.clip(w, 0.0, self.wmax).astype(float)

    def with_windows(self, windows):
        """A sibling curve over the same knots with different rounding."""
        return RoundedPWL(self.knots, windows)

    def _blends(self):
        for k, w in enumerate(self.windows, start=1):
            if w > 0:
                yield k, float(w)

    def __call__(self, s):
        s_in = np.asarray(s, float)
        sv = np.atleast_1d(s_in)
        q = np.interp(sv, self.s, self.q)            # the sharp polyline
        for k, w in self._blends():
            m = (sv > self.s[k] - w) & (sv < self.s[k] + w)
            if not m.any():
                continue
            t = (sv[m] - (self.s[k] - w)) / (2.0 * w)
            a = self.q[k] - self.slope[k - 1] * w    # value at the window edges,
            b = self.q[k] + self.slope[k] * w        # i.e. still ON the segments
            q[m] = (1 - t) ** 2 * a + 2 * (1 - t) * t * self.q[k] + t ** 2 * b
        return q.reshape(s_in.shape) if s_in.ndim else float(q[0])

    def deriv(self, s):
        """dq7/ds in NORMALISED s (multiply by 1/arc_len for rad per metre)."""
        s_in = np.asarray(s, float)
        sv = np.atleast_1d(s_in)
        i = np.clip(np.searchsorted(self.s, sv, side="right") - 1,
                    0, len(self.slope) - 1)
        d = self.slope[i].astype(float)
        for k, w in self._blends():
            m = (sv > self.s[k] - w) & (sv < self.s[k] + w)
            if not m.any():
                continue
            t = (sv[m] - (self.s[k] - w)) / (2.0 * w)
            d[m] = (1 - t) * self.slope[k - 1] + t * self.slope[k]
        return d.reshape(s_in.shape) if s_in.ndim else float(d[0])

    def corner_span(self, k):
        """(s_lo, s_hi) covered by the blend at interior knot k."""
        w = float(self.windows[k - 1])
        return self.s[k] - w, self.s[k] + w


def smooth_q7_of_s(knots, frac=ROUND_FRAC, windows=None):
    """PWL knots (K,2) -> a C1 function q7 = f(s).  See RoundedPWL.

    `frac` scales every corner's window to a fraction of the largest one that
    fits (scalar, or per-interior-knot); `windows` sets them in units of s
    directly and overrides `frac`.  Either way the result is clipped to what
    fits, so callers can ask for more than is possible without checking.

    frac = 1 rounds maximally — for equal-length segments the blends then meet
    at the segment midpoints and no straight piece survives between them, which
    is legal but throws away the compact description the polyline exists for.
    The default 0.5 keeps half of every segment straight.

    NOTE the result is NOT certified.  Rounding moves q7 off the certified
    polyline inside each window, and the band it moves into is not convex —
    a corner can be cut into a hole.  Run `certify` before believing it.
    """
    knots = np.asarray(knots, float)
    wmax = max_windows(knots)
    w = wmax * np.asarray(frac, float) if windows is None \
        else np.asarray(windows, float)
    return RoundedPWL(knots, w)


def _blend_at(curve, s_val, tol=0.0):
    """The rounded corner covering s_val, or None if the failure is not ours."""
    best, best_d = None, np.inf
    for k, w in curve._blends():
        d = abs(float(s_val) - curve.s[k])
        if d <= w + tol and d < best_d:
            best, best_d = k, d
    return best


def certify(stroke_pts, spec, curve, lat=None, sheet=None, ds=0.005,
            q_seed=None, h_inv=None, pen_ext=None, sigma_gate=SIGMA_GATE,
            margin_gate=MARGIN_GATE, jump=planner.JUMP_THRESH,
            max_bisect=MAX_BISECT, verbose=False):
    """Certify a smoothed q7(s) against the real kinematics, shrinking windows.

    THE SAME MACHINERY AS THE POLYLINE.  The curve is sampled at the stroke's
    execution resolution (`ds` metres) and chased with `pwl.chase_cc` — the
    identical helper `Corridor` screens candidate segments with and `backout`
    walks the final plan with — demanding at every sample a case-consistent IK
    solution on one branch, continuity (||dq||_inf <= jump), margin >= 0.15 and
    sigma_min >= 0.10.  Nothing about the smoother gets a softer test than the
    polyline it came from; in fact it gets a stricter one, because plan_pwl
    certifies at the lattice's s resolution and this runs at 5 mm.

    REPAIR IS LOCAL BECAUSE THE PRIMITIVE IS.  A failure at arc length s* is
    attributed to the blend whose window contains s*, and that window alone is
    halved.  Outside the windows the smoothed curve is q7-identical to the
    polyline and the chase is on the same branch, so the configuration there is
    the polyline's configuration exactly — which means a failure outside every
    window is not the smoother's doing and shrinking windows cannot fix it.
    That case is reported rather than iterated on.

    Halving converges to the sharp corner, which `plan_pwl` already certified,
    so the loop terminates: after `max_bisect` halvings the window is snapped
    to zero and the corner is left as the polyline had it.

    Returns everything `pwl.backout` returns (qs, sigmas, margins, tip_err,
    dqds, ...) plus:
        curve       the certified RoundedPWL (a copy; the input is untouched)
        windows     final half-windows, and windows0 as requested
        shrunk      (K-2,) bool, corners the bisection had to pull in
        n_bisect    how many halvings it took
        history     [(iteration, corner k, s of failure, reason, new window)]
        certified   True if the final chase passed every gate at every sample
    """
    curve = curve.with_windows(curve.windows)          # never mutate the input
    w0 = curve.windows.copy()
    setup = pwl.stroke_setup(stroke_pts, spec, curve, lat=lat, ds=ds,
                             sheet=sheet, q_seed=q_seed, h_inv=h_inv,
                             pen_ext=pen_ext)
    if setup["q_seed"] is None:
        return dict(ok=False, certified=False, curve=curve, windows=w0,
                    windows0=w0, shrunk=np.zeros(len(w0), bool), n_bisect=0,
                    history=[("seed", None, 0.0, "no IK at s=0", 0.0)],
                    fails=len(setup["pts"]), fallbacks=0, cut_index=-1)

    history, n_bisect = [], 0
    for it in range(max_bisect + 1):
        ch = pwl.chase_cc(setup["poses"], setup["q7"], setup["q_seed"],
                          pen_ext=setup["pen_ext"], margin_gate=margin_gate,
                          sigma_gate=sigma_gate, jump_gate=jump)
        if ch["ok"]:
            break
        i_fail = min(ch["stop_index"], len(setup["s"]) - 1)
        s_fail = float(setup["s"][i_fail])
        if it == max_bisect:             # last pass is verification only
            history.append((it, None, s_fail, ch["stop"]
                            + " (bisections exhausted)", 0.0))
            break
        k = _blend_at(curve, s_fail, tol=ds / max(setup["arc_len"], 1e-9))
        if k is None:                    # the sharp polyline fails here too
            history.append((it, None, s_fail, ch["stop"] + " (outside every "
                            "window — not the rounding)", 0.0))
            break
        w = 0.0 if it == max_bisect - 1 else float(curve.windows[k - 1]) / 2.0
        nw = curve.windows.copy()
        nw[k - 1] = w
        curve = curve.with_windows(nw)
        n_bisect += 1
        history.append((it, k, s_fail, ch["stop"], w))
        if verbose:
            print(f"    certify: {ch['stop']} at s={s_fail:.4f} -> corner {k} "
                  f"window {nw[k - 1]:.5f}")
        setup["q7"] = np.atleast_1d(np.asarray(curve(setup["s"]), float))

    out = dict(certified=False, curve=curve, windows=curve.windows.copy(),
               windows0=w0, shrunk=curve.windows < w0 - 1e-12,
               n_bisect=n_bisect, history=history)
    if not ch["n"]:
        out.update(ok=False, fails=ch["fails"], fallbacks=ch["fallbacks"],
                   cut_index=-1)
        return out
    rep = pwl.chase_report(ch, setup, jump=jump)
    rep.update(out)
    rep["certified"] = bool(rep["ok"] and rep["min_sigma"] >= sigma_gate
                            and rep["min_margin"] >= margin_gate
                            and rep["continuous"])
    return rep


def smooth_report(name, sm, raw=None):
    """Terse lines; with `raw` (a pwl.backout dict) the before/after comparison."""
    nb = int(sm["shrunk"].sum())
    w = sm["windows"]
    out = [f"  smooth: {len(w)} interior corners, windows "
           + ("(none — the plan is a single segment)" if not len(w) else
              " ".join(f"{x:.4f}" for x in w))
           + f"; {nb} shrunk, {sm['n_bisect']} bisections, "
           + ("CERTIFIED" if sm["certified"] else "NOT certified")]
    out.append(f"          min_sigma={sm['min_sigma']:.4f} "
               f"min_margin={sm['min_margin']:.3f} "
               f"tip_err={sm['tip_err']:.2e} m "
               f"max|dq/ds|={sm['max_dqds']:.2f} rad/m "
               f"travel={sm['sum_travel']:.2f} rad")
    if raw is not None:
        out.append(f"          vs raw PWL: sigma {raw['min_sigma']:.4f} -> "
                   f"{sm['min_sigma']:.4f}, margin {raw['min_margin']:.3f} -> "
                   f"{sm['min_margin']:.3f}, travel {raw['sum_travel']:.2f} -> "
                   f"{sm['sum_travel']:.2f} rad, max|dq/ds| "
                   f"{raw['max_dqds']:.2f} -> {sm['max_dqds']:.2f} rad/m, "
                   f"corner step in dq/ds {raw['max_dqds_jump']:.3f} -> "
                   f"{sm['max_dqds_jump']:.3f} rad/m per sample")
    return out
