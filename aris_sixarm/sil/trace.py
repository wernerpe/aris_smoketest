"""Per-tick recording, the metrics that answer the study, and a plot.

Every array is one row per 1 kHz tick.  Positions are metres, angles radians,
torques Nm, forces newtons.  Base-frame quantities carry `_base` in the
docstring below; world (canvas) quantities carry `_world`.  The paper normal
is world +z (`Paper.normal_world`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .setpoints import PHASES

_ARRAYS = ("t", "q", "dq", "tau", "q_null", "q_null_raw", "sp_pos_base",
           "sp_quat", "sp_pos_world", "filt_pos_base", "filt_quat",
           "tip_pos_base", "tip_quat", "tip_nom_world", "tip_act_world",
           "f_normal", "penetration", "in_contact", "phase_idx",
           "stroke_idx", "press_cmd_m", "track_err_m")

# rad.  A joint this close to its FR3 stop has no authority left in that
# direction and libfranka's own braking envelope is already reducing what it
# will allow (`frames.QD_MAX` caveat); the planner's comfort gate is 0.30 rad,
# thirty times this, so a tick inside 0.02 is a failure, not a tight fit.
NEAR_LIMIT_RAD = 0.02


@dataclass
class Trace:
    """Recorded run. Build with `Recorder`, or load with `Trace.load`."""

    t: np.ndarray                 # s
    q: np.ndarray                 # (N,7) rad
    dq: np.ndarray                # (N,7) rad/s
    tau: np.ndarray               # (N,7) Nm, the LAW's output (no gravity)
    q_null: np.ndarray            # (N,7) rad, the EFFECTIVE nullspace target
    q_null_raw: np.ndarray        # (N,7) rad, the reference AS RECEIVED
    sp_pos_base: np.ndarray       # (N,3) commanded equilibrium position
    sp_quat: np.ndarray           # (N,4) xyzw
    sp_pos_world: np.ndarray      # (N,3) the same point in the canvas frame
    filt_pos_base: np.ndarray     # (N,3) after the controller's low-pass
    filt_quat: np.ndarray         # (N,4)
    tip_pos_base: np.ndarray      # (N,3) NOMINAL tip = o_t_ee
    tip_quat: np.ndarray          # (N,4)
    tip_nom_world: np.ndarray     # (N,3)
    tip_act_world: np.ndarray     # (N,3) the PHYSICAL tip
    f_normal: np.ndarray          # (N,) N along the paper normal
    penetration: np.ndarray       # (N,) m, contact-query depth
    in_contact: np.ndarray        # (N,) bool, |f| > contact_force_eps
    phase_idx: np.ndarray         # (N,) index into `phase_names`
    stroke_idx: np.ndarray        # (N,) the CSV stroke this tick serves, -1 none
    press_cmd_m: np.ndarray       # (N,) signed offset from the CSV row
    track_err_m: np.ndarray       # (N,) |setpoint - nominal tip|, base frame
    paper_z: float = 0.0
    tip_error_m: float = 0.0
    phase_names: tuple = PHASES
    meta: dict = field(default_factory=dict)

    # -- masks -------------------------------------------------------------
    def phase_mask(self, name: str) -> np.ndarray:
        """(N,) bool: the ticks in phase `name`."""
        return self.phase_idx == self.phase_names.index(name)

    @property
    def draw(self) -> np.ndarray:
        return self.phase_mask("draw")

    # -- persistence -------------------------------------------------------
    def save(self, path) -> Path:
        """Write `trace.npz`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, paper_z=self.paper_z, tip_error_m=self.tip_error_m,
            phase_names=np.array(self.phase_names, dtype=object).astype(str),
            **{name: getattr(self, name) for name in _ARRAYS})
        return path

    @staticmethod
    def load(path) -> "Trace":
        d = np.load(path, allow_pickle=False)
        kw = {name: d[name] for name in _ARRAYS}
        return Trace(paper_z=float(d["paper_z"]),
                     tip_error_m=float(d["tip_error_m"]),
                     phase_names=tuple(str(s) for s in d["phase_names"]), **kw)

    def save_setpoint_log(self, path) -> Path:
        """Write the stream this run was driven by, replayable by
        `setpoints.SetpointLog` -- the (t, pose, q_ref) log contract §1 asks
        for so a closed-loop stage can be pushed through the same geometry."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t_s=self.t, pos=self.sp_pos_base, quat=self.sp_quat,
            q=self.q_null, press_cmd_m=self.press_cmd_m,
            phase_idx=self.phase_idx,
            phase_names=np.array(self.phase_names, dtype=object).astype(str))
        return path


class Recorder:
    """Accumulates one row per tick, then freezes into a `Trace`."""

    def __init__(self, paper_z: float, tip_error_m: float,
                 phase_names: tuple = PHASES):
        self.paper_z = float(paper_z)
        self.tip_error_m = float(tip_error_m)
        self.phase_names = tuple(phase_names)
        self._rows: dict[str, list] = {name: [] for name in _ARRAYS}

    def add(self, **kw) -> None:
        for name in _ARRAYS:
            self._rows[name].append(kw[name])

    def freeze(self, meta: dict | None = None) -> Trace:
        cols = {name: np.asarray(self._rows[name]) for name in _ARRAYS}
        cols["in_contact"] = cols["in_contact"].astype(bool)
        cols["phase_idx"] = cols["phase_idx"].astype(int)
        cols["stroke_idx"] = cols["stroke_idx"].astype(int)
        return Trace(paper_z=self.paper_z, tip_error_m=self.tip_error_m,
                     phase_names=self.phase_names, meta=meta or {}, **cols)


def metrics(trace: Trace) -> dict:
    """The numbers the study is about. All SI, all floats (JSON-friendly).

      touch_frac         fraction of DRAW ticks carrying real contact force.
                         The rig's own yardstick: the briefing's 34-run mean is
                         0.444 and a good day is ~30 % air (§10).
      f_mean_n           mean normal force over ALL draw ticks (air counted as
                         zero, which is what the paper sees).
      f_mean_contact_n   the same over contacting ticks only.
      f_max_n            peak normal force during draw.
      max_penetration_m  deepest the PHYSICAL tip went below the sheet, over
                         the whole run.  This is the paper's damage measure --
                         "paper tears from DEPTH, not force" (§10).
      mean_tip_height_err_m  mean over draw ticks of (nominal tip - commanded
                         setpoint) along the paper normal: the controller's own
                         z tracking error, positive = lagging above the command.
      mean_actual_tip_height_m  mean over draw ticks of (PHYSICAL tip - paper).
                         Positive = the pen is drawing in the air.
      xy_rms_m           rms in-plane distance between the nominal tip and the
                         commanded point during draw.
      nullspace_drift_rad  max |q - q_nullspace|_inf over the run.
      tau_sat_frac       fraction of (tick, joint) pairs at the magnitude clamp.
      near_limit_frac    fraction of ticks with ANY joint within
                         NEAR_LIMIT_RAD (0.02 rad) of an FR3 stop.
      first_near_limit_s the first time that happened (nan = never).  These two
                         are the "did the arm get stuck" pair: a spring that
                         has run out of authority is invisible in the force
                         trace and obvious here.
      qref_divergence_frac  fraction of ALL ticks -- not just draw ones, since
                         a posture step happens between strokes -- where the
                         raw joint reference was further than
                         `qref_divergence_rad` from the effective nullspace
                         target, i.e. where the plan asked for a posture the
                         controller was still slewing toward.  Nonzero means
                         the file is missing the transit that would have
                         realised the reconfiguration.
    """
    draw = trace.draw
    n_draw = int(draw.sum())
    normal_z = 2                     # the paper normal is world +z
    f = trace.f_normal
    contact = trace.in_contact & draw
    below = trace.paper_z - trace.tip_act_world[:, normal_z]
    dz = (trace.tip_nom_world[:, normal_z] - trace.sp_pos_world[:, normal_z])
    xy = np.linalg.norm(
        (trace.tip_nom_world - trace.sp_pos_world)[:, :2], axis=1)
    lim = np.asarray(trace.meta.get("max_torques", np.full(7, np.inf)), float)
    near = _near_limit(trace)
    diverged = np.abs(trace.q_null_raw - trace.q_null).max(axis=1) > \
        float(trace.meta.get("qref_divergence_rad", 0.5))

    def _mean(mask, values):
        return float(values[mask].mean()) if mask.any() else float("nan")

    return {
        "n_ticks": int(len(trace.t)),
        "duration_s": float(trace.t[-1] - trace.t[0]) if len(trace.t) else 0.0,
        "n_draw_ticks": n_draw,
        "touch_frac": float(contact.sum() / n_draw) if n_draw else float("nan"),
        "f_mean_n": _mean(draw, f),
        "f_mean_contact_n": _mean(contact, f),
        "f_max_n": float(f[draw].max()) if n_draw else float("nan"),
        "max_penetration_m": float(max(0.0, below.max())) if len(below) else 0.0,
        "mean_tip_height_err_m": _mean(draw, dz),
        "mean_actual_tip_height_m": _mean(
            draw, trace.tip_act_world[:, normal_z] - trace.paper_z),
        "xy_rms_m": (float(np.sqrt(np.mean(xy[draw] ** 2)))
                     if n_draw else float("nan")),
        "track_err_mean_m": _mean(draw, trace.track_err_m),
        "nullspace_drift_rad": float(
            np.abs(trace.q - trace.q_null).max()) if len(trace.t) else 0.0,
        "tau_sat_frac": float(np.mean(np.abs(trace.tau) >= lim - 1e-9)),
        "near_limit_frac": float(near.mean()) if len(near) else 0.0,
        "first_near_limit_s": (float(trace.t[np.argmax(near)]) if near.any()
                               else float("nan")),
        "qref_divergence_frac": (float(diverged.mean()) if len(diverged)
                                 else float("nan")),
    }


def _near_limit(trace: Trace) -> np.ndarray:
    """(N,) bool: any joint within `NEAR_LIMIT_RAD` of an FR3 stop."""
    from ..frames import FR3_MAX, FR3_MIN
    if not len(trace.t):
        return np.zeros(0, bool)
    slack = np.minimum(trace.q - FR3_MIN, FR3_MAX - trace.q)
    return slack.min(axis=1) <= NEAR_LIMIT_RAD


def per_stroke_metrics(trace: Trace) -> dict:
    """-> {stroke label: metrics} for every stroke that has DRAW ticks.

    The overall numbers hide the shape of a failure: a run that draws three
    strokes perfectly and then loses the arm reports the average of the two,
    which describes neither.  Keys are `s00`, `s01`, ... in stroke order.
    """
    out = {}
    for s in sorted(set(trace.stroke_idx[trace.draw].tolist())):
        mask = trace.stroke_idx == s
        sub = Trace(**{name: getattr(trace, name)[mask] for name in _ARRAYS},
                    paper_z=trace.paper_z, tip_error_m=trace.tip_error_m,
                    phase_names=trace.phase_names, meta=trace.meta)
        out[f"s{s:02d}"] = metrics(sub)
    return out


_METRIC_UNITS = {
    "touch_frac": ("touch fraction (draw)", "", 3),
    "f_mean_n": ("mean normal force (draw)", "N", 3),
    "f_mean_contact_n": ("mean force while touching", "N", 3),
    "f_max_n": ("peak normal force (draw)", "N", 3),
    "max_penetration_m": ("max tip depth below paper", "mm", 3),
    "mean_tip_height_err_m": ("mean z tracking error (draw)", "mm", 3),
    "mean_actual_tip_height_m": ("mean physical tip height (draw)", "mm", 3),
    "xy_rms_m": ("xy tracking error rms (draw)", "mm", 3),
    "track_err_mean_m": ("mean |pose error| (draw)", "mm", 3),
    "nullspace_drift_rad": ("nullspace drift |q-q_null|inf", "rad", 4),
    "tau_sat_frac": ("torque ticks at the clamp", "", 4),
    "near_limit_frac": ("ticks within 0.02 rad of a stop", "", 4),
    "first_near_limit_s": ("first tick near a stop", "s", 2),
    "qref_divergence_frac": ("ticks q_ref ahead of target", "", 4),
    "duration_s": ("simulated duration", "s", 2),
    "n_draw_ticks": ("draw ticks", "", 0),
}
# the compact per-stroke view: what tells a good stroke from a lost one
_PER_STROKE_KEYS = ("touch_frac", "f_mean_n", "near_limit_frac",
                    "qref_divergence_frac", "xy_rms_m", "tau_sat_frac",
                    "duration_s")
_MM = {"max_penetration_m", "mean_tip_height_err_m",
       "mean_actual_tip_height_m", "xy_rms_m", "track_err_mean_m"}


def format_metrics(named: dict, keys=None) -> str:
    """-> a fixed-width table of one or more metric dicts, keyed by run name."""
    names = list(named)
    keys = tuple(_METRIC_UNITS) if keys is None else tuple(keys)
    width = max(len(_METRIC_UNITS[k][0]) for k in keys)
    head = f"{'metric':<{width}}  {'unit':<4}" + "".join(
        f"  {n:>14}" for n in names)
    lines = [head, "-" * len(head)]
    for key in keys:
        label, unit, nd = _METRIC_UNITS[key]
        cells = []
        for n in names:
            v = named[n].get(key, float("nan"))
            v = v * 1e3 if key in _MM else v
            cells.append(f"  {v:>14.{nd}f}")
        lines.append(f"{label:<{width}}  {unit:<4}" + "".join(cells))
    return "\n".join(lines)


def format_per_stroke(trace: Trace) -> str:
    """-> the per-stroke table, one column per stroke."""
    return format_metrics(per_stroke_metrics(trace), _PER_STROKE_KEYS)


def _spans(trace: Trace):
    """-> [(phase name, t_start, t_end)] for each contiguous run of ticks."""
    idx = trace.phase_idx
    cuts = [0, *(np.flatnonzero(np.diff(idx)) + 1), len(idx)]
    return [(trace.phase_names[idx[a]], float(trace.t[a]), float(trace.t[b - 1]))
            for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def plot(trace: Trace, path) -> Path:
    """Four panels: tip height, contact force, xy error, joint torques."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = trace.t
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    for i, name in enumerate(trace.phase_names):
        mask = trace.phase_idx == i
        if not mask.any() or name == "draw":
            continue
        for a in ax:
            a.fill_between(t, 0, 1, where=mask, transform=a.get_xaxis_transform(),
                           color=f"C{i}", alpha=0.08, linewidth=0)
    for name, t0, t1 in _spans(trace):                # label the long phases
        if t1 - t0 < 0.3:
            continue
        ax[0].text(0.5 * (t0 + t1), 0.96, name, ha="center", va="top",
                   fontsize=7, color="0.35",
                   transform=ax[0].get_xaxis_transform())

    z0 = trace.paper_z
    ax[0].plot(t, 1e3 * (trace.tip_nom_world[:, 2] - z0), label="nominal tip")
    ax[0].plot(t, 1e3 * (trace.tip_act_world[:, 2] - z0), label="PHYSICAL tip")
    ax[0].plot(t, 1e3 * (trace.sp_pos_world[:, 2] - z0), "k--", lw=0.8,
               label="commanded (eq. pose)")
    ax[0].axhline(0.0, color="0.4", lw=0.8)
    ax[0].set_ylabel("height above paper [mm]")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"tip_error = {1e3 * trace.tip_error_m:+.1f} mm   "
                    f"({trace.meta.get('label', '')})")

    ax[1].plot(t, trace.f_normal, color="C3")
    ax[1].set_ylabel("contact normal force [N]")

    ax[2].plot(t, 1e3 * np.linalg.norm(
        (trace.tip_nom_world - trace.sp_pos_world)[:, :2], axis=1), color="C2")
    ax[2].set_ylabel("in-plane tracking error [mm]")

    for j in range(7):
        ax[3].plot(t, trace.tau[:, j], lw=0.7, label=f"j{j + 1}")
    ax[3].set_ylabel("commanded torque [Nm]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right", ncol=7, fontsize=7)

    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
