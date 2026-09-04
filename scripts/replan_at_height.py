#!/usr/bin/env python3
"""RE-PLAN THE WHOLE PIPELINE AT ANOTHER MOUNTING HEIGHT — report only.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/replan_at_height.py \
        --h 0.880 --atlas out/atlas_proposed_h0880_lat0588 \
        --parks out/park_search_h0880_lat0588.json \
        --placement out/csail_place_v16_placement.json \
        --out csail_schedule_h088_v17 -- <draw.py args...>

WHY THIS CAN BE DONE AT ALL, AND WHY IT IS STILL DANGEROUS.

`ArmSpec.T_world_base(h_inv=H_INV_DEFAULT)` returns EARLY for any spec that
carries its own pose (`self.R is not None`), which every `proposed` arm does:
the height is written into the spec by `layout.build_fleet`, and the `h_inv`
argument is dead for this rig.  Measured, not assumed — `--check` proves it by
asking one spec for its base at five different `h_inv` values and requiring the
same z every time.  `aris_sixarm.fleet.H_INV_DEFAULT` is still the LEGACY 1.00
and 68 signatures bind it as a default at def time, so if that early return did
NOT hold, every one of them would silently plan a metre up.

So the height travels with the FLEET OBJECT and with nothing else.  The danger
is that the object is bound by NAME in nine modules at import time
(`from .fleet import FLEET`), and a patch that misses one leaves that module
planning the shipped rig while the log says otherwise — the exact failure mode
`docs/DECISIONS.md` records for the mixed pen pair and for the test that
measured one tool and gated on another.

THEREFORE THE PATCH IS EXHAUSTIVE AND THEN PROVED.  Every module in
`sys.modules` is walked and every attribute that IS the shipped fleet (or the
shipped park dict) is rebound; then the walk is REPEATED and the run ABORTS if
a single stale reference survives or if any arm's base z is not the height
asked for.  Nothing on disk is touched: `layout.LAYOUT_PROPOSED` and
`layout.PARK_GRID_PROPOSED` are the shipped constants throughout, and this
process dies with its answer.

The park poses come from `scripts/height_sweep.py park`'s own output for that
height, so a re-plan uses the depot set that was SEARCHED there rather than the
0.940 literals, which land inside the ink at every lower height.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def build(h, parks_json):
    """The fleet and parks for one height. -> (fleet, {arm: q}, h)."""
    import feasible_workspace as fw
    from aris_sixarm import layout
    lay = layout.paired_grid(spacing=fw.SHIPPED_PITCH, rows=3, h=float(h))
    if parks_json:
        doc = json.loads(Path(parks_json).read_text())
        if not doc.get("certifies"):
            raise SystemExit(f"{parks_json}: that search found no fleet park "
                             "set clearing the gate; refusing to plan on it")
        parks = {int(a): np.asarray(v["q"], float)
                 for a, v in doc["best"].items()}
        src = f"{Path(parks_json).name} (searched at h = {doc['h']})"
    else:
        parks = layout.certified_park_poses(layout.build_fleet(lay),
                                            layout.PARK_GRID_PROPOSED)
        src = "PARK_GRID_PROPOSED re-derived (the 0.940 recipe)"
    return layout.build_fleet(lay, q_park=parks), parks, src


def prove_h_is_in_the_spec():
    """The early return this whole script rests on. -> raises if it moved."""
    from aris_sixarm import layout
    s = layout.FLEET_PROPOSED[31]
    zs = {round(float(s.T_world_base(v)[2, 3]), 9)
          for v in (None, 1.00, 0.94, 0.88, 0.50)}
    if len(zs) != 1:
        raise SystemExit(
            "ArmSpec.T_world_base now HONOURS h_inv for the proposed rig "
            f"(got {sorted(zs)}); the height no longer travels with the spec "
            "alone and this script's patch is unsound.  Stop.")
    return zs.pop()


def _ours(mod_name):
    return str(mod_name).split(".")[0] in (
        "aris_sixarm", "draw", "csail_schedule", "csail_allocate",
        "csail_trace", "csail_place", "feasible_workspace")


def _is_proposed_fleet(v, ids, h_ship):
    """The PROPOSED rig at its shipped height, whoever built the dict. -> bool.

    BY SHAPE, NOT BY IDENTITY, and that is the whole point: `fleet.FLEET` is a
    DIFFERENT dict object from `layout.FLEET_PROPOSED` holding the SAME specs,
    and six modules bind that second object.  An identity-only patch rebinds
    two names, leaves `scene_check`, `allocate`, `writing`, `coordination`,
    `atlas` and `idle` pointing at the shipped height, and reports success.

    Narrowed to the six proposed arms AT 0.940 so the other rigs' registries
    (`FLEET_SIXARM`, `FLEET_FINAL6`, …) are left alone: they are not what this
    run plans with, and replacing them would be a lie in any traceback.
    """
    if not isinstance(v, dict) or set(v) != ids:
        return False
    try:
        return all(hasattr(s, "T_world_base")
                   and abs(float(s.T_world_base()[2, 3]) - h_ship) < 1e-9
                   for s in v.values())
    except Exception:
        return False


def repoint(new_fleet, new_parks, old_parks, ids, h_ship):
    """Rebind the proposed fleet and the shipped parks. -> [what changed]."""
    hits = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not _ours(mod_name):
            continue
        for attr in list(vars(mod)):
            try:
                cur = getattr(mod, attr)
            except Exception:
                continue
            if _is_proposed_fleet(cur, ids, h_ship):
                setattr(mod, attr, new_fleet)
                hits.append(f"{mod_name}.{attr}=FLEET")
            elif cur is old_parks:
                setattr(mod, attr, new_parks)
                hits.append(f"{mod_name}.{attr}=PARKS")
    return hits


def other_rig_registries():
    """The registries that are NOT the proposed rig, by identity. -> [dict].

    They are keyed by the same six arm ids and they legitimately sit at other
    heights (the legacy booth at 1.00, `final6` at 0.776/0.922), so the audit
    has to exempt them by identity rather than by shape — and it has to take
    that identity BEFORE the patch, or a rebound name would look exempt.
    """
    out = []
    from aris_sixarm import fleet as _f
    for name in ("FLEET_SIXARM", "FLEET_FINAL"):
        v = getattr(_f, name, None)
        if isinstance(v, dict):
            out.append(v)
    try:
        from aris_sixarm import rig_final6 as _r
        for name in ("FLEET_FINAL6", "FLEET_FINAL6_OPT"):
            v = getattr(_r, name, None)
            if isinstance(v, dict):
                out.append(v)
    except Exception:
        pass
    return out


def audit(h, exempt):
    """Refuse to continue if ANY loaded proposed fleet is not at `h`. -> raises.

    Checks the SPECS, not the container, so a mapping this script never saw
    still fails the gate rather than passing it quietly.
    """
    from aris_sixarm import layout
    ids = set(layout.FLEET_PROPOSED)
    stale = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not _ours(mod_name):
            continue
        for attr in list(vars(mod)):
            try:
                cur = getattr(mod, attr)
            except Exception:
                continue
            if not isinstance(cur, dict) or set(cur) != ids:
                continue
            if any(cur is e for e in exempt):
                continue
            for a, s in cur.items():
                if not hasattr(s, "T_world_base"):
                    break
                z = float(s.T_world_base()[2, 3])
                if abs(z - h) > 1e-9:
                    stale.append(f"{mod_name}.{attr}[{a}] z={z}")
    if stale:
        raise SystemExit("A FLEET AT THE WRONG HEIGHT SURVIVES THE PATCH: "
                         f"{stale[:12]}\nRefusing to plan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--parks", default=None,
                    help="height_sweep.py park JSON for this height")
    ap.add_argument("--check", action="store_true",
                    help="patch, prove, print, and stop without planning")
    a, rest = ap.parse_known_args()
    # `--` survives `parse_known_args`, and argparse treats EVERYTHING after a
    # bare `--` as positional, so leaving it in turns every one of draw.py's
    # own flags into an unrecognised positional.
    rest = [v for v in rest if v != "--"]

    import draw                                        # scripts/draw.py
    from aris_sixarm import layout

    z_before = prove_h_is_in_the_spec()
    print(f"the spec is the truth: T_world_base ignores h_inv, z = {z_before}")

    ids, h_ship = set(layout.FLEET_PROPOSED), z_before
    exempt = other_rig_registries()
    old_parks = layout.Q_PARK_PROPOSED
    new_fleet, new_parks, src = build(a.h, a.parks)
    hits = repoint(new_fleet, new_parks, old_parks, ids, h_ship)
    layout.FLEET_PROPOSED = new_fleet
    layout.Q_PARK_PROPOSED = new_parks
    import feasible_workspace as fw
    layout.PARK_HOVER_PROPOSED = fw.park_hovers(new_fleet, new_parks, a.h)
    audit(a.h, exempt)

    print(f"RE-PLAN AT h = {a.h}  (report only; layout.py untouched)")
    print(f"  parks: {src}")
    print(f"  rebound {len(hits)} name(s): " + ", ".join(sorted(hits)))
    print(f"  bases: { {x: round(float(new_fleet[x].T_world_base()[2,3]), 4) for x in sorted(new_fleet)} }")
    print(f"  park hovers: {layout.PARK_HOVER_PROPOSED}")
    print(f"  atlas: {a.atlas}", flush=True)
    if a.check:
        return
    draw.main(["--atlas", a.atlas] + rest)


if __name__ == "__main__":
    main()
