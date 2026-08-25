"""The green-field layout study's registered artefacts stay sane.

The PROPOSED rig is a study output, not a build: these tests pin that it
stays env-selectable, constraint-clean, and plannable with the lateral tool
— so a later edit cannot silently break the recommendation the redesign is
based on (docs/LAYOUT_STUDY.md).
"""
import numpy as np

from aris_sixarm import fleet, frames, layout, stroke_api
from aris_sixarm.rig_final6 import SHEET_FINAL6


def test_proposed_rig_is_selectable_and_not_default():
    fl, sheet = fleet.rig("proposed")
    assert sorted(fl) == [2, 13, 17, 31, 71, 97]
    assert sheet == SHEET_FINAL6
    assert fleet.ACTIVE_RIG != "proposed"      # never the default
    assert "proposed" in fleet.RIG_NAMES


def test_proposed_layout_constraints_clean():
    assert layout.check_spacing(layout.LAYOUT_PROPOSED) == []
    W, H = SHEET_FINAL6
    for x, y in layout.LAYOUT_PROPOSED["floor"]:
        d = np.hypot(max(0.0 - x, x - W, 0.0), max(0.0 - y, y - H, 0.0))
        assert d >= layout.FLOOR_SETBACK[0] - 1e-9
    assert layout.LAYOUT_PROPOSED["h"] in (0.85, 0.922, 1.00)


def test_study_specs_are_green_field_on_merged_canvas():
    fl, _ = fleet.rig("proposed")
    for spec in fl.values():
        assert spec.static_obstacles() == []
        assert fleet.sheet_for(spec) == SHEET_FINAL6
        assert spec.mount in ("floor", "inv")
    assert sum(s.mount == "floor" for s in fl.values()) == 2
    assert sum(s.mount == "inv" for s in fl.values()) == 4


def test_proposed_inverted_arm_plans_lateral_stroke():
    fl, _ = fleet.rig("proposed")
    spec = fl[31]
    bx, by = spec.xy
    h = layout.LAYOUT_PROPOSED["h"]
    line = np.column_stack([np.linspace(bx + 0.30, bx + 0.60, 16),
                            np.full(16, by)])
    r = stroke_api.plan_stroke(line, spec,
                               dict(pen_lat=frames.PEN_LAT_HOLDER, h_inv=h))
    assert r["status"] == "ok", (r["status"], r.get("reason"))
    assert r["validation"]["ok"]
    assert r["min_sigma"] >= 0.10 and r["min_margin"] >= 0.15
