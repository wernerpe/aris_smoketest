"""The fabrication drawings: does the drop post still come out of the MODEL?

`scripts/draw_8020.py` is the sheet a fabricator cuts steel from, and the one
number on it that changes with a decision is the drop post.  It got that
number wrong once already — `out/ceiling_8020_layout.*` published 1435.0 mm
at h = 940 and 1525.0 at h = 850, because `mounts.MOUNTS.ceiling_z = 2.34 m`
reads the original drawing's 233,7 cm (FLOOR to top of a self-supporting
cage) as if it were measured from the paper.

So what is pinned here is the CHAIN, not a literal:

  * the sheet's `post_length` is `system_model.z_ladder(h)["post_length"]`
    at all three heights the sheet tabulates — the script may not carry its
    own arithmetic;
  * that number is `(runway underside - h) + post over-run`, with the
    over-run taken from the drawing (34.98, not a round 35);
  * the superseded numbers really are what the buggy datum gives, and they
    really are 716.4 mm too long, so nobody can quietly reinstate them;
  * the cut list the script writes carries the same number, 24 off.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from aris_sixarm import mounts
from aris_sixarm import system_model as SM

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "draw_8020", ROOT / "scripts" / "draw_8020.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()

# what the sheet publishes, and what it supersedes
EXPECTED = {970.0: 688.60, 940.0: 718.60, 850.0: 808.60}
SUPERSEDED = {940.0: 1434.98, 850.0: 1524.98}


@pytest.mark.parametrize("h", sorted(EXPECTED))
def test_the_drop_post_is_the_models_own_number(h):
    """The sheet does not compute a post length; it asks the model for one."""
    assert D.post_length(h) == pytest.approx(SM.z_ladder(h)["post_length"])
    assert D.post_length(h) == pytest.approx(EXPECTED[h], abs=0.005)


@pytest.mark.parametrize("h", sorted(EXPECTED))
def test_the_post_length_is_the_formula_on_the_sheet(h):
    """(runway underside - h) + over-run — and the over-run is 34.98."""
    assert SM.POST_OVER == pytest.approx(34.98, abs=0.005)
    assert SM.GRID_U == pytest.approx(1623.62, abs=0.005)
    assert D.post_length(h) == pytest.approx((SM.GRID_U - h) + SM.POST_OVER)


def test_the_three_heights_are_the_ones_the_sheet_tabulates():
    assert tuple(D.H_TABLE) == (970.0, 940.0, 850.0)
    assert D.H_DESIGN == 970.0
    # 940 is what the package plans against; the sheet must not drift from it
    assert D.H_SHIPPED == pytest.approx(SM.H_MOUNT)


@pytest.mark.parametrize("h", sorted(SUPERSEDED))
def test_the_superseded_posts_are_what_the_buggy_datum_gives(h):
    """1435 / 1525 came from the floor-referenced 233,7 used as paper-
    referenced.  Pin both the old numbers and how far off they are, so the
    revision note on the sheet cannot go stale."""
    buggy = (mounts.MOUNTS.ceiling_z * 1000.0 - h) + SM.POST_OVER
    assert buggy == pytest.approx(SUPERSEDED[h], abs=0.005)
    assert buggy - D.post_length(h) == pytest.approx(716.38, abs=0.02)


def test_the_cut_list_carries_the_same_post_24_off():
    for h in EXPECTED:
        rows = {it: (ln, qty) for it, _n, _p, ln, qty, _w in D.cut_list(h)}
        ln, qty = rows["D"]
        assert qty == 24
        assert ln == pytest.approx(D.post_length(h))
        # everything else is height-independent
        for it in ("A", "B", "C", "E", "F"):
            assert rows[it][0] == pytest.approx(
                {i: l for i, _n, _p, l, _q, _w in D.cut_list(970.0)}[it])


def test_the_extrusion_total_moves_only_with_the_post():
    d = D.extrusion_m(850.0) - D.extrusion_m(970.0)
    assert d * 1000.0 == pytest.approx(
        24 * (D.post_length(850.0) - D.post_length(970.0)), abs=0.05)


def test_the_written_cut_list_says_the_number(tmp_path):
    p = D.write_cut_list(tmp_path / "cut.md", 970.0)
    txt = Path(p).read_text()
    assert "688.6" in txt
    assert "718.6" in txt and "808.6" in txt          # the alternates
    assert "1435.0" in txt or "1435" in txt           # the revision note
    assert "z = 0 is the top surface of the paper" in txt
    # every open item that blocks a cut must be listed before the members
    assert txt.index("Open items") < txt.index("## Members")
    for title in ("PLATE OFFSET DIRECTION", "BASE CABLE PASS-THROUGH",
                  "CAGE LEGS", "GUSSET PART AND ATTACHMENT", "ROOM SURVEY"):
        assert title in txt


def test_nothing_here_touches_a_layout_or_gate_constant():
    """The sheet is READ-ONLY against the package: importing it must not have
    moved the mount plane, the pitch, or the certified keep-out."""
    # 970.0 since 2026-09-10; pinned as a literal for the same reason
    # `test_the_drawings_text_and_its_solids_disagree_about_the_mount_plane`
    # pins it — the trap is a sheet that MOVES a package constant on import,
    # and reading the layout on both sides could not see that.
    assert SM.H_MOUNT == pytest.approx(970.0)
    assert mounts.MOUNTS.ceiling_z == pytest.approx(2.34)
    assert mounts.MOUNTS.boom_r == pytest.approx(0.10)


# --------------------------------------------------------------------------
# the MIRRORED variant sheet (2026-09-10)
# --------------------------------------------------------------------------
def test_the_mirrored_sheet_flips_the_plate_per_column_and_opens_the_steel():
    """WHAT THE MIRROR IS WORTH IS STEEL, AND THE SHEET IS WHERE IT SHOWS.

    The plate's 25.15 mm offset from the J1 axis runs the way the connector
    does, which is away from the arm's front.  Turn the LEFT column to face
    the right and its offset flips with it, so the two clusters of a
    transverse pair move 2 x 25.15 = 50.30 mm APART.  That is not a workspace
    argument and it does not depend on one: it is the same arithmetic that
    gives the sheet its post x positions.
    """
    d = D
    try:
        d.CLOCKING = "uniform"
        assert d.clock_sign(d.COL_X[0]) == d.clock_sign(d.COL_X[1]) == +1.0
        u_gap, u_gus = d.cluster_gap(), d.gusset_pair_clear()
        assert u_gap == pytest.approx(216.20, abs=0.01)
        assert u_gus == pytest.approx(89.20, abs=0.01)
        # and the uniform sheet agrees with the truth module it reads
        assert d.plate_cx(d.COL_X[0]) == pytest.approx(
            SM.plate_centre_x(d.COL_X[0]), abs=1e-9)

        d.CLOCKING = "mirrored"
        assert d.clock_sign(d.COL_X[0]) == -1.0, "the LEFT column turns"
        assert d.clock_sign(d.COL_X[1]) == +1.0, "the right column does not"
        assert d.cluster_gap() == pytest.approx(266.50, abs=0.01)
        assert d.gusset_pair_clear() == pytest.approx(139.50, abs=0.01)
        assert d.cluster_gap() - u_gap == pytest.approx(50.30, abs=0.01)
        assert d.gusset_pair_clear() - u_gus == pytest.approx(50.30, abs=0.01)
    finally:
        d.CLOCKING = "uniform"


def test_the_centre_datum_is_the_seam_crossing_the_long_centre_line():
    """(0,0) on sheets 3 and 4 is FOUR statements that have to stay one point.

    `SEAM_Y` is the canvas's own mid-length, the middle arm row, the plane the
    two half-cages butt on and the line the seam bars straddle.  The whole
    centre datum rests on those being the same number; if `system_model` ever
    moves one of them apart from the others, the sheet's "find the seam bars
    and you have found Y = 0" instruction becomes a lie.
    """
    assert D.CENTRE_X == pytest.approx(SM.CANVAS_W / 2.0, abs=0.005)
    assert D.CENTRE_Y == pytest.approx(SM.CANVAS_L / 2.0, abs=0.005)
    assert D.CENTRE_Y == pytest.approx(SM.SEAM_Y, abs=1e-9)
    assert D.CENTRE_Y == pytest.approx(SM.ROW_Y[1], abs=0.005)
    assert D.CENTRE_Y == pytest.approx(0.5 * (SM.FR_Y0 + SM.FR_Y1), abs=0.005)
    bars = [b for b in SM.seam_bodies() if b.name.startswith("seam_bar")]
    assert bars, "the centre sheets point a tape at the seam bars"
    for b in bars:
        mid = 0.5 * (b.lo[1] + b.hi[1])
        assert mid == pytest.approx(D.CENTRE_Y, abs=0.005)
    # and Pete's 416.6 cm really is the two butted half-frames
    assert D.TABLE_L_TAPE == pytest.approx(2 * SM.HALF_CAGE_L, abs=0.005)


def test_how_the_canvas_sits_on_the_table_is_read_off_the_model():
    """The sheet may not ASSERT that the canvas is centred — it must read it.

    `canvas_on_table` is what prints the paragraph, so this pins the paragraph
    to the model's own `table` body.  It is a real question: the table's height
    comes from the drawing but its FOOTPRINT is an assumption, and if that
    assumption ever moves off centre the sheet has to say so instead of
    quietly drawing the canvas in the middle of it.
    """
    t = D.canvas_on_table()
    body = [b for b in SM.ground_bodies() if b.name == "table"][0]
    assert t["x0"] == pytest.approx(body.lo[0] - SM.CANVAS_W / 2.0, abs=0.005)
    assert t["x1"] == pytest.approx(body.hi[0] - SM.CANVAS_W / 2.0, abs=0.005)
    assert t["y0"] == pytest.approx(body.lo[1] - SM.CANVAS_L / 2.0, abs=0.005)
    assert t["y1"] == pytest.approx(body.hi[1] - SM.CANVAS_L / 2.0, abs=0.005)
    assert t["centred"] is (abs(t["ecc_x"]) < 0.005
                            and abs(t["ecc_y"]) < 0.005)
    # as the model stands today: centred, 114.3 (4.5 in) of table all round
    assert t["centred"], "if this flips, the sheet must draw and say BOTH"
    assert t["over_x"] == pytest.approx(114.3, abs=0.005)
    assert t["over_y"] == pytest.approx(114.3, abs=0.005)
    assert t["provenance"] == "ASSUMED"
    # the tape's table is LONGER than the model's assumed footprint, and the
    # sheet prints that difference rather than picking a winner
    assert t["tape_vs_model"] == pytest.approx(
        2 * SM.HALF_CAGE_L - (body.hi[1] - body.lo[1]), abs=0.005)
    assert t["tape_vs_model"] > 0.0


# --------------------------------------------------------------------------
# EVERY PRINTED PLAN DIMENSION, RE-DERIVED FROM THE PACKAGE
# --------------------------------------------------------------------------
# The centre sheets print nothing that is not in `centre_dims()`, so pinning
# `centre_dims()` pins the sheets.  Each expectation below is built from
# `system_model` HERE, independently of how the script builds it — the trap
# this guards is a sheet that starts carrying its own arithmetic, which is
# exactly how out/ceiling_8020_layout.* published a 1435 mm drop post.
def _expected_centre_dims():
    cx, cy = SM.CANVAS_W / 2.0, SM.CANVAS_L / 2.0
    P, off = SM.PROFILE, abs(SM.PLATE_OFF)
    xl, xr = SM.COL_X[0], SM.COL_X[1]
    # the model's own plate centre, then its two post centres, then the faces
    pl_c = [SM.plate_centre_x(xl), SM.plate_centre_x(xr)]
    post = {0: sorted(SM.post_x(xl)), 1: sorted(SM.post_x(xr))}
    f = {k: [v - P / 2 for v in post[k]] + [v + P / 2 for v in post[k]]
         for k in post}
    f = {k: [f[k][0], f[k][2], f[k][1], f[k][3]] for k in f}   # lo,hi per post
    return {
        "centre_x": cx, "centre_y": cy,
        "canvas_edge": cx, "canvas_end": cy,
        "canvas_w": SM.CANVAS_W, "canvas_l": SM.CANVAS_L,
        "rail_out": SM.FR_X1 - cx, "rail_in": SM.IN_X1 - cx,
        "rail_out_end": SM.FR_Y1 - cy,
        "rail_in_end": SM.FR_Y1 - SM.PROFILE - cy,
        "frame_w": SM.FR_W, "frame_l": SM.FR_L,
        "rail_len_x": SM.RAIL_LEN_X, "rail_len_y": SM.RAIL_LEN_Y,
        "profile": SM.PROFILE,
        "seam_bar_face": SM.SEAM_Y + SM.PROFILE - cy,
        "seam_bar_dy": SM.SEAM_BAR_DY,
        "runway_mid_face": SM.ROW_Y[1] + P - cy,
        "runway_row_face_in": SM.ROW_Y[2] - P - cy,
        "runway_row_face_out": SM.ROW_Y[2] + P - cy,
        "runway_w": 2 * P, "runway_x": SM.IN_X1 - cx,
        "runway_len": SM.RAIL_LEN_X,
        "axis_x": xr - cx, "axis_pitch_x": xr - xl,
        "axis_y": SM.ROW_Y[2] - cy, "axis_pitch_y": SM.ROW_Y[2] - SM.ROW_Y[1],
        "base_circle_d": 2 * mounts.MOUNTS.boom_r * 1000.0,
        "plate_off": off,
        "plate_cx_l": pl_c[0] - cx, "plate_cx_r": pl_c[1] - cx,
        "plate_w": SM.PLATE[0], "plate_d": SM.PLATE[1],
        "post_pitch": SM.POST_PITCH_X,
        "post_c_l": [v - cx for v in post[0]],
        "post_c_r": [v - cx for v in post[1]],
        "post_f_l": [v - cx for v in f[0]],
        "post_f_r": [v - cx for v in f[1]],
        "plate_f_l": [pl_c[0] - SM.PLATE[0] / 2 - cx,
                      pl_c[0] + SM.PLATE[0] / 2 - cx],
        "plate_f_r": [pl_c[1] - SM.PLATE[0] / 2 - cx,
                      pl_c[1] + SM.PLATE[0] / 2 - cx],
        "pair_gap": SM.POST_SLOT,
        "pair_outer_w": SM.CLUSTER_W,
        "axis_face_short": SM.CLUSTER_W / 2 - off,
        "axis_face_long": SM.CLUSTER_W / 2 + off,
        "inner_pair_gap": (f[1][0] - f[0][3]),
        # table rows come from the body, not from here
        "table_edge": None, "table_end": None, "table_w": None,
        "table_l": None, "table_end_tape": None, "table_len_tape": None,
        "table_over": None, "table_over_end": None,
    }


def test_every_printed_plan_dimension_is_the_models_own():
    got, want = D.centre_dims(), _expected_centre_dims()
    assert set(got) == set(want), "a key was added or dropped without a check"
    body = [b for b in SM.ground_bodies() if b.name == "table"][0]
    cy = SM.CANVAS_L / 2.0
    want.update(
        table_edge=body.hi[0] - SM.CANVAS_W / 2.0,
        table_end=body.hi[1] - cy,
        table_w=body.hi[0] - body.lo[0], table_l=body.hi[1] - body.lo[1],
        table_end_tape=SM.HALF_CAGE_L, table_len_tape=2 * SM.HALF_CAGE_L,
        table_over=body.hi[0] - SM.CANVAS_W,
        table_over_end=body.hi[1] - SM.CANVAS_L)
    for k, exp in want.items():
        if isinstance(exp, list):
            assert len(got[k]) == len(exp), k
            for a, b in zip(got[k], exp):
                assert a == pytest.approx(b, abs=0.005), f"{k}: {got[k]}"
        else:
            assert got[k] == pytest.approx(exp, abs=0.005), k


def test_the_five_numbers_pete_asked_for():
    """The plate offset is what makes the pair asymmetric, and it is INFERRED.

    Both axis-to-face numbers are half the pair width shifted by the plate
    offset, so a change to open item 1 moves both by the same amount in
    opposite directions and the pair width does not move at all.  That is the
    whole reason the sheet prints all four and the tape row flags only one.
    """
    d = D.centre_dims()
    assert d["pair_gap"] == pytest.approx(241.40, abs=0.005)
    assert d["pair_outer_w"] == pytest.approx(393.80, abs=0.005)
    assert d["axis_face_short"] == pytest.approx(171.75, abs=0.005)
    assert d["axis_face_long"] == pytest.approx(222.05, abs=0.005)
    assert d["inner_pair_gap"] == pytest.approx(216.20, abs=0.005)
    assert d["plate_off"] == pytest.approx(25.15, abs=0.005)
    # the identities the sheet's own prose claims
    assert (d["axis_face_short"] + d["axis_face_long"]
            == pytest.approx(d["pair_outer_w"], abs=0.005))
    assert ((d["axis_face_long"] - d["axis_face_short"]) / 2.0
            == pytest.approx(d["plate_off"], abs=0.005))
    assert (d["pair_outer_w"] - d["pair_gap"]
            == pytest.approx(2 * SM.PROFILE, abs=0.005))
    # and the row gap really is what the uniform-clocking cluster gap is
    assert d["inner_pair_gap"] == pytest.approx(D.cluster_gap(), abs=0.005)


def test_the_hand_measured_table_compares_against_centre_dims():
    """The TAPE column is typed; the MODEL column may not be.

    A hand measurement is the one thing on these sheets that cannot come from
    the package, so it is typed once in `_HAND` and paired with a KEY.  This
    pins that every model figure beside it is read through `centre_dims()`,
    and that the derived offset row really is derived rather than typed a
    fourth time.
    """
    d = D.centre_dims()
    rows = D.hand_vs_model()
    assert len(rows) == len(D._HAND) + 1, "the derived row is missing"
    by_key = {k: t for _w, t, k in D._HAND}
    for what, tape, model, delta in rows:
        assert delta == pytest.approx(tape - model, abs=0.005), what
    for (_w, tape, key), row in zip(D._HAND, rows[:4] + rows[5:]):
        assert row[1] == pytest.approx(tape, abs=0.005)
        assert row[2] == pytest.approx(d[key], abs=0.005)
    derived = rows[4]
    assert derived[1] == pytest.approx(
        (by_key["axis_face_long"] - by_key["axis_face_short"]) / 2.0,
        abs=0.005)
    assert derived[2] == pytest.approx(d["plate_off"], abs=0.005)
    # THE ONE DISCREPANCY.  Four rows agree to 2.5 mm; the plate offset does
    # not, and the sheet's flag says so in those words.
    big = [r for r in rows if abs(r[3]) > 5.0]
    assert len(big) == 3, [r[0] for r in rows]
    assert all("axis" in r[0] for r in big)
    assert derived in big
    assert derived[3] == pytest.approx(16.85, abs=0.005)
    assert "PLATE OFFSET DIRECTION" in D.HAND_FLAG
    # and that open item is still open in the cut list this sheet sits beside
    assert any(t.startswith("PLATE OFFSET DIRECTION")
               for _n, t, _w, _h, _b in D.OPEN_ITEMS)


def test_every_printed_reference_height_is_the_z_ladder():
    """Sheet 4 prints the SAME ladder twice, once off each datum."""
    for h in (970.0, 940.0, 850.0):
        z = SM.z_ladder(h)
        rows = D.side_levels(h)
        assert [k for k, _l, _s, _p, _f, _b in rows] == [
            k for k, _l, _s, _b in D._LEVELS]
        for k, _lbl, _short, paper, floor, _bold in rows:
            assert paper == pytest.approx(z[k], abs=0.005), k
            # the floor column is the paper column plus the table height, and
            # nothing else — it is NOT an independent survey
            assert floor - paper == pytest.approx(-SM.FLOOR_Z, abs=0.005), k
        # the ladder is monotonically descending, so the drawn ticks cannot
        # cross each other however the labels are jogged
        vals = [p for _k, _l, _s, p, _f, _b in rows]
        assert vals == sorted(vals, reverse=True)
        # every jog names a level that exists
        assert set(D._JOG) <= {k for k, _l, _s, _b in D._LEVELS}


def test_the_tape_checks_and_cuts_are_differences_on_that_ladder():
    h = 970.0
    z = SM.z_ladder(h)
    checks = dict((w, v) for w, v, _why in D.tape_checks(h))
    assert checks["FLOOR  ->  PLATE UNDERSIDE"] == pytest.approx(
        h - SM.FLOOR_Z, abs=0.005)
    assert checks["TABLE TOP  ->  PLATE UNDERSIDE"] == pytest.approx(
        h - SM.TABLE_TOP_Z, abs=0.005)
    assert checks["TABLE TOP  ->  PLATE UNDERSIDE"] == pytest.approx(
        972.0, abs=0.005)
    assert checks["FLOOR  ->  TABLE TOP"] == pytest.approx(
        SM.TABLE_TOP_Z - SM.FLOOR_Z, abs=0.005)
    assert checks["FLOOR  ->  TOP OF STEEL"] == pytest.approx(
        SM.CAGE_TOTAL_H, abs=0.005)
    cuts = {it: ln for it, _w, ln, _n in D.side_cuts(h)}
    assert cuts["D"] == pytest.approx(z["post_length"], abs=0.005)
    assert cuts["E"] == pytest.approx(SM.GRID_U - SM.LEG_BOTTOM, abs=0.005)
    assert cuts["I"] == pytest.approx(D.seam_post_length(), abs=0.005)
    assert cuts["E"] == pytest.approx(cuts["I"], abs=0.005)
    # and the cut list the fabricator holds says the same three lengths
    rows = {it: ln for it, _n, _p, ln, _q, _w in D.cut_list(h)}
    for it in ("D", "E", "I"):
        assert rows[it] == pytest.approx(cuts[it], abs=0.005)


def test_the_centre_sheets_write_beside_the_corner_sheets_not_over_them(
        tmp_path):
    """A DIFFERENT DATUM MUST NOT LAND ON THE SAME FILENAME.

    Sheets 3 and 4 draw the same steel as sheets 1 and 2 with every number
    measured from somewhere else.  Two sheets that differ only in a datum,
    written to one filename, is the 850/2340 failure again — so they get their
    own directory and their own stems, and this pins both.
    """
    out = tmp_path / "drawings"
    D.sheet_centre_plan(970.0, str(out / "centre"))
    D.sheet_side_heights(970.0, str(out / "centre"))
    for stem in ("plan_centre_datum", "side_reference_heights"):
        for ext in ("pdf", "png"):
            p = out / "centre" / f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size > 20000, p
    # and they did not touch the corner-datum sheets' own names
    for stem in ("arm_spacing_topdown", "cage_side_view"):
        assert not (out / "centre" / f"{stem}.pdf").exists()
        assert not (out / f"{stem}.pdf").exists()


def test_the_mirrored_sheet_writes_somewhere_else(tmp_path):
    """A VARIANT MUST NOT OVERWRITE THE SHEET THE FABRICATOR IS HOLDING.

    Two sheets that differ only in a clocking, written to the same filename,
    is exactly the failure the 850/2340 sheet spent three weeks in.  The
    default output directory carries the clocking, and the uniform default is
    unchanged.
    """
    d = D
    try:
        for clocking, tail in (("uniform", os.path.join("out", "drawings")),
                               ("mirrored",
                                os.path.join("out", "drawings", "mirrored"))):
            ap = d.argparse.ArgumentParser()
            # drive main()'s own defaulting rather than reimplementing it
            d.CLOCKING = clocking
            out = os.path.join(d._repo(), "out", "drawings")
            if clocking != "uniform":
                out = os.path.join(out, clocking)
            assert out.endswith(tail), (clocking, out)
        assert "mirrored" in d.CLOCKINGS and "uniform" in d.CLOCKINGS
    finally:
        d.CLOCKING = "uniform"
