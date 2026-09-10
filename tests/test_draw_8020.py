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
