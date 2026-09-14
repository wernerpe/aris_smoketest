"""THE PARTNER STANDOFF: an EXTRA requirement, and nothing else.

The one lever docs/V2_STAGED.md section 25 tests: a LEADER is held to
`gate + S` against its SAME-ROW follower's pose-invariant capsules, so the
follower's fixed links keep enough room for `paper.route` to fly its legs at
`paper.FRAME_FLOOR`.  The claims held down here:

  1. NO GATE CONSTANT MOVES.  `PAIR_MARGIN`, `STATIC_MARGIN` and `FRAME_FLOOR`
     are what they were.
  2. S = 0 IS A NO-OP, bit for bit: the same clearance, the same leg-cache
     signature, the same plan-memo room key, no standoff installed.
  3. S > 0 only ever LOWERS a clearance, never raises one, and it lowers it by
     exactly S where the pose-invariant set is what binds.
  4. THE SET IS THE POSE-INVARIANT ONE: the base column bands and the
     shoulder->elbow link, and nothing past the elbow.
  5. IT CANNOT LEAK.  A fresh `freeze` / `freeze_sets` / `thaw` clears it, so a
     bucket that asked for no standoff cannot inherit the previous bucket's.
  6. ONLY A LEADER OWES IT, AND ONLY TO A SAME-ROW FOLLOWER.
"""
import numpy as np
import pytest

from aris_sixarm import coordination, frozen, paper, rig_final, staged
from aris_sixarm import traces as traces_mod


# THE TESTS RUN ON WHATEVER FLEET IS CONFIGURED, which is the legacy three-arm
# rig unless `ARIS_RIG` says otherwise -- and they must, because the suite is
# never given the rig env.  Every claim below is about the MECHANISM, so any
# two arms in the fleet exercise it; the sweep's own numbers are the rig's.
OBS, PARTNER = sorted(staged.FLEET)[:2]


@pytest.fixture
def rig():
    fl = staged.FLEET
    pens = {a: fl[a].pen for a in fl}
    h_inv = staged.H_INV_DEFAULT
    parks = staged.shipped_parks(fl)
    yield fl, pens, h_inv, parks
    staged.thaw()


def _invariant_gap(P, partner, caps):
    """The observer's chain against ONE partner's pose-invariant rows. -> (N,)."""
    A, B, R = frozen._CAPS[partner][:3]
    m = frozen._INVARIANT[partner]
    A, B, R = A[m], B[m], R[m]
    worst = np.full(len(P), np.inf)
    for (i, j, r) in caps:
        d = coordination.seg_seg_dist(P[:, i][:, None, :], P[:, j][:, None, :],
                                      A[None, :, :], B[None, :, :])
        worst = np.minimum(worst, (d - (R[None, :] + r)).min(axis=1))
    return worst


def _chain(fl, pens, h_inv, arm, q):
    return coordination.chain_world(np.asarray(q, float).reshape(1, 7),
                                    fl[arm], h_inv, float(pens[arm]))


# ---------------------------------------------------------------------------
# 1.  the gates did not move
# ---------------------------------------------------------------------------
def test_no_gate_constant_moved():
    assert coordination.PAIR_MARGIN == 0.050
    assert rig_final.STATIC_MARGIN == pytest.approx(0.050)
    assert paper.FRAME_FLOOR == pytest.approx(0.063)
    assert staged.PAIR_MARGIN == coordination.PAIR_MARGIN


# ---------------------------------------------------------------------------
# 2.  S = 0 is a no-op
# ---------------------------------------------------------------------------
def test_zero_standoff_is_a_no_op(rig):
    fl, pens, h_inv, parks = rig
    a, p = OBS, PARTNER
    P = _chain(fl, pens, h_inv, a, parks[a])

    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False)
    base = frozen.partner_clearance(P).copy()
    assert frozen.standoff() == {}
    key0 = staged._room_key(None)

    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={p: 0.0})
    assert frozen.standoff() == {}, "a zero entry must not install a standoff"
    assert np.array_equal(frozen.partner_clearance(P), base)
    assert staged._room_key(None) == key0


def test_zero_standoff_keeps_the_leg_cache_signature(rig):
    fl, pens, h_inv, parks = rig
    poses = {int(x): np.asarray(q, float).reshape(1, 7)
             for x, q in parks.items() if int(x) != OBS}
    s0 = staged.leg_cache_signature(poses)
    assert staged.leg_cache_signature(poses, None, None) == s0
    assert staged.leg_cache_signature(poses, None, {}) == s0
    assert staged.leg_cache_signature(poses, None, {PARTNER: 0.0}) == s0
    assert staged.leg_cache_signature(poses, None, {PARTNER: 0.09}) != s0


def test_zero_standoff_keeps_the_room_key(rig):
    fl, pens, h_inv, parks = rig
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False)
    k0 = staged._room_key(None)
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={PARTNER: 0.11})
    assert staged._room_key(None) != k0
    assert staged._room_key(None).endswith(f"|S{PARTNER}:0.110000;")


# ---------------------------------------------------------------------------
# 3.  S > 0 only tightens, and by exactly S where it binds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [0.070, 0.090, 0.110, 0.130])
def test_standoff_only_tightens(rig, S):
    fl, pens, h_inv, parks = rig
    a, p = OBS, PARTNER
    P = _chain(fl, pens, h_inv, a, parks[a])
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False)
    base = frozen.partner_clearance(P).copy()
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={p: S})
    got = frozen.partner_clearance(P)
    assert np.all(got <= base + 1e-12)
    assert np.all(got >= base - S - 1e-12)


def test_standoff_is_exactly_min_of_the_room_and_the_set_less_S(rig):
    """The identity the whole mechanism is: `min(gap, invariant_gap - S)`.

    Not an approximation of one -- every call site compares the result against
    its own floor `f`, so this is precisely "`f` of the room and `f + S` of the
    named partner's pose-invariant capsules".
    """
    fl, pens, h_inv, parks = rig
    a, p, S = OBS, PARTNER, 0.13
    P = _chain(fl, pens, h_inv, a, parks[a])
    caps = (rig_final.STATIC_CAPSULES_LAT if P.shape[1] >= 11
            else rig_final.STATIC_CAPSULES)
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False)
    base = frozen.partner_clearance(P).copy()
    inv = _invariant_gap(P, p, caps)
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={p: S})
    got = frozen.partner_clearance(P)
    assert got == pytest.approx(np.minimum(base, inv - S), abs=1e-9)


def test_a_standoff_on_a_partner_nobody_froze_is_dropped(rig):
    fl, pens, h_inv, parks = rig
    # the observer cannot owe itself, and 9999 is not an arm
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={OBS: 0.13, 9999: 0.13})
    assert frozen.standoff() == {}


# ---------------------------------------------------------------------------
# 4.  the set is the pose-invariant one
# ---------------------------------------------------------------------------
def test_invariant_mask_is_base_column_plus_the_shoulder_link():
    tab, _ = coordination.known_pose_capsules(coordination.CAPSULES_LAT)
    m = frozen.invariant_mask(tab)
    got = [tuple(tab[k][:2]) for k in np.flatnonzero(m)]
    # the base column bands all run [0, 1]; the upper arm is [1, 3]
    assert got == [(0, 1), (0, 1), (0, 1), (1, 3)]
    # ...and nothing past the elbow is in it
    assert all(tuple(c[:2]) not in got
               for c in tab if c[0] >= 3), "no link past the elbow may be fixed"


def test_invariant_mask_tiles_pose_major(rig):
    """`freeze_sets` stacks N poses pose-major; the mask has to follow."""
    tab, _ = coordination.known_pose_capsules(coordination.CAPSULES_LAT)
    one = frozen.invariant_mask(tab)
    three = frozen.invariant_mask(tab, 3)
    assert len(three) == 3 * len(one)
    assert np.array_equal(three, np.tile(one, 3))


def test_the_frozen_block_and_its_mask_are_the_same_length(rig):
    fl, pens, h_inv, parks = rig
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={PARTNER: 0.09})
    blk = frozen._CAPS[31]
    assert len(frozen._INVARIANT[31]) == len(blk[2])
    assert int(frozen._INVARIANT[31].sum()) == 4


# ---------------------------------------------------------------------------
# 5.  it cannot leak
# ---------------------------------------------------------------------------
def test_a_fresh_freeze_clears_the_standoff(rig):
    fl, pens, h_inv, parks = rig
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={PARTNER: 0.13})
    assert frozen.standoff() == {PARTNER: 0.13}
    # the NEXT bucket asks for none and must get none
    staged.freeze_stage(PARTNER, parks, None, fl, pens, h_inv, leg_cache=False)
    assert frozen.standoff() == {}
    staged.freeze_partners(PARTNER, parks, fl, pens, h_inv, leg_cache=False)
    assert frozen.standoff() == {}


def test_thaw_clears_the_standoff(rig):
    fl, pens, h_inv, parks = rig
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={PARTNER: 0.13})
    staged.thaw()
    assert frozen.standoff() == {}
    assert not frozen.active()


def test_standoff_sig_is_empty_when_there_is_none(rig):
    fl, pens, h_inv, parks = rig
    staged.freeze_stage(OBS, parks, None, fl, pens, h_inv, leg_cache=False)
    assert frozen.standoff_sig() == ""


# ---------------------------------------------------------------------------
# 6.  only a leader owes it, and only to a same-row follower
# ---------------------------------------------------------------------------
def test_lf_standoffs_is_leader_to_same_row_follower_only():
    actives = (2, 13, 17, 31, 71, 97)
    roles = {13: "leader", 71: "leader", 2: "leader",
             17: "follower", 31: "follower", 97: "follower"}
    got = staged.lf_standoffs(actives, roles, 0.09)
    assert set(got) == {13, 71, 2}, "only leaders owe"
    for a, v in got.items():
        assert len(v) == 1
        p = next(iter(v))
        assert roles[p] == "follower"
        assert traces_mod.ROW_OF[p] == traces_mod.ROW_OF[a]
        assert v[p] == pytest.approx(0.09)
    # the shipped rows: 13/17, 31/71, 2/97
    assert got[71] == {31: pytest.approx(0.09)}


def test_lf_standoffs_is_empty_at_zero():
    actives = (2, 13, 17, 31, 71, 97)
    roles = {13: "leader", 71: "leader", 2: "leader",
             17: "follower", 31: "follower", 97: "follower"}
    assert staged.lf_standoffs(actives, roles, 0.0) == {}
    assert staged.lf_standoffs(actives, roles, None) == {}


def test_a_conducted_stage_owes_nothing():
    actives = (2, 13, 17, 31, 71, 97)
    roles = {a: "conductor" for a in actives}
    assert staged.lf_standoffs(actives, roles, 0.13) == {}


# ---------------------------------------------------------------------------
# 7.  the whole gate path sees it
# ---------------------------------------------------------------------------
def test_chain_clearance_sees_the_standoff(rig):
    """`frozen.chain_clearance` is the one seam every router gate reaches."""
    fl, pens, h_inv, parks = rig
    a, p, S = OBS, PARTNER, 0.13
    P = _chain(fl, pens, h_inv, a, parks[a])
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False)
    room = paper.static_boxes(a) if hasattr(paper, "static_boxes") else []
    base = float(frozen.chain_clearance(P, room)[0])
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={p: S})
    got = float(frozen.chain_clearance(P, room)[0])
    assert got <= base + 1e-12


def test_ink_vs_envelope_sees_the_standoff(rig):
    fl, pens, h_inv, parks = rig
    a, p, S = OBS, PARTNER, 0.13
    plan = dict(qs=np.asarray(parks[a], float).reshape(1, 7))
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False)
    base = staged.ink_vs_envelope(plan, fl[a], h_inv, pens[a])
    staged.freeze_stage(a, parks, None, fl, pens, h_inv, leg_cache=False,
                        standoff={p: S})
    got = staged.ink_vs_envelope(plan, fl[a], h_inv, pens[a])
    assert got <= base + 1e-12
