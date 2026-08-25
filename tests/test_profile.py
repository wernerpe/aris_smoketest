"""Execution-profile selection — what may ship, what may not, and the record.

Run: pytest tests/    (or python3 tests/test_profile.py)

`csail_schedule.select_profile` decides which of the four (qd_frac x cluster)
profiles a programme is drawn at by CONDUCTING the candidates, which costs
minutes to an hour each on a real drawing.  These tests drive it with fake
allocate/conduct/floor callables so that the decision procedure — order,
pruning, refusal handling, and the grid it records — is pinned in
milliseconds and independently of any one drawing.  The numbers below are
shaped after the two real cases the selector exists for:

  * the CSAIL logo, where qd_frac 0.60 with the cluster features certifies at
    77.792 s against 104.021 s on the shipped defaults, and
  * `bench`'s spiral, where the same 0.60 is REFUSED outright and the run has
    to fall back to 0.30 rather than ship an uncertified schedule.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from aris_sixarm import idle                                   # noqa: E402
import csail_schedule as cs                                    # noqa: E402


def _args(**kw):
    """The two attributes `select_profile` reads off the caller's args."""
    return types.SimpleNamespace(**dict(dict(pause=2.0, fps=12.0), **kw))


def _built(duration, ok=True, n_phases=1, clearance=0.0824):
    """A conducted phase list shaped like `build_phases`'s, and no more."""
    return [dict(sch=dict(duration=duration / n_phases),
                 rep=dict(ok=ok, min_clearance=clearance))
            for _ in range(n_phases)]


class Harness:
    """Fake allocate + floor + conduct, recording every conduct it is asked for.

    `floors` and `conducts` are keyed by profile name; a conduct value may be a
    number (a makespan that certifies), an exception (the conductor or a gate
    refusing), or a `built` list, which is what lets one test ship a timeline
    `scene_check` failed without going through `build_phase`'s own veto.
    """

    def __init__(self, floors, conducts, pens=None):
        self.floors, self.conducts = floors, conducts
        self.pens = pens or {2: 0.3}
        self.conducted, self.allocated = [], []

    def alloc(self, b, p):
        name = cs.profile_name(p)
        self.allocated.append(name)
        assert b.qd_frac == p["qd_frac"] and b.cluster == p["cluster"]
        return [dict(name=f"phase 1 @ {name}")], {}, self.pens

    def floor(self, b, phases, pens, alt=None):
        return self.floors[phases[0]["name"].split("@ ")[1]]

    def conduct(self, b, phases, dt, pens, alt=None):
        name = phases[0]["name"].split("@ ")[1]
        self.conducted.append(name)
        out = self.conducts[name]
        if isinstance(out, BaseException):
            raise out
        return out if isinstance(out, list) else _built(out)

    def run(self, a=None, **kw):
        return cs.select_profile(a or _args(), self.alloc, 1 / 48.0,
                                 conduct=self.conduct, floor=self.floor,
                                 verbose=False, **kw)


# ---------------------------------------------------------------------------
def test_only_a_certified_profile_may_ship():
    """The fastest profile does not ship unless scene_check signed it off.

    Two ways of failing, both from the fastest cell: a conductor that refuses
    outright, and — the one that matters, because it is the one a future
    conductor could reintroduce — a timeline that comes back with a makespan
    and a FAILED gate.  Neither may be shipped, and the selector may not fall
    silently back on the number they carry.
    """
    h = Harness(floors={"qd0.30": 81.8, "qd0.30+cluster": 90.7,
                        "qd0.60": 60.0, "qd0.60+cluster": 55.0},
                conducts={"qd0.30": 104.021, "qd0.30+cluster": 123.4,
                          # a gate failure that still reports a makespan
                          "qd0.60": _built(70.0, ok=False),
                          "qd0.60+cluster": idle.Unconductable(
                              "arm 2 cannot stop clear of 97 (-94 mm)")})
    sel = h.run()
    by = {r["profile"]: r for r in sel["grid"]}

    assert sel["chosen"]["profile"] == "qd0.30"
    assert sel["chosen"]["makespan_s"] == 104.021
    assert by["qd0.60"]["status"] == "refused" and by["qd0.60"]["reason"]
    assert by["qd0.60"]["makespan_s"] == 70.0      # measured, and still refused
    assert by["qd0.60+cluster"]["status"] == "refused"
    assert "cannot stop clear" in by["qd0.60+cluster"]["reason"]
    # and the ONE certified cell that was faster than nothing is what shipped
    assert [r["profile"] for r in sel["grid"] if r.get("chosen")] == ["qd0.30"]
    assert all(r["status"] == "certified" or not r.get("chosen")
               for r in sel["grid"])


def test_refusal_falls_back_to_the_next_certified_profile():
    """bench's spiral: 0.60 is refused after the whole ladder, 0.30 ships.

    The floors put both 0.60 cells first, so the fallback is not a matter of
    which one happened to be tried first: the selector conducts them, is
    refused, and carries on down the order rather than stopping at the first
    answer or shipping the refusal's floor as if it were a makespan.

    `qd0.60+cluster` is conducted before anything else has even been ALLOCATED
    (`csail_schedule.FIRST_PROFILE`, see docs/FAST_PLANNING.md §4), so that a
    picture whose usual winner does win has a runnable programme as early as
    possible.  When it is refused, as here, the remaining three fall back into
    floor order behind it and nothing else about the outcome changes.
    """
    boom = SystemExit("scene_check REFUSED phase 1: grey; nothing rendered")
    h = Harness(floors={"qd0.60": 56.7, "qd0.60+cluster": 58.0,
                        "qd0.30": 83.4, "qd0.30+cluster": 88.0},
                conducts={"qd0.60": idle.Unconductable("arm 2 cannot stop clear"),
                          "qd0.60+cluster": boom,
                          "qd0.30": 89.395833, "qd0.30+cluster": 95.0})
    sel = h.run(jobs=1)

    assert h.conducted == ["qd0.60+cluster", "qd0.60", "qd0.30", "qd0.30+cluster"]
    assert h.allocated[0] == "qd0.60+cluster", \
        "the provisional best must be allocated before the other three"
    assert sel["chosen"]["profile"] == "qd0.30"
    assert abs(sel["chosen"]["makespan_s"] - 89.395833) < 1e-9
    assert sel["chosen"]["qd_frac"] == 0.30 and not sel["chosen"]["cluster"]
    doc = cs.profile_json(sel)
    assert doc["chosen"] == "qd0.30" and doc["n_conducted"] == 4
    assert [r["status"] for r in doc["grid"]] == \
        ["certified", "certified", "refused", "refused"]

    # and when NOTHING certifies, the run stops instead of picking a loser
    h2 = Harness(floors=h.floors,
                 conducts={k: SystemExit("no") for k in h.floors})
    try:
        h2.run()
    except SystemExit as exc:
        assert "no execution profile could be certified" in str(exc)
    else:
        raise AssertionError("a run with four refusals shipped something")


def test_the_recorded_grid_is_the_conducts_that_happened():
    """All four cells recorded; every conducted number is the conduct's own.

    And the pruning is the claim it says it is: a cell is left unconducted only
    when its FLOOR — an exact lower bound on any schedule of that allocation —
    is already the incumbent's certified makespan or worse.  Turning the
    pruning off must not move the answer, only the bill.

    The cells are ALLOCATED with the provisional best first and REPORTED in the
    listed order: which one is finished first is a scheduling decision and the
    record of what was tried should not move with it.
    """
    floors = {"qd0.30": 81.789, "qd0.30+cluster": 90.716,
              "qd0.60": 62.0, "qd0.60+cluster": 58.0}
    conducts = {"qd0.30": 104.021, "qd0.30+cluster": 123.375,
                "qd0.60": 84.396, "qd0.60+cluster": 77.792}
    h = Harness(floors, conducts)
    sel = h.run(jobs=1)
    doc = json.loads(json.dumps(cs.profile_json(sel)))   # it has to survive JSON

    assert h.allocated == [cs.profile_name(p)
                           for p in cs.profile_order(cs.PROFILES)]
    assert h.allocated[0] == cs.FIRST_PROFILE
    assert [r["profile"] for r in doc["grid"]] == \
        [cs.profile_name(p) for p in cs.PROFILES]
    assert doc["chosen"] == "qd0.60+cluster"
    assert abs(doc["makespan_s"] - 77.792) < 1e-9

    ship = doc["makespan_s"]
    for r in doc["grid"]:
        assert r["floor_s"] == floors[r["profile"]]
        if r["status"] == "certified":
            assert r["profile"] in h.conducted
            assert abs(r["makespan_s"] - conducts[r["profile"]]) < 1e-9
            assert r["scene_check"] is True
        else:
            assert r["status"] == "pruned"
            assert r["profile"] not in h.conducted
            assert r["makespan_s"] is None
            assert r["floor_s"] >= ship, "pruned a cell that could have won"
    # the two 0.30 cells cannot beat 77.792 s even at their floors, so the
    # whole of this selection cost two conducts and not four
    assert h.conducted == ["qd0.60+cluster", "qd0.60"], h.conducted
    assert doc["n_conducted"] == 2

    full = Harness(floors, conducts).run(prune=False)
    assert full["chosen"]["profile"] == doc["chosen"]
    assert {r["profile"]: r["makespan_s"] for r in full["grid"]} == conducts


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("all profile-selection tests pass")
