"""The progress hook is a no-op with no sink, and a faithful record with one.

THE FIRST HALF IS THE IMPORTANT ONE.  The GUI is allowed to watch the planner
and is not allowed to change it, so "no sink attached" has to be provably free
of side effects: no events, no state, no exception, and — the one that would
actually bite — no PAYLOAD BUILT.  A call site that computes an expensive
summary and then hands it to a sink that is None has already spent the time,
and the run with the GUI off is no longer the run that was published.
"""
import pytest

from aris_sixarm import progress


@pytest.fixture(autouse=True)
def _no_sink():
    """Every test starts and ends with no sink, whatever it did in between."""
    progress.clear_sink()
    yield
    progress.clear_sink()


# --------------------------------------------------------------------------
def test_no_sink_emits_nothing_and_raises_nothing():
    assert not progress.active()
    progress.emit("item", "allocation", what="probe", stroke=3)
    progress.log("hello")
    progress.metric("x", 1.0)
    progress.progress("allocation", 1, 2)
    progress.item("allocation", what="placed")
    progress.artifact("f", "/tmp/f")
    with progress.stage("trace") as end:
        end["n_strokes"] = 4          # writing into the dead payload is legal
    with progress.substage("allocation", "probe"):
        pass
    assert progress.elapsed() == 0.0


def test_no_sink_does_not_build_the_payload():
    """`active()` is the guard every expensive call site is written behind."""
    calls = []

    def expensive():
        calls.append(1)
        return 42

    # the pattern used in aris_sixarm/allocate.py
    if progress.active():
        progress.item("allocation", what="probe", spans=expensive())
    assert calls == []

    with progress.recording():
        if progress.active():
            progress.item("allocation", what="probe", spans=expensive())
    assert calls == [1]


def test_recording_collects_and_restores():
    with progress.recording() as events:
        progress.log("one")
        progress.metric("sigma", 0.31, stage="allocation", unit="")
    assert [e["kind"] for e in events] == ["log", "metric"]
    assert events[0]["payload"]["msg"] == "one"
    assert events[1]["payload"] == {"name": "sigma", "value": 0.31, "unit": ""}
    assert not progress.active()          # restored


def test_recording_nests_and_restores_the_outer_sink():
    with progress.recording() as outer:
        progress.log("a")
        with progress.recording() as inner:
            progress.log("b")
        progress.log("c")
    assert [e["payload"]["msg"] for e in outer] == ["a", "c"]
    assert [e["payload"]["msg"] for e in inner] == ["b"]


def test_stage_brackets_carry_timing_and_the_closing_payload():
    with progress.recording() as ev:
        with progress.stage("trace", source="x.png") as end:
            end["n_strokes"] = 39
    assert [e["kind"] for e in ev] == ["stage_start", "stage_end"]
    assert ev[0]["stage"] == "trace"
    assert ev[0]["payload"] == {"source": "x.png"}
    assert ev[1]["payload"]["ok"] is True
    assert ev[1]["payload"]["n_strokes"] == 39
    assert ev[1]["payload"]["elapsed_s"] >= 0.0


def test_stage_closes_on_an_exception_and_re_raises():
    with progress.recording() as ev:
        with pytest.raises(ValueError):
            with progress.stage("allocation"):
                raise ValueError("the band collapsed")
    assert ev[-1]["kind"] == "stage_end"
    assert ev[-1]["payload"]["ok"] is False
    assert "ValueError" in ev[-1]["payload"]["error"]


def test_substage_events_carry_the_sub_name():
    with progress.recording() as ev:
        with progress.substage("allocation", "balance", n_placed=12) as end:
            end["rounds"] = 3
    assert [e["kind"] for e in ev] == ["substage_start", "substage_end"]
    assert ev[0]["payload"]["sub"] == "balance"
    assert ev[0]["payload"]["n_placed"] == 12
    assert ev[1]["payload"]["rounds"] == 3


def test_events_are_ordered_timed_and_stamped():
    with progress.recording() as ev:
        for i in range(5):
            progress.log(f"line {i}")
    assert [e["seq"] for e in ev] == [1, 2, 3, 4, 5]
    assert all(0.0 <= e["t"] for e in ev)
    assert all(e["t"] <= e2["t"] for e, e2 in zip(ev, ev[1:]))
    assert all(e["wall"] > 1.6e9 for e in ev)
    assert all(set(e) == {"seq", "t", "wall", "kind", "stage", "payload"}
               for e in ev)


def test_a_sink_that_raises_does_not_reach_the_planner():
    """A closed browser tab is not a planning failure."""
    def bad(_ev):
        raise RuntimeError("broken pipe")

    prev = progress.set_sink(bad)
    try:
        progress.log("this must not raise")
        with progress.stage("trace"):
            pass
    finally:
        progress.set_sink(prev)


def test_the_vocabulary_is_closed():
    """Every kind the reducer switches on is declared, and vice versa."""
    emitted = set()
    with progress.recording() as ev:
        progress.emit("job_start")
        progress.emit("job_end")
        with progress.stage("trace"):
            with progress.substage("trace", "s"):
                pass
        progress.progress("trace", 1, 2)
        progress.item("trace", what="x")
        progress.metric("m", 1)
        progress.log("l")
        progress.artifact("a", "/tmp/a")
    for e in ev:
        emitted.add(e["kind"])
    assert emitted == set(progress.KINDS)
