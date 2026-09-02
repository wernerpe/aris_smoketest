// The event reducer: one JSONL stream in, one plain object out.
//
// EVERY PANEL READS THIS AND NOTHING ELSE.  The stage timeline, the counters,
// the load bars, the live canvas and the "where the time goes" chart are four
// views of one reduction, so they cannot disagree about what the planner said.
// Replaying a finished job's events through the same reducer therefore
// reproduces exactly what a live watcher saw, which is the property that makes
// the JSONL on disk worth keeping.

export const STAGES = ["trace", "placement", "allocation", "conduction",
                       "scene_check"];

export const STAGE_LABEL = {
  trace: "trace", placement: "placement", allocation: "allocation",
  conduction: "conduction", scene_check: "scene check", export: "export",
};

// A stable colour per substage, so the "where the time goes" bar means the
// same thing between runs.
export const SUB_COLOR = {
  prefilter: "#4a6fa5", probe: "#4a8fd2", repair: "#5aa9c4",
  replan: "#cb6608", flycheck: "#a05a2c", merge: "#b0803a",
  balance: "#d2544a", guarantee: "#8a6fb0", park: "#7a5fa0",
  sequence: "#3fa46a", conduct: "#d8a13a", other: "#6d7480",
};

export function newState() {
  return {
    job: null,                 // the job record from /api/jobs
    started: null, ended: null, clock: 0,
    params: null, rig: "", tool: "",
    stages: {},                // name -> {status, t0, t1, elapsed, payload, end}
    subs: {},                  // stage -> {sub -> {seconds, open_t}}
    counters: {
      strokesTraced: 0, strokesProbed: 0, strokesTotal: 0,
      placed: 0, placedM: 0, dropped: 0,
      phasesDone: 0, phasesTotal: 0, checksOk: 0, checksBad: 0,
      inks: [], coverage: null, segments: 0,
    },
    progress: {},              // stage -> {done, total, label}
    loads: {},                 // arm -> seconds
    armMetres: {},             // arm -> metres (live, from placed spans)
    sheet: null, placement: null,
    strokes: new Map(),        // id -> {id,color,kind,pts:[[x,y]...]}
    placedSpans: [],           // {stroke, arm, s0, s1, lean}
    phases: [],                // conduct verdicts
    artifacts: [],             // {name, path}
    logs: [],                  // {t, level, msg, kind, stage}
    error: null, done: false, ok: null,
    seq: 0, dirty: {},
  };
}

function touch(s, ...what) { for (const w of what) s.dirty[w] = true; }

// JSON has no Infinity, so the server sends null for one (gui/jobs.finite).
const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : 0);

export function reduce(s, ev) {
  s.seq = ev.seq;
  s.clock = Math.max(s.clock, ev.t || 0);
  const st = ev.stage || "";
  const p = ev.payload || {};

  switch (ev.kind) {
    case "job_start":
      s.params = p.params; s.rig = p.rig; s.tool = p.tool;
      s.started = ev.wall;
      touch(s, "head");
      break;

    case "job_end":
      s.done = true; s.ok = !!p.ok; s.ended = ev.wall;
      if (p.error) s.error = p.error;
      if (p.bundle_error) s.bundleError = p.bundle_error;
      // A stage still marked running when the job ends never closed — the
      // planner was killed inside it.  Say so rather than leaving a bar
      // creeping forward for ever.
      for (const k of Object.keys(s.stages))
        if (s.stages[k].status === "running") s.stages[k].status = "aborted";
      touch(s, "head", "stages");
      break;

    case "stage_start":
      s.stages[st] = {status: "running", t0: ev.t, t1: null,
                      elapsed: 0, runs: (s.stages[st]?.runs || 0) + 1,
                      total: s.stages[st]?.total || 0, payload: p, end: null};
      touch(s, "stages");
      break;

    case "stage_end": {
      const cur = s.stages[st] || {runs: 1, total: 0};
      cur.status = p.ok === false ? "failed" : "done";
      cur.t1 = ev.t;
      cur.elapsed = p.elapsed_s || 0;
      cur.total = (cur.total || 0) + (p.elapsed_s || 0);
      cur.end = p;
      s.stages[st] = cur;
      if (st === "scene_check") {
        if (p.verdict_ok) s.counters.checksOk++; else s.counters.checksBad++;
      }
      if (st === "trace" && p.n_strokes) s.counters.strokesTraced = p.n_strokes;
      if (st === "allocation" && p.coverage_pct != null) {
        s.counters.coverage = p.coverage_pct;
        s.counters.segments = p.n_segments || s.counters.segments;
      }
      touch(s, "stages", "counters", "panels");
      break;
    }

    case "substage_start": {
      const m = s.subs[st] || (s.subs[st] = {});
      const e = m[p.sub] || (m[p.sub] = {seconds: 0, runs: 0});
      e.open_t = ev.t; e.runs++;
      s.openSub = {stage: st, sub: p.sub};
      touch(s, "stages");
      break;
    }

    case "substage_end": {
      const m = s.subs[st] || (s.subs[st] = {});
      const e = m[p.sub] || (m[p.sub] = {seconds: 0, runs: 1});
      e.seconds += p.elapsed_s || 0;
      e.open_t = null;
      if (s.openSub && s.openSub.sub === p.sub) s.openSub = null;
      Object.assign(e, {last: p});
      touch(s, "stages");
      break;
    }

    case "progress":
      s.progress[st] = {done: p.done, total: p.total, label: p.label};
      if (st === "allocation" && p.label === "probe") {
        s.counters.strokesProbed = p.done;
        s.counters.strokesTotal = p.total || s.counters.strokesTotal;
      }
      touch(s, "stages", "counters");
      break;

    case "item":
      reduceItem(s, st, p);
      break;

    case "metric":
      (s.metrics || (s.metrics = {}))[p.name] = p.value;
      touch(s, "counters");
      break;

    case "artifact":
      s.artifacts.push({name: p.name, path: p.path});
      touch(s, "panels");
      break;

    case "log":
      s.logs.push({t: ev.t, level: p.level, msg: p.msg, stage: st});
      if (s.logs.length > 6000) s.logs.splice(0, 2000);
      touch(s, "log");
      break;
  }
  return s;
}

function reduceItem(s, st, p) {
  switch (p.what) {
    case "sheet_strokes":
      s.sheet = p.sheet; s.placement = p.info;
      for (const k of p.strokes)
        s.strokes.set(k.id, {id: k.id, color: k.color, kind: k.kind,
                             length: k.length, pts: k.pts});
      s.counters.strokesTotal = p.strokes.length;
      s.counters.inks = [...new Set(p.strokes.map(x => x.color))];
      touch(s, "canvas", "counters");
      break;

    case "probe":
      // nothing to draw yet — a probe says who COULD, not who will.  The
      // per-stroke reach is kept so the live canvas can dim what no arm
      // certified, which is the first hint that a placement is wrong.
      s.reach = s.reach || new Map();
      s.reach.set(p.stroke, p.spans);
      touch(s, "canvas");
      break;

    case "placed": {
      // `num` because a non-finite planner answer arrives as JSON null (see
      // `gui/jobs.finite`), and `undefined + null` is NaN, which would poison
      // a running total for the rest of the job.
      const L = num(p.length_m);
      s.placedSpans.push({stroke: p.stroke, arm: p.arm,
                          s0: num(p.s_range[0]), s1: num(p.s_range[1]),
                          length: L, lean: num(p.lean_deg),
                          sigma: num(p.min_sigma)});
      s.counters.placed++;
      s.counters.placedM += L;
      s.armMetres[p.arm] = (s.armMetres[p.arm] || 0) + L;
      touch(s, "canvas", "counters");
      break;
    }

    case "loads":
      s.loads = p.loads_s || {};
      s.loadMax = {before: p.max_before_s, after: p.max_after_s};
      touch(s, "loads");
      break;

    case "sequenced":
      (s.sequenced || (s.sequenced = {}))[p.arm] = p;
      touch(s, "panels");
      break;

    case "allocated":
      s.alloc = p;
      s.counters.dropped = p.n_dropped;
      s.counters.segments = p.n_segments;
      // The balancer moves spans between arms AFTER they were placed, so the
      // live per-arm metres from `placed` events are a first draft; this is
      // the settled answer and it replaces them.
      s.armMetres = Object.fromEntries(
        Object.entries(p.arm_metres || {}).map(([k, v]) => [Number(k), v]));
      touch(s, "counters", "loads", "panels");
      break;

    case "phase":
      s.phases.push(p);
      s.counters.phasesDone = s.phases.length;
      touch(s, "panels", "counters");
      break;
  }
}

// --------------------------------------------------------------------------
// "where the time goes" — the number the algorithm iteration is driven from.
//
// A stage that CONTAINS another stage (scene_check runs inside conduction) is
// reported exclusive of it, or the pie adds up to more than the wall clock and
// every share is wrong.  The nesting is known statically because the pipeline
// is: `scene_check` is inside `conduction`, and nothing else nests.
const NESTED = {conduction: ["scene_check"]};

export function stageTimes(s) {
  const out = [];
  for (const name of [...STAGES, "export"]) {
    const st = s.stages[name];
    if (!st) continue;
    let total = st.total || 0;
    if (st.status === "running") total += Math.max(0, s.clock - st.t0);
    let excl = total;
    for (const inner of (NESTED[name] || [])) {
      const it = s.stages[inner];
      if (it) excl -= (it.total || 0);
    }
    out.push({name, total, exclusive: Math.max(0, excl), status: st.status,
              runs: st.runs || 1, subs: s.subs[name] || {}, end: st.end});
  }
  return out;
}
