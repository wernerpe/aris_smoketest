// The wiring.  Everything else is a component; this is what connects them.
//
// ONE REDUCTION, MANY VIEWS.  Events arrive on a websocket, go through
// `state.reduce`, and every panel re-renders from that one object.  A finished
// job replays its whole JSONL through the same reducer before the bundle is
// loaded, so a job watched live and the same job opened tomorrow show the
// identical progress history — which is the property that makes the recorded
// stream worth keeping at all.

import {api, openStream, fetchBin} from "./api.js";
import {newState, reduce, stageTimes} from "./state.js";
import {Panel, F, fmtS, esc} from "./panel.js";
import {Strip} from "./strip.js";
import {Panels} from "./panels.js";
import {Scene3D} from "./scene3d.js";
import {Program, chainPoints, witnessBetween} from "./program.js";
import * as fk from "./fk.js";

const el = (id) => document.getElementById(id);

const app = {
  cfg: null, state: newState(), jobId: null, ws: null, offset: 0,
  scene: null, sceneKey: null, prog: null, colors: new Map(),
  layers: {arms: 1, pens: 1, paper: 1, cage: 1, mounts: 1, strokes: 1, ink: 1,
           bases: 1, chains: 0},
};

// --------------------------------------------------------------------------
async function boot() {
  app.view = new Scene3D(el("view"));
  app.panel = new Panel(el("side"), {
    onStart: startJob, onCancel: cancelJob, onSelect: selectJob,
    onMeshcat: api.openMeshcat});
  app.strip = new Strip(el("stages"), el("counters"), el("loads"));
  app.strip.armColor = colorOf;
  app.panels = new Panels({
    tabs: el("tabs"), program: el("pane-program"), clear: el("pane-clear"),
    stroke: el("pane-stroke"), time: el("pane-time")},
    {onFrame: onFrame, onSelectSegment: onSelectSegment,
     onWitness: onWitness, colorOf});

  buildLayerToggles();

  app.cfg = await api.config();
  app.panel.build(app.cfg);
  const d = app.cfg.defaults;
  await loadScene(d.rig, d.tool);
  await refreshJobs();
  setInterval(refreshJobs, 3000);
  setInterval(() => {
    if (app.state.dirty.stages || app.state.dirty.counters ||
        app.state.dirty.loads) { app.strip.render(app.state); flushDirty(); }
  }, 250);

  // Deep link: /#<job id> opens that job, so a URL can be shared or
  // bookmarked.  With no hash, the newest job opens — the common case is
  // "I started something, then reloaded the page", and landing on an empty
  // viewer would make that look like the job had been lost.
  if (location.hash.length > 1) selectJob(location.hash.slice(1));
  else if (app.jobs && app.jobs.length) selectJob(app.jobs[0].id);
}

function flushDirty() { app.state.dirty = {}; }

// --------------------------------------------------------------------------
// the scene (rig-dependent, cached server-side by (rig, tool))
// --------------------------------------------------------------------------
async function loadScene(rig, tool) {
  const key = `${rig}/${tool}`;
  if (app.sceneKey === key) return;
  banner("");
  try {
    const doc = await api.scene(rig, tool);
    const buf = await fetchBin(`/api/scene.bin?rig=${rig}&tool=${tool}`);
    app.sceneDoc = doc;
    app.sceneBuf = buf;
    app.view.loadScene(doc, buf);
    app.sceneKey = key;
    app.colors = new Map(doc.arms.map(a => [a.arm, a.color_hex]));
    el("b-rig").textContent = `${doc.rig} · ${doc.tool} · h=${doc.h_inv}`;
    el("viewhud").innerHTML =
      `${doc.arms.length} arms · paper ${doc.sheet_m[0].toFixed(3)} × `
      + `${doc.sheet_m[1].toFixed(3)} m · ${doc.bodies.length} static bodies`;
    // WHAT THE TOOL IN THE PICTURE IS, in one line, because the difference
    // between "the model says 149 mm" and "the pen was measured at 149 mm" is
    // the difference between a drawing and a gate result.
    el("toolnote").textContent = doc.tool_model
      ? doc.tool_model.note
      : `tool: the planner's offsets only — ${(1000 * doc.pen_lat_m).toFixed(0)}`
        + ` mm lateral, ${(1000 * doc.pen_ext_m).toFixed(0)} mm axial;`
        + ` no holder model for this tool`;

    // THE FK GOLDEN CHECK.  `fk.js` is a second implementation of
    // `frames.link_frames_many`; the scene carries six joint vectors and the
    // link poses PYTHON computed for them, and if the two disagree at all the
    // viewer says so rather than drawing an arm that is subtly in the wrong
    // place.  See the header of web/viewer/js/fk.js.
    const T = new Float64Array(buf, doc.fk_check.T.offset,
                               doc.fk_check.T.length / 8);
    const chk = fk.checkAgainst(doc.fk_check.q, T);
    if (!chk.ok) {
      banner(`the viewer's forward kinematics disagrees with frames.py by `
             + `${chk.worst.toExponential(2)} — the arms drawn here are NOT `
             + `where the planner put them.  Fix web/viewer/js/fk.js.`);
    }
  } catch (e) {
    banner(`could not build the scene for ${key}: ${e.message}`);
  }
}

function banner(msg) {
  const b = el("banner");
  b.textContent = msg;
  b.style.display = msg ? "block" : "none";
}

function colorOf(arm) { return app.colors.get(Number(arm)) || "#8a8f99"; }

function buildLayerToggles() {
  const t = el("viewtools");
  t.innerHTML = "";
  // "pens" IS THE TOOL LAYER.  It used to hold the two-cylinder sketch of the
  // planner's (pen_lat, pen_ext); it now holds the modelled holder, and the
  // label says which so the button and the picture agree.
  const names = {arms: "arms", pens: "tool model", paper: "paper",
                 cage: "cage", mounts: "mounts", strokes: "strokes",
                 ink: "ink", bases: "bases", chains: "capsule chain"};
  for (const [k, label] of Object.entries(names)) {
    const b = F("button", {text: label,
                           class: app.layers[k] ? "on" : ""});
    b.addEventListener("click", () => {
      app.layers[k] = !app.layers[k];
      b.className = app.layers[k] ? "on" : "";
      app.view.setLayer(k, app.layers[k]);
    });
    t.appendChild(b);
    app.view.setLayer(k, !!app.layers[k]);
  }
  const home = F("button", {text: "reset view", onclick: () =>
    app.view.frameSheet(app.sceneDoc ? app.sceneDoc.sheet_m : [1.8, 3.6])});
  t.appendChild(home);
}

// --------------------------------------------------------------------------
// jobs
// --------------------------------------------------------------------------
async function refreshJobs() {
  try {
    const jobs = await api.listJobs();
    app.jobs = jobs;
    app.panel.setJobs(jobs, app.jobId);
    const cur = jobs.find(j => j.id === app.jobId);
    app.panel.setRunning(cur ? cur.live : false);
    if (cur) {
      el("b-job").textContent = `${cur.params.out || cur.id} · ${cur.status}`;
      el("b-job").className = "badge " +
        (cur.status === "done" ? "ok"
         : cur.status === "failed" ? "bad"
         : cur.live ? "live" : "");
    }
  } catch (e) { /* the server is restarting; the next tick will do */ }
}

async function startJob(params) {
  try {
    const job = await api.createJob(params);
    await refreshJobs();
    selectJob(job.id);
  } catch (e) {
    banner(`could not start the job: ${e.message}`);
  }
}

async function cancelJob() {
  if (!app.jobId) return;
  try { await api.cancelJob(app.jobId); } catch (e) { /* already gone */ }
  refreshJobs();
}

async function selectJob(id) {
  if (app.ws) { try { app.ws.close(); } catch (e) {} app.ws = null; }
  app.jobId = id;
  app.offset = 0;
  app.state = newState();
  app.prog = null;
  location.hash = id;
  el("log").innerHTML = "";
  app.panel.clearDay1();
  app.view.clearStrokes();
  app.view.setInk([]);
  app.panel.setJobs(app.jobs || [], id);

  let job;
  try { job = await api.getJob(id); }
  catch (e) { banner(`no such job: ${id}`); return; }

  // The rig and the tool are the JOB's, not the form's — opening an old job
  // must draw the room that job was planned for.
  await loadScene(job.params.rig || "proposed", job.params.tool || "lateral");

  app.ws = openStream(id, onEvents, onStreamClosed);
  el("b-conn").textContent = "streaming";
  el("b-conn").className = "badge live";
}

function onStreamClosed(jobDict) {
  el("b-conn").textContent = "idle";
  el("b-conn").className = "badge";
  if (jobDict) refreshJobs();
  maybeLoadBundle();
}

// --------------------------------------------------------------------------
// events
// --------------------------------------------------------------------------
function onEvents(events) {
  const s = app.state;
  let sawStrokes = false, newSpans = 0, jobEnded = false;
  for (const ev of events) {
    reduce(s, ev);
    if (ev.kind === "item" && ev.payload.what === "day1")
      app.panel.setDay1(ev.payload);
    if (ev.kind === "item" && ev.payload.what === "sheet_strokes") sawStrokes = true;
    if (ev.kind === "item" && ev.payload.what === "placed") newSpans++;
    if (ev.kind === "job_end") jobEnded = true;
    if (ev.kind === "log" || ev.kind === "stage_start" ||
        ev.kind === "stage_end" || ev.kind === "metric") appendLog(ev);
  }
  el("b-clock").textContent = fmtS(s.clock);
  app.strip.render(s);
  if (sawStrokes) app.view.setStrokes([...s.strokes.values()]);
  if (newSpans) {
    for (const sp of s.placedSpans.slice(-newSpans))
      app.view.addSpan(sp, colorOf(sp.arm));
  }
  // THE LIVE PANELS ARE THROTTLED AND THE STAGE BARS ARE NOT.  `renderLive`
  // rebuilds four panes of DOM including a several-hundred-row table, and the
  // batches arrive six times a second while the planner is talking; rebuilding
  // that on every batch is how a progress UI ends up costing more than the
  // thing it is watching.  The bars are cheap and stay immediate.
  if (s.dirty.panels || s.dirty.counters) {
    app.panels.setState(s);
    const now = performance.now();
    if (!app.prog && now - (app.lastPanels || 0) > 1000) {
      app.lastPanels = now;
      app.panels.renderLive(s);
    }
  }
  flushDirty();
  if (jobEnded) {
    refreshJobs();
    app.panels.setState(s);
    if (!app.prog) app.panels.renderLive(s);   // the throttle must not eat the
    setTimeout(maybeLoadBundle, 300);          // last state of a finished job
  }
}

const LOG_MAX = 3000;
function appendLog(ev) {
  const box = el("log");
  const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  const d = document.createElement("div");
  let cls = "e-info", txt;
  if (ev.kind === "log") {
    cls = ev.payload.level === "error" ? "e-error" : "e-info";
    txt = ev.payload.msg;
  } else if (ev.kind === "stage_start") {
    cls = "e-stage"; txt = `▶ ${ev.stage}`;
  } else if (ev.kind === "stage_end") {
    cls = "e-stage";
    txt = `■ ${ev.stage}  ${fmtS(ev.payload.elapsed_s)}`
        + (ev.payload.ok === false ? "  FAILED" : "")
        + (ev.payload.verdict_ok === false ? "  REFUSED" : "");
  } else if (ev.kind === "metric") {
    cls = "e-metric"; txt = `${ev.payload.name} = ${ev.payload.value}`;
  } else return;
  d.className = cls;
  d.textContent = `${ev.t.toFixed(1).padStart(7)}  ${txt}`;
  box.appendChild(d);
  while (box.childElementCount > LOG_MAX) box.removeChild(box.firstChild);
  if (atEnd) box.scrollTop = box.scrollHeight;
}

// --------------------------------------------------------------------------
// the finished programme
// --------------------------------------------------------------------------
async function maybeLoadBundle() {
  if (!app.jobId || app.prog) return;
  try {
    const p = await Program.load(app.jobId);
    if (p.meta.schema_version !== 1) {
      // The python side refuses an unknown key by name; this is the same
      // refusal on the browser side, and it is here rather than nowhere
      // because a viewer silently ignoring a field it does not know is
      // exactly the failure `program_schema` was written to end.
      banner(`this bundle is schema version ${p.meta.schema_version} and the `
             + `viewer speaks version 1 — re-export it with `
             + `scripts/export_viewer_bundle.py`);
      return;
    }
    app.prog = p;
    app.panels.setState(app.state);
    app.panels.setProgram(p);
    app.view.setInk(p.ink);
    // The bundle's own strokes replace the live ones: the live set is
    // decimated for the wire, this one is the record.
    if (p.strokes.length && p.strokes[0].pts.length)
      app.view.setStrokes(p.strokes);
    // target (faint grey, above) vs drawn (the plan's own tip path, in the
    // arm's colour) vs residual (red) — the three layers of the ink overlay
    for (const a of p.arms) {
      if (a.drawnXY && a.drawnOff && a.drawnOff.length > 1)
        app.view.addPlanned(a.arm, a.drawnXY, a.drawnOff, colorOf(a.arm));
      else
        for (const s of p.segments.filter(x => x.arm === a.arm))
          app.view.addSpan({stroke: s.stroke_id, arm: s.arm,
                            s0: s.s_range[0], s1: s.s_range[1]},
                           colorOf(s.arm));
    }
    app.view.addDropped(p.dropped);
    app.legend = `target <span style="color:#5a6068">grey</span> · drawn `
      + `<span style="color:#8a8f99">arm colour</span> · residual `
      + `<span style="color:#ff2fd0">magenta</span>`
      + ` — ${p.meta.coverage_pct.toFixed(2)} % of `
      + `${p.meta.traced_m.toFixed(3)} m drawn, ${p.dropped.length} holes`;
    onFrame(0);
    el("b-conn").textContent = `programme · ${p.F} frames`;
  } catch (e) {
    // A trace-only or refused run has no bundle, and that is not an error —
    // but a bundle that is there and will not load IS one, so it is at least
    // said out loud in the console rather than swallowed.
    if (!/404/.test(String(e.message))) {
      console.error("could not load the programme bundle:", e);
      banner(`the bundle for this job would not load: ${e.message}`);
    }
  }
}

function onFrame(frame) {
  const p = app.prog;
  if (!p) return;
  for (const a of p.arms) {
    const q = p.qAt(a.arm, frame);
    if (q) app.view.setJoints(a.arm, q);
  }
  app.view.setInkTime(p.timeAt(frame));
  // what each arm is drawing right now
  const bits = [];
  for (const a of p.arms) {
    const si = p.segAt(a.arm, frame);
    const seg = si >= 0 ? p.segByArm.get(a.arm)?.get(si) : null;
    bits.push(`<span style="color:${a.color_hex}">${a.arm}</span> `
      + (seg ? `stroke ${seg.stroke_id} [${seg.s_range[0].toFixed(2)}–`
             + `${seg.s_range[1].toFixed(2)}]` : "—"));
  }
  el("viewhud").innerHTML =
    `t = ${p.timeAt(frame).toFixed(2)} s &nbsp; ` + bits.join(" &nbsp; ")
    + (app.legend ? `<br>${app.legend}` : "");
}

function onSelectSegment(seg) {
  el("viewhud").innerHTML =
    `arm <span style="color:${colorOf(seg.arm)}">${seg.arm}</span>, stroke `
    + `${seg.stroke_id} [${seg.s_range[0].toFixed(3)}–`
    + `${seg.s_range[1].toFixed(3)}], ${seg.length_m.toFixed(3)} m, `
    + `σmin ${seg.min_sigma.toFixed(3)}, lean ${seg.lean_deg.toFixed(1)}° `
    + `of a ${seg.cone_deg.toFixed(1)}° cone`;
}

// The witness line: which two capsules made the dip, and where.
function onWitness(a, b, frame) {
  const p = app.prog, doc = app.sceneDoc;
  if (!p || !doc) return;
  const A = p.armById.get(a), B = p.armById.get(b);
  if (!A || !B) return;
  const PA = chainPoints(Float64Array.from(A.T_world_base), p.qAt(a, frame),
                         A.pen_ext_m, A.pen_lat_m);
  const PB = chainPoints(Float64Array.from(B.T_world_base), p.qAt(b, frame),
                         B.pen_ext_m, B.pen_lat_m);
  const w = witnessBetween(PA, PB, doc.radii);
  if (!w) return;
  app.view.showWitness(w.P, w.Q);
  app.view.setLayer("chains", true);
  app.layers.chains = 1;
  el("viewhud").innerHTML =
    `arm <span style="color:${colorOf(a)}">${a}</span> ↔ `
    + `<span style="color:${colorOf(b)}">${b}</span> at t = `
    + `${p.timeAt(frame).toFixed(2)} s: <b>${(1000 * w.d).toFixed(1)} mm</b> `
    + `(margin ${(1000 * p.margin).toFixed(0)} mm) — capsules `
    + `[${w.i[0]},${w.i[1]}] and [${w.j[0]},${w.j[1]}]`;
}

// THE ONE HANDLE ON THE PAGE.  Everything above is module-scoped, which is
// right — but it also means a console, a headless Chrome over CDP or a
// screenshot script has no way to aim the camera at a hand or step to a
// pen-down frame, and "open it and drag until it looks right" is not a thing a
// test can do.  Read-mostly: this is a debugging handle, not an API.
window.aris = app;

boot();
