// The left column: the parameter form, start/cancel, and the job list.
//
// THE FORM SENDS ONLY WHAT IT WAS ASKED TO CHANGE.  Every field starts at the
// value `/api/config` reports, and a field left at its default is still sent —
// but `aris_sixarm.gui.worker.ARGSPEC` is a WHITELIST, so a key the form does
// not know about cannot become a flag, and a flag the form does not send keeps
// `scripts/draw.py`'s own default.  A GUI that carried its own copy of a
// planner default would be a second place for a constant to live, which is how
// two numbers that must agree stop agreeing.

const F = (tag, attrs = {}, kids = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c) e.appendChild(c);
  return e;
};

// (key, label, kind, options)  — kind: text | num | check | select
const FIELDS = [
  ["Picture", [
    ["source", "source picture", "sources"],
    ["out", "output name", "text"],
    ["title", "title (figures/legend)", "text"],
    ["rig", "rig", "select", "rigs"],
    ["tool", "tool", "select", "tools"],
  ]],
  ["Trace", [
    ["inks", "inks (auto | N)", "text"],
    ["work_px", "working raster, long side px", "num"],
    ["min_len", "shortest stroke on paper, m", "num"],
    ["trace_only", "stop after the trace figure", "check"],
  ]],
  ["Placement", [
    ["placement", "auto | off | a placement json", "text"],
    ["target_width", "target width, m (placement off)", "num"],
    ["rotate", "rotation(s), deg", "text"],
    ["offset", "offset dx,dy m", "pair"],
    ["margin", "sheet margin, m", "num"],
    ["scales", "scale search lo,hi,n", "triple"],
    ["top", "translations allocated per cell", "num"],
    ["slack", "coverage a bigger placement may give up", "num"],
    ["jobs", "placement search processes", "num"],
    ["place_only", "stop after the placement search", "check"],
  ]],
  ["Allocation", [
    ["arms", "arms (all | 13,31,97)", "text"],
    ["atlas", "atlas dir (prefilter)", "select", "atlases"],
    ["max_probes", "plan calls per (stroke, arm)", "num"],
    ["tilt_max_deg", "pen-tilt rescue cone, deg", "num"],
    ["band_objective", "band objective", "select", "objectives"],
    ["sequencer", "sequencer", "select", "sequencers"],
    ["draw_speed", "draw speed, m/s", "num"],
    ["transit_speed", "transit speed, m/s", "num"],
    ["two_pass", "two passes with a pen swap", "check"],
    ["no_balance", "skip the min-max load balance", "check"],
    ["no_split", "no stroke splitting (allocation v1)", "check"],
    ["no_merge", "no remainder merging", "check"],
    ["no_rrt", "no C-space pen-up planner (much faster)", "check"],
    ["residual_passes", "residual passes over the holes", "num"],
  ]],
  ["Conduct & check", [
    ["arm_phases", "arm phasing (off | solo | disjoint)", "text"],
    ["idle_policy", "idle policy", "select", "policies"],
    ["qd_frac", "joint-speed cap fraction", "num"],
    ["fps", "animation fps", "num"],
    ["substeps", "coordination substeps per frame", "num"],
    ["subcheck", "scene_check re-sampling", "num"],
    ["image_jobs", "collision-image processes", "num"],
    ["min_coverage", "refuse below this coverage", "num"],
    ["select_profile", "conduct 4 profiles, ship the fastest", "check"],
    ["skip_unconductable", "skip a phase nobody can conduct", "check"],
    ["no_verify", "skip the unsplit A/B conduct", "check"],
  ]],
];

const OPTIONS = {
  objectives: ["maximin_sigma", "min_travel"],
  sequencers: ["opt", "nn"],
  policies: ["freeze", "home"],
};

// --------------------------------------------------------------------------
// HARDWARE DAY 1
//
// TWO BUTTONS, AND THEY RUN `scripts/day1.py`.  Not a second front door: the
// job the browser posts becomes the argument list a person would type
// (`gui/worker.build_day1_argv`), the CSVs land in `out/day1/` where
// docs/HARDWARE_DAY1.md says they are, and the PASS/FAIL line here is the same
// string `day1.py` prints in a terminal, built by the same formatter.
const DAY1_LINE = [
  ["arm", "arm", "select"],
  ["from", "from  x,y  (m, canvas datum)", "text"],
  ["to", "to  x,y  (m)", "text"],
  ["name", "file stem", "text"],
  ["hover", "fly it 30 mm ABOVE the paper (no ink, no contact)", "check"],
];

export class Day1 {
  constructor(onRun, onMeshcat) {
    this.onRun = onRun;
    this.onMeshcat = onMeshcat;
    this.inputs = {};
    this.last = null;
  }

  build(cfg) {
    const d = (cfg && cfg.day1) || {};
    this.cfg = d;
    const body = F("div", {class: "body"});
    body.appendChild(F("div", {class: "muted", text: d.note || ""}));

    for (const [key, label, kind] of DAY1_LINE) {
      let input;
      if (kind === "check") {
        input = F("input", {type: "checkbox"});
        input.checked = !!d[key];
        const wrap = F("label", {class: "check"}, [input]);
        wrap.appendChild(document.createTextNode(" " + label));
        body.appendChild(wrap);
      } else if (kind === "select") {
        input = F("select");
        for (const v of (d.arms || [31, 71])) {
          const o = F("option", {value: v, text: String(v)});
          if (Number(v) === Number(d[key])) o.selected = true;
          input.appendChild(o);
        }
        body.appendChild(F("div", {class: "row one"},
          [F("div", {}, [F("label", {text: label}), input])]));
      } else {
        input = F("input", {type: "text",
                            value: d[key] == null ? "" : String(d[key])});
        body.appendChild(F("div", {class: "row one"},
          [F("div", {}, [F("label", {text: label}), input])]));
      }
      this.inputs[key] = {el: input, kind};
    }

    this.lineBtn = F("button", {class: "primary", text: "Plan + certify line",
                                onclick: () => this.onRun(this.lineParams())});
    body.appendChild(F("div", {class: "btnrow one"}, [this.lineBtn]));

    this.variant = F("select");
    for (const v of (d.variants || ["alt", "concurrent", "hover"])) {
      const o = F("option", {value: v, text: v});
      if (v === d.variant) o.selected = true;
      this.variant.appendChild(o);
    }
    // THE WORD, AND IT IS NOW TWO RUNS BEHIND ONE CONTROL.  "both" is the
    // two-arm asset re-check this panel has always had and plans nothing; 31
    // or 71 is the SOLO word, which PLANS it here and now with the same
    // certified planner the line uses.  `variant` only means anything for
    // "both", and width / hover only for a solo arm -- the params builder
    // sends whichever applies and `gui/worker.build_day1_argv` refuses the
    // rest, so a stale field in the form cannot become a silent flag.
    this.wordArm = F("select");
    for (const v of ["both", ...(d.arms || [31, 71])]) {
      const o = F("option", {value: String(v), text: String(v)});
      if (String(v) === String(d.word_arm == null ? "both" : d.word_arm))
        o.selected = true;
      this.wordArm.appendChild(o);
    }
    this.wordWidth = F("input", {
      type: "text",
      value: String(d.word_width == null ? 0.55 : d.word_width)});
    this.wordHover = F("input", {type: "checkbox"});
    this.wordHover.checked = !!d.word_hover;
    const hoverWrap = F("label", {class: "check"}, [this.wordHover]);
    hoverWrap.appendChild(document.createTextNode(
      " word 30 mm ABOVE the paper (solo arm; no ink, no contact)"));
    this.wordBtn = F("button", {text: "Run word",
                                onclick: () => this.onRun(this.wordParams())});
    body.appendChild(F("div", {class: "row"}, [
      F("div", {}, [F("label", {text: "word arm"}), this.wordArm]),
      F("div", {}, [F("label", {text: "variant  (arm = both)"}),
                    this.variant])]));
    body.appendChild(F("div", {class: "row"}, [
      F("div", {}, [F("label", {text: "word width  m  (solo arm)"}),
                    this.wordWidth]),
      F("div", {}, [F("label", {text: " "}), this.wordBtn])]));
    body.appendChild(hoverWrap);

    // THE HIGH-QUALITY VIEW, AND IT IS A SECOND WINDOW.  The three.js viewer
    // to the right is built from primitives — boxes and cylinders, no mesh
    // files anywhere in `web/` — which is fine for reading a timeline and
    // wrong for judging a machine.  This hands the same npz to
    // `scripts/meshcat_drake.py`, which plays it through the system-model URDF
    // in Drake's meshcat with the real FR3 glTFs.  Nothing here changes the
    // viewer beside it.
    this.meshcatBtn = F("button", {text: "Open in Drake Meshcat",
                                   onclick: () => this._meshcat()});
    this.meshcatBtn.disabled = true;
    this.meshcatLink = F("a", {class: "muted", target: "_blank", text: ""});
    body.appendChild(F("div", {class: "btnrow one"}, [this.meshcatBtn]));
    body.appendChild(this.meshcatLink);

    this.out = F("div", {id: "day1out"});
    body.appendChild(this.out);

    const head = F("h3", {text: "Hardware day 1"});
    const g = F("div", {class: "group"}, [head, body]);
    head.addEventListener("click", () => g.classList.toggle("collapsed"));
    return g;
  }

  _val(k) { return String(this.inputs[k].el.value).trim(); }

  lineParams() {
    const d = this.cfg || {};
    const p = {day1: "line", rig: d.rig || "proposed", tool: d.tool || "lateral",
               arm: Number(this._val("arm")), from: this._val("from"),
               to: this._val("to"), name: this._val("name") || "line"};
    if (this.inputs.hover.el.checked) p.hover = d.hover_m || 0.03;
    return p;
  }

  wordParams() {
    const d = this.cfg || {};
    const p = {day1: "word", rig: d.rig || "proposed",
               tool: d.tool || "lateral"};
    const arm = String(this.wordArm.value);
    if (arm === "both") {                 // the two-arm asset; plans nothing
      p.variant = String(this.variant.value);
      return p;
    }
    p.arm = Number(arm);                  // the SOLO word; plans it now
    const w = Number(String(this.wordWidth.value).trim());
    if (Number.isFinite(w) && w > 0) p.width = w;
    if (this.wordHover.checked) p.hover = d.hover_m || 0.03;
    return p;
  }

  setRunning(live) {
    this.lineBtn.disabled = !!live;
    this.wordBtn.disabled = !!live;
    if (this.meshcatBtn)
      this.meshcatBtn.disabled = !!live || !this.meshcatParams();
  }

  // THE PORT COMES FROM THE SERVER, THE HOST FROM THE BROWSER.  The box calls
  // itself `frankastation` and the lab calls it
  // `frankastation.drl.csail.mit.edu`; whichever name got the person to this
  // page is the one that will reach the scene, so reuse it rather than trust
  // the server's idea of its own FQDN.
  async _meshcat() {
    const p = this.meshcatParams();
    if (!p || !this.onMeshcat) return;
    const b = this.meshcatBtn;
    const was = b.textContent;
    b.disabled = true;
    b.textContent = "starting...";
    try {
      const r = await this.onMeshcat(p);
      const url = `${location.protocol}//${location.hostname}:${r.port}/`;
      this.meshcatLink.href = url;
      this.meshcatLink.textContent = url;
      window.open(url, "_blank");
    } catch (e) {
      this.meshcatLink.textContent = `could not start it: ${e.message}`;
    } finally {
      b.textContent = was;
      b.disabled = !this.meshcatParams();
    }
  }

  clear() {
    this.last = null;
    if (this.meshcatBtn) this.meshcatBtn.disabled = true;
    if (this.out) this.out.innerHTML = "";
  }

  // WHERE THE MESHCAT BUTTON GETS ITS npz.  The day-1 verdict payload carries
  // `npz` and `program` (worker._day1_result), and until now `show` read only
  // the verdict line and the CSV list and dropped the rest.  Keeping the whole
  // record is what lets the button below name a file the server can open.
  // The three.js viewer is untouched and still loads itself from the bundle.
  meshcatParams() {
    const r = this.last, d = this.cfg || {};
    if (!r || !r.npz) return null;
    return {npz: r.npz, program: r.program || null,
            rig: d.rig || "proposed", tool: d.tool || "lateral",
            only_arms: (d.arms || [31, 71]).join(",")};
  }

  // The verdict, as `day1.py` printed it, and the files it left behind.
  show(r) {
    if (!this.out) return;
    this.last = r;
    if (this.meshcatBtn) this.meshcatBtn.disabled = !this.meshcatParams();
    this.out.innerHTML = "";
    this.out.appendChild(F("div", {
      class: "verdict " + (r.ok ? "ok" : "bad"), text: r.one_liner || ""}));
    for (const path of (r.csv || [])) {
      const row = F("div", {class: "filerow"});
      row.appendChild(F("code", {text: path}));
      const b = F("button", {text: "copy"});
      b.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(path); }
        catch (e) {                       // no clipboard on an http: origin
          const t = F("textarea", {}); t.value = path;
          document.body.appendChild(t); t.select();
          try { document.execCommand("copy"); } catch (e2) {}
          t.remove();
        }
        b.textContent = "copied"; setTimeout(() => b.textContent = "copy", 1200);
      });
      row.appendChild(b);
      this.out.appendChild(row);
    }
    if (!(r.csv || []).length)
      this.out.appendChild(F("div", {class: "muted", text: "no CSV written"}));
  }
}

export class Panel {
  constructor(el, {onStart, onCancel, onSelect, onMeshcat}) {
    this.el = el;
    this.cb = {onStart, onCancel, onSelect, onMeshcat};
    this.inputs = {};
    this.selected = null;
    // FIRST IN THE COLUMN, because on a hardware day it is the only thing
    // anybody touches; the planner form below it is the other days' panel.
    this.day1 = new Day1(onStart, onMeshcat);
  }

  build(cfg) {
    this.cfg = cfg;
    const d = cfg.defaults || {};
    this.el.innerHTML = "";
    this.el.appendChild(this.day1.build(cfg));
    const opts = Object.assign({}, OPTIONS, {
      rigs: cfg.rigs, tools: cfg.tools,
      atlases: [""].concat(cfg.atlases || []),
    });

    for (const [title, fields] of FIELDS) {
      const body = F("div", {class: "body"});
      for (const [key, label, kind, optKey] of fields) {
        body.appendChild(this._field(key, label, kind, opts[optKey], d[key],
                                     cfg));
      }
      const head = F("h3", {text: title});
      const g = F("div", {class: "group" + (title === "Picture" ||
                                            title === "Placement" ? "" :
                                            " collapsed")}, [head, body]);
      head.addEventListener("click", () => g.classList.toggle("collapsed"));
      this.el.appendChild(g);
    }

    this.startBtn = F("button", {class: "primary", text: "Plan",
                                 onclick: () => this.cb.onStart(this.values())});
    this.cancelBtn = F("button", {class: "danger", text: "Cancel",
                                  onclick: () => this.cb.onCancel()});
    this.cancelBtn.disabled = true;
    this.el.appendChild(F("div", {class: "btnrow"},
                          [this.startBtn, this.cancelBtn]));
    this.el.appendChild(F("h3", {text: "Jobs",
                                 style: "font-size:11px;text-transform:uppercase;"
                                        + "letter-spacing:.09em;color:#6d7480;"
                                        + "border-bottom:1px solid #2e333d;"
                                        + "padding-bottom:3px;margin-bottom:6px"}));
    this.jobList = F("div", {id: "joblist"});
    this.el.appendChild(this.jobList);
  }

  _field(key, label, kind, choices, def, cfg) {
    let input;
    if (kind === "check") {
      input = F("input", {type: "checkbox"});
      input.checked = !!def;
      const wrap = F("label", {class: "check"}, [input]);
      wrap.appendChild(document.createTextNode(" " + label));
      this.inputs[key] = {el: input, kind};
      return wrap;
    }
    if (kind === "select" || kind === "sources") {
      input = F("select");
      const list = kind === "sources"
        ? (cfg.sources || []).map(s => s.path) : (choices || []);
      for (const v of list) {
        const o = F("option", {value: v, text: v || "(none)"});
        if (v === def) o.selected = true;
        input.appendChild(o);
      }
      if (kind === "sources" && def && !list.includes(def)) {
        const o = F("option", {value: def, text: def});
        o.selected = true;
        input.insertBefore(o, input.firstChild);
      }
    } else if (kind === "pair" || kind === "triple") {
      input = F("input", {type: "text",
                          value: Array.isArray(def) ? def.join(",")
                               : (def == null ? "" : String(def))});
    } else {
      input = F("input", {type: kind === "num" ? "number" : "text",
                          step: "any",
                          value: def == null ? "" : String(def)});
    }
    this.inputs[key] = {el: input, kind};
    return F("div", {class: "row one"},
             [F("div", {}, [F("label", {text: label}), input])]);
  }

  values() {
    const out = {};
    for (const [key, {el, kind}] of Object.entries(this.inputs)) {
      if (kind === "check") { if (el.checked) out[key] = true; continue; }
      const raw = String(el.value).trim();
      if (raw === "") continue;
      if (kind === "num") { const v = Number(raw); if (!Number.isNaN(v)) out[key] = v; }
      else if (kind === "pair" || kind === "triple") {
        const parts = raw.split(",").map(x => Number(x.trim()));
        if (parts.every(x => !Number.isNaN(x))) out[key] = parts;
      } else out[key] = raw;
    }
    return out;
  }

  setRunning(live) {
    this.startBtn.disabled = !!live;
    this.cancelBtn.disabled = !live;
    this.day1.setRunning(live);
  }

  setDay1(result) { this.day1.show(result); }
  clearDay1() { this.day1.clear(); }

  setJobs(jobs, selectedId) {
    this.jobList.innerHTML = "";
    for (const j of jobs) {
      const when = new Date(j.created * 1000).toLocaleTimeString();
      const el = F("div", {
        class: "job" + (j.id === selectedId ? " sel" : ""),
        onclick: () => this.cb.onSelect(j.id),
      });
      el.innerHTML =
        `<span class="st ${j.status}">${j.status}</span>` +
        `<b>${esc(j.params.day1 ? _day1Label(j.params)
                                : (j.params.out || j.params.source || j.id))}</b><br>` +
        `<span class="t">${when} · ${fmtS(j.elapsed_s)} · ` +
        `${esc(j.params.rig || "")}/${esc(j.params.tool || "")}</span>`;
      this.jobList.appendChild(el);
    }
  }
}

function _day1Label(p) {
  if (p.day1 === "line") return `line ${p.name || ""}_${p.arm}`;
  if (p.arm) return `word arm ${p.arm}${p.hover ? " hover" : ""}`;
  return `word ${p.variant || "alt"}`;
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g,
    c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

export function fmtS(x) {
  if (x == null || Number.isNaN(x)) return "—";
  if (x < 60) return `${x.toFixed(1)} s`;
  if (x < 3600) return `${Math.floor(x / 60)}m ${Math.round(x % 60)}s`;
  return `${Math.floor(x / 3600)}h ${Math.round((x % 3600) / 60)}m`;
}

export {F};
