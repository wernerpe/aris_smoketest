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

export class Panel {
  constructor(el, {onStart, onCancel, onSelect}) {
    this.el = el;
    this.cb = {onStart, onCancel, onSelect};
    this.inputs = {};
    this.selected = null;
  }

  build(cfg) {
    this.cfg = cfg;
    const d = cfg.defaults || {};
    this.el.innerHTML = "";
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
  }

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
        `<b>${esc(j.params.out || j.params.source || j.id)}</b><br>` +
        `<span class="t">${when} · ${fmtS(j.elapsed_s)} · ` +
        `${esc(j.params.rig || "")}/${esc(j.params.tool || "")}</span>`;
      this.jobList.appendChild(el);
    }
  }
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
