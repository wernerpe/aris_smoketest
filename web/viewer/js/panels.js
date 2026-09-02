// The bottom-left panels: the per-arm scrubber, the clearance inspector, the
// per-segment table, and the timing breakdown.
//
// EVERY PANEL IS A VIEW OF THE SAME BUNDLE.  Clicking a dip in the clearance
// inspector moves the scrubber; moving the scrubber moves the arms in the 3D
// view and highlights the segment each arm is drawing; clicking a segment
// jumps the scrubber to it.  There is one clock (the frame index) and one
// selection (a segment), and all four panels read them, which is why a number
// in one panel can be trusted against a number in another.

import {F, fmtS, esc} from "./panel.js";
import {witnessBetween, chainPoints} from "./program.js";

export class Panels {
  constructor(els, hooks) {
    this.els = els;                     // {tabs, program, clear, stroke, time}
    this.hooks = hooks;                 // {onFrame, onSelectSegment, colorOf}
    this.tab = "program";
    this.frame = 0;
    this.playing = false;
    this.speed = 1;
    this._buildTabs();
  }

  _buildTabs() {
    const names = [["program", "Programme"], ["clear", "Clearance"],
                   ["stroke", "Segments"], ["time", "Where the time goes"]];
    this.els.tabs.innerHTML = "";
    this.btns = {};
    for (const [k, label] of names) {
      const b = F("button", {text: label, class: k === this.tab ? "on" : "",
                             onclick: () => this.show(k)});
      this.btns[k] = b;
      this.els.tabs.appendChild(b);
    }
  }

  show(k) {
    this.tab = k;
    for (const [n, b] of Object.entries(this.btns))
      b.className = n === k ? "on" : "";
    for (const n of ["program", "clear", "stroke", "time"])
      this.els[n].className = "pane" + (n === k ? " on" : "");
    if (this.prog) this.renderTab();
  }

  // ---- live: no programme yet, only events ---------------------------
  renderLive(state) {
    this.prog = null;
    this.els.program.innerHTML = livePhases(state);
    this.els.clear.innerHTML =
      '<div class="muted">The clearance inspector needs a finished '
      + 'programme — it is derived from the conducted timeline.  '
      + 'scene_check\'s per-phase verdict appears in the log and in the '
      + 'stage bar as soon as a phase is checked.</div>' + liveChecks(state);
    this.els.stroke.innerHTML = liveSpans(state, this.hooks.colorOf);
    this.els.time.innerHTML = timeTable(state);
  }

  // ---- a finished programme -------------------------------------------
  setProgram(prog) {
    this.prog = prog;
    this.frame = 0;
    this.renderTab();
    this.show(this.tab);
  }

  renderTab() {
    if (!this.prog) return;
    if (this.tab === "program") this._program();
    else if (this.tab === "clear") this._clearance();
    else if (this.tab === "stroke") this._segments();
    else this._time();
  }

  // ---------------- programme: player + per-arm lanes ------------------
  _program() {
    const p = this.prog, el = this.els.program;
    el.innerHTML = "";
    const play = F("button", {text: this.playing ? "❚❚" : "▶",
                              onclick: () => this.togglePlay()});
    this.playBtn = play;
    const scrub = F("input", {type: "range", id: "scrub", min: 0,
                              max: p.F - 1, value: this.frame, step: 1});
    scrub.addEventListener("input", () => this.setFrame(Number(scrub.value)));
    this.scrub = scrub;
    const speed = F("select", {style: "width:70px",
                               onchange: e => this.speed = Number(e.target.value)});
    for (const v of [0.25, 0.5, 1, 2, 4, 8]) {
      const o = F("option", {value: v, text: `${v}x`});
      if (v === this.speed) o.selected = true;
      speed.appendChild(o);
    }
    this.clockEl = F("span", {class: "muted", style: "min-width:118px",
                              text: ""});
    el.appendChild(F("div", {id: "player"},
                     [play, scrub, speed, this.clockEl]));

    const lanes = F("div", {id: "lanes"});
    this.lanes = [];
    for (const a of p.arms) {
      const c = F("canvas", {height: 15});
      c.addEventListener("click", ev => {
        const r = c.getBoundingClientRect();
        this.setFrame(Math.round((ev.clientX - r.left) / r.width * (p.F - 1)));
      });
      lanes.appendChild(F("div", {class: "lane"},
        [F("div", {class: "anm", text: String(a.arm),
                   style: `color:${a.color_hex}`}), c]));
      this.lanes.push({arm: a.arm, canvas: c});
    }
    el.appendChild(lanes);
    el.appendChild(F("div", {class: "muted",
      html: "solid = drawing (colour = the arm), hollow = pen-up transit, "
          + "dark = parked or paused; vertical rules are phase boundaries"}));
    el.appendChild(phaseTable(p));
    requestAnimationFrame(() => this._drawLanes());
    this.setFrame(this.frame);
  }

  _drawLanes() {
    const p = this.prog;
    if (!p || !this.lanes) return;
    for (const L of this.lanes) {
      const c = L.canvas, w = c.clientWidth || 600, h = 15;
      c.width = Math.max(1, Math.round(w * devicePixelRatio));
      c.height = Math.round(h * devicePixelRatio);
      const g = c.getContext("2d");
      g.scale(devicePixelRatio, devicePixelRatio);
      g.clearRect(0, 0, w, h);
      g.fillStyle = "#22262e";
      g.fillRect(0, 0, w, h);
      const a = p.armById.get(L.arm);
      const col = a.color_hex;
      for (let x = 0; x < w; x++) {
        const f0 = Math.floor(x / w * p.F);
        const f1 = Math.max(f0 + 1, Math.floor((x + 1) / w * p.F));
        let drawing = false, moving = false;
        for (let f = f0; f < f1 && f < p.F; f++) {
          if (a.seg[f] >= 0) drawing = true;
          else moving = true;
        }
        if (drawing) { g.fillStyle = col; g.fillRect(x, 2, 1, h - 4); }
        else if (moving) { g.fillStyle = "#3b414b"; g.fillRect(x, 6, 1, 3); }
      }
      g.strokeStyle = "#6d7480";
      for (const ph of p.doc.phases) {
        const x = ph.start_s * p.fps / p.F * w;
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
      }
      L.ctx = g; L.w = w;
    }
    this._cursor();
  }

  _cursor() {
    if (!this.lanes || !this.prog) return;
    for (const L of this.lanes) {
      if (!L.ctx) continue;
      // Redrawing the whole lane per frame would be six full repaints at 24
      // fps; instead the cursor is a DOM overlay the browser composites.
      if (!L.cur) {
        L.cur = F("div");
        L.cur.style.cssText = "position:absolute;top:0;bottom:0;width:1px;"
          + "background:#fff;pointer-events:none";
        L.canvas.parentElement.style.position = "relative";
        L.canvas.parentElement.appendChild(L.cur);
      }
      const frac = this.frame / Math.max(1, this.prog.F - 1);
      L.cur.style.left = `calc(46px + 6px + ${(100 * frac).toFixed(3)}% * `
        + `(1 - (52px / 100%)))`;
      L.cur.style.left = (52 + frac * (L.w || 1)) + "px";
    }
  }

  setFrame(f) {
    if (!this.prog) return;
    this.frame = Math.max(0, Math.min(this.prog.F - 1, Math.round(f)));
    if (this.scrub) this.scrub.value = this.frame;
    if (this.clockEl) {
      const t = this.prog.timeAt(this.frame);
      this.clockEl.textContent =
        `${t.toFixed(2)} s / ${this.prog.meta.duration_s.toFixed(1)} s`
        + `  (frame ${this.frame})`;
    }
    this._cursor();
    this.hooks.onFrame(this.frame);
  }

  togglePlay() {
    this.playing = !this.playing;
    if (this.playBtn) this.playBtn.textContent = this.playing ? "❚❚" : "▶";
    if (this.playing) {
      this._last = performance.now();
      const step = (now) => {
        if (!this.playing || !this.prog) return;
        const dt = (now - this._last) / 1000;
        this._last = now;
        let f = this.frame + dt * this.prog.fps * this.speed;
        if (f >= this.prog.F - 1) { f = 0; }
        this.setFrame(f);
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
  }

  // ---------------- clearance inspector --------------------------------
  _clearance() {
    const p = this.prog, el = this.els.clear;
    el.innerHTML = "";
    if (!p.clearance.length) {
      el.innerHTML = '<div class="muted">no clearance series in this bundle</div>';
      return;
    }
    el.appendChild(F("div", {class: "muted", html:
      `minimum distance between two arms' capsule sets, per frame; the dotted `
      + `line is the ${(1000 * p.margin).toFixed(0)} mm margin the programme `
      + `was certified against.  Click a dip to jump the scrubber there and `
      + `draw the witness.`}));
    const wrap = F("div", {id: "spark"});
    this.sparks = [];
    const rows = p.clearance.map(c => ({...c, min: minOf(c.d)}))
      .sort((a, b) => a.min - b.min);
    for (const c of rows) {
      const cv = F("canvas");
      const vv = F("div", {class: "v" + (c.min < p.margin ? " bad" : ""),
                           text: `${(1000 * c.min).toFixed(1)} mm`});
      const row = F("div", {class: "spk"}, [
        F("div", {html: `<span style="color:${this.hooks.colorOf(c.a)}">${c.a}</span>`
                       + `–<span style="color:${this.hooks.colorOf(c.b)}">${c.b}</span>`}),
        cv, vv]);
      cv.addEventListener("click", ev => {
        const r = cv.getBoundingClientRect();
        const i = Math.round((ev.clientX - r.left) / r.width * (c.d.length - 1));
        this.setFrame(i * p.clearStride);
        this.hooks.onWitness(c.a, c.b, Math.round(i * p.clearStride));
      });
      wrap.appendChild(row);
      this.sparks.push({c, canvas: cv, kind: "pair"});
    }
    for (const [arm, d] of Object.entries(p.selfD)) {
      const cv = F("canvas");
      const mn = minOf(d);
      const row = F("div", {class: "spk"}, [
        F("div", {html: `<span style="color:${this.hooks.colorOf(Number(arm))}">`
                       + `${arm}</span> self`}),
        cv,
        F("div", {class: "v" + (mn < p.selfMargin ? " bad" : ""),
                  text: `${(1000 * mn).toFixed(1)} mm`})]);
      cv.addEventListener("click", ev => {
        const r = cv.getBoundingClientRect();
        this.setFrame(Math.round((ev.clientX - r.left) / r.width *
                                 (d.length - 1)) * p.clearStride);
      });
      wrap.appendChild(row);
      this.sparks.push({c: {d, min: mn, self: arm}, canvas: cv, kind: "self"});
    }
    el.appendChild(wrap);
    requestAnimationFrame(() => this._drawSparks());
  }

  _drawSparks() {
    const p = this.prog;
    for (const s of this.sparks || []) {
      const cv = s.canvas, w = cv.clientWidth || 300, h = 22;
      cv.width = Math.max(1, Math.round(w * devicePixelRatio));
      cv.height = Math.round(h * devicePixelRatio);
      const g = cv.getContext("2d");
      g.scale(devicePixelRatio, devicePixelRatio);
      g.fillStyle = "#22262e"; g.fillRect(0, 0, w, h);
      const d = s.c.d, n = d.length;
      let hi = 0;
      for (let i = 0; i < n; i++) if (d[i] > hi && Number.isFinite(d[i])) hi = d[i];
      hi = Math.max(hi, 0.05);
      const margin = s.kind === "self" ? p.selfMargin : p.margin;
      const y = v => h - 1 - Math.max(0, Math.min(1, v / hi)) * (h - 2);
      g.strokeStyle = "#6d7480"; g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(0, y(margin)); g.lineTo(w, y(margin)); g.stroke();
      g.setLineDash([]);
      g.strokeStyle = s.c.min < margin ? "#d2544a" : "#4a8fd2";
      g.beginPath();
      for (let x = 0; x < w; x++) {
        const i0 = Math.floor(x / w * n), i1 = Math.max(i0 + 1,
                                                        Math.floor((x+1)/w*n));
        let m = Infinity;
        for (let i = i0; i < i1 && i < n; i++) if (d[i] < m) m = d[i];
        if (!Number.isFinite(m)) m = hi;
        if (x === 0) g.moveTo(x, y(m)); else g.lineTo(x, y(m));
      }
      g.stroke();
    }
  }

  // ---------------- segments -------------------------------------------
  _segments() {
    const p = this.prog, el = this.els.stroke;
    const rows = p.segments.slice().sort(
      (a, b) => a.arm - b.arm || a.index - b.index);
    el.innerHTML = "";
    el.appendChild(F("div", {class: "muted", html:
      "every certified span: click one to jump the scrubber to the moment it "
      + "is drawn.  <b>lean</b> is the pen tilt the plan COMMANDED and "
      + "<b>cone</b> the permission it was planned under — one name each, "
      + "which is the point of the typed schema."}));
    const t = F("table", {class: "k"});
    t.innerHTML = "<tr><th>arm</th><th>stroke</th><th>s-range</th>"
      + "<th>m</th><th>&sigma;min</th><th>margin</th><th>tip</th>"
      + "<th>lean&deg;</th><th>cone&deg;</th><th>draw s</th><th>ok</th></tr>";
    for (const s of rows) {
      const tr = F("tr", {style: "cursor:pointer"});
      tr.innerHTML =
        `<td style="color:${this.hooks.colorOf(s.arm)}">${s.arm}</td>` +
        `<td class="n">${s.stroke_id}</td>` +
        `<td class="n">${s.s_range[0].toFixed(3)}–${s.s_range[1].toFixed(3)}</td>` +
        `<td class="n">${s.length_m.toFixed(3)}</td>` +
        `<td class="n">${s.min_sigma.toFixed(3)}</td>` +
        `<td class="n">${s.min_margin.toFixed(3)}</td>` +
        `<td class="n">${(1000 * s.tip_err_m).toExponential(1)}</td>` +
        `<td class="n">${s.lean_deg.toFixed(1)}</td>` +
        `<td class="n">${s.cone_deg.toFixed(1)}</td>` +
        `<td class="n">${s.draw_time_s.toFixed(1)}</td>` +
        `<td>${s.plan_ok && s.validated ? "✓" : "✗"}</td>`;
      tr.addEventListener("click", () => {
        const f = firstFrameOf(p, s.arm, s.index);
        if (f >= 0) this.setFrame(f);
        this.hooks.onSelectSegment(s);
      });
      t.appendChild(tr);
    }
    el.appendChild(t);
  }

  // ---------------- timings ---------------------------------------------
  _time() { this.els.time.innerHTML = timeTable(this.state || {}, this.prog); }

  setState(s) { this.state = s; }
}

// --------------------------------------------------------------------------
function firstFrameOf(p, arm, segIndex) {
  const a = p.armById.get(arm);
  if (!a) return -1;
  for (let f = 0; f < p.F; f++) if (a.seg[f] === segIndex) return f;
  return -1;
}

function minOf(d) {
  let m = Infinity;
  for (let i = 0; i < d.length; i++) if (d[i] < m) m = d[i];
  return Number.isFinite(m) ? m : 0;
}

function phaseTable(p) {
  const t = F("table", {class: "k", style: "margin-top:8px"});
  let html = "<tr><th>phase</th><th>ink</th><th>start</th><th>duration</th>"
    + "<th>floor</th><th>min clearance</th><th>scene_check</th></tr>";
  for (const ph of p.doc.phases) {
    html += `<tr><td>${esc(ph.name)}</td><td>${esc(ph.ink || "")}</td>`
      + `<td class="n">${ph.start_s.toFixed(1)} s</td>`
      + `<td class="n">${ph.duration_s.toFixed(1)} s</td>`
      + `<td class="n">${ph.floor_s.toFixed(1)} s</td>`
      + `<td class="n">${(1000 * ph.min_clearance_m).toFixed(1)} mm</td>`
      + `<td style="color:${ph.scene_check_ok ? "#3fa46a" : "#d2544a"}">`
      + `${ph.scene_check_ok ? "PASS" : "FAIL"}</td></tr>`;
  }
  t.innerHTML = html;
  return t;
}

function livePhases(s) {
  if (!s.phases || !s.phases.length) {
    return '<div class="muted">no phase has been conducted yet.  '
      + 'The programme panel becomes a scrubber over the finished timeline '
      + 'once the job writes its bundle.</div>';
  }
  let h = '<table class="k"><tr><th>phase</th><th>ink</th><th>duration</th>'
    + '<th>min clearance</th><th>scene_check</th><th>split kept</th></tr>';
  for (const p of s.phases) {
    h += `<tr><td>${esc(p.name)}</td><td>${esc(p.ink || "")}</td>`
      + `<td class="n">${(p.duration_s || 0).toFixed(1)} s</td>`
      + `<td class="n">${(1000 * (p.min_clearance_m || 0)).toFixed(1)} mm</td>`
      + `<td style="color:${p.scene_check_ok ? "#3fa46a" : "#d2544a"}">`
      + `${p.scene_check_ok ? "PASS" : "FAIL"}</td>`
      + `<td>${p.split_kept ? "yes" : "no"}</td></tr>`;
  }
  return h + "</table>";
}

function liveChecks(s) {
  const st = s.stages && s.stages.scene_check;
  if (!st || !st.end) return "";
  const e = st.end;
  let h = '<table class="k" style="margin-top:8px">'
    + '<tr><th>pair</th><th>min clearance</th></tr>';
  const pp = Object.entries(e.per_pair_m || {}).sort((a, b) => a[1] - b[1]);
  for (const [k, v] of pp) {
    h += `<tr><td>${esc(k)}</td><td class="n"`
      + `${v < (e.margin_m || 0) ? ' style="color:#d2544a"' : ""}>`
      + `${(1000 * v).toFixed(1)} mm</td></tr>`;
  }
  return h + "</table>";
}

function liveSpans(s, colorOf) {
  if (!s.placedSpans || !s.placedSpans.length) {
    return '<div class="muted">no span has been certified yet</div>';
  }
  let h = '<table class="k"><tr><th>arm</th><th>stroke</th><th>s-range</th>'
    + '<th>m</th><th>&sigma;min</th><th>lean&deg;</th></tr>';
  for (const v of s.placedSpans.slice(-400).reverse()) {
    h += `<tr><td style="color:${colorOf(v.arm)}">${v.arm}</td>`
      + `<td class="n">${v.stroke}</td>`
      + `<td class="n">${v.s0.toFixed(3)}–${v.s1.toFixed(3)}</td>`
      + `<td class="n">${v.length.toFixed(3)}</td>`
      + `<td class="n">${(v.sigma || 0).toFixed(3)}</td>`
      + `<td class="n">${(v.lean || 0).toFixed(1)}</td></tr>`;
  }
  return h + "</table>";
}

function timeTable(s, prog) {
  const stages = s.stages || {};
  const subs = s.subs || {};
  const wall = Math.max(s.clock || 0, 0.001);
  let h = '<table class="k"><tr><th>stage</th><th>seconds</th><th>share</th>'
    + '<th>runs</th></tr>';
  const order = ["trace", "placement", "allocation", "conduction",
                 "scene_check", "export"];
  for (const name of order) {
    const st = stages[name];
    if (!st) continue;
    let tot = st.total || 0;
    if (name === "conduction" && stages.scene_check)
      tot -= stages.scene_check.total || 0;
    h += `<tr><td>${name}</td><td class="n">${fmtS(tot)}</td>`
      + `<td class="n">${(100 * tot / wall).toFixed(1)} %</td>`
      + `<td class="n">${st.runs || 1}</td></tr>`;
    const m = Object.entries(subs[name] || {}).sort((a, b) =>
      b[1].seconds - a[1].seconds);
    for (const [sub, v] of m) {
      h += `<tr><td style="padding-left:20px;color:#939aa6">${esc(sub)}</td>`
        + `<td class="n">${fmtS(v.seconds)}</td>`
        + `<td class="n">${(100 * v.seconds / wall).toFixed(1)} %</td>`
        + `<td class="n">${v.runs}</td></tr>`;
    }
  }
  h += `<tr><td><b>wall clock</b></td><td class="n"><b>${fmtS(wall)}</b></td>`
    + "<td></td><td></td></tr></table>";
  if (prog && prog.doc.timings) {
    h += '<div class="muted" style="margin-top:8px">The bundle also carries '
      + `the run's own recorded totals: allocation `
      + `${fmtS(prog.doc.timings.stages.allocation || 0)}, conduction `
      + `${fmtS(prog.doc.timings.stages.conduction || 0)}.</div>`;
  }
  return h;
}
