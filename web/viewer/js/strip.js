// The stage timeline, the live counters, and the "where the time goes" bar.
//
// THIS IS THE PANEL THE ALGORITHM ITERATION IS DRIVEN FROM.  Each stage gets
// one row; the row's fill is its share of the wall clock so far, and inside
// the fill the SUBSTAGES are drawn as coloured bands in proportion.  That is
// the whole finding of docs/FAST_PLANNING.md rendered continuously instead of
// once, after the fact, from a log: 93 % of an allocation inside `balance` is
// a red band filling most of the allocation row, visible ninety seconds into a
// run rather than an hour later.

import {STAGES, STAGE_LABEL, SUB_COLOR, stageTimes} from "./state.js";
import {F, fmtS} from "./panel.js";

export class Strip {
  constructor(stagesEl, countersEl, loadsEl) {
    this.stagesEl = stagesEl;
    this.countersEl = countersEl;
    this.loadsEl = loadsEl;
    this.rows = {};
    for (const name of STAGES.concat(["export"])) {
      const fill = F("div", {class: "fill"});
      const subs = F("div", {class: "subs"});
      const lbl = F("div", {class: "lbl"});
      const track = F("div", {class: "track"}, [fill, subs, lbl]);
      const el = F("div", {class: "stagebar idle"}, [
        F("div", {class: "nm", text: STAGE_LABEL[name] || name}),
        track,
        F("div", {class: "el", text: "—"}),
        F("div", {class: "ct", text: ""}),
      ]);
      this.rows[name] = {el, fill, subs, lbl,
                         elapsed: el.children[2], count: el.children[3]};
      this.stagesEl.appendChild(el);
    }
  }

  render(s) {
    const times = stageTimes(s);
    const wall = Math.max(s.clock, 0.001);
    const seen = new Set();
    for (const t of times) {
      seen.add(t.name);
      const r = this.rows[t.name];
      if (!r) continue;
      r.el.className = "stagebar " + t.status;
      const share = Math.min(1, t.exclusive / wall);
      // A running stage with a known item count fills by ITEMS, because that
      // is the only honest estimate of how far in it is; a running stage
      // without one fills by its share of the clock so far.
      const p = s.progress[t.name];
      const frac = (t.status === "running" && p && p.total)
        ? Math.min(1, p.done / p.total) : share;
      r.fill.style.width = (100 * frac).toFixed(1) + "%";

      // the substage bands, in proportion inside the row
      r.subs.innerHTML = "";
      const subs = Object.entries(t.subs);
      const tot = subs.reduce((a, [, v]) => a + v.seconds, 0);
      if (tot > 0) {
        for (const [name, v] of subs) {
          const b = F("div", {class: "sub"});
          b.style.width = (100 * v.seconds / Math.max(tot, 1e-9) *
                           Math.min(1, tot / Math.max(t.exclusive, 1e-9)))
                          .toFixed(2) + "%";
          b.style.background = SUB_COLOR[name] || SUB_COLOR.other;
          b.title = `${name}: ${fmtS(v.seconds)}` +
                    (v.runs > 1 ? ` over ${v.runs} runs` : "");
          r.subs.appendChild(b);
        }
      }
      r.lbl.textContent = t.status === "running" && p && p.total
        ? `${p.label || ""} ${p.done}/${p.total}` : "";
      r.elapsed.textContent = fmtS(t.exclusive);
      r.count.textContent = (t.runs > 1 ? `x${t.runs}` : "") +
        (t.exclusive > 0 ? ` ${(100 * t.exclusive / wall).toFixed(0)}%` : "");
    }
    for (const name of Object.keys(this.rows)) {
      if (!seen.has(name)) {
        const r = this.rows[name];
        r.el.className = "stagebar idle";
        r.fill.style.width = "0%";
        r.elapsed.textContent = "—";
        r.count.textContent = "";
      }
    }
    this._counters(s);
    this._loads(s);
  }

  _counters(s) {
    const c = s.counters;
    const bits = [];
    const put = (k, v) => bits.push(`${k} <b>${v}</b>`);
    if (c.strokesTraced) put("traced", c.strokesTraced);
    if (c.strokesTotal) put("on paper", c.strokesTotal);
    if (c.strokesProbed) put("probed", `${c.strokesProbed}/${c.strokesTotal}`);
    if (c.placed) put("spans placed", c.placed);
    if (c.placedM) put("ink", `${c.placedM.toFixed(2)} m`);
    if (c.segments) put("segments", c.segments);
    if (c.dropped != null && c.dropped > 0) put("holes", c.dropped);
    if (c.coverage != null) put("coverage", `${c.coverage.toFixed(2)} %`);
    if (c.phasesDone) put("phases conducted", c.phasesDone);
    if (c.checksOk || c.checksBad) {
      put("scene_check", `${c.checksOk} pass` +
          (c.checksBad ? ` / ${c.checksBad} FAIL` : ""));
    }
    if (s.alloc && s.alloc.timing) {
      const t = s.alloc.timing;
      const worst = Object.entries(t)
        .filter(([k]) => k !== "total")
        .sort((a, b) => b[1] - a[1])[0];
      if (worst) put("slowest allocation phase", `${worst[0]} ${fmtS(worst[1])}`);
    }
    this.countersEl.innerHTML = bits.join(" &nbsp;·&nbsp; ");
  }

  _loads(s) {
    const loads = Object.keys(s.loads || {}).length ? s.loads : null;
    const metres = s.armMetres || {};
    const keys = loads ? Object.keys(loads)
                       : Object.keys(metres).sort((a, b) => a - b);
    if (!keys.length) { this.loadsEl.innerHTML = ""; return; }
    const vals = keys.map(k => loads ? loads[k] : metres[k]);
    const max = Math.max(...vals, 1e-9);
    this.loadsEl.innerHTML = "";
    const unit = loads ? "s" : "m";
    for (let i = 0; i < keys.length; i++) {
      const h = Math.max(2, 26 * vals[i] / max);
      const bar = F("div", {class: "bar"});
      bar.style.height = h + "px";
      bar.style.background = this.armColor ? this.armColor(Number(keys[i]))
                                           : "#4a8fd2";
      const lb = F("div", {class: "lb"}, [bar, F("span", {text: keys[i]})]);
      lb.title = `arm ${keys[i]}: ${vals[i].toFixed(2)} ${unit}`;
      this.loadsEl.appendChild(lb);
    }
    const cap = F("span", {class: "muted",
                           text: loads ? "  per-arm nominal load (s)"
                                       : "  per-arm ink (m)"});
    cap.style.alignSelf = "center";
    this.loadsEl.appendChild(cap);
  }
}
