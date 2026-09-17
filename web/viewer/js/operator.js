// RUN ON ARM — the panel that talks to the operator box.
//
// FIVE BUTTONS IN THE ORDER A PERSON USES THEM: check the stack, copy the CSV
// across (a dry run, which copies nothing and prints the line that would run),
// RUN, and then the two stops.  Nothing here builds an ssh line or a
// supervisor line of its own: every command comes back from the server, which
// gets them from `aris_sixarm/gui/operator.py`, which reads
// `scripts/day1.py`'s own OPERATOR block.  One place knows the addresses.
//
// EVERY COMMAND IS ECHOED VERBATIM.  The pane below the buttons shows `$ <the
// exact argv>` before each result, so what a person reads here is what ran and
// can be pasted into a terminal unchanged.  That matters more than it sounds:
// on a hardware day the GUI and a shell are used alternately by two people.
//
// THE GATE IS THE SERVER'S.  This file disables the RUN button until the stack
// check is green AND `RUN <arm>` has been typed, but that is a courtesy — the
// same two conditions are checked again in `gui/operator.build_run_argv`, so
// the button being enabled is never what permits the motion.
//
// THE ABORT IS THE PHYSICAL E-STOP.  Hold keeps the checkpoint and Kill
// executor is the harder software stop; the panel says so in red, at the top,
// where it cannot be scrolled past.

import {F, esc} from "./panel.js";

export class RunOnArm {
  constructor(ops) {
    this.ops = ops || {};              // {config, check, copy, run, hold, kill,
                                       //  tail, tailRead, onJob}
    this.cfg = null;
    this.checked = {};                 // arm -> true once "STACK HEALTHY"
    this.started = {};                 // arm -> a run has been launched
    this.tailOffset = 0;
    this.tailArm = null;
    this.busy = false;
  }

  build(cfg) {
    const body = F("div", {class: "body"});
    this.body = body;
    // The rig and the tool the viewer is drawing: the pose comparison and the
    // Drake scene must be in the same room as the picture beside them.
    this.rig = ((cfg || {}).defaults || {}).rig || "proposed";
    this.tool = ((cfg || {}).defaults || {}).tool || "lateral";

    this.warn = F("div", {class: "opwarn", text:
      "ABORT = THE PHYSICAL E-STOP.  Hold keeps the checkpoint, Kill executor "
      + "is the harder stop; neither is an abort path."});
    body.appendChild(this.warn);
    this.where = F("div", {class: "muted", text: ""});
    body.appendChild(this.where);

    // ---- SITE SETUP: config/site.json, edited in place ------------------
    // ONE TRACKED FILE FOR THE INSTALLATION.  `scripts/day1.py` reads the same
    // json; this form is the same edit with fewer chances to mistype it.  The
    // slot -> arm choices are what Identify arms below is for.
    this.siteHost = F("input", {type: "text", value: ""});
    this.slotSel = {};                 // slot -> {arm, mounted, paper_z}
    const siteBox = F("div", {class: "opsite"});
    siteBox.appendChild(F("div", {class: "muted",
      text: "config/site.json — the whole installation, in one file"}));
    siteBox.appendChild(F("div", {class: "row one"},
      [F("div", {}, [F("label", {text: "operator  user@host"}),
                     this.siteHost])]));
    this.slotRows = F("div", {});
    siteBox.appendChild(this.slotRows);
    this.saveBtn = F("button", {text: "Save site",
                                onclick: () => this.saveSite()});
    siteBox.appendChild(F("div", {class: "btnrow one"}, [this.saveBtn]));
    body.appendChild(siteBox);

    // ---- IDENTIFY ARMS --------------------------------------------------
    // WHICH ROBOT IS IN WHICH POSITION IS A MEASUREMENT.  Every candidate id is
    // polled for its joints; the ones that answer are posed in the viewer with
    // the tool model, and re-polled every few seconds — move one by hand in
    // guiding mode and the row whose numbers change is the arm in front of you.
    this.idBtn = F("button", {text: "Identify arms",
                              onclick: () => this.toggleIdentify()});
    body.appendChild(F("div", {class: "btnrow one"}, [this.idBtn]));
    this.idBox = F("div", {id: "opident"});
    body.appendChild(this.idBox);

    this.armSel = F("select", {onchange: () => this.onSlot()});
    this.csv = F("input", {type: "text", value: "",
                           placeholder: "out/day1/<name>_<slot>.csv"});
    this.csv.addEventListener("input", () => this.csvTouched = true);
    body.appendChild(F("div", {class: "row"}, [
      F("div", {}, [F("label", {text: "slot  (the plan's position)"}),
                    this.armSel]),
      F("div", {}, [F("label", {text: "pathway CSV"}), this.csv])]));
    // THE MAPPING, WHEREVER THE BUTTONS ARE.  Never only in the site form:
    // "slot 31 -> arm 97" is the fact that decides which robot moves.
    this.mapLine = F("div", {class: "opmap", text: ""});
    body.appendChild(this.mapLine);

    this.checkBtn = F("button", {text: "Check stack",
                                 onclick: () => this.doCheck()});
    this.copyBtn = F("button", {text: "Copy to operator  (dry run)",
                                onclick: () => this.doCopy()});
    body.appendChild(F("div", {class: "btnrow one"}, [this.checkBtn]));
    body.appendChild(F("div", {class: "btnrow one"}, [this.copyBtn]));

    this.confirm = F("input", {type: "text", value: "",
                               placeholder: "RUN 31"});
    this.confirm.addEventListener("input", () => this.refreshGate());
    body.appendChild(F("div", {class: "row one"},
      [F("div", {}, [F("label", {text: "type RUN <arm> to arm the button"}),
                     this.confirm])]));

    this.runBtn = F("button", {class: "primary", text: "RUN  (observe mode)",
                               onclick: () => this.doRun()});
    this.runBtn.disabled = true;
    body.appendChild(F("div", {class: "btnrow one"}, [this.runBtn]));

    this.holdBtn = F("button", {text: "Hold", onclick: () => this.doHold()});
    this.killBtn = F("button", {class: "danger", text: "Kill executor",
                                onclick: () => this.doKill()});
    body.appendChild(F("div", {class: "row"}, [this.holdBtn, this.killBtn]));

    this.out = F("div", {id: "opout"});
    body.appendChild(this.out);

    this.tailBtn = F("button", {text: "Start log tail",
                                onclick: () => this.toggleTail()});
    body.appendChild(F("div", {class: "btnrow one"}, [this.tailBtn]));
    this.tailBox = F("pre", {id: "optail", text: ""});
    body.appendChild(this.tailBox);

    const head = F("h3", {text: "Run on arm"});
    const g = F("div", {class: "group"}, [head, body]);
    head.addEventListener("click", () => g.classList.toggle("collapsed"));
    this.loadConfig();
    return g;
  }

  // ---- what the server knows ------------------------------------------
  async loadConfig() {
    let c;
    try { c = await this.ops.config(); }
    catch (e) { this.say(`could not read the operator config: ${e.message}`,
                         "bad"); return; }
    this.cfg = c;
    const keep = this.armSel.value;
    this.armSel.innerHTML = "";
    for (const a of (c.slots || [31, 71]))
      this.armSel.appendChild(F("option", {value: String(a), text: String(a)}));
    if (keep) this.armSel.value = keep;
    this.siteHost.value = (c.site && c.site.operator
                           && c.site.operator.host) || c.host || "";
    this.buildSlotRows(c);
    // WHICH MACHINE, AND WHO SAID SO.  `config/site.json` is the installation;
    // when that file will not parse the server falls back to its own marked
    // defaults, and that is not a detail to hide.
    const bad = /will not parse|could not write/.test(String(c.source || ""));
    this.where.innerHTML =
      `operator <b>${esc(c.host)}</b> · supervisor <code>`
      + `${esc(c.supervisor)}</code> · h ${esc(c.h)} · `
      + (bad ? `<b class="bad">falling back to the GUI's own defaults</b> — `
               + `${esc(c.source)}`
             : `site <code>${esc(c.source)}</code>`);
    if (c.csv && !this.csv.value) this.csv.value = c.csv;
    for (const [arm, v] of Object.entries(c.checks || {}))
      if (v.ok && v.fresh) this.checked[Number(arm)] = true;
    this.refreshGate();
  }

  // The site form's per-slot row: which physical arm is in this position, is
  // that confirmed, and the paper plane measured for it.
  buildSlotRows(c) {
    this.slotRows.innerHTML = "";
    this.slotSel = {};
    for (const s of (c.slots || [31, 71])) {
      const m = (c.map || {})[String(s)] || {};
      const sel = F("select");
      for (const id of (c.ids || [31, 71, 97])) {
        const o = F("option", {value: String(id), text: String(id)});
        if (Number(id) === Number(m.arm)) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener("change", () => this.onSlot());
      const mounted = F("input", {type: "checkbox"});
      mounted.checked = !!m.mounted;
      const z = F("input", {type: "text",
                            value: m.paper_z == null ? "" : String(m.paper_z)});
      this.slotSel[s] = {arm: sel, mounted, paper_z: z};
      this.slotRows.appendChild(F("div", {class: "row"}, [
        F("div", {}, [F("label", {text: `${m.position || ("slot " + s)}`
                                        + ` (slot ${s}) = arm`}), sel]),
        F("div", {}, [F("label", {text: "paper z, m"}), z])]));
      const wrap = F("label", {class: "check"}, [mounted]);
      wrap.appendChild(document.createTextNode(
        ` slot ${s}: the mounting is CONFIRMED (somebody looked)`));
      this.slotRows.appendChild(wrap);
    }
  }

  async saveSite() {
    const slots = {};
    for (const [s, r] of Object.entries(this.slotSel)) {
      const z = Number(String(r.paper_z.value).trim());
      slots[s] = {arm: Number(r.arm.value), domain: Number(r.arm.value),
                  mounted: r.mounted.checked ? true : null,
                  paper_z: Number.isFinite(z) ? z : 0.0};
    }
    await this.withBusy(this.saveBtn, async () => {
      const r = await this.ops.site({
        operator: {host: String(this.siteHost.value).trim()}, slots});
      this.say(`saved ${r.source}`, "ok");
      for (const line of Object.values(r.map || {}))
        this.say(line.line || line, "");
      await this.loadConfig();
    });
  }

  // The SLOT the buttons act on, and the physical ARM that slot maps to.
  arm() { return Number(this.armSel.value || (this.cfg && this.cfg.slots[0])); }

  slot() { return this.arm(); }

  armId() {
    const m = ((this.cfg || {}).map || {})[String(this.slot())] || {};
    return Number(m.arm || this.slot());
  }

  // THE CSV THE LAST PASSING DAY-1 JOB WROTE.  The panel above this one hands
  // its verdict over as soon as one lands, so a person who has just certified
  // a line does not retype its path.  Only a PASS, and only if the box has not
  // been edited by hand.
  suggestCsv(result) {
    if (!result || !result.ok || !(result.csv || []).length) return;
    if (this.csvTouched) return;
    const want = result.csv.find(p => p.includes(`_${this.arm()}.`))
                 || result.csv[0];
    if (want) this.csv.value = want;
  }

  // Hold and Kill are for a run that is happening, and a page reload must not
  // hide them: any operator job in the list counts as "a run has started".
  noteJobs(jobs) {
    for (const j of jobs || [])
      if (j.params && j.params.operator === "run")
        this.started[Number(j.params.arm)] = true;
    this.refreshGate();
  }

  onSlot() {
    // The arm selector in the site form and the slot selector here are two
    // views of the same mapping; changing either re-reads it from the server so
    // the line under the buttons is never a guess.
    this.loadConfig();
  }

  refreshGate() {
    const s = this.slot(), a = this.armId();
    if (this.mapLine) {
      const m = ((this.cfg || {}).map || {})[String(s)] || {};
      this.mapLine.textContent = m.line || `slot ${s} -> arm ${a}`;
      this.mapLine.className = "opmap" + (m.mounted ? "" : " warn");
    }
    // GREEN CHECK AND TYPED WORDS.  The check is the PHYSICAL arm's (that is
    // the robot about to move); the typed token is the SLOT's (that is what the
    // person chose and what the plan is for).
    const armed = !!this.checked[a] &&
      String(this.confirm.value || "").trim() === `RUN ${s}`;
    if (this.runBtn) this.runBtn.disabled = !armed || this.busy;
    if (this.confirm) this.confirm.placeholder = `RUN ${s}`;
    const started = !!this.started[s];
    if (this.holdBtn) this.holdBtn.disabled = !started;
    if (this.killBtn) this.killBtn.disabled = !started;
    if (this.checkBtn)
      this.checkBtn.className = this.checked[a] ? "ok" : "";
  }

  // ---- the pane --------------------------------------------------------
  say(text, cls) {
    const d = F("div", {class: "opline " + (cls || ""), text: String(text)});
    this.out.appendChild(d);
    while (this.out.childElementCount > 200)
      this.out.removeChild(this.out.firstChild);
    this.out.scrollTop = this.out.scrollHeight;
    return d;
  }

  // EVERY RESULT IS THE COMMAND PLUS WHAT IT SAID.  Nothing is summarised
  // away: an ssh that could not find a route prints its own sentence and that
  // sentence is what the panel shows.
  show(r, okText) {
    this.say(`$ ${r.cmd}`, "cmd");
    if (r.output) this.say(r.output, r.ok ? "" : "bad");
    this.say(r.ok ? (okText || `exit 0  (${r.elapsed_s} s)`)
                  : `exit ${r.returncode}  (${r.elapsed_s} s)`,
             r.ok ? "ok" : "bad");
  }

  async withBusy(btn, fn) {
    if (this.busy) return null;
    this.busy = true;
    const was = btn.textContent;
    btn.textContent = "running...";
    btn.disabled = true;
    try { return await fn(); }
    catch (e) { this.say(`${e.message}`, "bad"); return null; }
    finally {
      this.busy = false;
      btn.textContent = was;
      btn.disabled = false;
      this.refreshGate();
    }
  }

  // ---- the buttons -----------------------------------------------------
  async doCheck() {
    const a = this.armId();
    await this.withBusy(this.checkBtn, async () => {
      const r = await this.ops.check(a);
      // GREEN ONLY ON "STACK HEALTHY".  A zero exit from a script that said
      // something else is not a healthy stack, and the raw output is shown
      // either way because `aris_hold.sh` was written for arms 13/17 and what
      // it prints for 31/71 is not something this panel may assume.
      this.checked[a] = !!r.healthy;
      this.show(r, `${r.healthy ? "STACK HEALTHY" : "no STACK HEALTHY in the "
                    + "output — the run button stays disabled"}`);
      if (!r.healthy) this.say("the raw output above is the whole answer: "
        + "aris_hold.sh was written for arms 13/17 and its wording for 31/71 "
        + "is not something this panel assumes", "muted");
      this.refreshGate();
    });
  }

  async doCopy() {
    const s = this.slot();
    await this.withBusy(this.copyBtn, async () => {
      const r = await this.ops.copy(s, this.csv.value.trim());
      if (r.mapping) this.say(r.mapping, "cmd");
      this.show(r, "dry run: nothing was copied, nothing ran");
    });
  }

  async doRun() {
    const s = this.slot();
    await this.withBusy(this.runBtn, async () => {
      const r = await this.ops.run(s, this.csv.value.trim(),
                                   this.confirm.value.trim());
      this.started[s] = true;
      if (r.mapping) this.say(r.mapping, "cmd");
      this.say(`$ ${r.cmd}`, "cmd");
      this.say(`job ${r.id} started — the run's own output is in the log pane `
               + `on the right`, "ok");
      this.confirm.value = "";            // one typed confirmation, one run
      if (this.ops.onJob) this.ops.onJob(r.id);
      this.refreshGate();
    });
  }

  async doHold() {
    const a = this.armId();
    await this.withBusy(this.holdBtn, async () => {
      this.show(await this.ops.hold(a), "held — the checkpoint is kept");
    });
  }

  async doKill() {
    const a = this.armId();
    await this.withBusy(this.killBtn, async () => {
      this.show(await this.ops.kill(a), "the executor was signalled");
    });
  }

  // ---- identify arms ---------------------------------------------------
  async toggleIdentify() {
    if (this.idTimer) {
      clearInterval(this.idTimer);
      this.idTimer = null;
      this.idBtn.textContent = "Identify arms";
      return;
    }
    this.idBtn.textContent = "Stop identifying";
    await this.pollIdentify();
    // EVERY FEW SECONDS, NOT ONCE.  The method is to move an arm by hand and
    // watch which id's numbers change; one snapshot cannot show that.
    this.idTimer = setInterval(() => this.pollIdentify(), 4000);
  }

  async pollIdentify() {
    let r;
    try { r = await this.ops.identify({rig: this.rig, tool: this.tool}); }
    catch (e) { this.say(`identify failed: ${e.message}`, "bad"); return; }
    this.idBox.innerHTML = "";
    for (const row of (r.arms || [])) {
      const head = F("div", {class: "idrow" + (row.reachable ? " ok" : " bad")});
      head.appendChild(F("span", {text: `arm ${row.arm}`}));
      head.appendChild(F("span", {class: "muted", text: row.ip || ""}));
      head.appendChild(F("span", {
        text: row.reachable ? "answered" : "no answer"}));
      if (row.slots && row.slots.length)
        head.appendChild(F("span", {class: "muted",
                                    text: `slot ${row.slots.join("+")}`}));
      this.idBox.appendChild(head);
      if (row.reachable) {
        this.idBox.appendChild(F("div", {class: "idq",
          text: row.q.map(x => x.toFixed(3)).join("  ")}));
        if (row.delta)
          this.idBox.appendChild(F("div", {class: "idq muted",
            text: "Δpark " + row.delta.map(x => x.toFixed(3)).join("  ")}));
        // ...and in the viewer, with the tool model, which is the whole point:
        // hold the picture next to the machine and read the holder's side off
        // both at once.
        if (this.ops.onPose) this.ops.onPose(row.arm, row.q);
        const b = F("button", {text: "Drake Meshcat", class: "wee"});
        b.addEventListener("click", () => this.poseInDrake(row.arm));
        this.idBox.appendChild(b);
      } else if (row.output) {
        this.idBox.appendChild(F("div", {class: "idq bad",
                                         text: row.output.slice(-400)}));
      }
    }
  }

  async poseInDrake(arm) {
    try {
      const r = await this.ops.pose({arm, meshcat: true, rig: this.rig,
                                     tool: this.tool});
      this.say(`$ ${r.cmd}`, "cmd");
      if (r.meshcat) this.say(`Drake scene on port ${r.meshcat.port}`, "ok");
      else this.say(r.meshcat_error || r.error || "no pose", "bad");
    } catch (e) { this.say(e.message, "bad"); }
  }

  // ---- the live log ----------------------------------------------------
  async toggleTail() {
    const a = this.armId();          // the log is the ROBOT's, not the slot's
    if (this.tailArm !== null) {
      const was = this.tailArm;
      this.tailArm = null;
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      try { await this.ops.tail(was, "stop"); } catch (e) {}
      this.tailBtn.textContent = "Start log tail";
      this.tailBox.textContent += `\n[tail stopped]\n`;
      return;
    }
    this.tailOffset = 0;
    this.tailBox.textContent = "";
    let r;
    try { r = await this.ops.tail(a, "start"); }
    catch (e) { this.say(`could not start the tail: ${e.message}`, "bad");
                return; }
    this.say(`$ ${r.cmd}`, "cmd");
    if (r.error) { this.say(r.error, "bad"); return; }
    this.tailArm = a;
    this.tailBtn.textContent = "Stop log tail";
    this.timer = setInterval(() => this.pollTail(), 1000);
    this.pollTail();
  }

  async pollTail() {
    if (this.tailArm === null) return;
    let r;
    try { r = await this.ops.tailRead(this.tailArm, this.tailOffset); }
    catch (e) { return; }
    this.tailOffset = r.offset;
    if (r.text) {
      const atEnd = this.tailBox.scrollTop + this.tailBox.clientHeight
                    >= this.tailBox.scrollHeight - 30;
      this.tailBox.textContent += r.text;
      if (atEnd) this.tailBox.scrollTop = this.tailBox.scrollHeight;
    }
    // A DEAD TAIL SAYS SO.  With no route to the operator box `ssh` exits
    // immediately; the pane must show that sentence and stop, not spin.
    if (!r.live) {
      clearInterval(this.timer);
      this.timer = null;
      this.tailArm = null;
      this.tailBtn.textContent = "Start log tail";
      this.tailBox.textContent +=
        `\n[the tail exited (${r.returncode}) — nothing is streaming]\n`;
      this.tailBox.scrollTop = this.tailBox.scrollHeight;
    }
  }
}
