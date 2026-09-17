"""RUN ON ARM — the GUI's half of the operator box, and nothing more.

THIS MODULE MOVES NO ARM BY ITSELF.  Every one of the five buttons the panel
offers is one of two things: an `ssh` to the operator box that ASKS it
something (is the stack healthy, hold, kill, tail the log), or a subprocess
call to `scripts/day1.py send`, which is the file that owns the delivery and
the run line.  Nothing here re-implements the chain
`day1.py send` prints — if the supervisor's arguments move, they move in one
file, and this panel shows whatever that file printed.

WHY `scripts/day1.py` IS A SUBPROCESS AND NOT AN IMPORT.  Importing it would
import the planner, and `aris_sixarm/__init__.py` binds the rig and the tool
the FIRST time the package is imported — a server that did that would be stuck
at one rig for its whole life (see the docstring of `gui/server.py`).  It is
also being edited on hardware days by whoever is at the rig, and a syntax error
in it must cost this panel one red box, not the GUI.

WHERE THE ADDRESSES COME FROM.  `scripts/day1.py`'s `OPERATOR` block is the one
place the repo knows about another machine, so this module READS it — by
parsing the file with `ast`, never by importing it, for the reason above.  What
that block does not carry (the executor's log file, the helper that defines
`arm_pkill`) is the small marked dict below, and the panel says which of the two
it is using.

THE GATE.  `build_run_argv` refuses to produce a `--live` argument list unless
the caller has typed `RUN <arm>` EXACTLY and a stack check for that arm has come
back "STACK HEALTHY" in this server's lifetime.  The browser disables the button
as well, but the button is a courtesy and this is the gate: a `curl` at the
endpoint meets the same refusal.  The physical e-stop remains the abort — no
software path here is one.
"""
from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY1_PY = ROOT / "scripts" / "day1.py"

# ONE TRACKED FILE IS THE SITE.  `config/site.json` is where the operator's
# address, which physical arm is bolted into which planner slot, the control-box
# IPs, the paper height and the executor's environment live — the GUI READS it
# and its "Site setup" form WRITES it, and `scripts/day1.py` is expected to read
# the same file.  Reconfiguring the installation is editing one json, on the
# machine, in the browser, and nothing else.
SITE_PATH = Path(os.environ.get("ARIS_SITE") or ROOT / "config" / "site.json")

# SLOTS ARE POSITIONS, IDS ARE ROBOTS, AND THEY NEED NOT MATCH.  The planner
# knows two middle POSITIONS, 31 (left) and 71 (right); what is bolted into them
# on 2026-09-17 is not known — "may be 97 and 71" — so the panel identifies the
# arms and the mapping is site configuration.  `ARM_ID` in every ssh line below
# is always a physical id (the DDS domain), never a slot.
SLOTS = (31, 71)
ARMS = SLOTS                       # kept: the older name for the slot list
CANDIDATE_IDS = (31, 71, 97)

# The last-resort defaults, used only to WRITE `config/site.json` the first time
# and when that file will not parse.  Seeded from `scripts/day1.py`'s OPERATOR
# block when that parses, so the two never disagree on day one.
LOCAL = dict(
    log="/tmp/rtff_draw_arm{arm}.log",          # the supervisor's own log
    arm_env="~/impedance_helpers/arm_env.sh",   # defines arm_pkill
    executor="rtff_pathway_exec",               # what `arm_pkill` is aimed at
    tail_lines=50,
)

FALLBACK = dict(host="diemut@192.168.50.2", store="~/RTff/pathway_persist",
                supervisor="~/RTff/draw_rtff_supervised.sh",
                stack_check="~/RTff/aris_hold.sh",
                remote="/tmp/impedance_pathway_arm{arm}.csv",
                log=LOCAL["log"], mode="fresh", force=("1.0", "2.5", "5"),
                arm_ips={31: "192.168.50.12", 71: "192.168.50.14",
                         97: "192.168.50.15"},
                slots={31: "left-middle", 71: "right-middle"})

# SSH THAT FAILS INSTEAD OF HANGING.  The lab's key-only ssh needs no options
# at all when the route is there; a workstation with NO route to 192.168.50.2
# would otherwise sit in the TCP handshake for two minutes with the button
# stuck at "running".  BatchMode refuses to ask for a password (there is none —
# the briefing says password auth is disabled) and ConnectTimeout turns "no
# route" into a sentence.  Both are ECHOED in the panel, so what is shown is
# what ran.
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")

HEALTHY = "STACK HEALTHY"        # the string `aris_hold.sh stack N` prints
CHECK_TTL_S = 900.0              # a green check goes stale after 15 minutes
CMD_TIMEOUT_S = 90.0             # every one-shot ssh is bounded


class GateRefused(ValueError):
    """The typed confirmation or the stack check is missing.  Not a crash."""


# ---------------------------------------------------------------------------
# configuration: day1.py's OPERATOR block, read without importing it
# ---------------------------------------------------------------------------
def read_day1_operator(path=DAY1_PY):
    """`OPERATOR = dict(...)` out of `scripts/day1.py`. -> (dict, source).

    PARSED, NOT IMPORTED.  See the module docstring.  Only literal keyword
    values are taken; `host` is `os.environ.get("ARIS_OPERATOR", "user@host")`
    in that file, which is not a literal, so its DEFAULT is lifted out of the
    call and the environment variable is honoured here exactly as it is there.
    """
    src, out = None, {}
    try:
        src = Path(path).read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as exc:
        return dict(FALLBACK), f"{type(exc).__name__}: {exc}"
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "OPERATOR"
                   for t in node.targets):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                out[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                # `os.environ.get("ARIS_OPERATOR", "diemut@192.168.50.2")`:
                # take the default it falls back to, and read the variable the
                # same way the script would.
                v = kw.value
                if (isinstance(v, ast.Call) and v.args
                        and len(v.args) == 2):
                    try:
                        name = ast.literal_eval(v.args[0])
                        dflt = ast.literal_eval(v.args[1])
                    except ValueError:
                        continue
                    out[kw.arg] = os.environ.get(str(name), dflt)
        break
    if not out.get("host") or not out.get("stack_check"):
        merged = dict(FALLBACK)
        merged.update(out)
        return merged, f"{Path(path).name} has no usable OPERATOR block"
    return out, str(Path(path).relative_to(ROOT))


# ---------------------------------------------------------------------------
# config/site.json — the one file that describes this installation
# ---------------------------------------------------------------------------
SITE_NOTE = (
    "The site.  ONE FILE: edit it here, in the GUI's Site setup form, or with "
    "an editor — scripts/day1.py and the GUI both read it.  `slots` maps a "
    "PLANNER SLOT (31 = left-middle, 71 = right-middle; the base frame the "
    "poses in a CSV are expressed in) to the PHYSICAL arm bolted into that "
    "position (its id IS its DDS domain and the ARM_ID the supervisor reads). "
    "Which is which is a measurement, not a fact of this repository: use the "
    "GUI's Identify arms view, or move one by hand and watch which id changes.")


def default_site():
    """Today's defaults, in `scripts/day1.py`'s own schema.

    THE SCHEMA IS THE OTHER FILE'S, and deliberately: `day1.py site` reads and
    edits this file too, so the GUI writes what that reader expects — `slots.
    <N>.arm` is the physical arm in the position, `arms.<id>.ip` is its control
    box, and the four strings the GUI needs and `day1.py` does not (`arm_env`,
    `executor`) are added as extra keys rather than as a second file.  Seeded
    from `day1.py`'s OPERATOR block when that still parses, so a checkout where
    the json is missing still agrees with the script.
    """
    op, _ = read_day1_operator()
    ips = {str(k): str(v) for k, v in
           (op.get("arm_ips") or FALLBACK["arm_ips"]).items()}
    names = {str(k): str(v) for k, v in
             (op.get("slots") or FALLBACK["slots"]).items()}
    return dict(
        _what=SITE_NOTE, _updated=time.strftime("%Y-%m-%d"),
        operator=dict(host=op.get("host", FALLBACK["host"])),
        h=0.970,
        slots={str(s): dict(position=names.get(str(s), f"slot {s}"),
                            # IDENTITY UNTIL SOMEBODY IDENTIFIES THE ARMS.  Not
                            # a claim: `mounted` null says nobody has confirmed
                            # it, and the run line prints the mapping every time.
                            arm=s, ip=ips.get(str(s), ""), domain=s,
                            paper_z=0.0, mounted=None)
               for s in SLOTS},
        arms={str(i): dict(ip=ips.get(str(i), ""), inverted=True)
              for i in CANDIDATE_IDS},
        remote_csv=op.get("remote", FALLBACK["remote"]),
        log=op.get("log", LOCAL["log"]),
        supervisor=op.get("supervisor", FALLBACK["supervisor"]),
        stack_check=op.get("stack_check", FALLBACK["stack_check"]),
        # THE GUI'S TWO EXTRA STRINGS.  `day1.py send` needs neither — they are
        # what the Identify / Kill buttons run — so they are named here and the
        # panel says they are the GUI's.
        arm_env=LOCAL["arm_env"], executor=LOCAL["executor"],
        force=list(op.get("force", FALLBACK["force"])),
        mode=str(op.get("mode", FALLBACK["mode"])),
        rtff_env=dict(op.get("env") or (
            ("RTFF_CONTACT_DESCEND", "0"), ("RTFF_FORCE_SIGN", "1"),
            ("RTFF_TRAVEL_SPEED", "0.02"), ("RTFF_MODE", "observe"),
            ("RTFF_DEPART_LIFT", "0"))))


def site(path=None):
    """`config/site.json`, created from the defaults if it is not there yet.

    -> (dict, source).  A file that will not parse is NOT silently replaced:
    the defaults are returned with the parse error as the source, so the panel
    says what happened instead of quietly running against different addresses.
    A file that IS there wins key by key, and only the keys it does not carry
    (the two the GUI added) come from the defaults.
    """
    p = Path(path or SITE_PATH)
    if not p.exists():
        d = default_site()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, indent=1) + "\n")
            return d, f"{_rel(p)} (created from today's defaults)"
        except OSError as exc:
            return d, f"could not write {p}: {exc}"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return default_site(), f"{_rel(p)} will not parse: {exc}"
    merged = default_site()
    for k, v in d.items():
        if k in ("slots", "arms", "operator") and isinstance(v, dict):
            for sub, spec in v.items():
                if isinstance(spec, dict):
                    base = dict(merged[k].get(str(sub), {}))
                    base.update(spec)
                    merged[k][str(sub)] = base
                else:
                    merged[k][str(sub)] = spec
        else:
            merged[k] = v
    return merged, _rel(p)


def save_site(patch, path=None):
    """Merge `patch` into `config/site.json` and write it. -> (dict, source).

    WRITTEN WHOLE AND ATOMICALLY, like every other record this repo keeps: a
    half-written site file read by `day1.py` mid-save would be a run against
    half an address.
    """
    p = Path(path or SITE_PATH)
    cur, _ = site(p)
    for k, v in (patch or {}).items():
        if k == "slots" and isinstance(v, dict):
            for s, spec in v.items():
                base = dict(cur["slots"].get(str(s), {}))
                base.update(spec or {})
                cur["slots"][str(s)] = base
        elif k == "operator" and isinstance(v, dict):
            cur["operator"].update(v)
        else:
            cur[k] = v
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=1) + "\n")
    tmp.replace(p)
    return site(p)


def _rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def arm_of(slot, st=None):
    """The PHYSICAL arm id bolted into a planner slot. -> int."""
    s = (st or site()[0])
    spec = s.get("slots", {}).get(str(_slot(slot)), {})
    try:
        return int(spec.get("arm") or _slot(slot))
    except (TypeError, ValueError):
        return _slot(slot)


def candidate_ids(st=None):
    """Every physical arm id this installation might have. -> [int]."""
    s = (st or site()[0])
    out = set()
    for k in (s.get("arms") or {}):
        try:
            out.add(int(k))
        except (TypeError, ValueError):
            continue
    return sorted(out or set(CANDIDATE_IDS))


def mapping_line(slot, st=None):
    """`slot 31 -> arm 97`, the sentence every run line must carry. -> str."""
    s = (st or site()[0])
    a = arm_of(slot, s)
    spec = s.get("slots", {}).get(str(_slot(slot)), {})
    where = spec.get("position") or f"slot {slot}"
    tail = "" if spec.get("mounted") else "  (MOUNTING NOT CONFIRMED)"
    return f"slot {_slot(slot)} -> arm {a}  ({where}){tail}"


def config():
    """Everything the panel needs to name a machine. -> dict."""
    st, source = site()
    return dict(
        site=st, source=source,
        slots=list(SLOTS), arms=list(SLOTS), ids=candidate_ids(st),
        host=st["operator"]["host"], supervisor=st["supervisor"],
        stack_check=st["stack_check"], log=st["log"],
        arm_env=st["arm_env"], executor=st["executor"],
        remote_csv=st.get("remote_csv"), h=st.get("h"),
        map={str(s): dict(st["slots"][str(s)], arm=arm_of(s, st),
                          line=mapping_line(s, st)) for s in SLOTS},
        as_arm_supported=as_arm_supported(),
        # SAID ON THE PANEL, NOT ONLY IN THE RUNBOOK.
        abort="The PHYSICAL E-STOP is the abort.  Hold keeps the checkpoint, "
              "Kill executor is the harder stop; neither is an abort path.",
        confirm_hint="type  RUN <slot>  to arm the run button",
        healthy=HEALTHY, check_ttl_s=CHECK_TTL_S)


def as_arm_supported(path=DAY1_PY):
    """Does `scripts/day1.py send` take `--as-arm` yet? -> bool.

    ASKED OF THE FILE, NOT ASSUMED.  The slot -> arm mapping is dispatched by
    that flag; a GUI that passed it to a `day1.py` which does not have it would
    turn a mapping into an argparse error at the worst moment.  Read rather
    than imported, for the reason in the module docstring.
    """
    try:
        return "--as-arm" in Path(path).read_text()
    except OSError:
        return False


def last_pass_csv(mgr, arm=None, scan=25):
    """The CSV of the newest day-1 job that PASSED. -> path str or "".

    THE FORM IS PRE-FILLED FROM THE RECORD, NOT FROM A GUESS.  `out/day1/` is
    full of files from refused runs, older words and yesterday's probes, and
    picking the newest one by mtime would sooner or later offer the panel a
    file nothing ever certified.  The job log is the record: a day-1 job that
    finished `done` emitted one `item` with `what == "day1"`, `ok` true, and
    the paths it wrote.  Only those are offered.
    """
    try:
        jobs = mgr.list(scan)
    except Exception:
        return ""
    for j in jobs:
        if not j.get("params", {}).get("day1") or j.get("status") != "done":
            continue
        try:
            events, _ = mgr.read_events(j["id"], 0)
        except Exception:
            continue
        for ev in reversed(events):
            p = ev.get("payload") or {}
            if ev.get("kind") != "item" or p.get("what") != "day1":
                continue
            if not p.get("ok") or not p.get("csv"):
                continue
            for path in p["csv"]:
                if arm is None or str(arm) in Path(path).stem.split("_")[-1]:
                    if Path(path).exists():
                        return str(path)
    return ""


# ---------------------------------------------------------------------------
# the argument lists.  A SLOT is a position in the plan; an ID is a robot.
# ---------------------------------------------------------------------------
def _slot(slot):
    s = int(slot)
    if s not in SLOTS:
        raise GateRefused(f"slot must be one of {list(SLOTS)}, got {slot!r}")
    return s


def _id(arm, st=None):
    """A PHYSICAL arm id — a DDS domain this installation might have."""
    a = int(arm)
    ids = set(candidate_ids(st) if st else CANDIDATE_IDS) | set(SLOTS)
    if a not in ids:
        raise GateRefused(f"arm id must be one of {sorted(ids)}, got {arm!r}")
    return a


# The older name, kept because the endpoints and the tests say `arm`: every
# ssh line below is addressed to a PHYSICAL id, so this is `_id`.
_arm = _id


def _ssh(host, remote):
    return ["ssh", *SSH_OPTS, str(host), str(remote)]


def check_argv(arm, cfg=None):
    """`ssh <host> 'bash ~/RTff/aris_hold.sh stack N'` -> argv."""
    c = cfg or config()
    return _ssh(c["host"], f"bash {c['stack_check']} stack {_arm(arm)}")


def hold_argv(arm, cfg=None):
    """The soft stop: it KEEPS the checkpoint, so the pass can resume."""
    c = cfg or config()
    return _ssh(c["host"], f"bash {c['stack_check']} hold {_arm(arm)}")


def kill_argv(arm, cfg=None):
    """The harder stop: kill the executor itself.  Still not the abort."""
    c = cfg or config()
    a = _arm(arm)
    return _ssh(c["host"], f"ARM_ID={a} source {c['arm_env']} && "
                           f"arm_pkill {c['executor']}")


def tail_argv(arm, cfg=None):
    """`ssh <host> 'tail -n 50 -f /tmp/rtff_draw_armN.log'` -> argv."""
    c = cfg or config()
    a = _arm(arm)
    return _ssh(c["host"], f"tail -n {LOCAL['tail_lines']} -f "
                           f"{c['log'].format(arm=a)}")


def send_argv(slot, csv, live=False, st=None):
    """`python scripts/day1.py send --arm SLOT --file CSV [--as-arm ID] ...`.

    A SUBPROCESS CALL TO THE FRONT DOOR, which is the whole contract with the
    other half of this system: the GUI never builds the scp line or the
    supervisor line itself, it runs the file that owns them and shows what it
    printed.  `--dry-run` copies nothing; `--live` is the only argument list on
    this page that can move an arm.

    `--arm` IS THE SLOT AND `--as-arm` IS THE ROBOT.  The CSV's poses are in the
    slot POSITION's base frame; the arm bolted into that position is whatever
    `config/site.json` says, and when it is not the same number the flag carries
    the difference.  If `day1.py` has no `--as-arm` yet, a non-identity mapping
    is REFUSED rather than dispatched to the wrong DDS domain.
    """
    s = _slot(slot)
    state = st if st is not None else site()[0]
    arm = arm_of(s, state)
    path = str(csv or "").strip()
    if not path:
        raise GateRefused("no CSV given — the panel needs the pathway file "
                          "the day-1 job wrote")
    argv = [sys.executable, str(DAY1_PY), "send", "--arm", str(s),
            "--file", path]
    if arm != s:
        if not as_arm_supported():
            raise GateRefused(
                f"{mapping_line(s, state)}, but `scripts/day1.py send` has no "
                f"--as-arm flag in this checkout.  Either the mapping is wrong "
                f"in config/site.json or that file has not landed yet; the GUI "
                f"will not dispatch a slot-31 plan to another DDS domain "
                f"silently.")
        argv += ["--as-arm", str(arm)]
    argv.append("--live" if live else "--dry-run")
    return argv


# ---------------------------------------------------------------------------
# SHOW CURRENT POSE — the arm's own joint state, read back over ssh
# ---------------------------------------------------------------------------
# WHY THIS BUTTON EXISTS.  The pen holder is bolted to the hand and which WAY
# it is bolted decides which side of the flange the tip comes out on — and the
# answer is not in any file here: `assets/system_model/installation_*.urdf` is
# a model of a holder, not a record of how this one was mounted this morning.
# So the panel reads the LIVE joints off the operator box, poses the same arm
# in the viewer with the tool model drawn, and a person holds the picture next
# to the machine.  It is a measurement, and the panel never claims otherwise:
# the raw `ros2 topic echo` output is shown whether or not it parsed.
#
# TWO TOPICS, IN ORDER.  `/joint_states` is what a standard driver publishes;
# the Franka broadcaster publishes `q` inside
# `/franka_robot_state_broadcaster/robot_state`.  The DDS domain is not a flag:
# `arm_env.sh` sets it from ARM_ID, which is why every one of these runs
# through that helper.
JOINT_TOPICS = ("/joint_states", "/franka_robot_state_broadcaster/robot_state")
JOINT_KEYS = ("position", "q")      # the list to read, per topic, in order


def pose_argv(arm, topic=JOINT_TOPICS[0], cfg=None):
    """One shot of the live joint state. -> argv."""
    c = cfg or config()
    a = _arm(arm)
    return _ssh(c["host"], f"ARM_ID={a} source {c['arm_env']} && "
                           f"timeout 5 ros2 topic echo --once {topic}")


def _floats(lines, i):
    """A YAML block or flow sequence starting at `lines[i]`. -> (values, next)."""
    out, n = [], len(lines)
    head = lines[i]
    if "[" in head:                                   # position: [0.1, 0.2, ...]
        body = head[head.index("[") + 1:]
        body = body[:body.index("]")] if "]" in body else body
        for tok in body.replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out, i + 1
    j = i + 1
    while j < n:
        s = lines[j].strip()
        if not s.startswith("- "):
            break
        try:
            out.append(float(s[2:].strip()))
        except ValueError:
            break
        j += 1
    return out, j


def parse_joint_state(text):
    """`ros2 topic echo` output -> dict(q, names, key) or None.

    HAND-ROLLED, AND ON PURPOSE.  The GUI machine has no ROS and need not gain
    a YAML dependency to read seven numbers; what comes back is a flat block of
    `key:` lines with `- value` lists under them, which is four lines of
    scanner.  Anything it cannot read is not guessed at — the caller shows the
    raw output instead.
    """
    lines = str(text or "").splitlines()
    names, vals, key = [], [], None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        head = s.split(":", 1)[0].strip() if ":" in s else ""
        if head == "name" and not names:
            got, i = _strings(lines, i)
            names = got
            continue
        if head in JOINT_KEYS and not vals:
            got, i = _floats(lines, i)
            if got:
                vals, key = got, head
                continue
        i += 1
    if not vals:
        return None
    # SEVEN ARM JOINTS, BY NAME WHEN THE NAMES ARE THERE.  `/joint_states` on a
    # hand-equipped FR3 also carries the two finger joints, and taking the
    # first seven entries of an unsorted list is how an arm gets drawn with a
    # gripper angle in joint 7.
    pick = []
    for idx, nm in enumerate(names):
        if idx >= len(vals) or "finger" in nm or "joint" not in nm:
            continue
        digits = "".join(ch for ch in nm.rsplit("joint", 1)[-1] if ch.isdigit())
        if digits and 1 <= int(digits) <= 7:
            pick.append((int(digits), vals[idx], nm))
    if len(pick) == 7:
        pick.sort()
        return dict(q=[v for _, v, _ in pick], names=[n for _, _, n in pick],
                    key=key)
    if len(vals) >= 7:
        return dict(q=[float(v) for v in vals[:7]],
                    names=names[:7] if len(names) >= 7 else [], key=key)
    return None


def _strings(lines, i):
    """The `name:` block. -> ([str], next index)."""
    out, n = [], len(lines)
    head = lines[i]
    if "[" in head:
        body = head[head.index("[") + 1:]
        body = body[:body.index("]")] if "]" in body else body
        return [t.strip().strip("'\"") for t in body.split(",") if t.strip()], \
            i + 1
    j = i + 1
    while j < n:
        s = lines[j].strip()
        if not s.startswith("- "):
            break
        out.append(s[2:].strip().strip("'\""))
        j += 1
    return out, j


def read_pose(arm, cfg=None, run=None):
    """The arm's live joints, or the reason there are none. -> dict.

    `run` is the command runner, so a test can hand it a fake and no ssh is
    ever attempted; the default is the real one.
    """
    runner = run or run_capture
    c = cfg or config()
    a = _arm(arm)
    tried = []
    for topic in JOINT_TOPICS:
        r = runner(pose_argv(a, topic, c))
        parsed = parse_joint_state(r["output"])
        tried.append(dict(topic=topic, cmd=r["cmd"], returncode=r["returncode"],
                          output=r["output"], parsed=bool(parsed)))
        if parsed:
            return dict(arm=a, ok=True, q=parsed["q"], names=parsed["names"],
                        key=parsed["key"], topic=topic, cmd=r["cmd"],
                        output=r["output"], tried=tried)
    last = tried[-1] if tried else {}
    return dict(arm=a, ok=False, q=None, names=[], key=None,
                topic=last.get("topic"), cmd=last.get("cmd"),
                output=last.get("output", ""), tried=tried,
                error="no joint vector in the output of either topic — the raw "
                      "text is above, unedited")


def identify(ids=None, cfg=None, run=None, st=None):
    """Poll every candidate arm id for its joints. -> [dict], newest first.

    WHICH ROBOT IS IN WHICH POSITION IS A MEASUREMENT.  Nothing in this
    repository knows — the briefing names control boxes .12/.14/.15 and DDS
    domains 31/71/97, and on 2026-09-17 the two MOUNTED arms "may be 97 and
    71".  So the panel asks all three, says which answered, and poses the ones
    that did in the viewer: move one by hand in guiding mode and the id whose
    numbers change is the id in front of you.  That is the whole method, and it
    is the only one that cannot be wrong about the hardware.
    """
    state = st if st is not None else site()[0]
    c = cfg or config()
    out = []
    for i in (ids or candidate_ids(state)):
        try:
            a = _id(i, state)
        except GateRefused:
            continue
        r = read_pose(a, c, run=run)
        out.append(dict(arm=a, reachable=bool(r.get("q")), q=r.get("q"),
                        cmd=r.get("cmd"), topic=r.get("topic"),
                        ip=((state.get("arms") or {}).get(str(a), {})
                            .get("ip", "")),
                        slots=[s for s in SLOTS if arm_of(s, state) == a],
                        output=str(r.get("output") or "")[-1200:],
                        error=r.get("error")))
    return out


def pose_npz(arm, q, parks, path):
    """A one-pose programme the Drake viewer can play. -> Path.

    NOT A NEW FILE FORMAT.  `scripts/meshcat_drake.py` plays a conducted npz,
    so the pose is written as the shortest possible one — two identical frames
    of every arm, the named arm at the measured joints and the others at their
    parks — and the existing `/api/meshcat` launch path opens it.  Nothing in
    that script changes for this button.
    """
    import numpy as np
    a = _arm(arm)
    arms = sorted(set(int(k) for k in parks) | {a})
    data = dict(arms=np.asarray(arms, int),
                drawing_arms=np.asarray([], int),
                fps=np.float64(2.0), stride=np.int64(1))
    for other in arms:
        qq = list(q) if other == a else list(parks.get(other,
                                                       parks.get(str(other),
                                                                 [0.0] * 7)))
        data[f"q_{other}"] = np.asarray([qq, qq], float)
        data[f"seg_{other}"] = np.asarray([-1, -1], int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def confirm_token(slot):
    """What must be typed to arm the run button.  THE SLOT, not the robot.

    The slot is what the operator chose in the form and what the plan is for;
    the physical id is site configuration and may change under them between one
    run and the next.  Asking a person to retype a number that moved on its own
    is how a confirmation becomes a reflex.
    """
    return f"RUN {_slot(slot)}"


def build_run_argv(params):
    """Job parameters -> the `--live` argv, or a refusal.  -> list[str].

    THE GATE, and it is here rather than in the browser because the browser is
    a courtesy.  Two conditions, both of them the operator's own act: a stack
    check that came back HEALTHY for the arm THIS SLOT dispatches to, and the
    words `RUN <slot>` typed by hand.  `stack_ok` is set by the server from its
    own record of the check it ran — never from the request body — so a client
    cannot assert it.
    """
    slot = _slot(params.get("slot", params.get("arm")))
    st = site()[0]
    arm = arm_of(slot, st)
    typed = str(params.get("confirm") or "").strip()
    if typed != confirm_token(slot):
        raise GateRefused(
            f"type {confirm_token(slot)!r} into the confirm box to run "
            f"{mapping_line(slot, st)} — got {typed!r}.  Nothing on this page "
            f"moves an arm without it.")
    if not params.get("stack_ok"):
        raise GateRefused(
            f"the stack check for arm {arm} ({mapping_line(slot, st)}) has not "
            f"come back {HEALTHY!r} in this session (or it has gone stale).  "
            f"Press Check stack first.")
    return send_argv(slot, params.get("csv"), live=True, st=st)


# ---------------------------------------------------------------------------
# running one command, once
# ---------------------------------------------------------------------------
def shown(argv):
    """The command, verbatim, the way the panel must echo it. -> str."""
    return shlex.join([str(a) for a in argv])


def run_capture(argv, timeout=CMD_TIMEOUT_S, cwd=ROOT, env=None):
    """One command, bounded, output and all. -> dict.

    NEVER RAISES ON THE COMMAND'S ACCOUNT.  A refusal, a timeout and a missing
    `ssh` binary are all RESULTS here — the panel's job is to show the operator
    what happened, and an exception would show them a 500 instead.
    """
    argv = [str(a) for a in argv]
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        rc = r.returncode
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode(errors="replace")
               if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        out += f"\n[no answer in {timeout:g} s — the command was killed]"
        rc = -1
    except OSError as exc:
        out, rc = f"{type(exc).__name__}: {exc}", -1
    return dict(cmd=shown(argv), argv=argv, returncode=rc, output=out,
                ok=rc == 0, elapsed_s=round(time.time() - t0, 2))


def is_healthy(text):
    """The one string that turns the Check-stack button green."""
    return HEALTHY in str(text or "")


# ---------------------------------------------------------------------------
# the live log: one `ssh … tail -f` per arm, read by byte offset
# ---------------------------------------------------------------------------
class Tails:
    """Background `tail -f` jobs, one per arm, each stoppable.

    A FILE, TAILED — the same answer `jobs.py` gives, for the same reasons: a
    pipe would deadlock the server the moment nobody drained it, would lose
    what arrived before the browser asked, and would need re-inventing for the
    second tab.  The child writes to `out/gui_operator/tail_arm<N>.log` and any
    number of readers seek into it.

    TRUNCATED AT EACH START, unlike a job log: this is a WINDOW onto the
    operator's own `/tmp/rtff_draw_armN.log`, which is the record.  Keeping
    yesterday's bytes here would only make a fresh pass hard to find.
    """

    def __init__(self, directory=None):
        self.dir = Path(directory or ROOT / "out" / "gui_operator")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs = {}
        self._cmds = {}

    def path(self, arm):
        return self.dir / f"tail_arm{_arm(arm)}.log"

    def start(self, arm, argv, spawn=subprocess.Popen):
        a = _arm(arm)
        self.stop(a)
        p = self.path(a)
        log = open(p, "wb", buffering=0)
        try:
            proc = spawn([str(x) for x in argv], cwd=str(ROOT), stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        except OSError as exc:
            log.close()
            p.write_text(f"could not start the tail: "
                         f"{type(exc).__name__}: {exc}\n")
            return dict(arm=a, live=False, cmd=shown(argv), path=str(p),
                        error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                log.close()
            except Exception:
                pass
        self._procs[a] = proc
        self._cmds[a] = shown(argv)
        return dict(arm=a, live=True, pid=proc.pid, cmd=self._cmds[a],
                    path=str(p))

    def stop(self, arm):
        a = _arm(arm)
        proc = self._procs.pop(a, None)
        if proc is None:
            return dict(arm=a, live=False, stopped=False)
        if proc.poll() is None:
            from .jobs import _kill_group           # one killer, one behaviour
            _kill_group(proc.pid)
        return dict(arm=a, live=False, stopped=True)

    def live(self, arm):
        proc = self._procs.get(_arm(arm))
        return proc is not None and proc.poll() is None

    def read(self, arm, offset=0):
        """New bytes since `offset`. -> dict(text, offset, live, ...).

        A DEAD TAIL IS NOT AN EMPTY ONE.  When there is no route to the
        operator box, `ssh` prints its refusal and exits; that text is in this
        file and `live` goes false, which is exactly what the panel must show
        rather than spinning forever on a pane that will never fill.
        """
        a = _arm(arm)
        p = self.path(a)
        off = max(0, int(offset or 0))
        text = ""
        if p.exists():
            size = p.stat().st_size
            if size < off:                        # restarted: read from zero
                off = 0
            if size > off:
                with open(p, "rb") as f:
                    f.seek(off)
                    blob = f.read(size - off)
                text = blob.decode(errors="replace")
                off = size
        proc = self._procs.get(a)
        rc = None if proc is None else proc.poll()
        return dict(arm=a, text=text, offset=off, live=self.live(a),
                    returncode=rc, cmd=self._cmds.get(a), path=str(p))

    def shutdown(self):
        for a in list(self._procs):
            self.stop(a)


# ---------------------------------------------------------------------------
# the worker half: `send --live`, streamed into the job's log
# ---------------------------------------------------------------------------
def run_operator(job_dir, params, progress):
    """The RUN button, as a job. -> (ok, error).

    ONE SUBPROCESS, ITS OUTPUT TEED LINE BY LINE.  `scripts/day1.py send
    --live` prints the scp it does, the stack-check line, the supervisor line
    and then whatever the run says over ssh; printing each line as it arrives
    puts all of it through `worker.Tee` and therefore into `events.jsonl`, so
    the browser's log pane and a terminal's show the same words in the same
    order.
    """
    st = site()[0]
    slot = _slot(params.get("slot", params.get("arm")))
    arm = arm_of(slot, st)
    argv = build_run_argv(params)                  # the gate, again
    progress.emit("job_start", "", params={k: v for k, v in params.items()
                                           if k != "confirm"},
                  argv=argv, slot=slot, arm=arm, pid=os.getpid(),
                  job_dir=str(job_dir))
    # THE MAPPING, ON EVERY RUN LINE.  A plan for a POSITION dispatched to a
    # ROBOT is the one thing about this system nobody may have to infer.
    print(f"RUN — {mapping_line(slot, st)}, observe mode, as "
          f"scripts/day1.py sends it.")
    print(f"ABORT: the physical e-stop.  Hold keeps the checkpoint; "
          f"Kill executor is the harder stop.")
    print(f"$ {shown(argv)}")
    rc = -1
    # THE STAGE IS "day1" AND NOT A NEW NAME.  `progress.STAGES` is a closed
    # vocabulary the viewer switches on; this run is the last step of the same
    # hardware day, so it goes in that lane rather than inventing a lane the
    # strip does not know how to draw.
    with progress.stage("day1", cmd=shown(argv), arm=arm):
        try:
            proc = subprocess.Popen(
                [str(a) for a in argv], cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                bufsize=1)
        except OSError as exc:
            print(f"could not start it: {type(exc).__name__}: {exc}")
            proc = None
        if proc is not None:
            for line in proc.stdout:
                print(line.rstrip("\n"))
            rc = proc.wait()
    ok = rc == 0
    err = None if ok else (f"`day1.py send --live` exited {rc} — the arm may "
                           f"not have run.  Read the log above and the "
                           f"operator's own tail.")
    progress.item("day1", what="operator", arm=arm, slot=slot, ok=ok,
                  cmd=shown(argv), returncode=rc,
                  csv=str(params.get("csv") or ""),
                  mapping=mapping_line(slot, st),
                  one_liner=(f"RAN {mapping_line(slot, st)}" if ok else
                             f"FAILED {mapping_line(slot, st)}: exit {rc}"))
    return ok, err
