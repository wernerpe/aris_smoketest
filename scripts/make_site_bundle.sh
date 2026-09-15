#!/usr/bin/env bash
# Build the self-contained site bundle: everything a machine on the
# installation needs to install the planning stack and plan at the as-built
# height, with no network and no access to this workstation.
#
#     scripts/make_site_bundle.sh                 # -> out/site_bundle_<date>.tar.gz
#     scripts/make_site_bundle.sh --no-wheel      # source-only IK (portable)
#     scripts/make_site_bundle.sh --with-legcache # + out/leg_cache (401 MB)
#
# WHAT GOES IN, AND WHY EACH PIECE IS THERE.
#
#   repo/         `git archive HEAD` of this repository -- the COMMITTED tree
#                 only.  Deliberately not the working tree: at the time of
#                 writing there are several untracked directories in it
#                 (aris_sixarm/export, aris_sixarm/sil, docker/) belonging to
#                 another session, and a bundle that silently shipped somebody
#                 else's in-progress work would be worse than one that is
#                 honestly incomplete.  Consequence, and it is a REAL one:
#                 **the impedance pathway CSV exporter is not in this bundle**
#                 because it is not committed.  See SITE_README.md and
#                 docs/SITE_SETUP.md section 3e.
#
#   ik_src/       The analytic IK C++ bindings (wernerpe/franka_analytical_ik)
#                 at a PINNED commit, plus a setuptools/pybind11 build shim.
#                 This is the one hard dependency that is not on PyPI.  The
#                 upstream build is bazel and its wheel target is pinned to
#                 `python_tag = "cp310"` while the planner runs on 3.12; the
#                 shim builds the same sources for whatever python runs pip,
#                 in about five seconds, with no bazel and no network.
#                 Verified bit-identical to the bazel-built .so over 300
#                 random poses (max abs difference 0.0).
#
#   ik_wheel/     A prebuilt wheel, IF this machine's python matches what the
#                 site machine will run.  A wheel is cp<ver>-specific AND
#                 glibc-specific; ik_src/ is the fallback that always works.
#
#   out/          The atlases and park searches the planner needs to plan
#                 without a 1-2 minute re-sweep, at BOTH candidate heights
#                 (0.850 and 0.970) until the as-built height is surveyed.
#                 Small: about 8 MB total.
#
#   SITE_README.md, smoke_plan.py   generated here; the 60-second version of
#                 docs/SITE_SETUP.md plus the install smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IK_REPO="${ARIS_IK_REPO:-/home/franka/aris_project/franka_analytical_ik}"
# Pinned: the commit that introduced the BATCH entry points (solve_batch,
# fk_batch, tip_jacobian_batch).  aris_sixarm.ik runs a python fallback loop
# without them, so an older commit is CORRECT but roughly an order of
# magnitude slower per lattice.  Do not float this.
IK_COMMIT="${ARIS_IK_COMMIT:-0d38d9667b7c1d45acd65e698911e76437405533}"

WITH_WHEEL=1
WITH_LEGCACHE=0
for a in "$@"; do
    case "$a" in
        --no-wheel)       WITH_WHEEL=0 ;;
        --with-legcache)  WITH_LEGCACHE=1 ;;
        -h|--help)        sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

DATE="$(date +%Y%m%d)"
NAME="site_bundle_${DATE}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/$NAME"
mkdir -p "$B"/{repo,ik_src,out}

cd "$ROOT"
SHA="$(git rev-parse HEAD)"
SHA_SHORT="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "== aris_sixarm  $BRANCH @ $SHA_SHORT"
if ! git diff --quiet HEAD -- . 2>/dev/null; then
    echo "   NOTE: the working tree has modifications; the bundle ships HEAD, not the tree."
fi
git archive --format=tar "HEAD" | tar -x -C "$B/repo"
echo "$SHA" > "$B/repo/.bundle_commit"

# ---------------------------------------------------------------- IK source
echo "== franka_analytical_ik @ ${IK_COMMIT:0:12}"
if [ ! -d "$IK_REPO/.git" ]; then
    echo "   ERROR: no IK checkout at $IK_REPO (set ARIS_IK_REPO)" >&2
    exit 1
fi
HAVE="$(git -C "$IK_REPO" rev-parse HEAD)"
if [ "$HAVE" != "$IK_COMMIT" ]; then
    echo "   NOTE: checkout is at ${HAVE:0:12}, bundling pinned ${IK_COMMIT:0:12}"
fi
git -C "$IK_REPO" archive --format=tar "$IK_COMMIT" | tar -x -C "$B/ik_src"
echo "$IK_COMMIT" > "$B/ik_src/.bundle_commit"

# The build shim.  Upstream ships only a bazel build whose wheel target is
# hardcoded to cp310; this builds the identical sources for the running python.
cat > "$B/ik_src/setup.py" <<'SETUP_EOF'
"""Portable build of the wernerpe/franka_analytical_ik pybind11 extension.

Replaces the upstream bazel build for site installs.  The bazel wheel target
is pinned to `python_tag = "cp310"` (see the repo's root BUILD file) while the
planner runs python 3.12, and a bazel toolchain fetch needs network plus a
large cache.  Same two source files, same six entry points; setuptools +
pybind11 build them for whatever python runs pip, in about five seconds.

    pip install ./ik_src

Set EIGEN_INCLUDE if Eigen is not at /usr/include/eigen3
(Ubuntu/Debian: `sudo apt install libeigen3-dev`).
"""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

EIGEN = os.environ.get("EIGEN_INCLUDE", "/usr/include/eigen3")

setup(
    name="franka_analytical_ik",
    version="1.0.0",
    packages=["franka_analytical_ik"],
    package_data={"franka_analytical_ik": ["*.pyi"]},
    ext_modules=[
        Pybind11Extension(
            "franka_analytical_ik._franka_ik",
            ["franka_analytical_ik/franka_analytic_ik_bindings.cpp"],
            include_dirs=[EIGEN, "franka_analytical_ik"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
SETUP_EOF

# Upstream's own pyproject is absent; bazel needed none.  Give pip a backend.
cat > "$B/ik_src/pyproject.toml" <<'PYP_EOF'
[build-system]
requires = ["setuptools>=61", "pybind11>=2.12", "wheel"]
build-backend = "setuptools.build_meta"
PYP_EOF

# ---------------------------------------------------------------- IK wheel
if [ "$WITH_WHEEL" = "1" ]; then
    PYV="$(python3 -c 'import sys;print("cp%d%d"%sys.version_info[:2])')"
    echo "== prebuilt wheel for ${PYV} (this machine's python3)"
    if python3 -c "import pybind11" >/dev/null 2>&1 \
       && [ -d "${EIGEN_INCLUDE:-/usr/include/eigen3}" ]; then
        mkdir -p "$B/ik_wheel"
        if python3 -m pip wheel "$B/ik_src" -w "$B/ik_wheel" --no-deps -q 2>/dev/null; then
            echo "   $(basename "$B"/ik_wheel/*.whl)"
            cat > "$B/ik_wheel/README" <<WHL_EOF
Prebuilt on $(hostname) $(date -Iseconds)
  $(python3 -VV | head -1)
  glibc $(ldd --version | head -1 | grep -o '[0-9]\+\.[0-9]\+$')
This wheel is valid ONLY for the same python minor version (${PYV}) and a
glibc at least this new.  Anything else: build from ../ik_src instead.
WHL_EOF
        else
            echo "   wheel build failed -- ik_src/ is the fallback, which always works"
            rmdir "$B/ik_wheel" 2>/dev/null || true
        fi
    else
        echo "   skipped (needs pybind11 + libeigen3-dev on this machine); ik_src/ covers it"
    fi
fi

# ------------------------------------------------------------ out/ payload
# REQUIRED to plan at a height: the per-arm atlas for that height AND the
# park search searched at that same height and tool.  Everything else in
# out/ is regenerable or is a report.  Atlases are per-arm .npz, so a
# two-arm fleet uses atlas_arm31.npz + atlas_arm71.npz out of the same dir.
echo "== out/ artefacts"
# NOTE the bundle lays these out FLAT under out/, so that `cp -r out/* repo/out/`
# on the site machine lands them at repo/out/<name> -- the paths the scripts
# expect.  Do not reintroduce a dirname here: it nests out/out/.
copy_if () {   # copy_if <name-under-out> <why>
    if [ -e "$ROOT/out/$1" ]; then
        cp -r "$ROOT/out/$1" "$B/out/"
        printf '   %-46s %s\n' "out/$1" "$2"
    else
        printf '   %-46s MISSING -- %s\n' "out/$1" "$2"
        echo "$1" >> "$B/out/.MISSING"
    fi
}
# h = 0.970 -- the CURRENT build sheet height (docs/BUILD_SHEET.md, re-issued
# 2026-09-10; it supersedes 850 and 940).  Complete set.
copy_if "atlas_proposed_h0970_lat0860"         "atlas, final tool"
copy_if "atlas_proposed_h0970_lat0860_gated63" "atlas, regated at 63"
copy_if "park_search_h0970_lat0860.json"       "park search, same h+tool"
copy_if "park_search_h0970_seam.json"          "park search, seam-aware"
copy_if "certified_area_h0970.json"            "certified hole-free block"
copy_if "fw_h0970_m50_map_seam.npz"            "seam-aware feasibility map"
copy_if "stage_parks_h0970_seam.json"          "staged park assignment"
# h = 0.850 -- the height the rig was BUILT at before the re-issue.  Carried
# until the survey says which is true.  NOTE the gap called out in
# docs/HARDWARE_LADDER.md rung 0: there is an atlas at the final tool but NO
# park search at it; park_search_h0850_lat0588 is the SUPERSEDED 0.0588 tool.
copy_if "atlas_proposed_h0850_lat0860"         "atlas, final tool"
copy_if "park_search_h0850_lat0588.json"       "park search -- OLD TOOL, see SITE_README"
copy_if "certified_area_h0850_drawpose.json"   "certified area"

if [ "$WITH_LEGCACHE" = "1" ]; then
    copy_if "leg_cache" "transit-leg cache (regenerable, large)"
fi

# ----------------------------------------------------------- smoke + readme
cat > "$B/smoke_plan.py" <<'SMOKE_EOF'
#!/usr/bin/env python3
"""Site-install smoke test: plan ONE stroke on arm 31 and certify it.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 smoke_plan.py

Exits 0 only if the certified-or-split contract returns `ok` AND the plan's
own independent validation passes.  Needs NO atlas and NO out/ artefact, so
it is the first thing to run after an install: it exercises numpy, scipy,
shapely, the analytic-IK C++ bindings (batch entry points included), the DP
planner and validate.py -- every piece the install has to get right.

Do NOT use scripts/demo_stroke.py as the smoke test.  It is stale at HEAD:
its two hardcoded strokes no longer plan end to end at the current rig/tool
constants, and it then dies on `KeyError: 'qs'` while reporting the split.
That is a demo bug, not an install failure.  docs/SITE_SETUP.md section 5.
"""
import sys

import numpy as np

from aris_sixarm import frames
from aris_sixarm.fleet import FLEET
from aris_sixarm.stroke_api import plan_stroke

ARM = 31
spec = FLEET[ARM]
bx, by = spec.xy
print(f"arm {ARM}: base xy = ({bx:.4f}, {by:.4f})")
print(f"tool: ext = {frames.ext_of(None):.7f} m, lat = {frames.lat_of(None):.7f} m")

# A short arc at a comfortable radius, well inside the arm's certified patch.
th = np.linspace(-0.25, 0.25, 60)
pts = np.column_stack([bx + 0.45 * np.cos(th), by + 0.45 * np.sin(th)])

res = plan_stroke(pts, spec)
status = res["status"]
print(f"status = {status!r}  ({res['arc_len']:.4f} m, {res['n_lattice']} lattice steps)")
if status != "ok":
    print(f"  FAILED: reason={res['reason']!r} fiber_cut_s={res['fiber_cut_s']}")
    sys.exit(1)

print(f"  {len(np.asarray(res['qs']))} joint samples over {res['total_time']:.2f} s")
print(f"  min_sigma  = {res['min_sigma']:.4f}   (strict-GO gate 0.14)")
print(f"  min_margin = {res['min_margin']:.4f} rad")
print(f"  max_step   = {res['max_step']:.4f} rad")
print(f"  tip_err    = {res['tip_err']:.2e} m  (validator tolerance 2e-3)")

val = res.get("validation")
ok = val.get("ok") if isinstance(val, dict) else None
viol = val.get("violations", []) if isinstance(val, dict) else []
print(f"  validation: ok={ok} violations={viol}")
if ok is False or viol or res["tip_err"] > 2e-3:
    print("  FAILED: independent validator rejected the plan")
    sys.exit(1)

print("SMOKE OK")
SMOKE_EOF

cat > "$B/SITE_README.md" <<README_EOF
# Aris six-arm planner — site bundle ${DATE}

Built $(date -Iseconds) on \`$(hostname)\`.
Repo \`${BRANCH}\` @ \`${SHA_SHORT}\` · IK \`${IK_COMMIT:0:12}\`

The full runbook is **\`repo/docs/SITE_SETUP.md\`** — read that. This file is
the sixty-second version and the list of what is in the box.

## Install (Ubuntu 24.04, python 3.12)

\`\`\`bash
sudo apt install -y build-essential libeigen3-dev python3-venv
python3 -m venv ~/aris_venv && source ~/aris_venv/bin/activate
pip install -r repo/requirements-site.txt
pip install ./ik_src          # or: pip install ik_wheel/*.whl  (same python only)
pip install -e ./repo
\`\`\`

## Verify — both must pass before anything is powered

\`\`\`bash
cd repo && python -m pytest tests/test_gates.py tests/test_hardware_prep.py -q
cd .. && ARIS_RIG=proposed ARIS_TOOL=lateral python smoke_plan.py
\`\`\`

Expect \`20 passed\` and \`SMOKE OK\`.

## What is in the box

| | |
|---|---|
| \`repo/\` | the planner at commit \`${SHA_SHORT}\`, committed tree only |
| \`ik_src/\` | analytic IK C++ bindings, pinned + a pip build shim (always works) |
| \`ik_wheel/\` | the same, prebuilt — only for this python minor + glibc |
| \`out/\` | atlases and park searches for h = 0.970 and h = 0.850 |
| \`smoke_plan.py\` | the one-stroke install smoke test |

Copy \`out/\` into the repo checkout before planning:
\`cp -r out/* repo/out/\`

## Three things this bundle does NOT contain — read before the day

1. **No impedance pathway CSV exporter.** The code that writes the
   \`stroke_idx,wp_idx,kind,x_m,…,intensity\` format the operator box actually
   consumes lives in \`aris_sixarm/export/\`, which is **uncommitted** on the
   build machine and therefore not in \`git archive HEAD\`. What IS here is
   \`Fr3BundleBackend.export\`, a joint-space npz for the \`fr3drivers\` stack,
   which that stack cannot fly (degree-1, discontinuous velocity) and which
   has no CLI. **If a pathway CSV is needed on the day, it must be committed
   and the bundle rebuilt, or the CSVs hand-carried.** SITE_SETUP.md §3e.
2. **No plan to export.** \`out/\` is gitignored; no schedule npz or program
   json is in the archive. Planning happens on site, or the plan is carried.
3. **No robot-facing code at all.** \`aris_sixarm\` has no rclpy, no libfranka,
   no socket. It ends at a file. It cannot move an arm.

## The one rule for the day

The **physical e-stop is the abort path.** The software gate brakes at
2 rad/s² and the watchdog latches at 20 mrad; the driver's own notes say
neither of those is the abort.
README_EOF

if [ -f "$B/out/.MISSING" ]; then
    echo "" >> "$B/SITE_README.md"
    echo "## Artefacts that were MISSING on the build machine" >> "$B/SITE_README.md"
    echo '```' >> "$B/SITE_README.md"
    cat "$B/out/.MISSING" >> "$B/SITE_README.md"
    echo '```' >> "$B/SITE_README.md"
    echo "Regenerate with \`scripts/height_sweep.py sweep\` / \`park\` — see SITE_SETUP.md §3c." \
        >> "$B/SITE_README.md"
fi

# ------------------------------------------------------------------- pack
mkdir -p "$ROOT/out"
TARBALL="$ROOT/out/${NAME}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo
echo "== $TARBALL"
echo "   $(du -h "$TARBALL" | cut -f1)   (uncompressed $(du -sh "$B" | cut -f1))"
echo "   sha256 $(sha256sum "$TARBALL" | cut -c1-16)…"
