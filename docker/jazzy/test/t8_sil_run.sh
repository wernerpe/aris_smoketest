#!/usr/bin/env bash
# T8 helper: ONE closed-loop run of the REAL executor against the Drake SIL.
#
#   bash t8_sil_run.sh <tag> <tip_error_m> <observe|closed|latch> [duration_s]
#
# Starts `aris_sixarm.sil.ros_node` (the contract §3 robot-state interface on
# top of the Drake plant of `aris_sixarm/sil/`), waits for it to publish, runs
# an UNMODIFIED /src/rtff/rtff_pathway_exec.py against it the way
# draw_rtff_supervised.sh runs it for an INVERTED arm, then SIGINTs the node so
# it writes its trace + summary.
#
# Everything lands in ${OUT}/t8/<tag>/ : sil/summary.json, sil/trace.npz,
# sil/setpoints_recv.npz, node.log, exec.log, force.csv.
#
# The arm/rig/tool come from the CSV's manifest, so the sim's base frame and
# paper plane are the CSV's by construction (the script fails if they are not).
set -o pipefail

TAG="${1:?tag}"
TIP_ERR="${2:?tip error, m}"
MODE="${3:?observe|closed|latch}"
DUR="${4:-${ARIS_T8_DURATION:-60}}"

TESTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ARIS_OUT:-/out}"
SIXARM="${ARIS_SIXARM_SRC:-/src/aris_sixarm}"
CSV="${ARIS_T8_CSV:-${SIXARM}/aris_sixarm/sil/examples/line10cm_arm31.csv}"
SILPY="${SIL_VENV:-/opt/sil-venv}/bin/python"
RUNDIR="${OUT}/t8/${TAG}"
mkdir -p "${RUNDIR}"

export PYTHONPATH="${SIXARM}${PYTHONPATH:+:${PYTHONPATH}}"
export ARIS_FRANKA_IK_PATH="${ARIS_FRANKA_IK_PATH:-/src/franka_analytical_ik/franka_analytical_ik}"
# Arm identity (briefing §5): domain = arm id.  Discovery stays LOCALHOST.
export ROS_DOMAIN_ID=31
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# ---- the CSV's own rig / arm / tool, and a paper-plane agreement check ------
read -r RIG ARM TOOL PZ_OK <<<"$(python3 - "${CSV}" <<'PYEOF'
import json, sys, csv
from pathlib import Path
csv_path = Path(sys.argv[1])
man = json.loads(csv_path.with_suffix(".manifest.json").read_text())
T = man["T_world_base"]
rows = [r for r in csv.DictReader(csv_path.open()) if r["kind"] == "draw"]
z_base = float(rows[0]["z_m"])
# world z of a draw row = row 3 of T_world_base applied to (x, y, z)
x, y = float(rows[0]["x_m"]), float(rows[0]["y_m"])
z_world = T[8] * x + T[9] * y + T[10] * z_base + T[11]
ok = "yes" if abs(z_world) < 1e-6 and abs(z_base - man["paper_z_base_m"]) < 1e-6 else "NO"
print(man["rig"], man["arm_id"], man["tool"]["name"], ok)
PYEOF
)"
echo "[t8:${TAG}] csv=$(basename "${CSV}") rig=${RIG} arm=${ARM} tool=${TOOL} " \
     "paper-plane agrees with the sim's world z=0: ${PZ_OK}"
if [ "${PZ_OK}" != "yes" ]; then
    echo "[t8:${TAG}] FATAL: the CSV's paper plane is not the sim's paper plane"
    exit 1
fi

# ---- the executor's surface memory ----------------------------------------
# `_table_mem_path` writes ~/RTff/data/surface_arm<ARM_ID>.json; the directory
# does not exist in this image, and without it the memory/table gate silently
# degrades to "no stored plane".  Create it, and CLEAR the store before every
# run: each run is a fresh pencil, and a plane remembered from the previous
# tip error would be rejected by RTFF_TABLE_GATE (2 mm) as a phantom -- which
# is correct behaviour on the rig and pure cross-talk between test runs.
mkdir -p "${HOME}/RTff/data"
rm -f "${HOME}/RTff/data/surface_arm31.json"

# ---- the plant ------------------------------------------------------------
NODE_LOG="${RUNDIR}/node.log"
EXEC_LOG="${RUNDIR}/exec.log"
rm -f "${RUNDIR}/sil/ready"
"${SILPY}" -m aris_sixarm.sil.ros_node \
    --csv "${CSV}" --rig "${RIG}" --arm "${ARM}" --tool "${TOOL}" \
    --tip-error "${TIP_ERR}" \
    --press 0.010 --dmax 0.012 --draw-speed 0.02 --travel-speed 0.02 \
    --start-hover 0.030 \
    --rt-factor "${ARIS_T8_RT_FACTOR:-1.0}" --state-rate 100 \
    --max-wall-s "$((DUR + 40))" --no-plot \
    --out "${RUNDIR}/sil" > "${NODE_LOG}" 2>&1 &
NODE_PID=$!

READY=0
for _ in $(seq 1 90); do
    [ -f "${RUNDIR}/sil/ready" ] && { READY=1; break; }
    kill -0 "${NODE_PID}" 2>/dev/null || break
    sleep 1
done
if [ "${READY}" != "1" ]; then
    echo "[t8:${TAG}] node never published a robot state:"
    tail -20 "${NODE_LOG}"
    kill "${NODE_PID}" 2>/dev/null; wait "${NODE_PID}" 2>/dev/null
    exit 1
fi
echo "[t8:${TAG}] SIL node ready ($(grep -c . "${NODE_LOG}") log lines)"

# ---- the executor, as draw_rtff_supervised.sh runs it for an INVERTED arm --
# GUI `_rtff_inverted_env` + briefing §8.5 / §9.  RTFF_CONTACT_DESCEND=0 is the
# inverted-arm setting: the supervisor then passes NEITHER --contact-descend
# NOR --lag-latch (both live in CONTACT_DESCEND_ARG), so the pen flies to the
# baked Z and the force loop is what holds contact.  RTFF_LAG_* are exported
# anyway, exactly as the dispatcher exports them.
COMMON_ENV=(
    ARM_ID=31 ARM_INVERSE=1
    RTFF_CONTACT_DESCEND=0 RTFF_FORCE_SIGN=1 RTFF_TRAVEL_SPEED=0.02
    RTFF_QREF=1
    RTFF_LAG_LATCH=1 RTFF_LAG_GAP=0.0010 RTFF_LAG_TICKS=3
    RTFF_HOVER=0.030 RTFF_MAX_SEEK=0.040 RTFF_LIFT_MAX=0.015
    RTFF_TDOWN_SPEED=0.002 RTFF_TDOWN_FAST=0.020 RTFF_TDOWN_SLOWZONE=0.002
    RTFF_LAND_RATE=0.010 RTFF_LAND_HOLD=1 RTFF_LAND_HOLD_M=0.006
    RTFF_TABLE_GATE=0.002 RTFF_LATCH_MAX_SHIFT=0.10
    RTFF_AIR_TRIM=1 RTFF_AIR_TRIM_MAX=0.012 RTFF_AIR_TRIM_PERSIST=1
    RTFF_KZ=800 RTFF_SLOWIN_F=0.25 RTFF_DEPART_LIFT_M=0.08
)
DESCEND_ARGS=()
if [ "${MODE}" = "latch" ]; then
    # EXTRA, NOT the live inverted configuration.  RTFF_CONTACT_DESCEND=0 is
    # what the GUI exports for arm 31, and draw_rtff_supervised.sh hangs BOTH
    # --contact-descend AND --lag-latch off that one variable -- so on the live
    # inverted arm the lag latch is unreachable by construction.  This run
    # forces the path back on (floor-arm settings, briefing §9: gap 1 mm,
    # 3 ticks) to answer what the latch itself does about a short pencil.
    DESCEND_ARGS=(--contact-descend --lag-latch
                  --lag-gap 0.0010 --lag-ticks 3
                  --contact-speed 0.002 --contact-gap 0.012
                  --contact-max 0.15)
fi
if [ "${MODE}" = "observe" ]; then
    # (a) open-loop depth, the walker's own press.  `--press` flat, no
    #     --press-min/--press-max, so _press_depth returns 10 mm everywhere.
    MODE_ARGS=(--mode observe --press 0.010 --force-sign 1)
else
    # (b) the production closed loop: the graphite band of briefing §9, with
    #     the GMINE_* -> flag mapping draw_rtff_supervised.sh performs
    #     (FMIN/FMAX/LEVELS, FCAP -> --f-max, DMAX -> --d-max,
    #      TFLOOR -> --min-press AND --draw-press-floor, KP, SLEW).
    MODE_ARGS=(--mode assist --closed-loop
               --force-min 0.7 --force-max 1.0 --force-levels 9 --force-sign 1
               --f-max 3.5 --d-max 0.012 --min-press 0.009
               --draw-press-floor 0.009 --contact-floor 0.2
               --kp 0.001 --slew 0.0012
               --spike-reject 2.0 --force-lp 0.05
               --air-trim --air-trim-max 0.012)
fi

echo "[t8:${TAG}] exec: ${MODE_ARGS[*]} ${DESCEND_ARGS[*]-}"
START=$(date +%s)
env "${COMMON_ENV[@]}" \
    timeout -s INT "${DUR}" python3 /src/rtff/rtff_pathway_exec.py \
        --csv "${CSV}" "${MODE_ARGS[@]}" ${DESCEND_ARGS[@]+"${DESCEND_ARGS[@]}"} \
        --draw-speed 0.02 --travel-speed 0.02 --hover 0.030 \
        --max-seek 0.040 --k-guess 800 \
        --log "${RUNDIR}/force.csv" \
        --progress-file "/tmp/rtff_progress_t8_${TAG}.txt" \
        --control-file "/tmp/rtff_control_t8_${TAG}" \
    > "${EXEC_LOG}" 2>&1
EXEC_RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "[t8:${TAG}] executor rc=${EXEC_RC} after ${ELAPSED}s (124 = timeout)"

# ---- let the plant settle a beat, then take its trace ----------------------
sleep 1
kill -INT "${NODE_PID}" 2>/dev/null
for _ in $(seq 1 40); do
    kill -0 "${NODE_PID}" 2>/dev/null || break
    sleep 1
done
kill -9 "${NODE_PID}" 2>/dev/null
wait "${NODE_PID}" 2>/dev/null

echo "${EXEC_RC}" > "${RUNDIR}/exec_rc.txt"
echo "[t8:${TAG}] node tail:"; tail -3 "${NODE_LOG}"
echo "[t8:${TAG}] exec tail:"; tail -4 "${EXEC_LOG}"
exit 0
