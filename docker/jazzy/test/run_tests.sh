#!/usr/bin/env bash
# In-container smoke tests T1..T8.  Called by ../test_smoke.sh; run it directly
# with:  docker compose run --rm dev bash /opt/aris_test/run_tests.sh
#
# Prints one PASS/FAIL line per test and a summary.  Exit code = number of
# FAILed tests (0 = all green).  T4 is EXPECTED to fail at activate -- see the
# README; it is scored on whether it fails for the RIGHT reason.
# NOTE: deliberately no `set -u` -- /opt/ros/jazzy/setup.bash dereferences
# AMENT_TRACE_SETUP_FILES unguarded and would abort the run.
set -o pipefail

TESTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ARIS_OUT:-/out}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}" 2>/dev/null || { OUT=/tmp/aris_out; LOGS=${OUT}/logs; mkdir -p "${LOGS}"; }

RESULTS=()
FAILS=0

record() {  # record <PASS|FAIL> <id> <message>
    local st="$1" id="$2"; shift 2
    RESULTS+=("${st}|${id}|$*")
    [ "${st}" = "FAIL" ] && FAILS=$((FAILS + 1))
    printf '\n[%s] %s: %s\n' "${st}" "${id}" "$*"
}

banner() { printf '\n==================== %s ====================\n' "$*"; }

source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
[ -f /opt/franka_ws/install/setup.bash ] && source /opt/franka_ws/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

# ---------------------------------------------------------------- T1 --------
banner "T1  franka_ros2 subset built against libfranka 0.20.5"
T1_LOG="${LOGS}/t1.log"
{
    echo "--- ros2 pkg prefix ---"
    for p in franka_msgs franka_semantic_components franka_hardware \
             franka_robot_state_broadcaster franka_description; do
        printf '%-32s ' "$p"
        ros2 pkg prefix "$p" 2>&1 || echo "MISSING"
    done
    echo "--- built libraries ---"
    ls -la /opt/franka_ws/install/franka_hardware/lib/*.so \
           /opt/franka_ws/install/franka_semantic_components/lib/*.so \
           /opt/franka_ws/install/franka_robot_state_broadcaster/lib/*.so 2>&1
    echo "--- libfranka version ---"
    ls -la /opt/franka_ws/install/libfranka/lib/libfranka.so* 2>&1
    cat /opt/franka_ws/install/libfranka/lib/cmake/Franka/FrankaConfigVersion.cmake 2>/dev/null \
        | grep -i 'PACKAGE_VERSION ' || true
    echo "--- franka_hardware links libfranka ---"
    ldd /opt/franka_ws/install/franka_hardware/lib/libfranka_hardware.so 2>&1 | grep -i franka || true
} > "${T1_LOG}" 2>&1
cat "${T1_LOG}"

T1_OK=1
for p in franka_msgs franka_semantic_components franka_hardware franka_robot_state_broadcaster; do
    ros2 pkg prefix "$p" >/dev/null 2>&1 || T1_OK=0
done
LF_VER="$(grep -oP 'set\(PACKAGE_VERSION "\K[0-9.]+' \
    /opt/franka_ws/install/libfranka/lib/cmake/Franka/FrankaConfigVersion.cmake 2>/dev/null || true)"
[ "${LF_VER}" = "0.20.5" ] || T1_OK=0
if [ "${T1_OK}" = "1" ]; then
    record PASS T1 "franka_msgs/semantic_components/hardware/robot_state_broadcaster present; libfranka ${LF_VER}"
else
    record FAIL T1 "missing packages or libfranka version mismatch (found '${LF_VER}') -- see ${T1_LOG}"
fi

# ---------------------------------------------------------------- T2 --------
banner "T2  patches 0001-0003 vs franka_ros2 d812aab"
if [ -f /opt/patch_report.txt ]; then
    cat /opt/patch_report.txt
    echo "--- base commit ---"
    git -C /opt/franka_ws/src/franka_ros2 log --oneline -1 2>&1
    N_CLEAN=$(grep -c '^APPLIED_CLEAN' /opt/patch_report.txt || true)
    N_CONF=$(grep -c '^CONFLICT' /opt/patch_report.txt || true)
    if [ "${N_CLEAN}" = "3" ] && [ "${N_CONF}" = "0" ]; then
        record PASS T2 "all 3 patches applied cleanly in order onto d812aab (git apply --check + git apply)"
    else
        record FAIL T2 "${N_CLEAN}/3 clean, ${N_CONF} conflict(s): $(tr '\n' ' ' < /opt/patch_report.txt)"
    fi
else
    record FAIL T2 "/opt/patch_report.txt missing -- image built without the patch step"
fi

# ---------------------------------------------------------------- T3 --------
banner "T3  cartesian_impedance_controller builds in /aris_ws + plugin discoverable"
CTRL_SRC="${ARIS_CONTROLLER_SRC:-/src/controller}"
T3_LOG="${LOGS}/t3_build.log"
echo "controller source: ${CTRL_SRC} ($(wc -l < "${CTRL_SRC}/src/cartesian_impedance_controller.cpp" 2>/dev/null || echo '?') lines of .cpp)"
mkdir -p /aris_ws && cd /aris_ws
MAKEFLAGS=-j8 colcon build \
    --parallel-workers "${ARIS_COLCON_WORKERS:-2}" \
    --base-paths "${CTRL_SRC}" \
    --event-handlers console_direct+ \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    > "${T3_LOG}" 2>&1
T3_BUILD_RC=$?
tail -25 "${T3_LOG}"

PLUGIN_NAME="cartesian_impedance_controller/CartesianImpedanceController"
PLUGIN_IDX="/aris_ws/install/cartesian_impedance_controller/share/ament_index/resource_index/controller_interface__pluginlib__plugin/cartesian_impedance_controller"
T3_OK=0
if [ "${T3_BUILD_RC}" = "0" ] && [ -f /aris_ws/install/cartesian_impedance_controller/lib/libcartesian_impedance_controller.so ]; then
    echo "--- pluginlib index entry ---"
    cat "${PLUGIN_IDX}" 2>&1
    echo "--- declared class ---"
    grep -o 'name="[^"]*"' /aris_ws/install/cartesian_impedance_controller/share/cartesian_impedance_controller/cartesian_impedance_controller.xml 2>/dev/null \
        || grep -o 'name="[^"]*"' "${CTRL_SRC}/cartesian_impedance_controller.xml"
    if grep -q "${PLUGIN_NAME}" "${CTRL_SRC}/cartesian_impedance_controller.xml" && [ -f "${PLUGIN_IDX}" ]; then
        T3_OK=1
    fi
fi
# shellcheck disable=SC1091
[ -f /aris_ws/install/setup.bash ] && source /aris_ws/install/setup.bash
if [ "${T3_OK}" = "1" ]; then
    record PASS T3 "built libcartesian_impedance_controller.so; pluginlib index exports ${PLUGIN_NAME} (dynamic check in T4)"
else
    record FAIL T3 "build rc=${T3_BUILD_RC} -- see ${T3_LOG}: $(grep -m3 -i 'error' "${T3_LOG}" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- T4 --------
banner "T4  load / configure / activate under ros2_control_node with MOCK hardware"
T4_LOG="${LOGS}/t4_cm.log"
T4_STEPS="${LOGS}/t4_steps.log"
: > "${T4_STEPS}"
CM_PID=""; RD_PID=""
cleanup_t4() {
    [ -n "${CM_PID}" ] && kill "${CM_PID}" 2>/dev/null
    [ -n "${RD_PID}" ] && kill "${RD_PID}" 2>/dev/null
    wait "${CM_PID}" "${RD_PID}" 2>/dev/null
    return 0
}
trap cleanup_t4 EXIT

if [ "${T3_OK}" != "1" ]; then
    record FAIL T4 "skipped -- T3 did not produce a loadable controller library"
else
    python3 "${TESTDIR}/robot_description_pub.py" "${TESTDIR}/fr3_mock.urdf" \
        >> "${T4_LOG}" 2>&1 &
    RD_PID=$!
    sleep 2
    # NOTE: do NOT pass the URDF as `-p robot_description:=<xml>` -- a multi-line
    # value makes rcl's argument parser abort ("at ./src/rcl/arguments.c:352").
    # Jazzy's controller_manager takes it from the latched topic anyway.
    ros2 run controller_manager ros2_control_node \
        --ros-args \
        --params-file "${TESTDIR}/controllers_mock.yaml" \
        >> "${T4_LOG}" 2>&1 &
    CM_PID=$!

    # wait for the controller_manager services (up to 40 s)
    CM_UP=0
    for _ in $(seq 1 40); do
        if ros2 service list 2>/dev/null | grep -q '/controller_manager/list_controllers'; then
            CM_UP=1; break
        fi
        sleep 1
    done
    echo "controller_manager up: ${CM_UP}"

    if [ "${CM_UP}" != "1" ]; then
        echo "--- controller_manager log (tail) ---"; tail -30 "${T4_LOG}"
        record FAIL T4 "controller_manager never came up -- see ${T4_LOG}"
    else
        {
            echo "### list_controller_types (dynamic pluginlib discovery)"
            ros2 control list_controller_types 2>&1 | grep -i cartesian || echo "NOT LISTED"
            echo
            echo "### STEP load_controller"
            ros2 control load_controller cartesian_impedance_controller 2>&1
            echo "rc=$?"
            echo
            echo "### STEP configure  (set_controller_state -> inactive)"
            ros2 control set_controller_state cartesian_impedance_controller inactive 2>&1
            echo "rc=$?"
            echo
            echo "### STEP activate   (set_controller_state -> active)"
            ros2 control set_controller_state cartesian_impedance_controller active 2>&1
            echo "rc=$?"
            echo
            echo "### list_controllers"
            ros2 control list_controllers 2>&1
        } >> "${T4_STEPS}" 2>&1
        cat "${T4_STEPS}"
        echo "--- controller_manager stderr (matching lines) ---"
        grep -iE 'cartesian|interface|not available|exception|error|fail' "${T4_LOG}" \
            | grep -v 'Received robot description' | tail -25

        # ros2cli colourises its output; strip ANSI before parsing.
        STRIP='s/\x1b\[[0-9;]*m//g'
        PLAIN="${LOGS}/t4_steps.plain.log"
        sed "${STRIP}" "${T4_STEPS}" > "${PLAIN}"

        T4_TYPE_OK=0
        grep -qi 'cartesian_impedance_controller/CartesianImpedanceController' "${PLAIN}" && T4_TYPE_OK=1
        T4_LOADED=0
        grep -qi 'Successfully loaded controller cartesian_impedance_controller' "${PLAIN}" && T4_LOADED=1
        T4_CONF=0
        grep -qi 'Successfully configured cartesian_impedance_controller' "${PLAIN}" && T4_CONF=1
        # last column of the controller's row in the `### list_controllers` section
        FINAL_STATE="$(awk '/^### list_controllers/{f=1} f && /^cartesian_impedance_controller[ \t]/{print $NF}' "${PLAIN}" | tail -1)"
        # the decisive controller_manager line
        CM_REASON="$(sed "${STRIP}" "${T4_LOG}" | grep -iE 'Unable to activate|Could not activate|Can not activate' | tail -1 | sed 's/^\[[^]]*\] \[[^]]*\] \[[^]]*\]: //')"
        echo "final controller state: ${FINAL_STATE:-<none>}"
        echo "activation refusal    : ${CM_REASON:-<none>}"

        if [ "${T4_TYPE_OK}" = "1" ] && [ "${T4_LOADED}" = "1" ] && \
           [ "${T4_CONF}" = "1" ] && [ "${FINAL_STATE}" = "inactive" ]; then
            record PASS T4 "EXPECTED: type discovered, load OK, configure OK, activate REFUSED, state stays 'inactive'. Reason: ${CM_REASON:-<none>}"
        elif [ "${FINAL_STATE}" = "active" ]; then
            record FAIL T4 "UNEXPECTED: controller reached 'active' on mock hardware -- the franka robot_model/robot_state interfaces must have come from somewhere. See ${T4_STEPS}"
        else
            record FAIL T4 "did not get past load: type_listed=${T4_TYPE_OK} loaded=${T4_LOADED} state='${FINAL_STATE:-unset}' -- see ${T4_STEPS} and ${T4_LOG}"
        fi
    fi
    cleanup_t4
    CM_PID=""; RD_PID=""
fi
trap - EXIT
sleep 2

# ---------------------------------------------------------------- T5 --------
banner "T5  rtff_pathway_exec.py --help  (rclpy / franka_msgs / numpy import)"
T5_LOG="${LOGS}/t5.log"
python3 /src/rtff/rtff_pathway_exec.py --help > "${T5_LOG}" 2>&1
T5_RC=$?
head -12 "${T5_LOG}"
if [ "${T5_RC}" = "0" ] && grep -q -- "--csv" "${T5_LOG}"; then
    record PASS T5 "--help exited 0; imports resolved (rclpy, franka_msgs.FrankaRobotState, numpy)"
else
    record FAIL T5 "rc=${T5_RC}: $(tail -3 "${T5_LOG}" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- T6 --------
banner "T6  executor end-to-end against a fake robot state, no hardware"
T6_FAKE="${LOGS}/t6_fake.log"
T6_EXEC="${LOGS}/t6_exec.log"
T6_LISTEN="${LOGS}/t6_listener.log"
T6_DURATION="${ARIS_T6_DURATION:-20}"

python3 "${TESTDIR}/fake_robot_state.py" --rate 100 --x 0.5 --y 0.0 --z 0.3 \
    > "${T6_FAKE}" 2>&1 &
FAKE_PID=$!
python3 "${TESTDIR}/eq_pose_listener.py" --duration "$((T6_DURATION + 6))" \
    --out "${OUT}/t6_eq_pose.json" > "${T6_LISTEN}" 2>&1 &
LISTEN_PID=$!
# FLAG-OFF PROOF (2026-09-09, ARIS2 contract 2): with neither --joint-ref nor
# RTFF_QREF=1 the executor must not even CREATE the joint-reference publisher.
# Snapshot the graph mid-run; T6 fails if the topic exists.
T6_TOPICS="${LOGS}/t6_topics.txt"
( sleep 8; ros2 topic list > "${T6_TOPICS}" 2>&1 ) &
TOPICS_PID=$!
sleep 3

# Most open-loop configuration the argparse offers:
#   --mode observe   : no force loop, "open-loop like BASE, just logs the force"
#   no --closed-loop, no --contact-descend, no --lag-latch, no --resume
timeout -s INT "${T6_DURATION}" \
    python3 /src/rtff/rtff_pathway_exec.py \
        --csv "${TESTDIR}/wp3.csv" \
        --mode observe \
        --travel-speed 0.08 --draw-speed 0.02 \
        --progress-file /tmp/rtff_progress_t6.txt \
        --control-file /tmp/rtff_control_t6 \
    > "${T6_EXEC}" 2>&1
T6_EXEC_RC=$?

wait "${LISTEN_PID}" 2>/dev/null
LISTEN_RC=$?
wait "${TOPICS_PID}" 2>/dev/null
kill "${FAKE_PID}" 2>/dev/null; wait "${FAKE_PID}" 2>/dev/null

echo "--- executor log ---"; tail -30 "${T6_EXEC}"
echo "--- listener summary ---"; grep EQ_POSE_SUMMARY "${T6_LISTEN}" || tail -5 "${T6_LISTEN}"
echo "--- fake broadcaster ---"; tail -3 "${T6_FAKE}"
echo "--- topics mid-run ---"; grep cartesian "${T6_TOPICS}" 2>/dev/null || echo "(no cartesian topics captured)"
echo "executor rc=${T6_EXEC_RC} (124 = killed by timeout, expected for a run that keeps going)"

EQ_COUNT="$(grep -o '"count": *[0-9]*' "${T6_LISTEN}" | head -1 | grep -o '[0-9]*' || echo 0)"
T6_QREF_SEEN=0
grep -q '/cartesian_impedance/joint_reference' "${T6_TOPICS}" 2>/dev/null && T6_QREF_SEEN=1
# an empty/failed snapshot must not pass as "topic absent": require the pose
# topic to be in it, which proves the list was taken while the executor ran.
T6_TOPICS_OK=0
grep -q '/cartesian_impedance/equilibrium_pose' "${T6_TOPICS}" 2>/dev/null && T6_TOPICS_OK=1
if [ "${EQ_COUNT:-0}" -gt 0 ] && [ "${T6_TOPICS_OK}" = "1" ] && [ "${T6_QREF_SEEN}" = "0" ]; then
    record PASS T6 "$(grep EQ_POSE_SUMMARY "${T6_LISTEN}" | head -1); flag OFF -> ros2 topic list mid-run has no /cartesian_impedance/joint_reference"
elif [ "${T6_QREF_SEEN}" = "1" ]; then
    record FAIL T6 "/cartesian_impedance/joint_reference EXISTS with the joint-ref flag off -- the gate leaks"
elif [ "${EQ_COUNT:-0}" -gt 0 ] && [ "${T6_TOPICS_OK}" != "1" ]; then
    record FAIL T6 "poses flowed but the mid-run 'ros2 topic list' snapshot is empty/invalid -- see ${T6_TOPICS}"
else
    record FAIL T6 "no PoseStamped on /cartesian_impedance/equilibrium_pose. Executor said: $(grep -iE 'error|warn|no robot state|waiting' "${T6_EXEC}" | tail -3 | tr '\n' ' ')"
fi

# ------------------------------------------------------------ T7 / T7b ------
# ARIS2 contract 2: with the joint reference ENABLED the executor publishes one
# sensor_msgs/JointState on /cartesian_impedance/joint_reference per
# equilibrium pose, interpolated at the same fraction as the pose.
#
# qref_run <tag> <csv> <exec-seconds> <env|cli> [extra joint_ref_listener args]
# Sets QREF_EXEC_RC / QREF_LISTEN_RC / QREF_TAG for the caller.  The listener is
# SIGINT'ed 2 s after the executor exits so it prints its verdict immediately
# instead of sitting out a fixed window.
qref_run() {
    local tag="$1" csv="$2" secs="$3" how="$4"; shift 4
    local fake="${LOGS}/${tag}_fake.log" ex="${LOGS}/${tag}_exec.log"
    local li="${LOGS}/${tag}_listener.log"
    python3 "${TESTDIR}/fake_robot_state.py" --rate 100 --x 0.5 --y 0.0 --z 0.3 \
        > "${fake}" 2>&1 &
    local fake_pid=$!
    python3 "${TESTDIR}/joint_ref_listener.py" --duration "$((secs + 20))" \
        --csv "${csv}" --out "${OUT}/${tag}_joint_ref.json" "$@" \
        > "${li}" 2>&1 &
    local li_pid=$!
    sleep 3
    if [ "${how}" = "cli" ]; then           # the --joint-ref switch
        timeout -s INT "${secs}" python3 /src/rtff/rtff_pathway_exec.py \
            --csv "${csv}" --mode observe --joint-ref \
            --travel-speed 0.08 --draw-speed 0.02 \
            --progress-file "/tmp/rtff_progress_${tag}.txt" \
            --control-file "/tmp/rtff_control_${tag}" > "${ex}" 2>&1
    else                                    # the RTFF_QREF=1 env switch
        RTFF_QREF=1 timeout -s INT "${secs}" python3 /src/rtff/rtff_pathway_exec.py \
            --csv "${csv}" --mode observe \
            --travel-speed 0.08 --draw-speed 0.02 \
            --progress-file "/tmp/rtff_progress_${tag}.txt" \
            --control-file "/tmp/rtff_control_${tag}" > "${ex}" 2>&1
    fi
    QREF_EXEC_RC=$?
    sleep 2
    kill -INT "${li_pid}" 2>/dev/null
    wait "${li_pid}" 2>/dev/null
    QREF_LISTEN_RC=$?
    kill "${fake_pid}" 2>/dev/null; wait "${fake_pid}" 2>/dev/null
    QREF_TAG="${tag}"
    echo "--- ${tag} executor (rc=${QREF_EXEC_RC}) ---"; tail -12 "${ex}"
    echo "--- ${tag} listener (rc=${QREF_LISTEN_RC}) ---"; tail -14 "${li}"
    return 0
}
qsum() {  # the decisive counters out of a listener log (head -4: the top-level
          # n_joint/n_pose/delta/ok, before the per-check "ok"s)
    grep -o '"n_joint": *[0-9]*\|"n_pose": *[0-9]*\|"delta": *-\?[0-9]*\|"ok": *[a-z]*' \
        "${LOGS}/$1_listener.log" | head -4 | tr '\n' ' '
}
complete() { grep -q "RTff pathway complete" "${LOGS}/$1_exec.log" && echo 1 || echo 0; }

banner "T7  joint reference ON (RTFF_QREF=1 and --joint-ref) on a v2 CSV"
qref_run t7 "${TESTDIR}/wp_v2.csv" "${ARIS_T7_DURATION:-35}" env
T7_EXEC_RC=${QREF_EXEC_RC}; T7_LI_RC=${QREF_LISTEN_RC}; T7_DONE=$(complete t7)
qref_run t7cli "${TESTDIR}/wp_v2.csv" "${ARIS_T7_CLI_DURATION:-20}" cli --partial
T7C_LI_RC=${QREF_LISTEN_RC}
echo "env run : rc=${T7_EXEC_RC} complete=${T7_DONE} listener=${T7_LI_RC} $(qsum t7)"
echo "cli run : rc=${QREF_EXEC_RC} listener=${T7C_LI_RC} $(qsum t7cli)"

if [ "${T7_LI_RC}" = "0" ] && [ "${T7_EXEC_RC}" = "0" ] && [ "${T7_DONE}" = "1" ] \
   && [ "${T7C_LI_RC}" = "0" ]; then
    record PASS T7 "RTFF_QREF=1: $(qsum t7)-- one JointState per pose, in hull, monotone, endpoints exact; --joint-ref CLI: $(qsum t7cli)"
else
    record FAIL T7 "env: rc=${T7_EXEC_RC} complete=${T7_DONE} listener=${T7_LI_RC} $(qsum t7)| cli: listener=${T7C_LI_RC} $(qsum t7cli)| $(grep -m3 FAIL "${LOGS}/t7_listener.log" | tr '\n' ' ')"
fi

banner "T7b  joint reference ON but the CSV is v1 (no q columns) -> NO references"
qref_run t7b "${TESTDIR}/wp3.csv" "${ARIS_T7B_DURATION:-35}" env --expect-none
T7B_EXEC_RC=${QREF_EXEC_RC}; T7B_LI_RC=${QREF_LISTEN_RC}; T7B_DONE=$(complete t7b)
echo "v1 run  : rc=${T7B_EXEC_RC} complete=${T7B_DONE} listener=${T7B_LI_RC} $(qsum t7b)"
if [ "${T7B_LI_RC}" = "0" ] && [ "${T7B_EXEC_RC}" = "0" ] && [ "${T7B_DONE}" = "1" ]; then
    record PASS T7b "v1 CSV with RTFF_QREF=1 ran to completion (rc=0) and published 0 joint references: $(qsum t7b)"
else
    record FAIL T7b "rc=${T7B_EXEC_RC} complete=${T7B_DONE} listener=${T7B_LI_RC} $(qsum t7b)| $(grep -m3 FAIL "${LOGS}/t7b_listener.log" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- T8 --------
# SOFTWARE IN THE LOOP, CLOSED (ARIS2 contract 3, 2026-09-09): the Drake plant
# of `aris_sixarm/sil/` wrapped as the franka robot-state broadcaster, driving
# the UNMODIFIED /src/rtff/rtff_pathway_exec.py over DDS, with a pencil that is
# 1 cm SHORTER than the robot believes.  Four runs on the same 10 cm stroke:
# open loop (--mode observe, flat 10 mm press) and the production closed loop
# (--mode assist --closed-loop, graphite band 0.7-1.0 N), each at tip_error 0
# and -0.010 m.  The question is what the executor's closed-loop half -- the
# force servo, the air-trim, the float/over-press sentinels -- DOES about a
# short pencil that the open-loop walker simply draws in the air with.
banner "T8  SIL closed loop: Drake plant + real executor, 1 cm short pencil"
T8_DIR="${OUT}/t8"
T8_LOG="${LOGS}/t8.log"
T8_DURATION="${ARIS_T8_DURATION:-60}"
if [ "${ARIS_T8_SKIP:-0}" = "1" ]; then
    record PASS T8 "SKIPPED on request (ARIS_T8_SKIP=1)"
elif [ ! -d /src/aris_sixarm/aris_sixarm/sil ]; then
    record FAIL T8 "/src/aris_sixarm is not mounted -- run ../test_smoke.sh, not a bare docker run"
elif [ ! -x "${SIL_VENV:-/opt/sil-venv}/bin/python" ]; then
    record FAIL T8 "no pydrake venv at ${SIL_VENV:-/opt/sil-venv} -- rebuild the image (docker/jazzy/build.sh)"
else
    rm -rf "${T8_DIR}"; mkdir -p "${T8_DIR}"
    T8_START=$(date +%s)
    {
        bash "${TESTDIR}/t8_sil_run.sh" observe_tip0      0.0    observe "${T8_DURATION}"
        bash "${TESTDIR}/t8_sil_run.sh" observe_tip-10mm -0.010  observe "${T8_DURATION}"
        bash "${TESTDIR}/t8_sil_run.sh" closed_tip0       0.0    closed  "${T8_DURATION}"
        bash "${TESTDIR}/t8_sil_run.sh" closed_tip-10mm  -0.010  closed  "${T8_DURATION}"
        # EXTRA: force the contact-descend + lag-latch path back on.  The live
        # inverted env (RTFF_CONTACT_DESCEND=0) cannot reach it -- the
        # supervisor hangs --lag-latch off the same variable -- so without
        # these two runs T8 could say nothing about the latch at all.
        if [ "${ARIS_T8_LATCH:-1}" = "1" ]; then
            bash "${TESTDIR}/t8_sil_run.sh" latch_tip0      0.0    latch "${T8_DURATION}"
            bash "${TESTDIR}/t8_sil_run.sh" latch_tip-10mm -0.010  latch "${T8_DURATION}"
        fi
    } 2>&1 | tee "${T8_LOG}"
    T8_WALL=$(( $(date +%s) - T8_START ))
    echo
    python3 "${TESTDIR}/t8_report.py" "${T8_DIR}" --json "${OUT}/t8_report.json" \
        | tee -a "${T8_LOG}"
    echo "T8 wall time: ${T8_WALL}s (budget ~600s)"

    # SCORING.  Not "the drawing looked good" -- the loop has to have CLOSED:
    #  * all four runs produced a trace with draw ticks,
    #  * the executor received state and streamed poses AND joint references
    #    (contract 2 + 3 both on the wire),
    #  * no run died of a stale/reflex timeout (that would mean the sim, not
    #    the executor, set the outcome),
    #  * the open-loop tip 0 run reproduces the host walker over the ticks
    #    the executor actually pressed: touch ~1.0 (the node's own draw rule
    #    also counts the 5 s touchdown DWELL this CSV's pen-up rows produce).
    T8_VERDICT="$(python3 - "${OUT}/t8_report.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
need = ["observe_tip0", "observe_tip-10mm", "closed_tip0", "closed_tip-10mm"]
bad = []
if not set(need) <= set(d):
    bad.append("missing runs: %s" % sorted(set(need) - set(d)))
# The lag-latch probes are diagnostics, not a gate: they run a configuration
# the live inverted arm cannot reach, so their outcome is reported, not scored.
for tag in [t for t in need if t in d]:
    p, e = d[tag]["plant"], d[tag]["exec"]
    if not p.get("n_draw_ticks"):
        bad.append("%s: no draw ticks in the trace" % tag)
    if not p.get("eq_pose_msgs"):
        bad.append("%s: no equilibrium poses reached the plant" % tag)
    if not p.get("joint_ref_msgs"):
        bad.append("%s: no joint references (contract 2 gate leaked shut)" % tag)
    if e.get("n_stale"):
        bad.append("%s: executor saw robot_state go stale" % tag)
    if (p.get("rt_factor") or 0) < 0.9:
        bad.append("%s: real-time factor %.2f < 0.9" % (tag, p.get("rt_factor") or 0))
ol = d.get("observe_tip0", {}).get("plant", {})
if (ol.get("touch_frac_pressed") or 0) < 0.95:
    bad.append("observe_tip0 touch(pressed) %.3f != the host walker's ~1.0"
               % (ol.get("touch_frac_pressed") or 0))
if bad:
    print("FAIL " + "; ".join(bad[:4]))
else:
    def one(t):
        p, e = d[t]["plant"], d[t]["exec"]
        return ("%s touch(pressed) %.3f f %.2f N tip %+.2f mm press %s mm rc %s"
                % (t, p.get("touch_frac_pressed", float("nan")),
                   p.get("f_mean_pressed_n", float("nan")),
                   p.get("tip_height_pressed_mm", float("nan")),
                   e.get("press_settled_mm"), e.get("exit_code")))
    print("PASS " + " | ".join(one(t) for t in need))
PYEOF
)"
    echo "${T8_VERDICT}"
    if [ "${T8_VERDICT:0:4}" = "PASS" ]; then
        record PASS T8 "${T8_VERDICT#PASS }"
    else
        record FAIL T8 "${T8_VERDICT#FAIL }  -- see ${T8_LOG} and ${T8_DIR}"
    fi
fi

# --------------------------------------------------------------- summary ----
banner "SUMMARY"
for r in "${RESULTS[@]}"; do
    printf '%-5s %-3s %s\n' "${r%%|*}" "$(echo "$r" | cut -d'|' -f2)" "$(echo "$r" | cut -d'|' -f3-)"
done
echo
echo "logs: ${LOGS}"
echo "${FAILS} failure(s)"
exit "${FAILS}"
