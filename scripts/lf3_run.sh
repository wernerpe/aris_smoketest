#!/usr/bin/env bash
# THE FULL PROGRAMME, WITH THE CUT AND THE ROW CONDUCTORS IN.
#
#   A, B   six arms, leaders then followers, the follower's refused pieces CUT
#          at the room boundary (`staged.split_at_room`) and its certified
#          clear stretches kept;
#   C      three TWO-ARM conductors, one per row, in parallel processes --
#          four priority orders each instead of 720, and a third of the pieces;
#   D      the dead-band ink, which no row conductor may take.
#
# Two settings, Pete's split sweep's two ends: +0.15 m and the whole bag.
# ROUTE JOBS x RUNS + CONDUCT JOBS <= 10 on this box (an animation render and a
# pytest may be running): 2 runs x 2 route jobs, 3 conductors each, staggered.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LINES=${LINES:-out/csail_schedule_h097_v19_strokes.json}
JOBS=${JOBS:-2}
CJOBS=${CJOBS:-3}
TAG=${TAG:-lf3}

run() {                         # run <tag> <extra args...>
    tag=$1; shift
    ARIS_RIG=proposed ARIS_TOOL=lateral setsid nohup "$PY" -u \
        -m aris_sixarm.staged --lines "$LINES" --pattern leader_follower \
        --route-jobs "$JOBS" --conduct-jobs "$CJOBS" "$@" \
        --json "out/staged_csail_h097_${TAG}_${tag}.json" \
        --programme "out/staged_csail_h097_${TAG}_${tag}_program.json" \
        > "out/staged_csail_h097_${TAG}_${tag}.log" 2>&1 < /dev/null &
    echo "  $tag -> pid $! -> out/staged_csail_h097_${TAG}_${tag}.log"
}

echo "leader/follower, split at the room boundary, per-row final pass:"
run s150  --split-m 0.15
run whole --whole-bag
wait
echo "done"
