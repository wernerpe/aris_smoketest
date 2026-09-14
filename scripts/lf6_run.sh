#!/usr/bin/env bash
# THE INTEGRATED RUN: every fix of 2026-09-14 in one programme.
#
#   A, B   six arms, leaders then followers, the leader owing its same-row
#          follower a PARTNER STANDOFF, the follower's refused pieces cut at
#          the room boundary;
#   C      three TWO-ARM row conductors composed SERIALLY, each planned with
#          the other rows' rooms known before any of them runs;
#   D      the dead-band ink, with the band retry (`staged.CONDUCT_BANDS`).
#
# Two settings, the two ends of Pete's split sweep, each with the standoff the
# sweep measured for it.
# ROUTE JOBS x RUNS + CONDUCT JOBS x RUNS <= 12 on this box.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LINES=${LINES:-out/csail_schedule_h097_v19_strokes.json}
JOBS=${JOBS:-3}
CJOBS=${CJOBS:-3}
CAP=${CAP:-1500}
TAG=${TAG:-lf6}

run() {                         # run <tag> <extra args...>
    tag=$1; shift
    ARIS_RIG=proposed ARIS_TOOL=lateral setsid nohup "$PY" -u \
        -m aris_sixarm.staged --lines "$LINES" --pattern leader_follower \
        --route-jobs "$JOBS" --conduct-jobs "$CJOBS" \
        --conduct-cap-s "$CAP" --row-compose serial "$@" \
        --json "out/staged_csail_h097_${TAG}_${tag}.json" \
        --programme "out/staged_csail_h097_${TAG}_${tag}_program.json" \
        > "out/staged_csail_h097_${TAG}_${tag}.log" 2>&1 < /dev/null &
    echo "  $tag -> pid $! -> out/staged_csail_h097_${TAG}_${tag}.log"
}

echo "the integrated programme, serial row conductors:"
run s150  --split-m 0.15 --partner-standoff 0.09
run whole --whole-bag --partner-standoff 0.11
wait
echo "done"
