#!/usr/bin/env bash
# THE BENCH CORPUS THROUGH THE STAGED PIPELINE.
#
# Pete, 2026-09-15: "the planner code will need to work for any drawing out of
# the box."  One picture at one placement is exactly the measurement a pipeline
# tuned to that picture passes, so the five drawings of docs/BENCH.md go through
# the SAME configuration the CSAIL logo ships on:
#
#   --split-m 0.15 --partner-standoff 0.09 --row-compose serial
#
# The line files come from `scripts/bench_lines.py --sheet installed`, which
# regenerates the pinned seeds into the paper the fleet actually has.
#
# WORKERS.  Each picture forks ROUTE jobs in stages A and B and CONDUCT jobs in
# stage C, and the two never overlap within one picture -- so N pictures at once
# cost N x max(JOBS, CJOBS) workers.  Three pictures at JOBS=2 CJOBS=3 is 9.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
JOBS=${JOBS:-2}
CJOBS=${CJOBS:-3}
CAP=${CAP:-1500}
NAMES=${NAMES:-"hatch scatter starburst spiral duotone"}
SPLIT=${SPLIT:-0.15}
STANDOFF=${STANDOFF:-0.09}
TILT=${TILT:-15}

run() {
    n=$1
    ARIS_RIG=proposed ARIS_TOOL=lateral setsid nohup "$PY" -u \
        -m aris_sixarm.staged --lines "out/bench_${n}_lines.json" \
        --pattern leader_follower --split-m "$SPLIT" \
        --partner-standoff "$STANDOFF" --row-compose serial \
        --tilt-max-deg "$TILT" \
        --route-jobs "$JOBS" --conduct-jobs "$CJOBS" --conduct-cap-s "$CAP" \
        --json "out/staged_bench_${n}.json" \
        --programme "out/staged_bench_${n}_program.json" \
        > "out/staged_bench_${n}.log" 2>&1 < /dev/null &
    echo "  $n -> pid $! -> out/staged_bench_${n}.log"
}

for n in $NAMES; do
    [ -f "out/bench_${n}_lines.json" ] || \
        "$PY" -m scripts.bench_lines --names "$n" --sheet installed
    run "$n"
done
wait
echo "done"
