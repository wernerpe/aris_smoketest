#!/usr/bin/env bash
# THE SPLIT SWEEP, as one command.  `traces.leader_follower_pattern`'s bag split
# is the pattern's one design freedom (docs/V2_STAGED.md section 22) and the
# only honest way to choose it is to fly it, so this runs the CSAIL logo at
# h = 0.970 at four settings at once and lets the numbers decide:
#
#   whole   Pete's literal baseline -- no split.  Every arm is offered its whole
#           cell in both roles, so its bag lands in the stage it plans first in
#           and a follower's remainder goes to the stage where it leads.
#   +000    the boundary on each arm's own base column: leader takes its half of
#           the contested middle, follower takes its outer strip.
#   +150    150 mm more leader, a narrower follower strip further from the partner.
#   +300    300 mm more leader -- nearly the zigzag's division of labour.
#
# ROUTE JOBS x RUNS <= 12.  `sequence.ROUTE_JOBS` forks the pen-up route screen
# and four runs sharing one box is exactly the case `--route-jobs` exists for.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LINES=${LINES:-out/csail_schedule_h097_v19_strokes.json}
JOBS=${JOBS:-3}

run() {                         # run <tag> <extra args...>
    tag=$1; shift
    ARIS_RIG=proposed ARIS_TOOL=lateral setsid nohup "$PY" -u \
        -m aris_sixarm.staged --lines "$LINES" --pattern leader_follower \
        --route-jobs "$JOBS" "$@" \
        --json "out/staged_csail_h097_lf_${tag}.json" \
        --programme "out/staged_csail_h097_program_lf_${tag}.json" \
        > "out/staged_csail_h097_lf_${tag}.log" 2>&1 < /dev/null &
    echo "  $tag -> pid $! -> out/staged_csail_h097_lf_${tag}.log"
}

echo "leader/follower split sweep on $LINES, ${JOBS} route jobs each:"
run whole --whole-bag
run s000  --split-m 0.00
run s150  --split-m 0.15
# THE FOURTH SLOT IS A CONTROL, NOT A FOURTH SPLIT.  If the follower keeps
# nothing at every split, the split is not what is refusing it, and the run that
# says so is the one with the per-piece ink gate turned back into a measurement:
# whatever flies here is ink whose LEGS route around the leader's trajectory and
# whose INK does not.  The difference between the two is the whole diagnosis.
run s150nogate --split-m 0.15 --no-follower-gate
wait
echo "done"
