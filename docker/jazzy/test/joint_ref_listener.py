#!/usr/bin/env python3
"""Counts and VALIDATES sensor_msgs/JointState on
/cartesian_impedance/joint_reference against the PoseStamped equilibrium
stream, for T7 / T7b.

The executor (ARIS2 contract 2) must publish one JointState per equilibrium
pose whenever the joint reference is enabled AND both endpoints of the current
segment carry q1..q7.  This node subscribes to BOTH topics from one process --
so the two counts are attached at the same instant and are directly comparable
-- and checks, against the endpoint rows of the pathway CSV itself:

  * one JointState per PoseStamped (|delta| <= --count-tol)
  * 7 positions, 7 zero velocities, names fr3_joint1..7
  * every position inside the convex hull of the first and last CSV rows
  * every joint monotone in the direction first-row -> last-row
  * the first reference == the first CSV row, the last == the last CSV row

--partial drops the last-reference check (for a run cut short on purpose).
--expect-none inverts everything: poses must flow and NO JointState may appear
(v1 CSV without joint columns, or the flag off).

Prints JOINT_REF_SUMMARY <json>, writes the same to --out, exits 0 on success.
"""
import argparse
import csv
import json
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

NAMES = ["fr3_joint%d" % i for i in range(1, 8)]


def csv_endpoints(path):
    """(first_row_q, last_row_q) from the CSV, or (None, None) for a v1 file."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            cols = [(row.get("q%d" % i) or "").strip() for i in range(1, 8)]
            rows.append([float(c) for c in cols] if all(cols) else None)
    if not rows or rows[0] is None or rows[-1] is None:
        return None, None
    return rows[0], rows[-1]


class JointRefListener(Node):
    def __init__(self, args):
        super().__init__("joint_ref_listener")
        self.n_pose = 0
        self.n_joint = 0
        self.first = None
        self.last = None
        self.bad_shape = 0
        self.bad_names = 0
        self.bad_vel = 0
        self.mono_break = None
        self.hull_break = None
        self.prev = None
        self.lo, self.hi, self.dirn = None, None, None
        self.qa, self.qb = csv_endpoints(args.csv)
        if self.qa is not None:
            self.lo = [min(a, b) for a, b in zip(self.qa, self.qb)]
            self.hi = [max(a, b) for a, b in zip(self.qa, self.qb)]
            self.dirn = [1.0 if b >= a else -1.0
                         for a, b in zip(self.qa, self.qb)]
        self.eps = args.eps
        # Executor publishes with rclpy defaults (RELIABLE, KEEP_LAST 10).
        self.create_subscription(PoseStamped, args.pose_topic, self.cb_pose, 10)
        self.create_subscription(JointState, args.topic, self.cb_joint, 10)
        self.get_logger().info("listening on %s + %s" % (args.topic, args.pose_topic))

    def cb_pose(self, msg):
        self.n_pose += 1

    def cb_joint(self, msg):
        self.n_joint += 1
        p = list(msg.position)
        if len(p) != 7:
            self.bad_shape += 1
            return
        if list(msg.name) != NAMES:
            self.bad_names += 1
        if list(msg.velocity) != [0.0] * 7:
            self.bad_vel += 1
        if self.first is None:
            self.first = [round(v, 9) for v in p]
        self.last = [round(v, 9) for v in p]
        if self.lo is not None and self.hull_break is None:
            for k in range(7):
                if not (self.lo[k] - self.eps <= p[k] <= self.hi[k] + self.eps):
                    self.hull_break = {"msg": self.n_joint, "joint": k + 1,
                                       "value": p[k],
                                       "bounds": [self.lo[k], self.hi[k]]}
                    break
        if self.prev is not None and self.dirn is not None \
                and self.mono_break is None:
            for k in range(7):
                if (p[k] - self.prev[k]) * self.dirn[k] < -self.eps:
                    self.mono_break = {"msg": self.n_joint, "joint": k + 1,
                                       "prev": self.prev[k], "cur": p[k]}
                    break
        self.prev = p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--csv", required=True,
                    help="the pathway the executor is running (endpoint rows)")
    ap.add_argument("--topic", default="/cartesian_impedance/joint_reference")
    ap.add_argument("--pose-topic", default="/cartesian_impedance/equilibrium_pose")
    ap.add_argument("--out", default="/out/t7_joint_ref.json")
    ap.add_argument("--count-tol", type=int, default=5,
                    help="allowed |JointState - PoseStamped| (discovery skew)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--partial", action="store_true",
                    help="run is cut short: skip the last-reference check")
    ap.add_argument("--expect-none", action="store_true",
                    help="assert poses flow but NO joint reference is published")
    args = ap.parse_args()

    rclpy.init()
    node = JointRefListener(args)
    t_end = time.time() + args.duration
    try:
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:      # ExternalShutdownException when SIGINT'ed
        print("listener spin ended: %r" % (exc,), file=sys.stderr)

    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if args.expect_none:
        chk("poses_flowed", node.n_pose > 0, "n_pose=%d" % node.n_pose)
        chk("no_joint_reference", node.n_joint == 0,
            "n_joint=%d" % node.n_joint)
    else:
        chk("joint_refs_flowed", node.n_joint > 0, "n_joint=%d" % node.n_joint)
        chk("count_matches_poses",
            abs(node.n_joint - node.n_pose) <= args.count_tol,
            "n_joint=%d n_pose=%d delta=%d tol=%d"
            % (node.n_joint, node.n_pose, node.n_joint - node.n_pose,
               args.count_tol))
        chk("seven_positions", node.bad_shape == 0,
            "%d messages without 7 positions" % node.bad_shape)
        chk("names_fr3_joint1_7", node.bad_names == 0,
            "%d messages with other names" % node.bad_names)
        chk("velocity_zeros", node.bad_vel == 0,
            "%d messages without 7 zero velocities" % node.bad_vel)
        chk("inside_convex_hull", node.hull_break is None,
            json.dumps(node.hull_break) if node.hull_break else
            "all %d inside [%s]" % (node.n_joint, "row0..rowN"))
        chk("monotone", node.mono_break is None,
            json.dumps(node.mono_break) if node.mono_break else
            "monotone over %d messages" % node.n_joint)
        if node.qa is None:
            chk("csv_has_joint_columns", False, "no q1..q7 in %s" % args.csv)
        else:
            chk("first_is_row0",
                node.first is not None
                and max(abs(a - b) for a, b in zip(node.first, node.qa)) < 1e-5,
                "first=%s row0=%s" % (node.first, node.qa))
            if args.partial:
                chk("last_is_rowN_SKIPPED", True, "--partial")
            else:
                chk("last_is_rowN",
                    node.last is not None
                    and max(abs(a - b) for a, b in zip(node.last, node.qb)) < 1e-5,
                    "last=%s rowN=%s" % (node.last, node.qb))

    summary = {
        "n_joint": node.n_joint,
        "n_pose": node.n_pose,
        "delta": node.n_joint - node.n_pose,
        "first": node.first,
        "last": node.last,
        "row0": node.qa,
        "rowN": node.qb,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    line = json.dumps(summary)
    print("JOINT_REF_SUMMARY " + line)
    for c in checks:
        print("  %-4s %-24s %s" % ("ok" if c["ok"] else "FAIL",
                                   c["check"], c["detail"]))
    try:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    except OSError as exc:
        print("could not write %s: %r" % (args.out, exc), file=sys.stderr)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
