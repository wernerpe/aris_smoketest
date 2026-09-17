#!/usr/bin/env python3
"""Counts PoseStamped on /cartesian_impedance/equilibrium_pose, for T6.

Runs for --duration seconds, then writes a one-line JSON summary to --out and
prints it.  Stands in for the controller: the executor's only output that
matters here is this topic.
"""
import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class EqPoseListener(Node):
    def __init__(self, topic):
        super().__init__("eq_pose_listener")
        self.count = 0
        self.first = None
        self.last = None
        self.t_first = None
        self.t_last = None
        # Executor publishes with rclpy defaults (RELIABLE, KEEP_LAST 10).
        self.create_subscription(PoseStamped, topic, self.cb, 10)
        self.get_logger().info("listening on %s" % topic)

    def cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        rec = {"x": round(p.x, 6), "y": round(p.y, 6), "z": round(p.z, 6),
               "qx": round(o.x, 6), "qy": round(o.y, 6),
               "qz": round(o.z, 6), "qw": round(o.w, 6),
               "frame_id": msg.header.frame_id}
        self.count += 1
        now = time.time()
        if self.first is None:
            self.first = rec
            self.t_first = now
        self.last = rec
        self.t_last = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--topic", default="/cartesian_impedance/equilibrium_pose")
    ap.add_argument("--out", default="/out/t6_eq_pose.json")
    args = ap.parse_args()

    rclpy.init()
    node = EqPoseListener(args.topic)
    t_end = time.time() + args.duration
    try:
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    summary = {
        "count": node.count,
        "first": node.first,
        "last": node.last,
        "span_s": (round(node.t_last - node.t_first, 3)
                   if node.count > 1 else 0.0),
        "rate_hz": (round((node.count - 1) / (node.t_last - node.t_first), 2)
                    if node.count > 1 and node.t_last > node.t_first else 0.0),
    }
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    line = json.dumps(summary)
    print("EQ_POSE_SUMMARY " + line)
    try:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    except OSError as exc:
        print("could not write %s: %r" % (args.out, exc), file=sys.stderr)
    sys.exit(0 if summary["count"] > 0 else 1)


if __name__ == "__main__":
    main()
