#!/usr/bin/env python3
"""Publishes a URDF string as std_msgs/String with TRANSIENT_LOCAL durability.

ros2_control's controller_manager (Jazzy) takes the robot description from a
latched topic rather than a parameter.  robot_state_publisher normally provides
it; this stands in for it so T4 needs no extra apt package and no TF tree.

Publishes to /robot_description and /controller_manager/robot_description so it
works whichever the controller_manager is configured to read.
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--topics", nargs="+",
                    default=["/robot_description",
                             "/controller_manager/robot_description"])
    args = ap.parse_args()

    with open(args.urdf) as f:
        urdf = f.read()

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

    rclpy.init()
    node = Node("robot_description_pub")
    msg = String()
    msg.data = urdf
    pubs = [node.create_publisher(String, t, qos) for t in args.topics]
    for p in pubs:
        p.publish(msg)
    node.get_logger().info("published %d bytes of URDF on %s"
                           % (len(urdf), ", ".join(args.topics)))
    # TRANSIENT_LOCAL already serves late joiners; a few retries cover a
    # controller_manager that starts much later.  Stop after that -- the CM logs
    # a warning for every redundant description it receives.
    state = {"n": 0}

    def retry():
        state["n"] += 1
        if state["n"] > 4:
            return
        for pub in pubs:
            pub.publish(msg)

    node.create_timer(2.0, retry)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
