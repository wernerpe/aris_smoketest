#!/usr/bin/env python3
"""Fake franka_robot_state_broadcaster, for T6.  No hardware, no libfranka.

Publishes, at 100 Hz, exactly the three topics rtff_pathway_exec.py subscribes
to, with the QoS it expects (BEST_EFFORT / KEEP_LAST / depth 10):

  /franka_robot_state_broadcaster/robot_state                  franka_msgs/FrankaRobotState
  /franka_robot_state_broadcaster/current_pose                 geometry_msgs/PoseStamped
  /franka_robot_state_broadcaster/external_wrench_in_base_frame geometry_msgs/WrenchStamped

The pose is static: identity rotation at [0.5, 0.0, 0.3] in fr3_link0.  The
wrench is zero.  Nothing here closes a loop -- the executor is being watched,
not driven.
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from franka_msgs.msg import FrankaRobotState
from geometry_msgs.msg import PoseStamped, WrenchStamped

JOINTS = ["fr3_joint%d" % i for i in range(1, 8)]
Q_HOME = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]


class FakeBroadcaster(Node):
    def __init__(self, args):
        super().__init__("fake_franka_robot_state_broadcaster")
        self.args = args
        self.frame = args.frame
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        ns = "/franka_robot_state_broadcaster"
        self.pub_state = self.create_publisher(FrankaRobotState, ns + "/robot_state", qos)
        self.pub_pose = self.create_publisher(PoseStamped, ns + "/current_pose", qos)
        self.pub_wrench = self.create_publisher(
            WrenchStamped, ns + "/external_wrench_in_base_frame", qos)
        self.n = 0
        self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info(
            "fake broadcaster up: pose=[%.3f, %.3f, %.3f] identity quat, zero wrench, %.0f Hz"
            % (args.x, args.y, args.z, args.rate))

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _pose(self):
        p = PoseStamped()
        p.header.stamp = self._stamp()
        p.header.frame_id = self.frame
        p.pose.position.x = self.args.x
        p.pose.position.y = self.args.y
        p.pose.position.z = self.args.z
        # identity rotation
        p.pose.orientation.x = 0.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 1.0
        return p

    def _wrench(self):
        w = WrenchStamped()
        w.header.stamp = self._stamp()
        w.header.frame_id = self.frame
        w.wrench.force.x = 0.0
        w.wrench.force.y = 0.0
        w.wrench.force.z = 0.0
        w.wrench.torque.x = 0.0
        w.wrench.torque.y = 0.0
        w.wrench.torque.z = 0.0
        return w

    def tick(self):
        stamp = self._stamp()
        pose = self._pose()
        wrench = self._wrench()

        st = FrankaRobotState()
        st.header.stamp = stamp
        st.header.frame_id = self.frame
        st.o_t_ee = pose
        st.o_t_ee_d = pose
        st.o_t_ee_c = pose
        st.o_f_ext_hat_k = wrench
        st.k_f_ext_hat_k = wrench
        for js in (st.measured_joint_state, st.desired_joint_state):
            js.header.stamp = stamp
            js.name = list(JOINTS)
            js.position = list(Q_HOME)
            js.velocity = [0.0] * 7
            js.effort = [0.0] * 7
        st.robot_mode = FrankaRobotState.ROBOT_MODE_MOVE
        st.control_command_success_rate = 1.0
        st.time = float(self.n) / self.args.rate

        self.pub_state.publish(st)
        self.pub_pose.publish(pose)
        self.pub_wrench.publish(wrench)
        self.n += 1
        if self.n % (10 * int(self.args.rate)) == 0:
            self.get_logger().info("published %d state samples" % self.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--x", type=float, default=0.5)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=0.3)
    ap.add_argument("--frame", default="fr3_link0")
    args = ap.parse_args()

    rclpy.init()
    node = FakeBroadcaster(args)
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
