#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
from scipy.interpolate import CubicSpline
from ur_kinematics import ur_kinematics
from scipy.spatial.transform import Rotation as R


# ░░░░░░░░░░░░░░░░░░░░░░░░
#   UR5e IK (analytisch)
# ░░░░░░░░░░░░░░░░░░░░░░░░
ik = ur_kinematics.URKinematics("ur5e")

def pose_to_T(x, y, z, quat=[0, 1, 0, 0]):
    """ quaternion + xyz → 4x4 matrix """
    T = np.eye(4)
    T[0:3, 0:3] = R.from_quat(quat).as_matrix()
    T[0:3, 3] = [x, y, z]
    return T

def solve_ik(x, y, z, quat=[0, 1, 0, 0]):
    """ returns joint angles for UR5e """
    T = pose_to_T(x, y, z, quat)
    sols = ik.inverse(T)

    for s in sols:
        if ik.is_valid(s):
            return s.tolist()
    raise RuntimeError("Keine gültige IK-Lösung.")


# ░░░░░░░░░░░░░░░░░░░░░░░░
#     ROS2 NODE
# ░░░░░░░░░░░░░░░░░░░░░░░░
class Ur5eSplineNode(Node):

    def __init__(self):
        super().__init__("ur5e_spline_node")

        self.pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.timer = self.create_timer(1.0, self.run)
        self.sent = False

    def run(self):
        if self.sent:
            return

        self.get_logger().info("Erzeuge 20 cm Spline ...")

        # ░░░░░░░░░░░░░░░░░░░░░░
        #   SPLINE-PUNKTE
        # ░░░░░░░░░░░░░░░░░░░░░░
        # Diesen Bogen kannst du beliebig verändern.
        # Gesamtlänge = 0.20 m
        P0 = np.array([0.3, 0.0, 0.3])
        P1 = np.array([0.35, 0.10, 0.3])
        P2 = np.array([0.45, -0.05, 0.3])
        P3 = np.array([0.50, 0.0, 0.3])

        # Sicherheitscheck der Länge
        rough_len = (
            np.linalg.norm(P1 - P0) +
            np.linalg.norm(P2 - P1) +
            np.linalg.norm(P3 - P2)
        )
        assert abs(rough_len - 0.20) < 0.02, "Spline ist nicht ca. 20 cm lang!"

        # Parameter t
        t = [0.0, 0.33, 0.66, 1.0]
        x_s = CubicSpline(t, [P0[0], P1[0], P2[0], P3[0]])
        y_s = CubicSpline(t, [P0[1], P1[1], P2[1], P3[1]])
        z_s = CubicSpline(t, [P0[2], P1[2], P2[2], P3[2]])

        # Feine Abtastung
        samples = 200
        T = np.linspace(0, 1, samples)

        path = np.array([[x_s(t), y_s(t), z_s(t)] for t in T])

        # ░░░░░░░░░░░░░░░░░░░░░░
        #   TRAJECTORY
        # ░░░░░░░░░░░░░░░░░░░░░░
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        speed = 0.05  # 5 cm/s
        t_acc = 0.0

        for i in range(len(path)):
            x, y, z = path[i]

            q = solve_ik(x, y, z)

            if i > 0:
                dist = np.linalg.norm(path[i] - path[i - 1])
                t_acc += dist / speed

            p = JointTrajectoryPoint()
            p.positions = q
            p.time_from_start = Duration(
                sec=int(t_acc),
                nanosec=int((t_acc % 1) * 1e9)
            )
            traj.points.append(p)

        self.pub.publish(traj)
        self.get_logger().info("Spline gesendet!")
        self.sent = True


def main(args=None):
    rclpy.init(args=args)
    node = Ur5eSplineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
