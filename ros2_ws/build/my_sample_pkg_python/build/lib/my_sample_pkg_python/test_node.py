#!/usr/bin/env python3

import rclpy 
import math
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64, String

class Pencil_And_Startpoint(Node):
    def __init__(self):
        super().__init__('pencil_to_nozzle')

        # Publisher auf den aktiven Controller
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        self.gripper_pub = self.create_publisher(Float64, '/gripper/command', 10)
        self.script_pub = self.create_publisher(String, '/script_command', 10)

        # Timer ruft einmalig die Trajektorie auf
        self.timer = self.create_timer(2.0, self.send_trajectory)
        self.sent = False

    def send_trajectory(self):
        if self.sent:
            return  # Nur einmal senden

        self.traj = JointTrajectory()
        self.traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.time = 0
        self.go_get_pencil()

        self.publisher_.publish(self.traj)
        self.sent = True

    def go_get_pencil(self):
        pointLuft = JointTrajectoryPoint()
        pointLuft.positions = [
            math.radians(76.15),
            math.radians(-68.27),
            math.radians(86.27),
            math.radians(-98.62),
            math.radians(-85.50),
            math.radians(13.34)
        ]
        self.time += 4
        pointLuft.time_from_start.sec = self.time
        pointLuft.velocities = [0.0] * 6
        pointLuft.accelerations = [0.0] * 6

        pointDruck = JointTrajectoryPoint()
        pointDruck.positions = [
            math.radians(74.41),
            math.radians(-89.83),
            math.radians(85.32),
            math.radians(-75.56),
            math.radians(-85.76),
            math.radians(11.53)
        ]
        self.time += 5
        pointDruck.time_from_start.sec = self.time
        pointDruck.velocities = [0.0] * 6
        pointDruck.accelerations = [0.0] * 6

        self.traj.points.append(pointLuft)
        self.traj.points.append(pointDruck)

def main(args=None):
    rclpy.init(args=args)
    node = Pencil_And_Startpoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
