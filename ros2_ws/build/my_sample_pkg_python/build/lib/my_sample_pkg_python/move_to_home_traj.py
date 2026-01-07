#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class Move_To_Home(Node):
    def __init__(self):
        super().__init__('move_to_home')

        # Action Client zum MoveGroup Controller
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/follow_joint_trajectory'
        )

        # Home-Position: UR5e Standard-Home-Joints (rad)
        self.home_joints = [
            0.0,           # shoulder_pan_joint
            -1.57,         # shoulder_lift_joint
            1.57,          # elbow_joint
            -1.57,         # wrist_1_joint
            -1.57,         # wrist_2_joint
            0.0            # wrist_3_joint
        ]

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.get_logger().info("Verbinde mit FollowJointTrajectory Action Server...")
        self._client.wait_for_server()
        self.get_logger().info("Action Server gefunden. Fahre zur Home-Position...")

        # Bewegung ausführen
        self.send_home_trajectory()

    def send_home_trajectory(self):
        goal_msg = FollowJointTrajectory.Goal()

        # Joint-Namen
        goal_msg.trajectory.joint_names = self.joint_names

        # Ein Punkt: Home-Joints
        point = JointTrajectoryPoint()
        point.positions = self.home_joints
        point.time_from_start.sec = 5  # 5 Sekunden Fahrtzeit
        goal_msg.trajectory.points.append(point)

        # Senden
        send_goal_future = self._client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Ziel abgelehnt")
            return
        self.get_logger().info("Ziel akzeptiert, Bewegung startet...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("UR5e ist in der Home-Position angekommen.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = Move_To_Home()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
