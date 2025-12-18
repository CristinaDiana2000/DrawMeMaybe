#!/usr/bin/env python3

from enum import Enum
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

class RobotState(Enum):
    INIT = 0
    MOVE_HOME = 1
    DONE = 2

class UR5eMoveGroupNode(Node):
    def __init__(self):
        super().__init__('move_moveit')

        # ActionClient zum MoveGroup ActionServer
        self._action_client = ActionClient(self, MoveGroup, '/move_group')
        self.get_logger().info("Warte auf MoveGroup ActionServer...")
        self._action_client.wait_for_server()
        self.get_logger().info("MoveGroup ActionServer erreichbar")

        self.state = RobotState.INIT
        self.create_timer(1.0, self.state_machine_callback)

    def state_machine_callback(self):
        if self.state == RobotState.INIT:
            self.get_logger().info("INIT: Vorbereitung abgeschlossen.")
            self.state = RobotState.MOVE_HOME

        elif self.state == RobotState.MOVE_HOME:
            self.get_logger().info("MOVE_HOME: Fahre zur Home-Position...")
            self.send_pose_goal()
            self.state = RobotState.DONE

        elif self.state == RobotState.DONE:
            self.get_logger().info("DONE: Home-Position erreicht.")

    def send_pose_goal(self):
        # -------- Beispiel-Pose --------
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = 0.28
        pose_goal.pose.position.y = -0.2
        pose_goal.pose.position.z = 0.5
        pose_goal.pose.orientation.w = 1.0

        # -------- Constraints --------
        pos_constraint = PositionConstraint()
        pos_constraint.header = pose_goal.header
        pos_constraint.link_name = "tool0"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.001]  # sehr kleine Toleranz
        pos_constraint.constraint_region.primitives.append(primitive)
        pos_constraint.constraint_region.primitive_poses.append(pose_goal.pose)
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose_goal.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = pose_goal.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)

        # -------- MoveGroup Goal --------
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.goal_constraints.append(goal_constraints)
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5

        # -------- Goal senden --------
        self.get_logger().info("Sende Pose an MoveGroup...")
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal wurde abgelehnt")
            return

        self.get_logger().info("Goal akzeptiert, warte auf Result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.error_code.val == 1:
            self.get_logger().info("Bewegung erfolgreich geplant und ausgeführt")
        else:
            self.get_logger().error(f"Planung/Execution fehlgeschlagen (Code {result.error_code.val})")


def main():
    rclpy.init()
    node = UR5eMoveGroupNode()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
