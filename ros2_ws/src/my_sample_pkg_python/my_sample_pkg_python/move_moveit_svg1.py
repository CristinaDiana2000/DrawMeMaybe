#!/usr/bin/env python3

from enum import Enum
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from svgpathtools import svg2paths, Arc, Line, QuadraticBezier, CubicBezier

svg_file = "./svg-file.svg"


class RobotState(Enum):
    INIT = 0
    MOVE_HOME = 1
    MOVE_SVG = 2
    DONE = 3

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
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "base_link"
            pose_goal.pose.position.x = 0.28
            pose_goal.pose.position.y = -0.2
            pose_goal.pose.position.z = 0.5
            pose_goal.pose.orientation.x = 1.0
            pose_goal.pose.orientation.y = 1.0
            pose_goal.pose.orientation.z = 1.0
            pose_goal.pose.orientation.w = 1.0
            
            self.send_pose_goal(pose_goal)
            
            self.state = RobotState.MOVE_SVG

        elif self.state == RobotState.MOVE_SVG:
            paths, attributes = svg2paths(svg_file)

            i = 0
            for (path, attr) in zip(paths, attributes):
                #stroke = attr['stroke'] #Die Farbe der Linie ist aber unwichtig
                #print "Path", i, "with color", stroke, "of length", round(path.length())

                #move_to_paint()
                try:
                    #get_paint(stroke)
                    #move_to_canvas()
                    send_way_points(self, paths)
                except Exception as e:
                    #print "ERROR:", e
                    raw_input("Press enter to continue... ")

                i += 1

            self.state = RobotState.DONE

        elif self.state == RobotState.DONE:
            self.get_logger().info("DONE: Home-Position erreicht.")

    def send_pose_goal(self, msg:PoseStamped):
        # -------- Beispiel-Pose --------
        #pose_goal = PoseStamped()
        #pose_goal.header.frame_id = "base_link"
        #pose_goal.pose.position.x = 0.28
        #pose_goal.pose.position.y = -0.2
        #pose_goal.pose.position.z = 0.5
        #pose_goal.pose.orientation.x = 1.0
        #pose_goal.pose.orientation.y = 1.0
        #pose_goal.pose.orientation.z = 1.0
        #pose_goal.pose.orientation.w = 1.0
        
        self.get_logger().info(f"publishing: {msg}")
        self.publisher.publish(msg)

        # -------- Constraints --------
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = msg.header.frame_id
        #pos_constraint.header = pose_goal.header
        pos_constraint.link_name = "tool0"
        
        pos_constraint.constraint_region.primitives.append(SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[0.02, 0.02, 0.02]
        ))
        pos_constraint.constraint_region.primitive_poses.append(msg.pose)
        pos_constraint.weight=1.0

        #primitive = SolidPrimitive()
        #primitive.type = SolidPrimitive.SPHERE
        #primitive.dimensions = [0.001]  # sehr kleine Toleranz
        #pos_constraint.constraint_region.primitives.append(primitive)
        #pos_constraint.constraint_region.primitive_poses.append(pose_goal.pose)
        #pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = msg.header.frame_id
        #ori_constraint.header = pose_goal.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = msg.pose.orientation
        #ori_constraint.orientation = pose_goal.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.05
        ori_constraint.absolute_y_axis_tolerance = 0.05
        ori_constraint.absolute_z_axis_tolerance = 0.05
        ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)
        goal_constraints.name = "cartesian_goal"

        # -------- MoveGroup Goal --------
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.goal_constraints.append(goal_constraints)
        goal_msg.request.num_planning_attempts = 3
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5
        goal_msg.request.workspace_parameters = WorkspaceParameters()
        goal_msg.planning.options.plan_only = False
        goal_msg.planning.options.look_around = False

        # -------- Goal senden --------
        self.get_logger().info("Sende Pose an MoveGroup...")
        future = self._action_client.send_goal_async(goal_msg, self.feedback_cb)
        future.add_done_callback(self.goal_response_cb)
        self.get_logger().info("MoveGroup goal was send ...")
        #rclpy.spin_until_future_complete(self, future)
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

    def send_way_points(self, paths):

        # -------- Constraints --------
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = msg.header.frame_id
        #pos_constraint.header = pose_goal.header
        pos_constraint.link_name = "tool0"
        
        pos_constraint.constraint_region.primitives.append(SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[0.02, 0.02, 0.02]
        ))
        pos_constraint.constraint_region.primitive_poses.append(msg.pose)
        pos_constraint.weight=1.0

        #primitive = SolidPrimitive()
        #primitive.type = SolidPrimitive.SPHERE
        #primitive.dimensions = [0.001]  # sehr kleine Toleranz
        #pos_constraint.constraint_region.primitives.append(primitive)
        #pos_constraint.constraint_region.primitive_poses.append(pose_goal.pose)
        #pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = msg.header.frame_id
        #ori_constraint.header = pose_goal.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = msg.pose.orientation
        #ori_constraint.orientation = pose_goal.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.05
        ori_constraint.absolute_y_axis_tolerance = 0.05
        ori_constraint.absolute_z_axis_tolerance = 0.05
        ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)
        goal_constraints.name = "cartesian_goal"

        # -------- MoveGroup Goal --------
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.goal_constraints.append(goal_constraints)
        goal_msg.request.num_planning_attempts = 3
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5
        goal_msg.request.workspace_parameters = WorkspaceParameters()
        goal_msg.planning.options.plan_only = False
        goal_msg.planning.options.look_around = False

        #print "Set canvas coordinate system"
        r.set_csys(canvas_coordinates)

        # TODO: check current position
        #print "  Distance to canvas:", r._get_joints_dist(j_canvas_above)

        #print "Paint path"
        for sub in path.continuous_subpaths():
            #print "  Paint continuous sub path with length %smm" % (round(sub.length()))
            r.movel((sub.start.real / 1e3, sub.start.imag / 1e3, -hover, 0, 0, 0), acc=a, vel=v)
            poses = []
            acc_dist = 0
            for seg in sub:
                if isinstance(seg, Line):
                    #print "    ", seg, "length:", seg.length()
                    poses.append((seg.start.real / 1e3, seg.start.imag / 1e3, offset + feed * acc_dist / 1e3, 0, 0, 0))
                elif isinstance(seg, Arc) or isinstance(seg, QuadraticBezier) or isinstance(seg, CubicBezier):
                    # one point every curve_interp_step, but at least two points
                    step = min(curve_interp_step * 1e3 / seg.length(), 0.5)
                    points = [seg.point(t) for t in np.arange(0, 1, step)]
                    # TODO acc_dist should be incremented from point to point:
                    poses.extend([(p.real / 1e3, p.imag / 1e3, offset + feed * acc_dist / 1e3, 0, 0, 0) for p in points])
                acc_dist += seg.length()
            poses.append((sub.end.real / 1e3, sub.end.imag / 1e3, offset, 0, 0, 0))
            poses.append((sub.end.real / 1e3, sub.end.imag / 1e3, -hover, 0, 0, 0))
            r.movels(poses, acc=a, vel=v/4, threshold=0.001)
        # If we are on left side of canvas move to save position first
        r.movel((0.6, 0.3, -hover, 0, 0, 0), acc=a, vel=v)

        # -------- Goal senden --------
        self.get_logger().info("Sende Pose an MoveGroup...")
        future = self._action_client.send_goal_async(goal_msg, self.feedback_cb)
        future.add_done_callback(self.goal_response_cb)
        self.get_logger().info("MoveGroup goal was send ...")
        #rclpy.spin_until_future_complete(self, future)
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
