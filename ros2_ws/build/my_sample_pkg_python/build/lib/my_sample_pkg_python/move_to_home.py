#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from enum import Enum
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory


class RobotState(Enum):
    INIT = 0
    MOVE_HOME = 1
    MOVE_P2 = 2
    MOVE_P3 = 3
    MOVE_P4 = 4
    MOVE_P5 = 5
    MOVE_P6 = 6
    MOVE_P7 = 7
    MOVE_P8 = 8
    DONE = 9

class Move_To_Home(Node):
    def __init__(self):
        super().__init__('move_to_home')

        self.state = RobotState.INIT

        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.home_joint_positions = [-1.58, -1.57, 1.57, -1.57, -1.57, 0.0]

        self.get_logger().info("halli ich bins")

        #self.create_timer(1.0, self.timer_callback)
        self.create_timer(1.0, self.state_machine_callback)

    #Methoden   
    def timer_callback(self):
        self.get_logger().info("da läuft was")
    
    def state_machine_callback(self):
        if self.state == RobotState.INIT:
            self.get_logger().info("INIT: Vorbereitung abgeschlossen.")
            self.state = RobotState.MOVE_HOME

        elif self.state == RobotState.MOVE_HOME:
            self.get_logger().info("MOVE_HOME: Fahre zur Home-Position...")
            self.move_to_home()
            self.state = RobotState.DONE

        elif self.state == RobotState.DONE:
            self.get_logger().info("DONE: Home-Position erreicht.")
    
    def move_to_home(self):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup Action Server nicht erreichbar!")
            return

        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory = JointTrajectory()
        trajectory.trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        #point = JointTrajectoryPoint()
        if self.state == RobotState.MOVE_HOME:
            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-90),
                math.radians(-90),
                math.radians(90),
                math.radians(-90),
                math.radians(-90),
                math.radians(0)
            ] 
            point.time_from_start.sec = 5

            trajectory.trajectory.points.append(point)


        #point.positions = self.home_joint_positions
        

        self.get_logger().info("Sende Ziel an FollowJointTrajectory...")
        self.action_client.send_goal_async(trajectory)

        return

        goal_msg = MoveGroup.Goal()
        # Ziel-Jointwerte setzen
        goal_msg.request.group_name = "manipulator"
        joint_constraints = []
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        for name, pos in zip(joint_names, self.home_joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(Constraints(joint_constraints=joint_constraints))

        self.get_logger().info("Sende Ziel an MoveIt2...")
        self.action_client.send_goal_async(goal_msg)
    

def main(args=None):
    rclpy.init(args=args)

    node = Move_To_Home()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
