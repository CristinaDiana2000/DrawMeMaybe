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
from scipy.interpolate import CubicSpline


class RobotState(Enum):
    INIT = 0
    MOVE_HOME = 1
    DONE = 9

class Move_To_Home(Node):
    def __init__(self):
        super().__init__('move_to_home')

        self.state = RobotState.INIT

        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')

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
                math.radians(0),
                math.radians(-90),
                math.radians(0),
                math.radians(-90),
                math.radians(0),
                math.radians(0)
            ] 

            point.time_from_start.sec = 2
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-96.0),
                math.radians(-78.93),
                math.radians(119.79),
                math.radians(-130.12),
                math.radians(-88.85),
                math.radians(-5.81)
            ] 
            point.time_from_start.sec = 8
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-113.75),
                math.radians(-49.59),
                math.radians(73.91),
                math.radians(-113.31),
                math.radians(-89.11),
                math.radians(-23.67)
            ] 

            point.time_from_start.sec = 10
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-94.53),
                math.radians(-51.74),
                math.radians(77.40),
                math.radians(-114.97),
                math.radians(-88.82),
                math.radians(-4.45)
            ] 

            point.time_from_start.sec = 12
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-123.69),
                math.radians(-76.0),
                math.radians(115.88),
                math.radians(-128.75),
                math.radians(-89.31),
                math.radians(-33.57)
            ] 

            point.time_from_start.sec = 14
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-96.0),
                math.radians(-78.93),
                math.radians(119.79),
                math.radians(-130.12),
                math.radians(-88.85),
                math.radians(-5.81)
            ] 

            point.time_from_start.sec = 16
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-94.53),
                math.radians(-51.74),
                math.radians(77.40),
                math.radians(-114.97),
                math.radians(-88.82),
                math.radians(-4.45)
            ] 

            point.time_from_start.sec = 18
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-101.80),
                math.radians(-36.02),
                math.radians(48.67),
                math.radians(-101.38),
                math.radians(-88.90),
                math.radians(-11.76)
            ] 

            point.time_from_start.sec = 20
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-101.80),
                math.radians(-36.02),
                math.radians(48.67),
                math.radians(-101.38),
                math.radians(-88.90),
                math.radians(-11.76)
            ] 

            point.time_from_start.sec = 22
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-113.75),
                math.radians(-49.59),
                math.radians(73.91),
                math.radians(-113.31),
                math.radians(-89.11),
                math.radians(-23.67)
            ] 

            point.time_from_start.sec = 24
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(-123.69),
                math.radians(-76.0),
                math.radians(115.88),
                math.radians(-128.75),
                math.radians(-89.31),
                math.radians(-33.57)
            ] 

            point.time_from_start.sec = 26
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

            point = JointTrajectoryPoint()
            point.positions =[
                math.radians(0),
                math.radians(-90),
                math.radians(0),
                math.radians(-90),
                math.radians(0),
                math.radians(0)
            ] 

            point.time_from_start.sec = 32
            point.velocities = [0.0]*6
            point.accelerations = [0.0]*6

            trajectory.trajectory.points.append(point)

        #point.positions = self.home_joint_positions
        

        self.get_logger().info("Sende Ziel an FollowJointTrajectory...")
        self.action_client.send_goal_async(trajectory)

def main(args=None):
    rclpy.init(args=args)

    node = Move_To_Home()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()