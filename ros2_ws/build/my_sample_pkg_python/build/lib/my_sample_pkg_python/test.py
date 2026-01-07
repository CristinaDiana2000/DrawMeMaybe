import rclpy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from scipy.interpolate import CubicSpline

def create_trajectory_client():
    # Connect to the action server
    client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()
    return client

def generate_spline_trajectory(times, positions):
    # Create a cubic spline interpolator
    cs = CubicSpline(times, positions, bc_type='clamped')
    
    # Define a new set of time points (e.g., more dense than the input times)
    time_points = np.linspace(times[0], times[-1], num=100)
    
    # Compute the spline at these time points
    spline_positions = cs(time_points)
    
    return time_points, spline_positions

def create_trajectory_goal(time_points, spline_positions, joint_names):
    # Create the trajectory message
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    
    # Create the trajectory points
    trajectory_points = []
    for t, pos in zip(time_points, spline_positions):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(t)
        point.positions = pos.tolist()  # Convert numpy array to list for ROS message
        trajectory_points.append(point)
    
    trajectory.points = trajectory_points
    
    # Create the goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = trajectory
    
    return goal

def send_trajectory(client, goal):
    client.send_goal(goal)
    
    # Wait for the action to finish (optional: set timeout)
    client.wait_for_result(rospy.Duration(30.0))
    
    # Check if it succeeded
    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Trajectory executed successfully!")
    else:
        rospy.logwarn("Trajectory execution failed.")

def main():
    rospy.init_node('trajectory_spline_example')

    # Define joint names (for example)
    joint_names = ["joint_1", "joint_2", "joint_3"]

    # Define the times and positions for the spline
    times = [0, 1, 2, 3, 4]  # example times in seconds
    positions = [
        [0.0, 0.5, 0.0],  # joint positions at t=0
        [0.5, 0.7, 0.2],  # joint positions at t=1
        [0.6, 0.8, 0.4],  # joint positions at t=2
        [0.7, 0.9, 0.5],  # joint positions at t=3
        [1.0, 1.0, 0.6]   # joint positions at t=4
    ]

    # Generate the smooth spline trajectory
    time_points, spline_positions = generate_spline_trajectory(times, positions)

    # Create the action client
    client = create_trajectory_client()

    # Create the trajectory goal
    goal = create_trajectory_goal(time_points, spline_positions, joint_names)

    # Send the goal to the action server
    send_trajectory(client, goal)

if __name__ == "__main__":
    main()