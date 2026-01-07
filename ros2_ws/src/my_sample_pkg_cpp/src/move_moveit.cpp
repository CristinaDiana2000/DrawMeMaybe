#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <cstdio>
//necessary memory included
#include <memory>
//rclcpp - Ros Client Libirary for C++ package
#include <rclcpp/rclcpp.hpp>
//this is necessart to plan  and execute motion
#include <moveit/move_group_interface/move_group_interface.h>
//for converting Euler anglles to quaternions
#include <tf2_geometry_msgs/tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <moveit_msgs/msg/display_robot_state.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group");
//static const rclpp::Logger Logger = rclpp::get_logger("move_group");

int main(int argc, char **argv){
	rclcpp::init(argc, argv);
	rclcpp::NodeOptions node_options;
	node_options.automatically_declare_parameters_from_overrides(true);
	auto move_group_node =rclcpp::Node::make_shared("move_group_interface_tutorial", node_options);

	rclcpp:: executors::SingleThreadedExecutor executor;
	executor.add_node(move_group_node);
	std::thread([&executor]() {executor.spin(); }) .detach();
	static const std::string PLANNING_GROUP_ARM = "ur_manipulator";
	moveit::planning_interface::MoveGroupInterface move_group_arm(move_group_node, PLANNING_GROUP_ARM);
	const moveit::core::JointModelGroup *joint_model_group_arm =
	move_group_arm. getCurrentState()->getJointModelGroup(PLANNING_GROUP_ARM);

	//Get Current State
	moveit::core::RobotStatePtr current_state_arm =
	move_group_arm.getCurrentState(10);
	std::vector<double> joint_group_positions_arm;
	current_state_arm->copyJointGroupPositions(joint_model_group_arm, joint_group_positions_arm);
	move_group_arm.setStartStateToCurrentState();

	//Go Home
	RCLCPP_INFO(LOGGER, "Going Home");
	double deg2rad = M_PI / 180.0;
	joint_group_positions_arm[0] = -76.25 * deg2rad; //Shoulder Pan
	joint_group_positions_arm[1] = -81.76 * deg2rad; //Shoulder Lift
	joint_group_positions_arm[2] = 110.62 * deg2rad; //Elbow
	joint_group_positions_arm[3] = 238.40 * deg2rad; //Wrist 1
	joint_group_positions_arm[4] = -92.08 * deg2rad; //Wrist 2
	//joint_group_positions_arm[5] = 0.00* deg2rad; //Wrist 3
	move_group_arm.setJointValueTarget(joint_group_positions_arm);
	moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;
	bool success_arm = (move_group_arm.plan(my_plan_arm) == moveit::core::MoveItErrorCode::SUCCESS);
	move_group_arm.execute(my_plan_arm);

	//Pregrasp
	RCLCPP_INFO(LOGGER, "Pregrasp Position");

	moveit_msgs::msg::OrientationConstraint ori_constraint;
	geometry_msgs::msg::Pose target_pose1;

	target_pose1.orientation.x = 0.9996;
	target_pose1.orientation.y = -0.0177;
	target_pose1.orientation.z = -0.0020;
	target_pose1.orientation.w = -0.0212;
	target_pose1.position.x = 0.136;
	target_pose1.position.y = -0.447;
	target_pose1.position.z = 0.312;

	ori_constraint.header.frame_id = move_group_arm.getPlanningFrame();
	ori_constraint.link_name = move_group_arm.getEndEffectorLink();

	// gewünschte Orientierung (z. B. aus target_pose1)
	ori_constraint.orientation = target_pose1.orientation;

	// Toleranzen (rad)
	ori_constraint.absolute_x_axis_tolerance = 0.05;
	ori_constraint.absolute_y_axis_tolerance = 0.05;
	ori_constraint.absolute_z_axis_tolerance = 0.05;

	ori_constraint.weight = 1.0;

	moveit_msgs::msg::Constraints path_constraints;
	path_constraints.orientation_constraints.push_back(ori_constraint);

	move_group_arm.setPathConstraints(path_constraints);

	move_group_arm.setPoseTarget(target_pose1);
	success_arm = (move_group_arm.plan(my_plan_arm) ==
	moveit::core::MoveItErrorCode::SUCCESS);
	move_group_arm.execute(my_plan_arm);

	//Approach
	RCLCPP_INFO(LOGGER, "Approach to object!");
    std::vector<geometry_msgs::msg::Pose> approach_waypoints;

    geometry_msgs::msg::Pose start_pose = move_group_arm.getCurrentPose().pose;
    geometry_msgs::msg::Pose p1 = start_pose;
    p1.position.z -= 0.01;
    approach_waypoints.push_back(p1);

    geometry_msgs::msg::Pose p2 = p1;
    p2.position.z -= 0.01;
    approach_waypoints.push_back(p2);

    moveit_msgs::msg::RobotTrajectory trajectory_approach;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;
    double fraction = move_group_arm.computeCartesianPath(approach_waypoints, eef_step, jump_threshold, trajectory_approach);
    move_group_arm.execute(trajectory_approach);

    if (fraction < 0.99)
        RCLCPP_WARN(LOGGER, "Nur %.1f %% des Pfades geplant!", fraction * 100.0);

	//Retreat
	RCLCPP_INFO(LOGGER, "Retreat from object!");
    std::vector<geometry_msgs::msg::Pose> retreat_waypoints;

    geometry_msgs::msg::Pose retreat_start = move_group_arm.getCurrentPose().pose;
    geometry_msgs::msg::Pose r1 = retreat_start;
    r1.position.y += 0.03;
    retreat_waypoints.push_back(r1);

    geometry_msgs::msg::Pose r2 = r1;
    r2.position.y += 0.03;
    retreat_waypoints.push_back(r2);

    moveit_msgs::msg::RobotTrajectory trajectory_retreat;
    fraction = move_group_arm.computeCartesianPath(retreat_waypoints, eef_step, jump_threshold, trajectory_retreat);
    move_group_arm.execute(trajectory_retreat);

	rclcpp::shutdown();
	return 0;
}