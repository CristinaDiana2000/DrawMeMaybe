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

//#include <moveit/move_group_interface/move_group_interface.hpp>
//#include <moveit/planning_scene_interface/planning_scene_interface.hpp>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

//#include <moveit_visual_tools/moveit_visual_tools.h>

// All source files that use ROS logging should define a file-specific
// static const rclcpp::Logger named LOGGER, located at the top of the file
// and inside the namespace with the narrowest scope (if there is one)
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_first_node");

int main(int argc, char ** argv)
{
  //This is a standard line for intializing rclcpp
		//This is a standard line for intializing rclcpp
		//rclcpp - Ros Client Library for C++ package
		rclcpp::init(argc, argv);
		auto const node = std::make_shared<rclcpp::Node>
		(
			"move_program", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
		);

		//Over here, we create a ros logger
		auto const logger = rclcpp::get_logger("move_program");

		//Over here, we create the MoveIt MoveGroup Interface this is the main object for performing motion
		//we need to specify our node and the Planning Group "panda_arm" this planning group is the same 
		//Planning Group in RViz, under the MotionPlanning -> Planning Request
		
		moveit::planning_interface::MoveGroupInterface MoveGroupInterface(node, "ur_manipulator");//"panda_arm" ist der Name der Gruppe von joins

		//here, we define the goal pose

		//over here, we define the orientation
		tf2::Quaternion tf2_quart;

		//create a quartion object by specifying roll, pitch, and yaw angles, these angles should be specified in radius
		tf2_quart.setRPY(0,0,-3.14/2);

		//Convert tf2::Quaternion zu geometry_msgs::msg::Quaternion
		geometry_msgs::msg::Quaternion msg_quart = tf2::toMsg(tf2_quart);

		//Set a goal pose, you can change the position, however, try a avoid self-collesion of joints
		geometry_msgs::msg::Pose GoalPose;
		GoalPose.orientation = msg_quart;
		GoalPose.position.x = 0.3;
		GoalPose.position.y = -0.3;
		GoalPose.position.z = 0.4;

		MoveGroupInterface.setPoseTarget(GoalPose);

		//Create a plan that will move the robot to goal pose
		moveit::planning_interface::MoveGroupInterface::Plan plan1;
		auto const outcome = static_cast<bool>(MoveGroupInterface.plan(plan1));
		//Execute the plan
		if(outcome)
		{
			MoveGroupInterface.execute(plan1);
		}
		else
		{
			RCLCPP_ERROR(logger, "We were not able to plan and execute");
		}

		//Shutdown ros
		rclcpp::shutdown();
  return 0;
}
