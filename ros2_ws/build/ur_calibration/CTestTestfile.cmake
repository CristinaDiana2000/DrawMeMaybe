# CMake generated Testfile for 
# Source directory: /home/ros2/DrawMeMaybe/ros2_ws/src/src/Universal_Robots_ROS2_Driver/ur_calibration
# Build directory: /home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(calibration_test "/usr/bin/python3" "-u" "/home/ros2/ros2_humble/install/ament_cmake_test/share/ament_cmake_test/cmake/run_test.py" "/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration/test_results/ur_calibration/calibration_test.gtest.xml" "--package-name" "ur_calibration" "--output-file" "/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration/ament_cmake_gmock/calibration_test.txt" "--command" "/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration/calibration_test" "--gtest_output=xml:/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration/test_results/ur_calibration/calibration_test.gtest.xml")
set_tests_properties(calibration_test PROPERTIES  LABELS "gmock" REQUIRED_FILES "/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration/calibration_test" TIMEOUT "60" WORKING_DIRECTORY "/home/ros2/DrawMeMaybe/ros2_ws/build/ur_calibration" _BACKTRACE_TRIPLES "/home/ros2/ros2_humble/install/ament_cmake_test/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/home/ros2/ros2_humble/install/ament_cmake_gmock/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;106;ament_add_test;/home/ros2/ros2_humble/install/ament_cmake_gmock/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;52;_ament_add_gmock;/home/ros2/DrawMeMaybe/ros2_ws/src/src/Universal_Robots_ROS2_Driver/ur_calibration/CMakeLists.txt;81;ament_add_gmock;/home/ros2/DrawMeMaybe/ros2_ws/src/src/Universal_Robots_ROS2_Driver/ur_calibration/CMakeLists.txt;0;")
subdirs("gmock")
subdirs("gtest")
