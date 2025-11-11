#Diese Node nimmt die Linien aus der SVG-Datei und plant die linen ohne abzusetzen
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import moveit_commander
import sys
from svgpathtools import svg2paths
import numpy as np
from tf_transformations import quaternion_from_matrix

class UR5eSVGOrientedNode(Node):
    def __init__(self):
        super().__init__('ur5e_svg_oriented_node')

        # --- MoveIt initialisieren ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "ur_manipulator"
        self.move_group = MoveGroupCommander(self.group_name)

        # --- Punkte aus SVG laden ---
        filename = '/home/user/ros2_ws/src/my_ur5e_svg/line.svg'
        self.points = self.load_svg_points(filename)
        self.get_logger().info(f"{len(self.points)} Punkte aus SVG geladen.")

        # --- Bewegung starten ---
        self.follow_path()

    def load_svg_points(self, filename, scale=0.001, z_height=0.2):
        paths, _ = svg2paths(filename)
        poses = []

        for path in paths:
            n = 100  # Anzahl von Punkten pro Pfad
            for i in range(n):
                t = i / (n - 1)
                p = path.point(t)
                x = p.real * scale
                y = p.imag * scale
                z = z_height
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                poses.append(pose)

        # Orientierungen anpassen
        oriented_poses = self.add_orientations(poses) #berechnet zwischen zwei aufeinanderfolgenden Punkten eine Tangentenrichtung → das ist die Bewegungsrichtun
        return oriented_poses

    def add_orientations(self, poses):
        oriented = []
        for i in range(len(poses) - 1):
            p1 = np.array([poses[i].position.x, poses[i].position.y, poses[i].position.z])
            p2 = np.array([poses[i+1].position.x, poses[i+1].position.y, poses[i+1].position.z])
            v = p2 - p1
            v_norm = v / np.linalg.norm(v)

            # Wir nehmen an: Roboter-X-Achse = Bewegungsrichtung
            # Also: Erzeuge eine Orientierung, bei der X = v_norm
            # und Z zeigt nach oben (0, 0, 1)
            z_axis = np.array([0, 0, 1.0])
            y_axis = np.cross(z_axis, v_norm)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(v_norm, y_axis)

            rot = np.eye(4)
            rot[0:3, 0] = v_norm
            rot[0:3, 1] = y_axis
            rot[0:3, 2] = z_axis
            quat = quaternion_from_matrix(rot)#wandelt die Rotationsmatrix in ein Quaternion um (für ROS).

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p1
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            oriented.append(pose)
        oriented.append(poses[-1])  # letzten Punkt hinzufügen
        return oriented

    def follow_path(self):
        waypoints = self.points
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,   # Liste von Pose-Objekten
            0.005,       # Schrittweite [m]
            0.0          # Sprunggrenze
        )
        self.get_logger().info(f"Kartesischer Pfad geplant (Erfolgsrate: {fraction*100:.1f}%)")
        self.move_group.execute(plan, wait=True)

def main(args=None):
    rclpy.init(args=args)
    node = UR5eSVGOrientedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

