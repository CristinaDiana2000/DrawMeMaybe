#diese Node zeichent Linein mit Absetzen in Z-Richtung
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

class UR5eSVGLiftDrawNode(Node):
    def __init__(self):
        super().__init__('ur5e_svg_lift_draw_node')

        # --- MoveIt initialisieren ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "ur_manipulator"
        self.move_group = MoveGroupCommander(self.group_name)

        # --- Parameter ---
        self.svg_file = '/home/user/ros2_ws/src/my_ur5e_svg/line.svg'
        self.z_draw = 0.20     # Zeichenhöhe
        self.z_lift = 0.30     # Abhebehöhe
        self.scale = 0.001     # 1 SVG-Pixel = 1 mm

        # --- SVG laden ---
        self.paths, _ = svg2paths(self.svg_file)
        self.get_logger().info(f"{len(self.paths)} Pfade aus SVG geladen.")

        # --- Bewegung starten ---
        self.draw_all_paths()

    # --------------------------
    # SVG → Liste von Punkten
    # --------------------------
    def sample_path(self, path, n=80):
        """Unterteilt einen SVG-Pfad in n Punkte."""
        pts = []
        for i in range(n):
            t = i / (n - 1)
            p = path.point(t)
            x = p.real * self.scale
            y = p.imag * self.scale
            pts.append(np.array([x, y, 0.0]))
        return pts

    # --------------------------
    # Orientierung (Tangenten)
    # --------------------------
    def orientation_from_tangent(self, v):
        """Erzeugt Quaternion aus Tangente (x-Achse entlang Bewegung)."""
        v_norm = v / np.linalg.norm(v)
        z_axis = np.array([0, 0, 1.0])
        y_axis = np.cross(z_axis, v_norm)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(v_norm, y_axis)

        rot = np.eye(4)
        rot[0:3, 0] = v_norm
        rot[0:3, 1] = y_axis
        rot[0:3, 2] = z_axis
        q = quaternion_from_matrix(rot)
        return q

    # --------------------------
    # Bewegung eines Pfads
    # --------------------------
    def draw_path(self, points):
        poses = []

        for i in range(len(points)):
            pose = Pose()
            pose.position.x = points[i][0]
            pose.position.y = points[i][1]
            pose.position.z = self.z_draw

            if i < len(points) - 1:
                v = points[i+1] - points[i]
                q = self.orientation_from_tangent(v)
            else:
                q = self.orientation_from_tangent(points[i] - points[i-1])

            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            poses.append(pose)

        # Plane kartesische Bewegung entlang dieser Punkte
        (plan, fraction) = self.move_group.compute_cartesian_path(
            poses,
            0.005,
            0.0
        )
        self.get_logger().info(f"Pfad geplant (Erfolgsrate: {fraction*100:.1f}%)")
        self.move_group.execute(plan, wait=True)

    # --------------------------
    # Lift & Draw Logik
    # --------------------------
    def draw_all_paths(self):
        for idx, path in enumerate(self.paths):
            self.get_logger().info(f"--- Pfad {idx+1}/{len(self.paths)} ---")

            points = self.sample_path(path)

            # Startposition
            start = points[0]
            lift_pose = Pose()
            lift_pose.position.x = start[0]
            lift_pose.position.y = start[1]
            lift_pose.position.z = self.z_lift
            lift_pose.orientation.w = 1.0

            # Hinfahren (über Linie)
            self.move_group.set_pose_target(lift_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            # Absenken auf Zeichenhöhe
            draw_pose = Pose()
            draw_pose.position.x = start[0]
            draw_pose.position.y = start[1]
            draw_pose.position.z = self.z_draw
            draw_pose.orientation.w = 1.0
            self.move_group.set_pose_target(draw_pose)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            # Jetzt den Pfad abfahren
            self.draw_path(points)

            # Wieder anheben
            end = points[-1]
            lift_end = Pose()
            lift_end.position.x = end[0]
            lift_end.position.y = end[1]
            lift_end.position.z = self.z_lift
            lift_end.orientation.w = 1.0
            self.move_group.set_pose_target(lift_end)
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

def main(args=None):
    rclpy.init(args=args)
    node = UR5eSVGLiftDrawNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

