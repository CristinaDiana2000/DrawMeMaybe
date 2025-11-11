#List Punkte aus der SVG-Datei aus und plant die linie
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import moveit_commander
import sys
from svgpathtools import svg2paths

class UR5eSVGLineNode(Node):
    def __init__(self):
        super().__init__('ur5e_svg_line_node')

        # --- MoveIt initialisieren ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "ur_manipulator"
        self.move_group = MoveGroupCommander(self.group_name)

        # --- Pfadpunkte aus SVG laden ---
        filename = '/home/user/ros2_ws/src/my_ur5e_svg/line.svg'
        self.points = self.load_svg_points(filename)
        self.get_logger().info(f"{len(self.points)} Punkte aus SVG geladen.")

        # --- Bewegung starten ---
        self.follow_path()

    def load_svg_points(self, filename, scale=0.001, z_height=0.2):
        """
        Liest eine SVG-Datei ein und konvertiert die Linienpfade in kartesische Punkte.
        scale: SVG-Einheiten → Meter (z. B. 1 px = 1 mm → 0.001 m)
        z_height: konstante Z-Höhe über dem Tisch
        """
        paths, attributes = svg2paths(filename)
        points = []

        for path in paths:
            # Pfad in kleine Schritte aufteilen (z. B. 50 Punkte pro Kurve)
            for i in range(50):
                pos = path.point(i / 49.0)
                x = pos.real * scale
                y = pos.imag * scale
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z_height
                pose.orientation.w = 1.0  # einfache Orientierung
                points.append(pose)
        return points

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
    node = UR5eSVGLineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

