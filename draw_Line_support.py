#diese Node zeichet eine Linie mit Unterstützungspunkte
import rclpy
from rclpy.node import Node
from moveit_commander import MoveGroupComander

class DrawLine(Node):

      def __init__(self):

      super().__init__('draw_line')

      self.arm = MoveGroupCommander('manipulator')

      waypoints = []
      #aktuelle Pose holen
      pose = self.arm.get_current_pose().pose

      for i in range(5):
          wpose.position.x += 0.02 #kleine Schritte in x
          waypoints.append(copy.deepcopy(wpose))

      
      (plan, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, 0.0)
      self.arm.execute(plan, wait=True)



      if success:
        self.get_logger().info('Linie gezeichnet')
      else:
        self.get_logger().warn('Bewegung fehlgeschlagen.')

def main(args=None):
      rclpy.init(args=args)
      node = Drawline()
      rclpy.shutdown()

if __name__ == '__main__':
      main()
