#diese Node zeichet eine Linie ohne Unterstützungspunkte
import rclpy
from rclpy.node import Node
from moveit_commander import MoveGroupComander

class DrawLine(Node):

      def __init__(self):

      super().__init__('draw_line') #Konstuktor der dem Node einen Namen gibt

      self.arm = MoveGroupCommander('manipulator')

      #aktuelle Pose holen
      pose = self.arm.get_current_pose().pose

      #Startphase speichern
      start_pose = pose

      #Zielpose definieren - z.B. 10cm in x-Richtung weiter
      pose_target = start_pose
      pose_target.position.x += 0.10 #10 cm Linie

      #Planen und ausführen
      self.arm.set_pose_target(pose_target)
      success = self.arm.go(wait=True)
      self.arm.stop()
      self.arm.clear_pose_targets()

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
