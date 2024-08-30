import rospy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg


class GripperController(object):
    def __init__(self):
        self.pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output,queue_size=10)

        self.activate_gripper()

    def activate_gripper(self):
        # activate
        self.command = outputMsg.Robotiq2FGripper_robot_output()
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rSP  = 255
        self.command.rFR  = 150
        self.pub.publish(self.command)

    def gripper_control(self,width):
        # 0 < width < 140
        value = (-11/7)*width + 220
        self.command.rPR = int(value)

        self.pub.publish(self.command)
        rospy.sleep(0.5)
        

if __name__ == "__main__":
    grippercontroller = GripperController()
    grippercontroller.gripper_control(130)