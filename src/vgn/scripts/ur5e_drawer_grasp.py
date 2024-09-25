import geometry_msgs.msg
import numpy as np
import rospy
from std_srvs.srv import SetBool, Empty

from robot_helpers.ros.conversions import *
from robot_helpers.ros.robotiq_gripper import GripperController
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.ros import tf
from robot_helpers.spatial import Rotation, Transform
from vgn.rviz import Visualizer
from vgn.srv import GetMapCloud, GetSceneCloud, PredictGrasps, PredictGraspsRequest
from vgn.utils import from_grasp_config_msg


class Ur5eGraspController(object):
    def __init__(self):
        self.load_parameters()
        self.lookup_transforms()
        self.init_robot_connection()
        self.init_services()
        self.vis = Visualizer()
        rospy.sleep(1.0)  # Wait for all connections to be established
        rospy.loginfo("Ready to take action")

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id")
        self.ee_frame = rospy.get_param("~ee_frame_id")
        self.task_frame = rospy.get_param("~task_frame_id")
        self.ee_grasp_offset = Transform.from_list(rospy.get_param("~ee_grasp_offset"))
        self.grasp_ee_offset = self.ee_grasp_offset.inv()
        self.scan_joints = rospy.get_param("~scan_joints")

    def lookup_transforms(self):
        tf.init()
        self.T_base_task = tf.lookup(self.base_frame, self.task_frame)

    def init_robot_connection(self):
        self.gripper = GripperController()
        self.moveit = MoveItClient("manipulator")
        self.moveit.move_group.set_end_effector_link(self.ee_frame)

        # Add a box to the planning scene to avoid collisions with the table.
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = -0.12
        msg.pose.position.y = -1
        msg.pose.position.z = 0.47
        #self.moveit.scene.add_box("table", msg, size=(1, 0.02, 1))

    def init_services(self):
        self.reset_map = rospy.ServiceProxy("reset_map", Empty)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)
        self.get_scene_cloud = rospy.ServiceProxy("get_scene_cloud", GetSceneCloud)
        self.get_map_cloud = rospy.ServiceProxy("get_map_cloud", GetMapCloud)
        self.predict_grasps = rospy.ServiceProxy("predict_grasps", PredictGrasps)

    def run(self):
        self.vis.clear()
        self.gripper.gripper_control(100)

        self.vis.roi(self.task_frame, 0.15)

        rospy.loginfo("Reconstructing scene")
        self.scan_scene()
        self.get_scene_cloud()
        res = self.get_map_cloud()

        rospy.loginfo("Planning grasps")
        req = PredictGraspsRequest(res.voxel_size, res.map_cloud)
        res = self.predict_grasps(req)
        rospy.loginfo(f"res:{res}")

        if len(res.grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        # Deserialize result
        grasps = []
        for msg in res.grasps:
            grasp, _ = from_grasp_config_msg(msg)
            grasps.append(grasp)

        # Select the highest grasp
        scores = [grasp.pose.translation[2] for grasp in grasps]
        grasp = grasps[np.argmax(scores)]
        rospy.loginfo(f"grasp:{grasp}")
        rospy.loginfo(f"scores:{scores}")
        self.vis.grasp(self.task_frame, grasp, -1)

        # Execute grasp
        rospy.loginfo("Executing grasp")
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        success = self.execute_grasp(grasp)
        
    def scan_scene(self):
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        self.reset_map()
        self.toggle_integration(True)
        for joint_target in self.scan_joints:
            self.moveit.goto(joint_target)
        self.toggle_integration(False)

    def execute_grasp(self, grasp):
        # Transform to base frame
        grasp.pose = self.T_base_task * grasp.pose
        
        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        T_base_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.1, 0])
        T_grasp_grasp = Transform(Rotation.identity(), [0.0, 0.03, 0])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.22, 0])
        T_base_pregrasp = T_grasp_pregrasp * T_base_grasp
        T_base_retreat = T_grasp_retreat * T_base_grasp
        T_base_grasp = T_grasp_grasp * T_base_grasp
        
        rospy.loginfo("a")
        self.moveit.goto(T_base_pregrasp * self.grasp_ee_offset, velocity_scaling=0.2)
        rospy.loginfo("b")
        self.moveit.gotoL(T_base_grasp * self.grasp_ee_offset)
        rospy.loginfo("c")
        self.gripper.gripper_control(0)
        rospy.loginfo("d")
        self.moveit.gotoL(T_base_retreat * self.grasp_ee_offset)
        self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        rospy.sleep(2)
        #ピッキング
        self.moveit.goto([1.645, -1.862, -2.275, 0.993, 1.490, 0])
        self.gripper.gripper_control(80)
        self.moveit.goto([1.306, -2.288, -0.616, -1.721, 2.064, -0.275])
        self.moveit.goto([1.620, -1.899, -1.252, -1.151, 1.551, 0.049])
        self.moveit.goto([2.071, -1.838, -1.405, -1.159, 0.890, 0.303])
        self.moveit.goto([1.604, -2.431, -0.403, -1.880, 1.575, 0.033])
        rospy.sleep(2)
        self.moveit.goto([1.739, -2.601, -0.188, -1.936, 1.278, 0.951])
        self.moveit.goto([1.739, -2.497, -0.611, -1.619, 1.277, 0.953])
        self.gripper.gripper_control(0)
        self.moveit.goto([1.739, -2.601, -0.188, -1.936, 1.278, 0.951])
        self.moveit.goto([2.537, -1.556, -1.669, -1.488, 1.575, -0.018])
        self.gripper.gripper_control(80)
        self.moveit.goto([1.645, -1.862, -2.275, 0.993, 1.490, 0])
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        rospy.sleep(2)



        #閉める
        self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        self.moveit.gotoL(T_base_retreat * self.grasp_ee_offset)
        self.gripper.gripper_control(0)
        self.moveit.gotoL(T_base_grasp * self.grasp_ee_offset)
        self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])



        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        #self.moveit.goto(T_base_lift * self.grasp_ee_offset)
        

        return 


if __name__ == "__main__":
    rospy.init_node("ur5e_grasp")
    controller = Ur5eGraspController()
    while True:
        controller.run()
