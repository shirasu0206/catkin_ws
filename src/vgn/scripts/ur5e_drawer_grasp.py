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
        msg.pose.position.y = -0.87
        msg.pose.position.z = 0.195
        self.moveit.scene.add_box("table", msg, size=(0.347, 0.02, 0.347))

    def init_services(self):
        self.reset_map = rospy.ServiceProxy("reset_map", Empty)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)
        self.get_scene_cloud = rospy.ServiceProxy("get_scene_cloud", GetSceneCloud)
        self.get_map_cloud = rospy.ServiceProxy("get_map_cloud", GetMapCloud)
        self.predict_grasps = rospy.ServiceProxy("predict_grasps", PredictGrasps)

    def run(self):
        self.vis.clear()
        self.gripper.gripper_control(100)

        self.vis.roi(self.task_frame, 0.34)

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
        self.moveit.goto([1.7301 ,-1.5518, -1.9879, -0.2827, 1.5261, -0.0168])
        #success = self.execute_grasp(grasp)
        #ここから仮のgraspプログラム
        grasp_pose_data = res.grasps[0].pose

        # 位置情報を抽出
        position_x = -(grasp_pose_data.position.x -0.12)
        position_y = -(-grasp_pose_data.position.z +0.87 -0.2)
        position_z = grasp_pose_data.position.y +0.195

        # 姿勢（クォータニオン）情報を抽出
        #orientation_x = grasp_pose_data.orientation.x
        orientation_x = 0
        orientation_y = 1
        orientation_z = 0
        orientation_w = 0
        
        # Transformオブジェクトを作成
        target_transform = Transform(
            Rotation.from_quat([orientation_x, orientation_y, orientation_z, orientation_w]), 
            [position_x, position_y, position_z]
        )
        rospy.loginfo(f"target:{target_transform}")
        self.moveit.goto(target_transform)

        '''
        rospy.loginfo("Dropping object")
        self.moveit.goto([3.1415, -1.5708, -1.5708, -1.5708, 1.5708, 0.0])
        self.gripper.gripper_control(100)

        self.moveit.goto([1.7301 ,-1.5518, -1.9879, -0.2827, 1.5261, -0.0168])
        '''
    def scan_scene(self):
        self.moveit.goto([1.7301 ,-1.5518, -1.9879, -0.2827, 1.5261, -0.0168])
        self.reset_map()
        self.toggle_integration(True)
        for joint_target in self.scan_joints:
            self.moveit.goto(joint_target)
        self.toggle_integration(False)

    def execute_grasp(self, grasp):
        # Transform to base frame
        rospy.loginfo(f"res:{res}")
        rospy.loginfo(f"grasp.pose:{grasp.pose}")
        grasp.pose = self.T_base_task * grasp.pose
        rospy.loginfo(f"grasp:{grasp}")
        rospy.loginfo(f"grasp.pose:{grasp.pose}")
        
        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        T_base_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.05, 0])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.2, 0])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.moveit.goto(T_base_pregrasp * self.grasp_ee_offset, velocity_scaling=0.2)
        self.moveit.gotoL(T_base_grasp * self.grasp_ee_offset)
        self.gripper.gripper_control(0)
        self.moveit.gotoL(T_base_retreat * self.grasp_ee_offset)

        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.moveit.goto(T_base_lift * self.grasp_ee_offset)

        return 


if __name__ == "__main__":
    rospy.init_node("ur5e_grasp")
    controller = Ur5eGraspController()
    while True:
        controller.run()
