
#!/usr/bin/env python

import geometry_msgs.msg
import numpy as np
import rospy
from drawer.srv import PlaneDetection  # PlaneDetectionをインポート

from std_srvs.srv import Empty, SetBool

import tf2_ros
import tf2_geometry_msgs  # これが必要

from robot_helpers.ros.conversions import *
from robot_helpers.ros.robotiq_gripper import GripperController
from robot_helpers.ros.moveit import MoveItClient
from robot_helpers.ros import tf
from robot_helpers.spatial import Rotation, Transform
from vgn.rviz import Visualizer
from vgn.srv import GetMapCloud, GetSceneCloud, PredictGrasps, PredictGraspsRequest
from vgn.utils import from_grasp_config_msg

class UR5eMover:
    def __init__(self):
        # 初期化部分
        self.load_parameters()

        # tf2バッファとリスナーの初期化を先に行う
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lookup_transforms()  # その後、lookup_transforms を呼び出す

        self.init_robot_connection()
        self.init_services()
        self.br = tf2_ros.TransformBroadcaster()

        self.vis = Visualizer()  

        rospy.sleep(1.0)
        rospy.loginfo("UR5e moving for drawer operation")

    def load_parameters(self):
        # パラメータ読み込み
        self.base_frame = rospy.get_param("/ur5e_drawer/base_frame_id")
        self.ee_frame = rospy.get_param("/ur5e_drawer/ee_frame_id")
        self.task_frame = rospy.get_param("/ur5e_drawer/task_frame_id")
        #self.camera_frame = rospy.get_param("/ur5e_drawer/camera_frame_id", "camera_link")
        self.camera_frame = rospy.get_param("/ur5e_drawer/camera_frame_id", "camera_depth_frame")

        self.grasp_ee_offset = Transform.from_list(rospy.get_param("/ur5e_drawer/ee_grasp_offset", [0, 0, 0.7071068, 0.7071068, 0.0, 0.0, 0]))
        self.scan_joints = rospy.get_param("/ur5e_drawer/scan_joints", [])

    def init_robot_connection(self):
        # ロボットとの接続初期化
        self.gripper = GripperController()
        self.moveit = MoveItClient("manipulator")
        self.moveit.move_group.set_end_effector_link(self.ee_frame)

    def init_services(self):
        # サービスの初期化
        self.reset_map = rospy.ServiceProxy("reset_map", Empty)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)
        self.get_scene_cloud = rospy.ServiceProxy("get_scene_cloud", GetSceneCloud)
        self.get_map_cloud = rospy.ServiceProxy("get_map_cloud", GetMapCloud)
        self.predict_grasps = rospy.ServiceProxy("predict_grasps", PredictGrasps)

    def get_plane_coordinates(self):
        rospy.wait_for_service('/start_detection')
        try:
            detection_service = rospy.ServiceProxy('/start_detection', PlaneDetection)
            response = detection_service()
            if response.x != 0.0 or response.y != 0.0 or response.z != 0.0:
                plane_x = response.x
                plane_y = response.y
                plane_z = response.z
                qx = response.qx
                qy = response.qy
                qz = response.qz
                qw = response.qw
                width = response.width
                height = response.height
                drawer_type = response.drawer_type  # 引き出しの種類を取得
                rospy.loginfo(f"平面の座標を取得（カメラ座標系）: x={plane_x}, y={plane_y}, z={plane_z}")
                rospy.loginfo(f"平面のクォータニオンを取得（カメラ座標系）: qx={qx}, qy={qy}, qz={qz}, qw={qw}")
                rospy.loginfo(f"平面のサイズ: 幅={width} m, 高さ={height} m")
                rospy.loginfo(f"引き出しの種類: {drawer_type}")

                # PoseStampedメッセージを作成
                pose_stamped = geometry_msgs.msg.PoseStamped()
                pose_stamped.header.frame_id = self.camera_frame
                pose_stamped.header.stamp = rospy.Time(0)
                pose_stamped.pose.position.x = plane_x
                pose_stamped.pose.position.y = plane_y
                pose_stamped.pose.position.z = plane_z
                pose_stamped.pose.orientation.x = qx
                pose_stamped.pose.orientation.y = qy
                pose_stamped.pose.orientation.z = qz
                pose_stamped.pose.orientation.w = qw

                # 座標変換を実行
                target_frame = self.base_frame
                transformed_pose = self.tf_buffer.transform(pose_stamped, target_frame, rospy.Duration(1.0))

                # 変換後の座標とクォータニオンを取得
                trans_x = transformed_pose.pose.position.x
                trans_y = transformed_pose.pose.position.y
                trans_z = transformed_pose.pose.position.z
                trans_qx = transformed_pose.pose.orientation.x
                trans_qy = transformed_pose.pose.orientation.y
                trans_qz = transformed_pose.pose.orientation.z
                trans_qw = transformed_pose.pose.orientation.w

                rospy.loginfo(f"平面の座標を取得（ベース座標系）: x={trans_x}, y={trans_y}, z={trans_z}")
                rospy.loginfo(f"平面のクォータニオンを取得（ベース座標系）: qx={trans_qx}, qy={trans_qy}, qz={trans_qz}, qw={trans_qw}")

                return trans_x, trans_y, trans_z, trans_qx, trans_qy, trans_qz, trans_qw, width, height, drawer_type
            else:
                rospy.logerr("平面の座標が無効です")
                return None
        except (rospy.ServiceException, tf2_ros.TransformException) as e:
            rospy.logerr(f"サービスまたは座標変換の呼び出しに失敗しました: {e}")
            return None

    def broadcast_plane_frame(self, plane_x, plane_y, plane_z, qx, qy, qz, qw):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.base_frame  # ベースフレーム
        t.child_frame_id = "detected_plane"  # 新しいフレーム名
        t.transform.translation.x = plane_x
        t.transform.translation.y = plane_y
        t.transform.translation.z = plane_z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.br.sendTransform(t)

    def handle_convex_drawer(self):
        rospy.loginfo("Convexタイプの引き出しに対するtask_frameを設定中...")
        result = self.get_plane_coordinates()
        if result is not None:
            plane_x, plane_y, plane_z, qx, qy, qz, qw, width, height, drawer_type = result
            self.broadcast_task_frame(plane_x, plane_y, plane_z, qx, qy, qz, qw)
            rospy.loginfo(f"task_frame を設定しました: 位置=({plane_x}, {plane_y}, {plane_z}), 姿勢=({qx}, {qy}, {qz}, {qw})")
        else:
            rospy.logerr("平面の座標を取得できませんでした。task_frame の設定に失敗しました。")
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
    def broadcast_task_frame(self, plane_x, plane_y, plane_z, qx, qy, qz, qw):
        """
        task_frame を指定された平面の座標でブロードキャストします。
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.base_frame  # ベースフレームからの相対位置
        t.child_frame_id = self.task_frame  # task_frame を再設定

        # 平面の座標とクォータニオンを設定
        t.transform.translation.x = plane_x -0.075
        t.transform.translation.y = plane_y -0.02
        t.transform.translation.z = plane_z
        t.transform.rotation.x = qx
        t.transform.rotation.y = abs(qy)
        t.transform.rotation.z = abs(qz)
        t.transform.rotation.w = qw

        # task_frame の変換をブロードキャスト
        self.br.sendTransform(t)  # selfを使用してブロードキャスト
        rospy.loginfo(f"task_frame をブロードキャストしました: 位置=({plane_x}, {plane_y}, {plane_z}), 姿勢=({qx}, {qy}, {qz}, {qw})")

    def lookup_transforms(self):
        # task_frame の座標を取得する
        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, self.task_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # 手動でTransformオブジェクトに変換
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self.T_base_task = Transform(
                Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]), 
                [translation.x, translation.y, translation.z]
            )

            rospy.loginfo(f"T_base_task: {self.T_base_task}")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"座標変換の取得に失敗しました: {e}")



    def scan_scene(self):
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        self.reset_map()
        self.toggle_integration(True)
        for joint_target in self.scan_joints:
            self.moveit.goto(joint_target)
        self.toggle_integration(False)

    def execute_grasp(self, grasp):
        grasp.pose = self.T_base_task * grasp.pose
        
        # Ensure that the camera is pointing forward.
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        T_base_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.15, 0])
        T_grasp_grasp = Transform(Rotation.identity(), [0.0, 0.06, 0])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.18, 0])
        T_base_pregrasp = T_grasp_pregrasp * T_base_grasp
        T_base_retreat = T_grasp_retreat * T_base_grasp
        T_base_grasp = T_grasp_grasp * T_base_grasp
        
        self.gripper.gripper_control(100)
        self.moveit.goto(T_base_pregrasp * self.grasp_ee_offset, velocity_scaling=0.2)
        self.moveit.gotoL(T_base_grasp * self.grasp_ee_offset)
        self.gripper.gripper_control(0)
        self.moveit.gotoL(T_base_retreat * self.grasp_ee_offset)
        self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        '''
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
        
        '''

        return 


    def run(self):
        self.vis.clear()
        #self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2 , 1.624, 1.491, 0])
        result = self.get_plane_coordinates()
        if result is not None:
            plane_x, plane_y, plane_z, qx, qy, qz, qw, width, height, drawer_type = result
            rospy.loginfo(f"取得した平面の座標とクォータニオン（ベース座標系）: ({plane_x}, {plane_y}, {plane_z}), ({qx}, {qy}, {qz}, {qw})")
            rospy.loginfo(f"取得した平面のサイズ: 幅={width} m, 高さ={height} m")
            rospy.loginfo(f"取得した引き出しの種類: {drawer_type}")

            # 座標系をブロードキャスト
            self.broadcast_plane_frame(plane_x, plane_y, plane_z, qx, qy, qz, qw)

            # 取得した引き出しの種類に応じて動作を変更
            if drawer_type == "Convex":
                rospy.loginfo("Convexタイプの引き出しを検出しました。")
                rospy.loginfo("Convexタイプの引き出しに対する処理を開始します。")
                self.handle_convex_drawer()
                
                # ここにロボットの動作を記述
            elif drawer_type == "Concave":
                # Concaveタイプの引き出しに対する処理
                rospy.loginfo("Concaveタイプの引き出しを検出しました。")
                # ここにロボットの動作を記述
            elif drawer_type == "SideConcave":
                # SideConvexタイプの引き出しに対する処理
                rospy.loginfo("SideConcaveタイプの引き出しを検出しました。")
                # ここにロボットの動作を記述
            else:
                rospy.logwarn("未知の引き出しタイプです。")
                # 必要に応じてエラー処理やデフォルトの動作を追加

            # 必要に応じて待機またはループ
            rospy.sleep(1.0)
        else:
            rospy.logerr("平面の座標とクォータニオンの取得に失敗しました")

if __name__ == "__main__":
    rospy.init_node("ur5e_move")
    controller = UR5eMover()
    controller.run()
