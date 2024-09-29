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

class UR5eMover:
    def __init__(self):
        # 初期化部分
        self.load_parameters()
        self.lookup_transforms()
        self.init_robot_connection()
        self.init_services()
        self.br = tf2_ros.TransformBroadcaster()

        # tf2バッファとリスナーの初期化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.sleep(1.0)
        rospy.loginfo("UR5e moving for drawer operation")

    def load_parameters(self):
        # パラメータ読み込み
        self.base_frame = rospy.get_param("/ur5e_drawer/base_frame_id", "base_link")
        self.ee_frame = rospy.get_param("/ur5e_drawer/ee_frame_id", "ee_link")
        self.task_frame = rospy.get_param("/ur5e_drawer/task_frame_id", "task_frame")
        #self.camera_frame = rospy.get_param("/ur5e_drawer/camera_frame_id", "camera_link")
        self.camera_frame = rospy.get_param("/ur5e_drawer/camera_frame_id", "camera_depth_frame")

        self.ee_grasp_offset = Transform.from_list(rospy.get_param("/ur5e_drawer/ee_grasp_offset", [0, 0, 0]))
        self.scan_joints = rospy.get_param("/ur5e_drawer/scan_joints", [])

    def lookup_transforms(self):
        # 座標変換の初期化
        tf.init()
        self.T_base_task = tf.lookup(self.base_frame, self.task_frame)

    def init_robot_connection(self):
        # ロボットとの接続初期化
        self.gripper = GripperController()
        self.moveit = MoveItClient("manipulator")
        self.moveit.move_group.set_end_effector_link(self.ee_frame)

    def init_services(self):
        # サービスの初期化
        self.reset_map = rospy.ServiceProxy("reset_map", Empty)
        self.toggle_integration = rospy.ServiceProxy("toggle_integration", SetBool)

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
                rospy.loginfo(f"平面の座標を取得（カメラ座標系）: x={plane_x}, y={plane_y}, z={plane_z}")
                rospy.loginfo(f"平面のクォータニオンを取得（カメラ座標系）: qx={qx}, qy={qy}, qz={qz}, qw={qw}")

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

                return trans_x, trans_y, trans_z, trans_qx, trans_qy, trans_qz, trans_qw
            else:
                rospy.logerr("平面の座標が無効です")
                return None
        except (rospy.ServiceException, tf2_ros.TransformException) as e:
            rospy.logerr(f"サービスまたは座標変換の呼び出しに失敗しました: {e}")
            return None

    def broadcast_plane_frame(self, plane_x, plane_y, plane_z, qx, qy, qz, qw):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.base_frame  # 親フレーム（ロボットのベースフレーム）
        t.child_frame_id = "detected_plane"  # 新しいフレーム名
        t.transform.translation.x = plane_x
        t.transform.translation.y = plane_y
        t.transform.translation.z = plane_z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.br.sendTransform(t)

    def run(self):
        # メイン動作
        self.gripper.gripper_control(100)
        self.moveit.goto([1.647, -2.568, -2.2, 1.624, 1.491, 0])
        result = self.get_plane_coordinates()
        if result is not None:
            plane_x, plane_y, plane_z, qx, qy, qz, qw = result
            rospy.loginfo(f"取得した平面の座標とクォータニオン（ベース座標系）: ({plane_x}, {plane_y}, {plane_z}), ({qx}, {qy}, {qz}, {qw})")
            
            # 座標系をブロードキャスト
            self.broadcast_plane_frame(plane_x, plane_y, plane_z, qx, qy, qz, qw)
            
            # 必要に応じて待機またはループ
            rospy.sleep(1.0)
        else:
            rospy.logerr("平面の座標とクォータニオンの取得に失敗しました")

if __name__ == "__main__":
    rospy.init_node("ur5e_move")
    controller = UR5eMover()
    controller.run()
