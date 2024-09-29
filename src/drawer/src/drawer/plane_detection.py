#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

class PlaneDetection:
    def __init__(self):
        self.bridge = CvBridge()
        # RealSenseカメラの内部パラメータを設定
        self.cx = 312.6048889160156  # 画像の中心x座標
        self.cy = 241.56346130371094  # 画像の中心y座標
        self.fx = 594.2120971679688  # x軸の焦点距離
        self.fy = 594.2120971679688  # y軸の焦点距離

    def process_segmentation(self, segmentation_mask_msg, depth_image):
        rospy.loginfo("セグメンテーションマスクを受信しました。")

        # ROSメッセージがNumPy配列でない場合のみ変換
        if isinstance(segmentation_mask_msg, Image):
            try:
                segmentation_mask = self.bridge.imgmsg_to_cv2(segmentation_mask_msg, desired_encoding="mono8")
            except CvBridgeError as e:
                rospy.logerr("セグメンテーションマスクの変換に失敗しました: %s", str(e))
                return None, None, None, None
        else:
            segmentation_mask = segmentation_mask_msg

        # 型とサイズを確認
        if not isinstance(segmentation_mask, np.ndarray):
            rospy.logerr("セグメンテーションマスクはNumPy配列ではありません。")
            return None, None, None, None

        if len(segmentation_mask.shape) != 2:
            rospy.logerr("セグメンテーションマスクの形状が不正です。")
            return None, None, None, None

        # セグメンテーションマスクで領域を抽出
        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=segmentation_mask)

        # 有効な深度値の点を収集
        points = []
        for v in range(0, masked_depth.shape[0], 2):  # 解像度を1/2にして処理速度を向上
            for u in range(0, masked_depth.shape[1], 2):  # 同上
                z = masked_depth[v, u]
                if z > 0 and z < 10000:  # 10m未満の有効な深度値のみ使用
                    x = (u - self.cx) * z / self.fx / 1000.0
                    y = (v - self.cy) * z / self.fy / 1000.0
                    z = z / 1000.0
                    points.append([x, y, z])

        points = np.array(points)
        if len(points) == 0:
            rospy.logwarn("有効な点がありません。")
            return None, None, None, None

        # RANSACによる平面推定
        X = points[:, :2]
        y = points[:, 2]

        model = LinearRegression()
        ransac = RANSACRegressor(model, random_state=42, max_trials=1000000, residual_threshold=0.002)

        ransac.fit(X, y)

        # 平面の大きさを計算
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])

        # 平面のパラメータを計算
        inlier_mask = ransac.inlier_mask_
        plane_points = points[inlier_mask]
        centroid = np.mean(plane_points, axis=0)

        # 平面の法線ベクトルを計算
        centered_points = plane_points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)

        # 左下隅の座標を計算 (x, y, z)
        bottom_left_corner = np.array([
            np.min(plane_points[:, 0]),  # X方向で最小の値
            np.max(plane_points[:, 1]),  # Y方向で最大の値（カメラ座標系で手前方向）
            np.min(plane_points[:, 2])   # Z方向で最小の値（奥行き）
        ])

        # クォータニオンの計算
        reference_vector = np.array([0, 0, 1])  # カメラ座標系での基準ベクトル（Z軸方向）
        rotation_axis = np.cross(reference_vector, normal)  # 法線ベクトルと基準ベクトルの外積
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-8:
            if np.dot(reference_vector, normal) > 0:
                rotation = R.from_quat([0, 0, 0, 1])
            else:
                rotation = R.from_rotvec(np.pi * np.array([1, 0, 0]))  # X軸周りに180度回転
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(reference_vector, normal), -1.0, 1.0))
            rotation = R.from_rotvec(rotation_angle * rotation_axis)

        # 最終クォータニオンを取得
        qx, qy, qz, qw = rotation.as_quat()

        return bottom_left_corner, (qx, qy, qz, qw), width, height
