#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation as R

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

        # ROSメッセージをNumPy配列に変換
        try:
            segmentation_mask = self.bridge.imgmsg_to_cv2(segmentation_mask_msg, desired_encoding="mono8")
        except CvBridgeError as e:
            rospy.logerr("セグメンテーションマスクの変換に失敗しました: %s", str(e))
            return None, None

        # 型とサイズを確認
        if not isinstance(segmentation_mask, np.ndarray):
            rospy.logerr("セグメンテーションマスクはNumPy配列ではありません。")
            return None, None

        if len(segmentation_mask.shape) != 2:
            rospy.logerr("セグメンテーションマスクの形状が不正です。")
            return None, None

        # セグメンテーションマスクで領域を抽出
        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=segmentation_mask)

        # RANSACによる平面推定
        points = []
        for v in range(masked_depth.shape[0]):
            for u in range(masked_depth.shape[1]):
                z = masked_depth[v, u]
                if z > 0:
                    x = (u - self.cx) * z / self.fx / 1000.0  # ミリメートルからメートルに変換
                    y = (v - self.cy) * z / self.fy / 1000.0  # ミリメートルからメートルに変換
                    z = z / 1000.0  # z軸もミリメートルからメートルに変換
                    points.append([x, y, z])

        points = np.array(points)
        if len(points) == 0:
            rospy.logwarn("有効な点がありません。")
            return None

        # RANSACによる平面推定
        X = points[:, :2]  # x, y座標
        y = points[:, 2]   # z座標

        model = LinearRegression()
        ransac = RANSACRegressor(model, random_state=42)
        ransac.fit(X, y)

        # 推定された平面のパラメータを取得
        inlier_mask = ransac.inlier_mask_
        plane_points = points[inlier_mask]

        # 平面の法線ベクトルを計算
        centroid = np.mean(plane_points, axis=0)
        centered_points = plane_points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)  # 法線ベクトルを正規化

        # 平面の左下の座標を計算
        bottom_left_corner = np.array([
            np.min(plane_points[:, 0]),  # xが最小の値
            np.max(plane_points[:, 1]),  # yが最大の値
            np.min(plane_points[:, 2])   # zが最小の値（任意で調整可能）
        ]) # 最小のx, y, z座標を持つ点を取得

        # クォータニオンの計算
        reference_vector = np.array([0, 0, 1])  # 参照ベクトル（Z軸）

        # 回転軸の計算
        rotation_axis = np.cross(reference_vector, normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-8:
            # ベクトルが平行な場合
            if np.dot(reference_vector, normal) > 0:
                # 同じ方向（回転なし）
                rotation = R.from_quat([0, 0, 0, 1])
            else:
                # 逆方向（180度回転）
                rotation = R.from_rotvec(np.pi * np.array([1, 0, 0]))  # X軸周りに180度回転
        else:
            # 回転軸を正規化
            rotation_axis = rotation_axis / rotation_axis_norm
            # 回転角の計算
            rotation_angle = np.arccos(np.clip(np.dot(reference_vector, normal), -1.0, 1.0))
            # 回転を定義
            rotation = R.from_rotvec(rotation_angle * rotation_axis)

        # クォータニオンを取得（順序は[x, y, z, w]）
        qx, qy, qz, qw = rotation.as_quat()

        return bottom_left_corner, (qx, qy, qz, qw)  # 左下の座標とクォータニオンを返す
