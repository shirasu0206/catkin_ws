#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DrawerDetection:
    def __init__(self):
        self.bridge = CvBridge()
        # RealSenseカメラの内部パラメータを設定（必要に応じて調整）
        self.cx = 312.6048889160156
        self.cy = 241.56346130371094
        self.fx = 594.2120971679688
        self.fy = 594.2120971679688

    def detect_drawer_type(self, segmentation_mask, depth_image_np, position, quaternion):
        # segmentation_maskがNumPy配列でない場合はエラーを発生させる
        if not isinstance(segmentation_mask, np.ndarray):
            rospy.logerr("Segmentation mask is not a NumPy array")
            return "Unknown"
        
        # depth_image_npもNumPy配列であることを確認
        if not isinstance(depth_image_np, np.ndarray):
            rospy.logerr("Depth image is not a NumPy array")
            return "Unknown"

        # 深度画像とセグメンテーションマスクから点群を取得
        points = self.extract_point_cloud(segmentation_mask, depth_image_np)

        if points.size == 0:
            rospy.logwarn("有効な点がありません。")
            return "Unknown"

        # 平面の方程式を取得
        plane_normal, plane_d = self.get_plane_equation(position, quaternion)

        # 平面からの距離を計算
        # 平面からの距離を計算し、手前（正の方向）か奥（負の方向）かを区別
        distances = self.calculate_point_plane_distances(points, plane_normal, plane_d)

        # 手前の2〜3cmにある点（正の距離）
        front_outlier_points = points[(distances > 0.005) & (distances < 0.04)]
        rospy.loginfo(f"手前の外れ点の数: {len(front_outlier_points)}")

        points = []
        invalid_depth_points = []  # 深度が取れていない点を保存するリスト
        for v in range(segmentation_mask.shape[0]):
            for u in range(segmentation_mask.shape[1]):
                if segmentation_mask[v, u] > 0:  # マスクされている領域のみ処理
                    z = depth_image_np[v, u]
                    if z > 0:
                        x = (u - self.cx) * z / self.fx / 1000.0
                        y = (v - self.cy) * z / self.fy / 1000.0
                        z = z / 1000.0
                        points.append([x, y, z])
                    else:
                        # 深度が取れていない点（z = 0）を追加
                        invalid_depth_points.append([u, v])

        points = np.array(points)
        invalid_depth_points = np.array(invalid_depth_points)  # 無効な点もNumPy配列に変換

        # 深度が取れていない点を表示
        rospy.loginfo(f"深度が取れていない点の数: {len(invalid_depth_points)}")

        # 奥の2〜3cmにある点（負の距離）
        back_outlier_points = points[(distances < -0.03) & (distances > -0.04)]

        # 無効な点も含める（無効な点はカメラ座標系で再構成するか、適切な扱いを決定）
        if invalid_depth_points.size > 0:
            # 無効な点の座標をカメラ座標系に変換する必要がある場合
            invalid_points = []
            for uv in invalid_depth_points:
                u, v = uv
                # z=0なので、仮にz=0として座標変換
                x = (u - self.cx) * 0 / self.fx
                y = (v - self.cy) * 0 / self.fy
                invalid_points.append([x, y, 0])
            back_outlier_points = np.concatenate([back_outlier_points, np.array(invalid_points)], axis=0)

        # 奥の外れ点の数を表示
        rospy.loginfo(f"奥の外れ点の数: {len(back_outlier_points)}")

        # 凸型の取っ手の検出
        if self.detect_convex_handle( front_outlier_points, segmentation_mask):
            rospy.loginfo("Convex型の取っ手を検出しました。")
            return "Convex"

        # 凹型の取っ手（穴）の検出
        if self.detect_concave_handle(back_outlier_points, segmentation_mask):  # ここでsegmentation_maskも渡す
            rospy.loginfo("Concave型の取っ手を検出しました。")
            return "Concave"

        # それ以外の場合はSideConcaveとする
        rospy.loginfo("SideConcave型の取っ手を検出しました。")
        return "SideConcave"


    def extract_point_cloud(self, segmentation_mask, depth_image_np):
        points = []
        for v in range(segmentation_mask.shape[0]):
            for u in range(segmentation_mask.shape[1]):
                if segmentation_mask[v, u] > 0:
                    z = depth_image_np[v, u]
                    if z > 0:
                        x = (u - self.cx) * z / self.fx / 1000.0
                        y = (v - self.cy) * z / self.fy / 1000.0
                        z = z / 1000.0
                        points.append([x, y, z])
        return np.array(points)

    def get_plane_equation(self, position, quaternion):
        # クォータニオンから法線ベクトルを取得
        import tf
        quaternion_list = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        euler = tf.transformations.euler_from_quaternion(quaternion_list)
        normal = np.array([0, 0, 1])
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion_list)[:3, :3]
        plane_normal = rotation_matrix.dot(normal)

        # 平面の方程式 ax + by + cz + d = 0 の d を計算
        plane_d = -np.dot(plane_normal, position)

        return plane_normal, plane_d

    def calculate_point_plane_distances(self, points, plane_normal, plane_d):
        # 平面から各点までの距離を計算
        distances = (np.dot(points, plane_normal) + plane_d) / np.linalg.norm(plane_normal)
        return distances

    def detect_convex_handle(self, convex_outlier_points, segmentation_mask):
        if len(convex_outlier_points) == 0:
            rospy.loginfo("convex=O")
            return False

        # クラスタリングを行い、大きなクラスターがあるかを確認
        clustering = DBSCAN(eps=0.015, min_samples=3).fit(convex_outlier_points)
        labels = clustering.labels_

        # ノイズでないクラスタの数を取得
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clustered_points = np.sum(labels != -1)  # クラスタに属する点数
        n_mask_points = np.sum(segmentation_mask > 0)  # セグメンテーションマスク内の全点数

        # 点数のログを出力
        rospy.loginfo(f"Convex: クラスタに属する点数: {n_clustered_points}, マスク内の点数: {n_mask_points}")

        # クラスタに属する点数がセグメンテーションマスクの全体点数の2.4%以上であればConvexと判断
        if n_clusters > 0 and (n_clustered_points / n_mask_points) > 0.00001:  # 閾値2.4%（調整可能）
            return True
        else:
            return False

    def detect_concave_handle(self, concave_outlier_points, segmentation_mask):
        if len(concave_outlier_points) == 0:
            rospy.loginfo("concave=O")
            return False

        # クラスタリングを行い、大きなクラスターがあるかを確認
        clustering = DBSCAN(eps=0.03, min_samples=3).fit(concave_outlier_points)
        labels = clustering.labels_

        # クラスタ内の点数とセグメンテーションマスク内の点数を取得
        n_clustered_points = np.sum(labels != -1)  # クラスタに属する点数
        n_mask_points = np.sum(segmentation_mask > 0)  # セグメンテーションマスク内の全点数

        # 点数のログを出力
        rospy.loginfo(f"Concave: クラスタに属する点数: {n_clustered_points}, マスク内の点数: {n_mask_points}")

        # クラスタ内の点数がセグメンテーションマスク内の全点数の1%以上であればConcaveと判断
        if n_clustered_points / n_mask_points > 0.01:  # 1%以上がクラスタに属していればConcave
            return True
        else:
            return False
