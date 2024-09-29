#!/usr/bin/env python

import rospy
from drawer.srv import PlaneDetection, PlaneDetectionResponse  # PlaneDetectionに変更
from drawer.yolo_detection import YoloDetection 
from drawer.plane_detection import PlaneDetection as PlaneDetectionLogic  # 名前を変えてインポート
from drawer.drawer_detection import DrawerDetection  # DrawerDetectionのインポート
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DrawerServer:
    def __init__(self):
        # サービスを初期化
        self.service = rospy.Service('/start_detection', PlaneDetection, self.handle_detection)
        self.yolo_detector = YoloDetection()
        self.plane_detector = PlaneDetectionLogic()
        self.drawer_detector = DrawerDetection()  # DrawerDetectionのインスタンスを作成
        self.bridge = CvBridge()

    def handle_detection(self, req):
        rospy.loginfo("YOLOによるセグメンテーションを開始します。")

        # YOLOによるセグメンテーションを実行
        yolo_results = self.yolo_detector.handle_detection(req)

        # セグメンテーションマスクを取得
        segmentation_mask = yolo_results.mask  # 1つ目のセグメンテーションマスクを使用

        # segmentation_maskがROSのImageメッセージである場合、NumPy配列に変換
        if isinstance(segmentation_mask, Image):
            try:
                segmentation_mask = self.bridge.imgmsg_to_cv2(segmentation_mask, desired_encoding="mono8")
            except CvBridgeError as e:
                rospy.logerr(f"セグメンテーションマスクの変換に失敗しました: {e}")
                return PlaneDetectionResponse(
                    x=0.0, y=0.0, z=0.0,
                    qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                    width=0.0, height=0.0,
                    drawer_type="Unknown",
                    result="Segmentation mask conversion failed"
                )

        # segmentation_maskのコピーを作成
        segmentation_mask_for_plane = segmentation_mask.copy()

        # depth_image を取得
        try:
            depth_msg = rospy.wait_for_message('/camera/depth/image_rect_raw', Image, timeout=5.0)
            depth_image_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")  # Depth imageをNumPy配列に変換
        except CvBridgeError as e:
            rospy.logerr(f"Depth画像を取得できませんでした: {e}")
            return PlaneDetectionResponse(
                x=0.0, y=0.0, z=0.0,
                qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                width=0.0, height=0.0,
                drawer_type="Unknown",
                result="Depth image not available"
            )

        # 平面推定を実行（コピーを使用）
        position, quaternion, width, height = self.plane_detector.process_segmentation(segmentation_mask_for_plane, depth_image_np)

        # segmentation_maskの型を確認
        rospy.loginfo(f"セグメンテーションマスクの型（平面推定後）: {type(segmentation_mask)}")

        # 引き出しタイプの推定を実行（オリジナルのsegmentation_maskを使用）
        drawer_type = self.drawer_detector.detect_drawer_type(segmentation_mask, depth_image_np, position, quaternion)

        if position is not None:
            rospy.loginfo(f"平面推定: {position}, クォータニオン: {quaternion}, 幅: {width}, 高さ: {height}, 引き出しタイプ: {drawer_type}")
            return PlaneDetectionResponse(
                x=position[0],
                y=position[1],
                z=position[2],
                qx=quaternion[0],
                qy=quaternion[1],
                qz=quaternion[2],
                qw=quaternion[3],
                width=width,
                height=height,
                drawer_type=drawer_type,
                result="Detection successful"
            )
        else:
            rospy.logerr("平面推定に失敗しました")
            return PlaneDetectionResponse(
                x=0.0,
                y=0.0,
                z=0.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
                width=0.0,
                height=0.0,
                drawer_type="Unknown",
                result="Plane estimation failed"
            )

if __name__ == '__main__':
    rospy.init_node('drawer_server')
    server = DrawerServer()
    rospy.spin()
