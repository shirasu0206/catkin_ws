#!/usr/bin/env python

import rospy
from drawer.srv import PlaneDetection, PlaneDetectionResponse  # PlaneDetectionに変更
from drawer.yolo_detection import YoloDetection 
from drawer.plane_detection import PlaneDetection as PlaneDetectionLogic  # 名前を変えてインポート
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DrawerServer:
    def __init__(self):
        # サービスを初期化
        self.service = rospy.Service('/start_detection', PlaneDetection, self.handle_detection)  # サービス名をPlaneDetectionに変更
        self.yolo_detector = YoloDetection()
        self.plane_detector = PlaneDetectionLogic()  # PlaneDetectionクラスのインスタンスを作成
        self.bridge = CvBridge()

    def handle_detection(self, req):
        rospy.loginfo("YOLOによるセグメンテーションを開始します。")

        # YOLOによるセグメンテーションを実行
        yolo_results = self.yolo_detector.handle_detection(req)

        # セグメンテーションマスクを取得
        segmentation_mask = yolo_results.mask  # 1つ目のセグメンテーションマスクを使用

        # depth_image を取得
        try:
            depth_msg = rospy.wait_for_message('/camera/depth/image_rect_raw', Image, timeout=5.0)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")  # Depth imageのエンコーディング
        except rospy.ROSException as e:
            rospy.logerr(f"Depth画像を取得できませんでした: {e}")
            return PlaneDetectionResponse(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0, width=0.0, height=0.0, result="Depth image not available")

        # 平面推定を実行
        position, quaternion, width, height = self.plane_detector.process_segmentation(segmentation_mask, depth_image)

        if position is not None:
            rospy.loginfo(f"平面推定: {position}, クォータニオン: {quaternion}, 幅: {width}, 高さ: {height}")
            return PlaneDetectionResponse(
                x=position[0], 
                y=position[1], 
                z=position[2], 
                qx=quaternion[0], 
                qy=quaternion[1], 
                qz=quaternion[2], 
                qw=quaternion[3], 
                width=width,   # 幅を返す
                height=height, # 高さを返す
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
                width=0.0,   # 幅をデフォルトで0として返す
                height=0.0,  # 高さをデフォルトで0として返す
                result="Plane estimation failed"
            )

if __name__ == '__main__':
    rospy.init_node('drawer_server')
    server = DrawerServer()
    rospy.spin()
