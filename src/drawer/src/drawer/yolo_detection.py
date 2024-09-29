#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
from drawer.srv import Detection, DetectionResponse
import numpy as np

class YoloDetection:
    def __init__(self):
        model_path = "/home/syu_kato/catkin_ws/src/drawer/assets/models/best.pt"
        self.model = YOLO(model_path, task="segment")  # "segment" タスクを指定
        self.bridge = CvBridge()
        self.save_dir = '/home/syu_kato/catkin_ws/src/drawer/result/yolo'
        os.makedirs(self.save_dir, exist_ok=True)

    def handle_detection(self, req):
        rospy.loginfo("セグメンテーションリクエストを受信しました")

        # カメラから1枚の画像を取得
        img_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5.0)
        image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        # YOLOによるセグメンテーションを実行
        results = self.model(image)

        # マスクを取得
        if len(results[0].masks.data) > 1:
            selected_mask = results[0].masks.data[1].cpu().numpy()  # 2番目のマスクを選択
        else:
            selected_mask = results[0].masks.data[0].cpu().numpy()  # マスクが1つの場合、最初のものを使う

        rospy.loginfo(f"セグメンテーションマスクの型: {type(selected_mask)}")
        rospy.loginfo(f"セグメンテーションマスクの形状: {selected_mask.shape}")

        # マスクを0-255の範囲にスケールし、uint8型に変換
        selected_mask = (selected_mask * 255).astype(np.uint8)

        # セグメンテーションマスクを保存
        result_path = os.path.join(self.save_dir, "segmented_result.jpg")
        annotated_image = results[0].plot()  # 結果を描画
        cv2.imwrite(result_path, annotated_image)  # セグメンテーション画像を保存
        rospy.loginfo(f"セグメンテーション結果を {self.save_dir} に保存しました")

        # マスクをROSのImage形式に変換
        mask_image = self.bridge.cv2_to_imgmsg(selected_mask, encoding="mono8")

        # マスクをレスポンスとして返す
        return DetectionResponse(mask=mask_image, result="Segmentation successful")


if __name__ == "__main__":
    rospy.init_node('yolo_detection')
    server = YoloDetection()
    rospy.spin()
