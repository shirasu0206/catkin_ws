cmake_minimum_required(VERSION 3.0.2)
project(drawer)

## 必要なパッケージを指定
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs  # sensor_msgs を追加
  visualization_msgs
  message_generation
)

## Pythonパッケージの設定
catkin_python_setup()

## サービスファイルを追加
add_service_files(
  FILES
  YoloDetection.srv     # 追加
  PlaneDetection.srv     # 追加
  DrawerDetection.srv    # 追加
  Convex.srv             # 追加
  Concave.srv            # 追加
  SideConcave.srv       # 追加
)

## 依存関係を設定
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs  # sensor_msgs を追加
)

## catkin specific configuration ##
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime  # sensor_msgs と message_runtime を追加
)

## スクリプトのインストール
catkin_install_python(PROGRAMS
  nodes/drawer_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Pythonモジュールのインストール（srcディレクトリ内のPythonコードをインストール）
install(DIRECTORY src/drawer/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/drawer
)

## Launchファイルのインストール
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## インクルードディレクトリの指定
include_directories(
  ${catkin_INCLUDE_DIRS}
)
