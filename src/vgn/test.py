import rospy
from robot_helpers.spatial import Transform, Rotation
from robot_helpers.ros.moveit import MoveItClient

class SimpleMoveItController:
    def __init__(self):
        # MoveItClientを初期化
        self.moveit = MoveItClient("manipulator")

    def get_current_pose(self):
        # 現在のエンドエフェクタの位置と姿勢を取得
        current_pose = self.moveit.move_group.get_current_pose().pose
        
        # 位置と姿勢をログに表示
        rospy.loginfo(f"Current Position: x={current_pose.position.x}, y={current_pose.position.y}, z={current_pose.position.z}")
        rospy.loginfo(f"Current Orientation: x={current_pose.orientation.x}, y={current_pose.orientation.y}, z={current_pose.orientation.z}, w={current_pose.orientation.w}")
        
        # Transformオブジェクトとして返す（必要に応じて）
        current_transform = Transform(
            Rotation.from_quat([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]), 
            [current_pose.position.x, current_pose.position.y, current_pose.position.z]
        )
        
        return current_transform

if __name__ == "__main__":
    # ROSノードの初期化
    rospy.init_node("simple_moveit_controller")

    # SimpleMoveItControllerのインスタンスを作成
    controller = SimpleMoveItController()

    # 現在のエンドエフェクタの位置と姿勢を取得して表示
    current_transform = controller.get_current_pose()

    # ここではROSスピンは必要ありませんが、他のROS操作をする場合はスピンを使います
    # rospy.spin()
