import subprocess
import time

# コマンドを順番に実行
proc1 = subprocess.Popen(["roslaunch", "vgn", "ur5e_drawer_grasp.launch"])
time.sleep(2)  # 5秒待つ（必要に応じて調整）
proc2 = subprocess.Popen(["python3", "scripts/ur5e_drawer_grasp.py"])

# キー入力待ち
input("プロセスを終了するにはEnterキーを押してください...")

# プロセスを終了

proc1.terminate()
proc2.terminate()
# プロセスが完全に終了するまで待つ

proc1.wait()
proc2.wait()

print("全てのプロセスが終了しました。")

# コマンドを順番に実行
proc3 = subprocess.Popen(["roslaunch", "vgn", "ur5e_inside_grasp.launch"])
time.sleep(2)  # 5秒待つ（必要に応じて調整）
proc4 = subprocess.Popen(["python3", "scripts/ur5e_inside_grasp.py"])


# プロセスを終了
proc3.terminate()
proc4.terminate()

# プロセスが完全に終了するまで待つ
proc3.wait()
proc4.wait()
