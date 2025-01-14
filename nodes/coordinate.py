import matplotlib.pyplot as plt
import numpy as np
import os
import json


# JSON形式の座標を読み込む関数
def load_coordinates(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.loads(file.read().strip())  # JSON形式として読み込み
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON data in file: {filepath}")
        return []


# ロボットの軌跡をプロットする関数
def plot_robot_trajectory(robot_coords, color, label, arrow_step=4, scale=0.6, markersize=0, headwidth=7):
    x, y = zip(*robot_coords)
    plt.plot(x, y, marker='o', label=label, linestyle='-', color=color, markersize=markersize)
    # 矢印を間引いて追加
    for i in range(0, len(robot_coords) - 1, arrow_step):  # `arrow_step` 間隔で矢印を描画
        dx = robot_coords[i + 1][0] - robot_coords[i][0]
        dy = robot_coords[i + 1][1] - robot_coords[i][1]
        plt.quiver(robot_coords[i][0], robot_coords[i][1], dx, dy, angles='xy', scale_units='xy', 
                   scale=scale, color=color, alpha=0.9, headwidth=headwidth)

if __name__ == '__main__':
    # ディレクトリが存在しない場合は作成
    Path = os.path.dirname(os.path.realpath(__file__)) + '/result'
    if not os.path.exists(Path):
        os.makedirs(Path)

    # テキストファイルから座標を読み込む
    robot_0 = load_coordinates(os.path.join(Path, 'coordinate_robot0.txt'))
    robot_1 = load_coordinates(os.path.join(Path, 'coordinate_robot1.txt'))
    robot_2 = load_coordinates(os.path.join(Path, 'coordinate_robot2.txt'))

    # プロット設定
    plt.figure(figsize=(6, 6))

    for i in range(len(robot_0)):
        # 毎回図をクリア
        plt.clf()

        # 各ロボットのプロット
        plot_robot_trajectory(robot_0[i], 'purple', 'Robot 0')
        plot_robot_trajectory(robot_1[i], 'green', 'Robot 1')
        plot_robot_trajectory(robot_2[i], 'orange', 'Robot 2')

        # グラフの範囲設定
        plt.xlim(-0.9, 0.9)
        plt.ylim(0, 1.8)
        plt.xticks(np.arange(-0.9, 0.91, 0.3))  # X軸の目盛りを設定
        plt.yticks(np.arange(0, 1.81, 0.3))  # X軸の目盛りを設定

        # 線の追加
        plt.plot([0.6, 0.9], [0.0, 0.3], linestyle='-', color='orange', linewidth=3) # robot2のゴール
        plt.plot([-0.6, -0.9], [0.0, 0.3], linestyle='-', color='green', linewidth=3) # robot1のゴール
        plt.plot([0.6, 0.9], [1.8, 1.5], linestyle='-', color='red', linewidth=3) # robot3のゴール
        plt.plot([-0.6, -0.9], [1.8, 1.5], linestyle='-', color='purple', linewidth=3) # robot0のゴール

        # グラフのラベルとタイトル
        plt.xlabel('X Coordinate [m]')
        plt.ylabel('Y Coordinate [m]')
        plt.title('Robot Trajectories')
        plt.legend()
        plt.grid(True)

        # 枠線を太くする
        ax = plt.gca()
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # グラフの保存
        plt.savefig(f'{Path}/plot_{i}.png')
