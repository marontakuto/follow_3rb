# goal_3rb

Gazeboで群ロボットの回避行動の獲得を実現するためのリポジトリです。
- ロボット3台
- turtlebot3 burger
- Python 3.8.10
- 深層強化学習
- Optuna(上位群の割合が可変的なTPE)
- ハイパーパラメータと報酬関数の最適化


## 環境構築
1. [こちらのサイト](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)へアクセス(章番号の表示で3.1.1が1.1.1のようになっていることがあります)
2. [3. Quick Start Guide]上部の「Kinetic」「Melodic」「Noetic」... より「Noetic」を選択し、3.1.1~3.1.4のコマンドを順に実行する(3.1.4の「Click here to expand more details about building TurtleBot3 package from source.」も実行する)
3. [9. Machine Learning]上部の「Kinetic」「Melodic」「Noetic」... より「Melodic」を選択し、9.1.5のコマンドを順番に入力する(numpyのコマンドは無視)
4. [6. Simulation]6.1.2のコマンドを実行してワールドとturtlebot3 burgerが出力されればOK
5. cd ~/catkin_ws/turtlebot3_learning_machine/turtlebot3_dqn
6. git clone https://github.com/marontakuto/goal_3rb.git
7. ターミナルを2つ開く
8. roslaunch turtlebot3_dqn world_goal_3rb.launch # 1つ目のターミナルで実行
9. roslaunch turtlebot3_dqn exe_goal_3rb.launch # 2つ目のターミナルで実行
10. 実機で動かす場合はIPアドレスなどの設定のために以下を実行してください
11. sudo nano .bashrc # 実行後に最下層を以下のように変更
<pre>
export ROS_MASTER_URI=http://'your_pc_ip':11311 # <your_pc_ip>にはifconfigを実行して確認した「wl...:」のIPアドレスを入力
export ROS_HOSTNAME='your_pc_ip'
</pre>
12. source .bashrc # 変更内容の適用
