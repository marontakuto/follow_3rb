#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import random
import math
import time
from math import pi
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from collections import deque
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import *
import cv2
from sensor_msgs.msg import Image, CompressedImage
import ros_numpy

class Env():
    def __init__(self, mode, robot_n, lidar_num, input_list, teleport, 
                 r_collision, r_just, r_near, r_goal, r_cost, r_passive, 
                 Target, mask_switch, display_image_normal, display_image_mask, 
                 display_rb):
        
        self.mode = mode
        self.robot_n = robot_n
        self.lidar_num = lidar_num
        self.input_list = input_list
        self.teleport = teleport
        self.previous_cam_list = deque([])
        self.previous_lidar_list = deque([])
        self.previous2_cam_list = deque([])
        self.previous2_lidar_list = deque([])
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # カメラ画像のマスク処理
        self.mask_switch = mask_switch
        self.display_image_normal = display_image_normal
        self.display_image_mask = display_image_mask
        self.display_rb = display_rb

        # Optunaで選択された報酬値
        self.r_collision = r_collision
        self.r_just = r_just
        self.r_near = r_near
        self.r_goal = r_goal
        self.r_cost = r_cost
        self.r_passive = r_passive
        self.Target = Target

        # LiDARについての設定
        self.lidar_max = 2 # 対象のworldにおいて取りうるlidarの最大値(simの貫通対策や正規化に使用)
        self.lidar_min = 0.12 # lidarの最小測距値[m]
        self.range_margin = self.lidar_min + 0.03 # 衝突として処理される距離[m] 0.02

        # 初期のゴールの色
        if self.robot_n == 0:
            self.goal_color = 'purple'
        elif self.robot_n == 1:
            self.goal_color = 'green'
        elif self.robot_n == 2:
            self.goal_color = 'yellow'
        
        self.previous_goal = None

    def get_lidar(self, retake=False): # lidar情報の取得
        if retake:
            self.scan = None
        
        if self.scan is None:

            scan = None
            while scan is None:
                try:
                    scan = rospy.wait_for_message('scan', LaserScan, timeout=1) # LiDAR値の取得(1deg刻み360方向の距離情報を取得)
                except:
                    self.stop()
                    pass
            
            data_range = [] # 取得したLiDAR値を修正して格納するリスト
            for i in range(len(scan.ranges)):
                if scan.ranges[i] == float('Inf'): # 最大より遠いなら3.5(LiDARの規格で取得できる最大値)
                    data_range.append(3.5)
                if np.isnan(scan.ranges[i]): # 最小より近いなら0
                    data_range.append(0)
                
                if self.mode == 'sim':
                    if scan.ranges[i] > self.lidar_max: # フィールドで観測できるLiDAR値を超えていたら0
                        data_range.append(0)
                    else:
                        data_range.append(scan.ranges[i]) # 取得した値をそのまま利用
                else:
                    data_range.append(scan.ranges[i]) # 実機では取得した値をそのまま利用

            # lidar値を[360/(self.lidar_num)]deg刻み[self.lidar_num]方向で取得
            use_list = [] # 計算に利用するLiDAR値を格納するリスト
            for i in range(self.lidar_num):
                index = (len(data_range) // self.lidar_num) * i
                scan = max(data_range[index - 2], data_range[index - 1], data_range[index], data_range[index + 1], data_range[index + 2]) # 実機の飛び値対策(値を取得できず0になる場合があるため前後2度で最大の値を採用)
                use_list.append(scan)
            
            self.scan = use_list

        return self.scan

    def get_camera(self, retake=False): # camera画像取得

        if retake:
            self.img = None
        
        if self.img is None:
            img = None
            while img is None:
                try:
                    if self.mode == 'sim':
                       img = rospy.wait_for_message('usb_cam/image_raw', Image, timeout=1) # シミュレーション用(生データ)
                    else:
                       img = rospy.wait_for_message('usb_cam/image_raw/compressed', CompressedImage, timeout=1) # 実機用(圧縮データ)
                except:
                    self.stop()
                    pass
            
            if self.mode == 'sim':
                img = ros_numpy.numpify(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # カラー画像(BGR)
            else:
                img = np.frombuffer(img.data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR) # カラー画像(BGR)
            
            img = cv2.resize(img, (48, 27)) # 取得した画像を48×27[pixel]に変更

            if self.display_image_normal and self.robot_n in self.display_rb:
                self.display_image(img, f'camera_normal_{self.robot_n}')
            
            self.img = img

        return self.img
    
    def getState(self): # 情報取得

        collision = False
        goal = False
        state_list = [] # 入力する情報を格納するリスト

        ### 画像の取得と処理 ###
        img = self.get_camera() # カメラ画像の取得
        img, goal_num = self.goal_mask(img) # 目標ゴールを緑に, 他のゴールを黒に変換 + ゴールの画素数取得

        if goal_num > 300:
            goal = True

            # 1step前もゴールだった場合は前回と同じ画像を読み込んでいる可能性があるため再度画像取得する
            if self.previous_goal:
                while goal_num > 300:
                    img = self.get_camera(retake=True)
                    img, goal_num = self.goal_mask(img)
                goal = False

            # 目標のゴールを反対側のゴールに設定
            if goal and not self.teleport:
                if self.goal_color == 'red':
                    self.goal_color = 'green'
                elif self.goal_color == 'green':
                    self.goal_color = 'red'
                elif self.goal_color == 'yellow':
                    self.goal_color = 'purple'
                elif self.goal_color == 'purple':
                    self.goal_color = 'yellow'
        
        self.previous_goal = goal
        
        #### LiDAR情報の取得と処理 ###
        scan = self.get_lidar() # LiDAR値の取得

        if self.range_margin >= min(scan):
            collision = True
            if self.mode == 'real': # 実機実験におけるLiDARの飛び値の処理
                scan_true = [element_cont for element_num, element_cont in enumerate(scan) if element_cont != 0]
                if scan.count(0) >= 1 and self.range_margin < min(scan_true): # (飛び値が存在する)and(飛び値を除いた場合は衝突判定ではない)
                    collision = False
        
        # 入力するカメラ画像の処理
        if ('cam' in self.input_list) or ('previous_cam' in self.input_list) or ('previous2_cam' in self.input_list):
            input_img = np.asarray(img, dtype=np.float32)
            input_img /= 255.0 # 画像の各ピクセルを255で割ることで0~1の値に正規化
            input_img = np.asarray(input_img.flatten())
            input_img = input_img.tolist()

            state_list = state_list + input_img # [現在]

            if 'previous_cam' in self.input_list:
                self.previous_cam_list.append(input_img)
                if len(self.previous_cam_list) > 2: # [1step前 現在]の状態に保つ
                    self.previous_cam_list.popleft() # 左端の要素を削除(2step前の情報を削除)
                past_cam = self.previous_cam_list[0] # 1step前の画像
                state_list = state_list + past_cam # [現在 1step前]

                if 'previous2_cam' in self.input_list:
                    self.previous2_cam_list.append(input_img)
                    if len(self.previous2_cam_list) > 3: # [2step前 1step前 現在]の状態に保つ
                        self.previous2_cam_list.popleft() # 左端の要素を削除(3step前の情報を削除)
                    past_cam = self.previous2_cam_list[0] # 2step前の画像
                    state_list = state_list + past_cam # [現在 1step前 2step前]

        # 入力するLiDAR値の処理
        if ('lidar' in self.input_list) or ('previous_lidar' in self.input_list) or ('previous2_lidar' in self.input_list):
            input_scan = [] # 正規化したLiDAR値を格納するリスト
            for i in range(len(scan)): # lidar値の正規化
                input_scan.append((scan[i] - self.range_margin) / (self.lidar_max - self.range_margin))

            state_list = state_list + input_scan # [画像] + [現在]

            if 'previous_lidar' in self.input_list:
                self.previous_lidar_list.append(input_scan)
                if len(self.previous_lidar_list) > 2: # [1step前 現在]の状態に保つ
                    self.previous_lidar_list.popleft() # 左端の要素を削除(2step前の情報を削除)
                past_scan = self.previous_lidar_list[0] # 1step前のLiDAR値
                state_list = state_list + past_scan # [画像] + [現在 1step前]

                if 'previous2_lidar' in self.input_list:
                    self.previous2_lidar_list.append(input_scan)
                    if len(self.previous2_lidar_list) > 3: # [2step前 1step前 現在]の状態に保つ
                        self.previous2_lidar_list.popleft() # 左端の要素を削除(3step前の情報を削除)
                    past_scan = self.previous2_lidar_list[0] # 2step前のLiDAR値
                    state_list = state_list + past_scan # [画像] + [現在 1step前 2step前]
        
        return state_list, scan, input_scan, collision, goal, goal_num
   
    def setReward(self, scan, collision, goal, goal_num,  action):

        reward = 0
        color_num = 0
        just_count = 0
        color_num = goal_num

        if self.Target == 'both' or self.Target == 'reward':
            if goal:
                reward += self.r_goal + self.r_cost
                just_count = 1
            elif collision:
                reward -= self.r_collision
            if action in [3, 4]:
                reward -= self.r_passive
            reward -= self.r_cost
            reward += goal_num * self.r_just
            reward -= min(1 / (min(scan) + 0.01), 7) * self.r_near
        else:
            if goal:
                reward += 50 + 10 # r_goal + r_cost
                just_count = 1
            elif collision:
                reward -= 50 # r_collision
            if action in [3, 4]:
                reward -= 50 # r_passive
            reward -= 10 # r_cost
            reward += goal_num * 1 # r_just
            reward -= min(1 / (min(scan) + 0.01), 7) * 1 # r_near
        
        return reward, color_num, just_count

    def step(self, action, test): # 1stepの行動

        self.img = None
        self.scan = None

        vel_cmd = Twist()

        "最大速度 x: 0.22[m/s], z: 2.84[rad/s](162.72[deg/s])"
        "z値 0.785375[rad/s] = 45[deg/s], 1.57075[rad/s] = 90[deg/s], 2.356125[rad/s] = 135[deg/s]"
        "行動時間は行動を決定してから次の行動が決まるまでであるため1秒もない"

        if action == 0: # 左折
            vel_cmd.linear.x = 0.15 # 直進方向[m/s]
            vel_cmd.angular.z = 1.57 # 回転方向 [rad/s]
        
        elif action == 1: # 直進
            vel_cmd.linear.x = 0.10
            vel_cmd.angular.z = 0

        elif action == 2: # 右折
            vel_cmd.linear.x = 0.15
            vel_cmd.angular.z = -1.57
        
        elif action == 3: # 左旋回
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 1.57
        
        elif action == 4: # 右旋回
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = -1.57
        
        self.pub_cmd_vel.publish(vel_cmd) # 実行
        state_list, scan, input_scan, collision, goal, goal_num = self.getState() # 状態観測
        reward, color_num, just_count = self.setReward(scan, collision, goal, goal_num, action) # 報酬計算

        if not test: # テスト時でないときの処理
            if (collision or goal) and not self.teleport:
                self.restart() # 進行方向への向き直し
            elif collision or goal:
                self.relocation() # 空いているエリアへの再配置
        
        return np.array(state_list), reward, color_num, just_count, collision, goal, input_scan

    def reset(self):
        self.img = None
        self.scan = None
        state_list, _, _, _, _, _ = self.getState()
        return np.array(state_list)
    
    def restart(self): # 障害物から離れるように動いて安全を確保

        self.stop()
        vel_cmd = Twist()
        front_side = list(range(0, self.lidar_num * 1 // 4 + 1)) + list(range(self.lidar_num * 3 // 4, self.lidar_num))
        left_side = list(range(0, self.lidar_num * 1 // 2 + 1))

        data_range = self.get_lidar(retake=True)

        while True:
            if data_range.index(min(data_range)) in left_side: # 左側に障害物がある時
                vel_cmd.angular.z = -pi / 2 # 右回転[rad/s]
            else:
                vel_cmd.angular.z = pi / 2 # 左回転[rad/s]
            
            if data_range.index(min(data_range)) in front_side: # 前方に障害物がある時
                vel_cmd.linear.x = -0.1 # 後退[m/s]
                vel_cmd.angular.z = vel_cmd.angular.z * -1 # 回転方向反転[rad/s]
            else:
                vel_cmd.linear.x = 0.1 # 前進[m/s]
            
            self.pub_cmd_vel.publish(vel_cmd) # 実行

            data_range = self.get_lidar(retake=True)

            if min(data_range) > self.range_margin + 0.1: # LiDAR値が衝突判定の距離より余裕がある時
                self.stop()
                break
    
    def set_robot(self, num): # 指定位置にロボットを配置

        self.stop()

        # テスト時の目標ゴールの再設定
        if 0 <= num <= 100:
            if self.robot_n == 0:
                self.goal_color = 'purple'
            elif self.robot_n == 1:
                self.goal_color = 'green'
            elif self.robot_n == 2:
                self.goal_color = 'yellow'
        
        # 配置場所の定義
        a = [0.55, 0.9, 0.02, 3.14] # 上
        b = [0.55, 0.35, 0.02, 2.355] # 右上
        c = [0.0, 0.35, 0.02, 1.57] # 右
        d = [-0.55, 0.35, 0.02, 0.785] # 右下
        e = [-0.55, 0.9, 0.02, 0.0] # 下
        f = [-0.55, 1.45, 0.02, -0.785] # 左下
        g = [0.0, 1.45, 0.02, -1.57] # 左
        h = [0.55, 1.45, 0.02, -2.355] # 左上

        if num == 0: # 初期位置
            if self.robot_n == 0:
                XYZyaw = b
            elif self.robot_n == 1:
                XYZyaw = h
            elif self.robot_n == 2:
                XYZyaw = f
        
        # 以下テスト用
        if num in [1, 2]:
            if self.robot_n == 0:
                XYZyaw = b
            elif self.robot_n == 1:
                XYZyaw = h
            elif self.robot_n == 2:
                XYZyaw = f
        
        elif num in [3, 4]:
            if self.robot_n == 0:
                XYZyaw = a
            elif self.robot_n == 1:
                XYZyaw = g
            elif self.robot_n == 2:
                XYZyaw = e
        
        elif num in [5, 6]:
            if self.robot_n == 0:
                XYZyaw = c
            elif self.robot_n == 1:
                XYZyaw = a
            elif self.robot_n == 2:
                XYZyaw = g
        
        # フィールド外
        elif num == 102: # フィールド外の右側
            if self.robot_n == 0:
                XYZyaw = [-0.55, -0.3, 0.02, 0] # 下
            elif self.robot_n == 1:
                XYZyaw = [0.0, -0.3, 0.02, 3.14] # 中央
            elif self.robot_n == 2:
                XYZyaw = [0.55, -0.3, 0.02, 0] # 上
        
        elif num == 103: # フィールド外の左側
            if self.robot_n == 0:
                XYZyaw = [0.55, 2.1, 0.02, 0] # 下
            elif self.robot_n == 1:
                XYZyaw = [0.0, 2.1, 0.02, 3.14] # 中央
            elif self.robot_n == 2:
                XYZyaw = [-0.55, 2.1, 0.02, 0] # 上
        
        elif num == 104: # フィールド外の下側
            if self.robot_n == 0:
                XYZyaw = [-1.2, 1.45, 0.02, 0] # 左
            elif self.robot_n == 1:
                XYZyaw = [-1.2, 0.9, 0.02, 3.14] # 中央
            elif self.robot_n == 2:
                XYZyaw = [-1.2, 0.35, 0.02, 0] # 右
        
        elif num == 105: # フィールド外の上側
            if self.robot_n == 0:
                XYZyaw = [1.2, 0.35, 0.02, 0] # 右
            elif self.robot_n == 1:
                XYZyaw = [1.2, 0.9, 0.02, 3.14] # 中央
            elif self.robot_n == 2:
                XYZyaw = [1.2, 1.45, 0.02, 0] # 左

        # 空いたエリアへのロボットの配置用[relocation()]
        if num == 1001:
            XYZyaw = a
        elif num == 1002:
            XYZyaw = b
        elif num == 1003:
            XYZyaw = c
        elif num == 1004:
            XYZyaw = d
        elif num == 1005:
            XYZyaw = e
        elif num == 1006:
            XYZyaw = f
        elif num == 1007:
            XYZyaw = g
        elif num == 1008:
            XYZyaw = h
        
        state_msg = ModelState()
        state_msg.model_name = 'tb3_{}'.format(self.robot_n)
        state_msg.pose.position.x = XYZyaw[0]
        state_msg.pose.position.y = XYZyaw[1]
        state_msg.pose.position.z = XYZyaw[2]
        q = quaternion_from_euler(0, 0, XYZyaw[3])
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(state_msg)

        if 0 <= num <= 100 or 1001 <= num <= 1100:
            time.sleep(0.1) # 配置後すぐに行動させた場合は配置前の情報が使われることがあるため数秒待機

    # 以降追加システム
    def goal_mask(self, img): # 目標ゴールを緑に, 他のゴールを黒に変換

        goal_num = 0

        # 画像をHSV色空間に変換
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 色範囲を定義(HSVの値)
        color_ranges = {
            "red": [(0, 50, 50), (10, 255, 255)], # 赤(低域)
            "red2": [(170, 150, 90), (180, 255, 255)], # 赤(高域)
            "green": [(50, 50, 50), (70, 255, 255)], # 緑
            "yellow": [(20, 50, 50), (30, 255, 255)], # 黄
            "purple": [(130, 50, 50), (160, 255, 255)] # 紫
        }

        # 色の変換
        for color, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
            if color == self.goal_color or (self.goal_color == 'red' and color in ['red', 'red2']):
                goal_num += cv2.countNonZero(mask)
                changed_color = [0, 255, 0] # ゴールは緑色に変換
            else:
                changed_color = [0, 0, 0] # 他の色は黒に変換
            if self.mask_switch:
                img[mask > 0] = changed_color

        # 画像の出力
        if self.display_image_mask and self.robot_n in self.display_rb:
            self.display_image(img, f'camera_mask_{self.robot_n}')
        
        return img, goal_num
    
    def display_image(self, img, name): # カメラ画像の出力

        # アスペクト比を維持してリサイズ
        magnification = 10 # 出力倍率
        height, width = img.shape[:2]
        target_width, target_height = width * magnification, height * magnification # 出力サイズ(width, height)
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        disp_img = cv2.resize(img, (new_width, new_height))

        # ウィンドウを表示
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, target_width, target_height)
        cv2.imshow(name, disp_img)
        cv2.waitKey(1)

    def stop(self): # ロボットの停止
        vel_cmd = Twist()
        vel_cmd.linear.x = 0 # 直進方向[m/s]
        vel_cmd.angular.z = 0  # 回転方向[rad/s]
        self.pub_cmd_vel.publish(vel_cmd) # 実行
    
    def robot_coordinate(self): # ロボットの座標を取得
        ros_data = None
        while ros_data is None:
            try:
                ros_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1) # ROSデータの取得
            except:
                pass
        
        tb3_0 = ros_data.name.index('tb3_0') # robot0のデータの配列番号
        tb3_1 = ros_data.name.index('tb3_1') # robot1のデータの配列番号
        tb3_2 = ros_data.name.index('tb3_2') # robot2のデータの配列番号

        rb0 = np.array([ros_data.pose[tb3_0].position.x, ros_data.pose[tb3_0].position.y], dtype='float') # robot0の座標
        rb1 = np.array([ros_data.pose[tb3_1].position.x, ros_data.pose[tb3_1].position.y], dtype='float') # robot1の座標
        rb2 = np.array([ros_data.pose[tb3_2].position.x, ros_data.pose[tb3_2].position.y], dtype='float') # robot2の座標

        return rb0, rb1, rb2

    def relocation(self): # 衝突時、ほかロボットの座標を観測し、空いている座標へ配置

        exist_erea = [] # ロボットが存在するエリアを格納するリスト
        teleport_area = 0

        # 各ロボットの座標
        rb0, rb1, rb2 = self.robot_coordinate()

        if self.robot_n == 0:
            coordinate_list = [rb1, rb2]
        elif self.robot_n == 1:
            coordinate_list = [rb0, rb2]
        elif self.robot_n == 2:
            coordinate_list = [rb0, rb1]

        for coordinate in coordinate_list:
            if 0.3 <= coordinate[0] <= 0.9 and 0.6 <= coordinate[1] <= 1.2: # 上のエリアに存在するか
                exist_erea.append(1)
            elif 0.3 <= coordinate[0] <= 0.9 and 0.0 <= coordinate[1] <= 0.6: # 右上のエリアに存在するか
                exist_erea.append(2)
            elif -0.3 <= coordinate[0] <= 0.3 and 0.0 <= coordinate[1] <= 0.6: # 右のエリアに存在するか
                exist_erea.append(3)
            elif -0.9 <= coordinate[0] <= -0.3 and 0.0 <= coordinate[1] <= 0.6: # 右下のエリアに存在するか
                exist_erea.append(4)
            elif -0.9 <= coordinate[0] <= -0.3 and 0.6 <= coordinate[1] <= 1.2: # 下のエリアに存在するか
                exist_erea.append(5)
            elif -0.9 <= coordinate[0] <= -0.3 and 1.2 <= coordinate[1] <= 1.8: # 左下のエリアに存在するか
                exist_erea.append(6)
            elif -0.3 <= coordinate[0] <= 0.3 and 1.2 <= coordinate[1] <= 1.8: # 左のエリアに存在するか
                exist_erea.append(7)
            elif 0.3 <= coordinate[0] <= 0.9 and 1.2 <= coordinate[1] <= 1.8: # 左上のエリアに存在するか
                exist_erea.append(8)
        
        # 空いているエリア
        empty_area = [x for x in list(range(8, 0, -1)) if x not in exist_erea]

        if self.goal_color == 'red':
            target_list = [4, 5, 3]
        elif self.goal_color == 'green':
            target_list = [8, 1, 7]
        elif self.goal_color == 'yellow':
            target_list = [6, 7, 5]
        elif self.goal_color == 'purple':
            target_list = [2, 3, 1]
        
        for value in target_list:
            if value in empty_area:
                teleport_area = value
                break
        if teleport_area == 0:
            teleport_area = empty_area[0]
        
        # テレポート
        self.set_robot(teleport_area + 1000)

    def area_judge(self, terms, area): # ロボットのエリア内外判定
        exist = False
        judge_list = []
        rb0, rb1, rb2 = self.robot_coordinate() # ロボットの座標を取得

        # エリアの座標を定義
        if area == 'right':
            area_coordinate = [-0.9, 0.9, -1.8, 0.0] # [x_最小, x_最大, y_最小, y_最大]
        elif area == 'left':
            area_coordinate = [-0.9, 0.9, 1.8, 3.6]
        elif area == 'lower':
            area_coordinate = [-2.7, -0.9, 0.0, 1.8]
        elif area == 'upper':
            area_coordinate = [0.9, 2.7, 0.0, 1.8]
        
        # 他のロボットの座標を格納
        if self.robot_n == 0:
            judge_robot = [rb1, rb2]
        elif self.robot_n == 1:
            judge_robot = [rb0, rb2]
        elif self.robot_n == 2:
            judge_robot = [rb0, rb1]
        
        # 他のロボットのエリア内外判定
        for rb in judge_robot:
            judge_list.append(area_coordinate[0] < rb[0] < area_coordinate[1] and area_coordinate[2] < rb[1] < area_coordinate[3])
        
        if terms == 'hard' and (judge_list[0] and judge_list[1] and judge_list[2]): # 他の全ロボットがエリアに存在する時
            exist = True
        elif terms == 'soft' and (judge_list[0] or judge_list[1] or judge_list[2]): # 他のロボットが1台でもエリアに存在する時
            exist = True

        return exist

    # 以降リカバリー方策
    def recovery_change_action(self, e, input_scan, lidar_num, action, state, model): # LiDARの数値が低い方向への行動を避ける

        ### ユーザー設定パラメータ ###
        threshold = 0.2 # 動きを変える距離(LiDAR値)[m]
        probabilistic = True # True: リカバリー方策を確率的に利用する, False: リカバリー方策を必ず利用する
        initial_probability = 1.0 # 最初の確率
        finish_episode = 25 # 方策を適応する最後のエピソード
        mode_change_episode = 11 # 行動変更のトリガーをLiDAR値からQ値に変えるエピソード
        ############################

        # リカバリー方策の利用判定
        if not probabilistic and e <= finish_episode: # 必ず利用
            pass
        elif random.random() < round(initial_probability - (initial_probability / finish_episode) * (e - 1), 3): # 確率で利用(確率は線形減少)
            pass
        else:
            return action
        
        change_action = False
        bad_action = []

        # 方向の定義
        left = list(range(2, lidar_num // 4 + 1)) # LiDARの前方左側
        forward = [0, 1, lidar_num - 1] # LiDARの前方とする要素番号(左右数度ずつ)
        right = list(range(lidar_num * 3 // 4, lidar_num - 1)) # LiDARの前方右側

        # LiDARのリストで条件に合う要素を格納したリストをインスタンス化(element_num:要素番号, element_cont:要素内容)
        low_lidar = [element_num for element_num, element_cont in enumerate(input_scan) if element_cont <= threshold]

        # 指定したリストと条件に合う要素のリストで同じ数字があった場合は行動を変更する(actionを 0は左折, 1は直進, 2は右折 に設定する必要あり)
        if set(left) & set(low_lidar) != set():
            bad_action.append(0)
            if action == 0:
                change_action = True
        if set(forward) & set(low_lidar) != set():
            bad_action.append(1)
            if action == 1:
                change_action = True
        if set(right) & set(low_lidar) != set():
            bad_action.append(2)
            if action == 2:
                change_action = True
        
        # 行動を変更
        if change_action:
            if e < mode_change_episode: # LiDAR値による行動の変更
                # 各方向のLiDAR値
                front_scan = input_scan[0:left[-1] + 1] + input_scan[right[0]:lidar_num]
                left_scan = input_scan[left[0]:left[-1] + 1]
                forward_scan = input_scan[0:left[0]] + input_scan[right[-1] + 1:lidar_num]
                right_scan = input_scan[right[0]:right[-1] + 1]
                scan_list = [left_scan, forward_scan, right_scan]
                if len(bad_action) == 3: # 全方向のLiDAR値が低い場合はLiDAR値が最大の方向へ
                    if max(front_scan) in left_scan:
                        action = 0
                    elif max(front_scan) in forward_scan:
                        action = 1
                    elif max(front_scan) in right_scan:
                        action = 2
                elif len(bad_action) == 2: # 2方向のLiDAR値が低い場合は残りの方向へ
                    action = (set([0, 1, 2]) - set(bad_action)).pop()
                elif len(bad_action) == 1: # 1方向のLiDAR値が低い場合は残りのLiDAR値が大きい方向へ
                    action_candidate = list(set([0, 1, 2]) - set(bad_action))
                    if max(scan_list[action_candidate[0]]) > max(scan_list[action_candidate[1]]):
                        action = action_candidate[0]
                    else:
                        action = action_candidate[1]
            else: # Q値による行動の変更
                net_out = model.forward(state.unsqueeze(0).to('cuda:0')) # ネットワークの出力
                q_values = net_out.q_values.cpu().detach().numpy().tolist()[0] # Q値
                if len(bad_action) == 3: # 全方向のLiDAR値が低い場合はQ値が最大の方向へ
                    action = q_values.index(max(q_values))
                elif len(bad_action) == 2: # 2方向のLiDAR値が低い場合は残りの方向へ
                    action = (set([0, 1, 2]) - set(bad_action)).pop()
                elif len(bad_action) == 1: # 1方向のLiDAR値が低い場合は残りのQ値が大きい方向へ
                    action_candidate = list(set([0, 1, 2]) - set(bad_action))
                    if q_values[action_candidate[0]] > q_values[action_candidate[1]]:
                        action = action_candidate[0]
                    else:
                        action = action_candidate[1]

        return action
