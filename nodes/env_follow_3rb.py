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

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Env():
    def __init__(self, mode, robot_n, lidar_num, input_lidar, lidar_past_step, 
                 input_cam, cam_past_step, teleport, 
                 r_collision, r_near, r_center, r_just, distance, 
                 trials, display_image_normal, display_image_count, 
                 display_rb, cam_width, cam_height):
        
        self.mode = mode
        self.robot_n = robot_n
        self.lidar_num = lidar_num
        self.input_lidar = input_lidar
        self.lidar_past_step = lidar_past_step
        self.input_cam = input_cam
        self.cam_past_step = cam_past_step
        self.teleport = teleport

        self.cam_list = deque([])
        self.lidar_list = deque([])
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        if self.mode=='sim':
            self.sub_img=rospy.Subscriber('usb_cam/image_raw',Image,self.pass_img,queue_size=10) #これがないと上手く画像取得できない(原因不明) 実機はなし
        else:
            self.sub_img=rospy.Subscriber('usb_cam/image_raw/compressed',CompressedImage,self.pass_img,queue_size=10) #これがないと上手く画像取得できない(原因不明) 実機用

        # カメラ画像
        self.display_image_normal = display_image_normal
        self.display_image_count = display_image_count
        self.display_rb = display_rb
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.collision_img_past = False

        # Optunaで選択された値
        self.r_collision = r_collision
        self.r_near = r_near
        self.r_center = r_center
        self.r_just = r_just
        self.distance = distance
        self.trials = trials

        # LiDARについての設定
        self.lidar_max = 2 # 対象のworldにおいて取りうるlidarの最大値(simの貫通対策や正規化に使用)
        self.lidar_min = 0.12 # lidarの最小測距値[m]
        self.range_margin = self.lidar_min + 0.02 # 衝突として処理される距離[m] 0.02

    def pass_img(self, img):#画像正常取得用callback
        pass

    def get_clock(self):
        if self.mode == 'sim':
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('/clock', Clock, timeout=10)
                except:
                    time.sleep(2)
                    print('clock waiting...')
                    pass
            secs = data.clock.secs
            nsecs = data.clock.nsecs/10**9
            return secs+nsecs
        else:
            return time.time()

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
                    if self.mode=='sim':
                       img = rospy.wait_for_message('usb_cam/image_raw', Image, timeout=1) # 非圧縮データ(シミュレーションで圧縮データを扱うと学習性能が落ちる)
                    else:
                       img = rospy.wait_for_message('usb_cam/image_raw/compressed', CompressedImage, timeout=1) # 圧縮データ
                except:
                    self.stop()
                    pass

            if self.mode=='sim':
                img = ros_numpy.numpify(img) # RGB
            else:
                img = np.frombuffer(img.data, np.uint8) # データ参照
                img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
            
            img = cv2.resize(img, (self.cam_width, self.cam_height)) # 取得した画像をcam_width×cam_height[pixel]に変更
            
            self.img = img

        return self.img
    
    def get_count(self, img, scope):
        
        # 色の範囲
        outside_lower = np.array([0, 0, 0], dtype=np.uint8) # 黒
        outside_upper = np.array([255, 255, 80], dtype=np.uint8)
        inside_lower = np.array([0, 150, 90], dtype=np.uint8) # 橙
        inside_upper = np.array([20, 255, 255], dtype=np.uint8)
        robot_blue_lower = np.array([85, 100, 50], dtype=np.uint8) # 青
        robot_blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        robot_green_lower = np.array([55, 100, 60], dtype=np.uint8) # 緑
        robot_green_upper = np.array([85, 255, 255], dtype=np.uint8)

        # 注目領域の指定
        if scope == 'middle':
            img = img[:, len(img[0])*4//9:len(img[0])*5//9]
        elif scope == 'middle2':
            img = img[:, len(img[0])*1//3:len(img[0])*2//3]
        elif scope == 'middle3':
            img = img[:, len(img[0])*1//5:len(img[0])*4//5]
        elif scope == 'left':
            img = img[:, 0:len(img[0])*1//9]
        elif scope == 'right':
            img = img[:, len(img[0])*8//9:len(img[0])*9//9]
        elif scope == 'side':
            img = np.hstack((img[:, :len(img[1])*1//5], img[:, len(img[1])*4//5:]))
        
        # 画像をHSV色空間に変換
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # マスク処理
        outside = cv2.inRange(img_hsv, outside_lower, outside_upper)
        inside = cv2.inRange(img_hsv, inside_lower, inside_upper)
        robot_blue = cv2.inRange(img_hsv, robot_blue_lower, robot_blue_upper)
        robot_green = cv2.inRange(img_hsv, robot_green_lower, robot_green_upper)

        # 摘出した画素数を算出
        outside_num = np.count_nonzero(outside)
        inside_num = np.count_nonzero(inside)
        robot_blue_num = np.count_nonzero(robot_blue)
        robot_green_num = np.count_nonzero(robot_green)

        # 注目領域の出力
        if self.display_image_count and self.robot_n in self.display_rb:
            self.display_image(img, f'camera_count_{self.robot_n}_{scope}')
        
        return outside_num, inside_num, robot_blue_num, robot_green_num
    
    def getState(self): # 情報取得

        collision = False
        collision_img = False
        collision_lidar = False
        state_list = [] # 入力する情報を格納するリスト

        ### 画像の取得と処理 ###
        img = self.get_camera() # カメラ画像の取得
        if 'img_past' in locals():
            while True: # 画像が1step前と同じ場合は画像を取得し続ける
                if np.array_equal(np.array(img), img_past):
                    self.stop()
                    img = self.get_camera()
                else:
                    break
        img_past = np.array(img) # このstepの画像を保存

        if self.display_image_normal and self.robot_n in self.display_rb:
            self.display_image(img, f'camera_normal_{self.robot_n}')
        
        _, _, robot_blue_num, robot_green_num = self.get_count(img, scope='all')
        if 500 <= robot_blue_num or 500 <= robot_green_num: # 画像の衝突判定
            collision_img = True
            if self.collision_img_past:
                collision_img = False
            self.collision_img_past = True
        else:
            self.collision_img_past = False
        
        #### LiDAR情報の取得と処理 ###
        scan = self.get_lidar() # LiDAR値の取得

        if self.range_margin >= min(scan):
            collision_lidar = True
            if self.mode == 'real': # 実機実験におけるLiDARの飛び値の処理
                scan_true = [element_cont for element_num, element_cont in enumerate(scan) if element_cont != 0]
                if scan.count(0) >= 1 and self.range_margin < min(scan_true): # (飛び値が存在する)and(飛び値を除いた場合は衝突判定ではない)
                    collision_lidar = False
        
        # 衝突判定
        collision = collision_img or collision_lidar
        
        # 入力するカメラ画像の処理
        if self.input_cam:
            input_img = np.asarray(img, dtype=np.float32)
            input_img /= 255.0 # 画像の各ピクセルを255で割ることで0~1の値に正規化
            input_img = np.asarray(input_img.flatten())
            input_img = input_img.tolist()

            if len(self.cam_list) == (self.cam_past_step + 1):
                if np.array_equal(input_img, self.cam_list[0]):
                    pass
                else:
                    self.cam_list.appendleft(input_img)
            else:
                self.cam_list.appendleft(input_img)

            if len(self.cam_list) > (self.cam_past_step + 1):
                self.cam_list.pop()

            state_list_cam = [item for sublist in self.cam_list for item in sublist]
            for i in range((self.cam_past_step + 1) - len(self.cam_list)):
                state_list_cam = self.cam_list[0] + state_list_cam

        # 入力するLiDAR値の処理
        if self.input_lidar:
            input_scan = [] # 正規化したLiDAR値を格納するリスト
            for i in range(len(scan)): # lidar値の正規化
                input_scan.append((scan[i] - self.range_margin) / (self.lidar_max - self.range_margin))
            
            if len(self.lidar_list) == (self.lidar_past_step + 1):
                if np.array_equal(input_scan, self.lidar_list[0]):
                    pass
                else:
                    self.lidar_list.appendleft(input_scan)
            else:
                self.lidar_list.appendleft(input_scan)

            if len(self.lidar_list) > (self.lidar_past_step + 1):
                self.lidar_list.pop()

            state_list_lidar = [item for sublist in self.lidar_list for item in sublist]
            for i in range((self.lidar_past_step + 1) - len(self.lidar_list)):
                state_list_lidar = self.lidar_list[0] + state_list_lidar
        
        state_list = state_list_cam + state_list_lidar
        
        return state_list, collision
   
    def setReward(self, collision, action):

        reward = 0
        color_num = 0
        just_count = 0
        lidar_value_left = round(self.scan[round(len(self.scan)*1/4)], 3)
        lidar_value_right = round(self.scan[round(len(self.scan)*3/4)], 3)
        _, _, robot_blue_num, robot_green_num = self.get_count(self.img, scope='middle3')
        color_num = robot_blue_num + robot_green_num

        if collision:
            reward -= self.r_collision

        if self.robot_n == 0: # robot0
            if abs(lidar_value_left - lidar_value_right) <= 0.04 and action == 1:
                reward += self.r_center
        
        elif self.robot_n != 0: # robot1, 2
            if abs(lidar_value_left - lidar_value_right) <= 0.04:
                reward += self.r_center
            if (30 <= robot_blue_num <= 130) or (30 <= robot_green_num <= 130):
                reward += self.r_just
                just_count = 1
            #     if self.robot_n in self.display_rb:
            #         print(f"\033[31m{robot_blue_num}, {robot_green_num}\033[0m")
            # else:
            #     if self.robot_n in self.display_rb:
            #         print(robot_blue_num, robot_green_num)
            if robot_green_num > 130 or robot_blue_num > 130:
                reward -= self.r_near
        
        return reward, color_num, just_count

    def step(self, action, test): # 1stepの処理

        self.img = None
        self.scan = None

        vel_cmd = Twist()

        "最大速度 x: 0.22[m/s], z: 2.84[rad/s](162.72[deg/s])"
        "z値 0.785375[rad/s] = 45[deg/s], 1.57075[rad/s] = 90[deg/s], 2.356125[rad/s] = 135[deg/s]"
        "行動時間は行動を決定してから次の行動が決まるまでであるため1秒もない"

        if action == 0: # 左折
            vel_cmd.linear.x = 0.17 # 直進方向[m/s]
            vel_cmd.angular.z = 0.78 # 回転方向 [rad/s]
        
        elif action == 1: # 高速直進
            vel_cmd.linear.x = 0.17
            vel_cmd.angular.z = 0

        elif action == 2: # 右折
            vel_cmd.linear.x = 0.17
            vel_cmd.angular.z = -0.78
        
        elif action == 3: # 低速直進
            vel_cmd.linear.x = 0.08
            vel_cmd.angular.z = 0
        
        if self.robot_n == 0 and action != 3:
            vel_cmd.linear.x = vel_cmd.linear.x * 0.7
        
        self.pub_cmd_vel.publish(vel_cmd) # 実行
        state_list, collision = self.getState() # 状態観測
        reward, color_num, just_count = self.setReward(collision, action) # 報酬計算

        if collision and not test:
            if self.mode == 'real' or not self.teleport:
                self.restart() # 進行方向への向き直し
            else:
                self.relocation() # 空いているエリアへの再配置
        elif collision and test:
            self.stop()
        
        return np.array(state_list), reward, color_num, just_count, collision

    def reset(self):
        self.img = None
        self.scan = None
        state_list, _ = self.getState()
        return np.array(state_list)
    
    def restart(self):

        threshold = 60 # 衝突した相手を決定する画素数
        
        vel_cmd = Twist()
        self.stop()

        # 周囲の状況を把握して適切な退避行動を取る
        while True:

            # 正面に衝突対象がくるか背面に衝突対象がくるまで左回転させる
            while True:
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = pi/3
                self.pub_cmd_vel.publish(vel_cmd)

                data_range = self.get_lidar(retake=True)

                if data_range.index(min(data_range)) == 0:
                   wall = 'front'
                   break
                elif data_range.index(min(data_range)) == round(len(data_range)//2):
                   wall = 'back'
                   break
            self.stop()
            
            if wall == 'front': # 正面衝突した場合は後退させる
                start_time = self.get_clock()
                while self.get_clock() - start_time < 1.2:
                    vel_cmd.linear.x = -0.10
                    vel_cmd.angular.z = 0
                    self.pub_cmd_vel.publish(vel_cmd) #実行
                    data_range = self.get_lidar(retake=True)
            elif wall == 'back': # 後ろから衝突された場合は前進する
                start_time = self.get_clock()
                while self.get_clock() - start_time < 1.2:
                    vel_cmd.linear.x = 0.10
                    vel_cmd.angular.z = 0
                    self.pub_cmd_vel.publish(vel_cmd) #実行
                    data_range = self.get_lidar(retake=True)
            self.stop()

            # 衝突相手の断定
            side = 'none'
            img = self.get_camera(retake=True)
            outside_num, inside_num, robot_blue_num, robot_green_num = self.get_count(img, scope='middle2')
            if robot_blue_num >= threshold:
                side = 'robot'
            elif robot_green_num >= threshold:
                side = 'robot'
            elif inside_num >= threshold:
                side = 'inside'
            elif outside_num >= threshold:
                side = 'outside'
            
            # 衝突相手に応じた退避行動
            while True:
                if side == 'robot':
                    # オレンジの障害物が中央に映るまで左回転
                    vel_cmd.linear.x = 0
                    vel_cmd.angular.z = pi/3.7
                    self.pub_cmd_vel.publish(vel_cmd)
                    img = self.get_camera(retake=True)
                    _, inside_num, _, _ = self.get_count(img, scope='middle')
                    while inside_num < 15:
                        self.pub_cmd_vel.publish(vel_cmd)
                        img = self.get_camera(retake=True)
                        _, inside_num, _, _ = self.get_count(img, scope='middle')

                    # 左半回転
                    vel_cmd.linear.x = 0
                    vel_cmd.angular.z = pi/2
                    self.pub_cmd_vel.publish(vel_cmd)
                    start_time = self.get_clock()
                    while self.get_clock() - start_time < 1.0:
                        pass
                    break

                elif side =='inside':
                    # 左半回転
                    vel_cmd.linear.x = 0
                    vel_cmd.angular.z = pi/2
                    self.pub_cmd_vel.publish(vel_cmd)
                    start_time=self.get_clock()
                    while self.get_clock() - start_time < 1.0:
                        pass
                    break
                
                elif side == 'outside':
                    # オレンジの障害物が右側に映るまで右回転
                    vel_cmd.linear.x = 0
                    vel_cmd.angular.z = -pi/3.7
                    self.pub_cmd_vel.publish(vel_cmd)
                    img = self.get_camera(retake=True)
                    _, inside_num, _, _ = self.get_count(img, scope='right')
                    while inside_num < 10:
                        self.pub_cmd_vel.publish(vel_cmd)
                        img = self.get_camera(retake=True)
                        _, inside_num, _, _ = self.get_count(img, scope='right')
                    break

                else:
                    break
            self.stop()

            # restart直後に衝突判定にならないように最低2cm余裕をもたせる
            data_range = self.get_lidar(retake=True)
            img = self.get_camera(retake=True)
            outside_num, inside_num, robot_blue_num, robot_green_num = self.get_count(img, scope='all')
            if min(data_range) > self.range_margin + 0.02 and (50 > robot_blue_num and 50 > robot_green_num):
                break
    
    def set_robot(self, num): # 指定位置にロボットを配置

        self.stop()
        
        # 配置場所の定義
        a = [0.55, 0.9, 0.02, -1.57]  # 上
        b = [0.55, 0.35, 0.02, 3.14]  # 右上
        c = [0.0, 0.35, 0.02, 3.14]   # 右
        d = [-0.55, 0.35, 0.02, 1.57] # 右下
        e = [-0.55, 0.9, 0.02, 1.57]  # 下
        f = [-0.55, 1.45, 0.02, 0.0]  # 左下
        g = [0.0, 1.45, 0.02, 0.0]    # 左
        h = [0.55, 1.45, 0.02, -1.57] # 左上

        test_first = [0.2, 0.35, 0.02, 3.14] # 右
        test_second = [0.55, 0.9, 0.02, -2.355] # 上
        test_third = [0.55, 1.45, 0.02, -1.57] # 左上

        if num == 0: # 初期位置
            if self.robot_n == 0:
                XYZyaw = test_first
            elif self.robot_n == 1:
                XYZyaw = test_second
            elif self.robot_n == 2:
                XYZyaw = test_third
        
        # 以下テスト用
        if num in [1, 2]:
            if self.robot_n == 0:
                XYZyaw = test_first
            elif self.robot_n == 1:
                XYZyaw = test_second
            elif self.robot_n == 2:
                XYZyaw = test_third
        
        elif num in [3, 4]:
            if self.robot_n == 0:
                XYZyaw = test_first
            elif self.robot_n == 1:
                XYZyaw = test_third
            elif self.robot_n == 2:
                XYZyaw = test_second
        
        elif num in [5, 6]:
            if self.robot_n == 0:
                XYZyaw = test_first
            elif self.robot_n == 1:
                XYZyaw = test_second
            elif self.robot_n == 2:
                XYZyaw = test_third
        
        elif num in [7, 8]:
            if self.robot_n == 0:
                XYZyaw = test_first
            elif self.robot_n == 1:
                XYZyaw = test_third
            elif self.robot_n == 2:
                XYZyaw = test_second
        
        # フィールド外
        elif num == 102: # フィールド外の右側
            if self.robot_n == 0:
                XYZyaw = [-0.55, -0.3, 0.02, 0] # 下
            elif self.robot_n == 1:
                XYZyaw = [0.0, -0.3, 0.02, 0] # 中央
            elif self.robot_n == 2:
                XYZyaw = [0.55, -0.3, 0.02, 0] # 上
        
        elif num == 103: # フィールド外の左側
            if self.robot_n == 0:
                XYZyaw = [0.55, 2.1, 0.02, 0] # 下
            elif self.robot_n == 1:
                XYZyaw = [0.0, 2.1, 0.02, 0] # 中央
            elif self.robot_n == 2:
                XYZyaw = [-0.55, 2.1, 0.02, 0] # 上
        
        elif num == 104: # フィールド外の下側
            if self.robot_n == 0:
                XYZyaw = [-1.2, 1.45, 0.02, 0] # 左
            elif self.robot_n == 1:
                XYZyaw = [-1.2, 0.9, 0.02, 0] # 中央
            elif self.robot_n == 2:
                XYZyaw = [-1.2, 0.35, 0.02, 0] # 右
        
        elif num == 105: # フィールド外の上側
            if self.robot_n == 0:
                XYZyaw = [1.2, 0.35, 0.02, 0] # 右
            elif self.robot_n == 1:
                XYZyaw = [1.2, 0.9, 0.02, 0] # 中央
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
    def coordinate_file(self):
        f_coordinate_file =  os.path.dirname(os.path.realpath(__file__)) + '/result/' # os.path.dirname(os.path.realpath(__file__)) ← カレントディレクトリのパス
        self.f_coordinate_name = f_coordinate_file + 'coordinate_robot' + str(self.robot_n) + '.txt'
        if not os.path.exists(f_coordinate_file):
            os.makedirs(f_coordinate_file)
        with open(self.f_coordinate_name, 'w') as f: # ファイルに属性を書き込む
            f.writelines('[')

    def coordinate_get(self): # ロボットの座標の記録
        ros_data = None
        while ros_data is None:
            try:
                ros_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1) # ROSデータの取得
            except:
                pass
        index = ros_data.name.index(f'tb3_{self.robot_n}') # ロボットのデータの配列番号
        coordinate = [ros_data.pose[index].position.x, ros_data.pose[index].position.y] # ロボットの座標
        self.path.append(coordinate)  # 座標をリストに追加
    
    def coordinate_recode(self, flag_last):
        if flag_last:
            text = [str(self.path) + ']\n']
        else:
            text = [str(self.path) + ', ']
        
        with open(self.f_coordinate_name, 'a') as f:
            f.writelines(text)
        self.path = []

    def display_image(self, img, name): # カメラ画像の出力

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

        rb0, rb1, rb2 = self.robot_coordinate() # 各ロボットの座標を取得

        # 居場所を特定するロボットを設定
        if self.robot_n == 0:
            robot_coordinate_list = [rb0, rb1, rb2]
        elif self.robot_n == 1:
            robot_coordinate_list = [rb0, rb0, rb2]
        elif self.robot_n == 2:
            robot_coordinate_list = [rb0, rb0, rb1]
        
        # 確認するエリアを設定
        area_coordinate_list = [
            [0.3, 0.9, 0.6, 1.2], # 上
            [0.3, 0.9, 0.0, 0.6], # 右上
            [-0.3, 0.3, 0.0, 0.6], # 右
            [-0.9, -0.3, 0.0, 0.6], # 右下
            [-0.9, -0.3, 0.6, 1.2], # 下
            [-0.9, -0.3, 1.2, 1.8], # 左下
            [-0.3, 0.3, 1.2, 1.8], # 左
            [0.3, 0.9, 1.2, 1.8] # 左上
            ]

        # 各ロボットの各エリアで内外判定を行う
        for robot_coordinate in robot_coordinate_list:
            for area_coordinate in area_coordinate_list:
                if area_coordinate[0] <= robot_coordinate[0] <= area_coordinate[1] and area_coordinate[2] <= robot_coordinate[1] <= area_coordinate[3]:
                    exist_erea.append(area_coordinate_list.index(area_coordinate) + 1)
                    break
        
        if len(exist_erea) == 2:
            rb0_area = exist_erea.pop(0) # 先頭のロボットの居場所を取り出す
        
        empty_area = [x for x in list(range(len(area_coordinate_list), 0, -1)) if x not in exist_erea] # 空いているエリア

        if len(exist_erea) == 2:
            smaller = [x for x in empty_area if x < rb0_area] # 先頭のロボットに続いた後ろのエリア
            larger = [x for x in empty_area if x > rb0_area] # さらに後ろのエリア
            empty_area = smaller + larger # 先頭のロボットの後ろのエリアを一番に置き換える

        # 配置する位置の決定
        if self.robot_n == 0:
            teleport_area = empty_area[-1]
        elif self.robot_n in [1, 2]:
            teleport_area = empty_area[self.robot_n - 1]
        
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
        
        if terms == 'hard' and (judge_list[0] and judge_list[1]): # 他の全ロボットがエリアに存在する時
            exist = True
        elif terms == 'soft' and (judge_list[0] or judge_list[1]): # 他のロボットが1台でもエリアに存在する時
            exist = True

        return exist

    # 以降リカバリー方策
    def recovery_change_action(self, e, lidar_num, action, state, model): # LiDARの数値が低い方向への行動を避ける

        ### ユーザー設定パラメータ ###
        threshold = self.distance # 動きを変える距離(LiDAR値)[m]
        probabilistic = False # True: リカバリー方策を確率的に利用する, False: リカバリー方策を必ず利用する
        initial_probability = 1.0 # 最初の確率
        finish_episode = 999 # 方策を適応する最後のエピソード
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
        front_deg = 40 # 正面とする角度の定義[deg]
        lidar_deg = 360 // lidar_num # 1要素間の角度[deg]
        front = list(range(0, (front_deg // 2) // lidar_deg + 1)) + list(range(lidar_num - ((front_deg // 2) // lidar_deg), lidar_num)) # LiDARの正面とする要素番号
        left = list(range((front_deg // 2) // lidar_deg + 1, lidar_num // 4 + 1)) # LiDARの前方左側
        right = list(range(lidar_num * 3 // 4, lidar_num - ((front_deg // 2) // lidar_deg))) # LiDARの前方右側

        # LiDARのリストで条件に合う要素を格納したリストをインスタンス化(element_num:要素番号, element_cont:要素内容)
        low_lidar = [element_num for element_num, element_cont in enumerate(self.scan) if element_cont <= threshold]

        # 指定したリストと条件に合う要素のリストで同じ数字があった場合は行動を変更する(actionを 0は左折, 1は直進, 2は右折 に設定する必要あり)
        if set(left) & set(low_lidar) != set():
            bad_action.append(0)
            if action == 0:
                change_action = True
        if set(front) & set(low_lidar) != set():
            bad_action.append(1)
            if action == 1 or action == 3:
                change_action = True
        if set(right) & set(low_lidar) != set():
            bad_action.append(2)
            if action == 2:
                change_action = True
        
        # 行動を変更
        if change_action:
            if e < mode_change_episode: # LiDAR値による行動の変更
                # 各方向のLiDAR値
                front_scan = self.scan[0:left[-1] + 1] + self.scan[right[0]:lidar_num]
                left_scan = self.scan[left[0]:left[-1] + 1]
                forward_scan = self.scan[0:left[0]] + self.scan[right[-1] + 1:lidar_num]
                right_scan = self.scan[right[0]:right[-1] + 1]
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
            if action == 1:
                action = 3

        return action
