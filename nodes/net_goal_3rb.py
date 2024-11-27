# -*- coding: utf-8 -*-

"""
このファイルではネットワークの構造を決めています
"""

from __future__ import division
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Q_Func(nn.Module):
    def __init__(self, n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True) #第1引数*第1引数で注目して第2引数のストライドで最大値プーリングをする

        # convolution

        # [Conv2d]
        # 第1引数個のチャンネルをそれぞれの第3引数*第3引数のカーネルで畳み込んだもの
        # の和を取って1チャンネルにする.<=1セット分の工程
        # 1セット(第1引数個)のカーネルを第2引数セット用意して,
        # 1セットずつ畳み込むことで第2引数個のチャンネルを得る.
        # これらを自動で行うためチャンネルの入出力数とカーネルサイズのみの設定で良い (default => stride=1)
        
        # #1セット
        # self.conv1_1 = nn.Conv2d(n_input_channels, 16, 17) #48*27*3=>32*11*16
        # nn.init.kaiming_normal_(self.conv1_1.weight)
        
        # self.conv1_2 = nn.Conv2d(16, 64, 10) #32*11*16=>23*2*64
        # nn.init.kaiming_normal_(self.conv1_2.weight)

        # #pooling2*2=>12*1*64

        # self.img_input = 12*1*64

        # #2セット
        # self.conv1_1 = nn.Conv2d(n_input_channels, 8, 5) #48*27*6=>44*23*8
        # nn.init.kaiming_normal_(self.conv1_1.weight)
        
        # self.conv1_2 = nn.Conv2d(8, 16, 5) #44*23*8=>40*19*16  
        # nn.init.kaiming_normal_(self.conv1_2.weight)

        # #pooling2*2=>20*10*16

        # self.conv2_1 = nn.Conv2d(16, 64, 5) #20*10*16=>16*6*64
        # nn.init.kaiming_normal_(self.conv2_1.weight)

        # self.conv2_2 = nn.Conv2d(64, 128, 5) #16*6*64=>12*2*128
        # nn.init.kaiming_normal_(self.conv2_2.weight)

        # #pooling2*2=>6*1*128

        # self.img_input = 6*1*128

        # #3セット
        # self.conv1_1 = nn.Conv2d(n_input_channels, 8, 3) #48*27*3=>46*25*8
        # nn.init.kaiming_normal_(self.conv1_1.weight)
        
        # self.conv1_2 = nn.Conv2d(8, 16, 3) #46*25*8=>44*23*16  
        # nn.init.kaiming_normal_(self.conv1_2.weight)

        # #pooling2*2=>22*12*16

        # self.conv2_1 = nn.Conv2d(16, 32, 3) #22*12*16=>20*10*32
        # nn.init.kaiming_normal_(self.conv2_1.weight)

        # self.conv2_2 = nn.Conv2d(32, 64, 3) #20*10*32=>18*8*64
        # nn.init.kaiming_normal_(self.conv2_2.weight)

        # #pooling2*2=>9*4*64

        # self.conv3_1 = nn.Conv2d(64, 128, 2) #9*4*64=>8*3*128
        # nn.init.kaiming_normal_(self.conv3_1.weight)

        # self.conv3_2 = nn.Conv2d(128, 256, 2) #8*3*128=>7*2*256
        # nn.init.kaiming_normal_(self.conv3_2.weight)

        # #pooling2*2=>4*1*256

        # self.img_input = 4*1*256

        #手動で最適化したもの
        self.conv1_1 = nn.Conv2d(n_input_channels, 32, 8) #48*27*6=>41*20*32
        nn.init.kaiming_normal_(self.conv1_1.weight)

        #pooling2*2=>21*10*32

        self.conv2_1 = nn.Conv2d(32, 32, 8) #21*10*32=>14*3*32
        nn.init.kaiming_normal_(self.conv2_1.weight)

        #pooling2*2=>7*2*32

        self.img_input = 7*2*32

        # fully-connected layer
        self.al1 = nn.Linear(self.img_input + n_added_input, 256)
        nn.init.kaiming_normal_(self.al1.weight)

        # self.al2 = nn.Linear(512, 512)
        # nn.init.kaiming_normal_(self.al2.weight)

        # self.al3 = nn.Linear(512, 450)
        # nn.init.kaiming_normal_(self.al3.weight)
        
        self.al5 = nn.Linear(256, n_actions)
        nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        # convolution
        conv_activation=F.relu
        h = conv_activation(self.conv1_1(img))
        # h = conv_activation(self.conv1_2(h))
        h = self.pool(h)
        h = conv_activation(self.conv2_1(h))
        # h = conv_activation(self.conv2_2(h))
        h = self.pool(h)
        # h = conv_activation(self.conv3_1(h))
        # h = conv_activation(self.conv3_2(h))
        # h = self.pool(h)
        
        # reshape(fully-connected layerで扱える形にする)
        h = h.view(-1,self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h,sen), axis=1)
        
        # fully-connected layer
        full_activation=F.relu
        h = full_activation(self.al1(h))
        # h = full_activation(self.al2(h))
        # h = full_activation(self.al3(h))
        q = pfrl.action_value.DiscreteActionValue(self.al5(h)) #Q値を出力

        return q

class Q_Func_Optuna(nn.Module):
    def __init__(self, 
    conv_num,mid_layer_num,mid_units1,mid_units2,mid_units3,cnv_act,ful_act,
    n_actions,n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.conv_num = conv_num
        self.mid_layer_num = mid_layer_num
        self.cnv_act = cnv_act
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Q_Func_Optuna, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        # convolution
        if conv_num == 1:
            self.conv1_1 = nn.Conv2d(n_input_channels, 16, 17) #48*27*3=>32*11*16
            nn.init.kaiming_normal_(self.conv1_1.weight)
            
            self.conv1_2 = nn.Conv2d(16, 64, 10) #32*11*16=>23*2*64
            nn.init.kaiming_normal_(self.conv1_2.weight)

            #pooling2*2=>12*1*64

            self.img_input = 12*1*64

        elif conv_num == 2:
            self.conv1_1 = nn.Conv2d(n_input_channels, 8, 5) #48*27*3=>44*23*8
            nn.init.kaiming_normal_(self.conv1_1.weight)
            
            self.conv1_2 = nn.Conv2d(8, 16, 5) #44*23*8=>40*19*16  
            nn.init.kaiming_normal_(self.conv1_2.weight)

            #pooling2*2=>20*10*16

            self.conv2_1 = nn.Conv2d(16, 64, 5) #20*10*16=>16*6*64
            nn.init.kaiming_normal_(self.conv2_1.weight)

            self.conv2_2 = nn.Conv2d(64, 128, 5) #16*6*64=>12*2*128
            nn.init.kaiming_normal_(self.conv2_2.weight)

            #pooling2*2=>6*1*128

            self.img_input = 6*1*128

        elif conv_num == 3:
            self.conv1_1 = nn.Conv2d(n_input_channels, 8, 3) #48*27*3=>46*25*8
            nn.init.kaiming_normal_(self.conv1_1.weight)
            
            self.conv1_2 = nn.Conv2d(8, 16, 3) #46*25*8=>44*23*16  
            nn.init.kaiming_normal_(self.conv1_2.weight)

            #pooling2*2=>22*12*16

            self.conv2_1 = nn.Conv2d(16, 32, 3) #22*12*16=>20*10*32
            nn.init.kaiming_normal_(self.conv2_1.weight)

            self.conv2_2 = nn.Conv2d(32, 64, 3) #20*10*32=>18*8*64
            nn.init.kaiming_normal_(self.conv2_2.weight)

            #pooling2*2=>9*4*64

            self.conv3_1 = nn.Conv2d(64, 128, 2) #9*4*64=>8*3*128
            nn.init.kaiming_normal_(self.conv3_1.weight)

            self.conv3_2 = nn.Conv2d(128, 256, 2) #8*3*128=>7*2*256
            nn.init.kaiming_normal_(self.conv3_2.weight)

            #pooling2*2=>4*1*256

            self.img_input = 4*1*256

        # fully-connected layer
        if mid_layer_num == 1:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(mid_units1, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif mid_layer_num == 2:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif mid_layer_num == 3:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h,sen), axis=1)

        # fully-connected layer
        if self.mid_layer_num == 1:
            h = self.ful_act(self.al1(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))
        elif self.mid_layer_num == 2:
            h = self.ful_act(self.al1(h))
            h = self.ful_act(self.al2(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))
        elif self.mid_layer_num == 3:
            h = self.ful_act(self.al1(h))
            h = self.ful_act(self.al2(h))
            h = self.ful_act(self.al3(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))

        return q

class MLP(nn.Module):
    def __init__(self, n_actions,n_input=1296,n_added_input=0):
        self.n_actions = n_actions
        self.n_input = n_input
        self.n_added_input = n_added_input
        super(MLP, self).__init__()

        # fully-connected layer
        self.al1 = nn.Linear(n_input + n_added_input, 863)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(863, 635)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(635, 411)
        nn.init.kaiming_normal_(self.al3.weight)
        
        self.al5 = nn.Linear(411, n_actions)
        nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):#state:入力情報
        # fully-connected layer
        h = F.relu(self.al1(h))
        h = F.relu(self.al2(h))
        h = F.relu(self.al3(h))
        q = pfrl.action_value.DiscreteActionValue(self.al5(h))

        return q

class MLP_Optuna(nn.Module):
    def __init__(self, 
    mid_layer_num,mid_units1,mid_units2,mid_units3,ful_act,
    n_actions,n_input=1296,n_added_input=0):
        self.mid_layer_num = mid_layer_num
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input = n_input
        self.n_added_input = n_added_input
        super(MLP_Optuna, self).__init__()

        # fully-connected layer
        if mid_layer_num == 1:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(mid_units1, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif mid_layer_num == 2:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif mid_layer_num == 3:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

    def forward(self, state):
        # fully-connected layer
        if self.mid_layer_num == 1:
            h = self.ful_act(self.al1(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))
        elif self.mid_layer_num == 2:
            h = self.ful_act(self.al1(h))
            h = self.ful_act(self.al2(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))
        elif self.mid_layer_num == 3:
            h = self.ful_act(self.al1(h))
            h = self.ful_act(self.al2(h))
            h = self.ful_act(self.al3(h))
            q = pfrl.action_value.DiscreteActionValue(self.al5(h))

        return q

class Dueling_Q_Func(nn.Module):
    def __init__(self, n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Dueling_Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv_num = 2
        self.mid_layer_num = 2
        self.cnv_act = F.relu
        self.ful_act = F.relu

        # convolution
        if self.conv_num == 1:
            channels=[16, 64] #各畳込みでのカーネル枚数
            kernels=[5, 5] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels=[8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels=[5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels=[8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels=[3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        # Advantage
        if self.mid_layer_num == 1:
            units=[580] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(units[0], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        elif self.mid_layer_num == 2:
            units=[512, 512] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(units[1], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        elif self.mid_layer_num == 3:
            units=[512, 512, 512] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(units[1], units[2])
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(units[2], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        # State Value
        if self.mid_layer_num == 1:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl5 = nn.Linear(units[0], 1)
            nn.init.kaiming_normal_(self.vl5.weight)
        
        elif self.mid_layer_num == 2:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.vl2.weight)

            self.vl5 = nn.Linear(units[1], 1)
            nn.init.kaiming_normal_(self.vl5.weight)
        
        elif self.mid_layer_num == 3:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.vl2.weight)

            self.vl3 = nn.Linear(units[1], units[2])
            nn.init.kaiming_normal_(self.vl3.weight)

            self.vl5 = nn.Linear(units[2], 1)
            nn.init.kaiming_normal_(self.vl5.weight)

    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
            # img = F.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h,sen), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            ha = self.ful_act(self.al1(h))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 2:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 3:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ha = self.ful_act(self.al3(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            hs = self.ful_act(self.vl3(hs))
            ys = self.vl5(hs)

        batch_size = img.shape[0]#=1
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = pfrl.action_value.DiscreteActionValue(ya + ys)

        return q

class Dueling_Q_Func_Optuna(nn.Module):
    def __init__(self, 
    conv_num, mid_layer_num, mid_units1, mid_units2, mid_units3, cnv_act, ful_act, 
    n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.conv_num = conv_num
        self.mid_layer_num = mid_layer_num
        self.cnv_act = cnv_act
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Dueling_Q_Func_Optuna, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # convolution
        if self.conv_num == 1:
            channels = [16, 64] #各畳込みでのカーネル枚数
            kernels = [5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels = [8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels = [5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels = [8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        # Advantage
        if self.mid_layer_num == 1:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(mid_units1, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 2:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 3:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        # State Value
        if self.mid_layer_num == 1:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl5 = nn.Linear(mid_units1, 1)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 2:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.vl5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 3:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.vl3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.vl5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:, :-self.n_added_input]
            sen = state[:, -self.n_added_input:]
        else:
            img = state
        
        img = torch.reshape(img, (-1, self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h, sen), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            ha = self.ful_act(self.al1(h))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 2:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 3:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ha = self.ful_act(self.al3(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            hs = self.ful_act(self.vl3(hs))
            ys = self.vl5(hs)

        batch_size = img.shape[0]
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = pfrl.action_value.DiscreteActionValue(ya + ys)

        return q

# 畳み込み・プーリングを終えた画像の1次元入力数の計算
def calculate(img_width, img_height, channels, kernels, pool_info):
    cnv_num = len(channels) # 畳み込み回数
    pool_interval = pool_info[0] # プーリングする間隔(何回の畳み込みごとか)
    for i in range(cnv_num):
        img_width = img_width - (kernels[i] - 1)
        img_height = img_height - (kernels[i] - 1)
        if (i + 1) % pool_interval == 0:
            img_width = math.ceil(img_width / 2)
            img_height = math.ceil(img_height / 2)
    img_input = img_width * img_height * channels[-1]
    # print(img_width, img_height, img_input)
    return img_input