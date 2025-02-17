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

class Dueling_Q_Func_Optuna(nn.Module):
    def __init__(self, 
    conv_pool, mid_layer_num, mid_units1, mid_units2, mid_units3, cnv_act, ful_act, 
    n_actions, n_input_channels, n_added_input, img_width, img_height):
        self.conv_pool = conv_pool
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
        if self.conv_pool == 1:
            channels = [8, 16, 32, 64, 128, 256] # 各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3, 2, 2] # 各畳込みでのカーネルサイズ
            self.conv1 = nn.Conv2d(n_input_channels, channels[0], kernels[0]) # 46*25*8
            nn.init.kaiming_normal_(self.conv1.weight)
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1]) # 44*23*16
            nn.init.kaiming_normal_(self.conv2.weight)
            # 22*12*16
            self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2]) # 20*10*32
            nn.init.kaiming_normal_(self.conv3.weight)
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernels[3]) # 18*8*64
            nn.init.kaiming_normal_(self.conv4.weight)
            # 9*4*64
            self.conv5 = nn.Conv2d(channels[3], channels[4], kernels[4]) # 8*3*128
            nn.init.kaiming_normal_(self.conv5.weight)
            self.conv6 = nn.Conv2d(channels[4], channels[5], kernels[5]) # 7*2*256
            nn.init.kaiming_normal_(self.conv6.weight)
            # 4*1*256
            self.img_input = 4*1*256
        
        elif self.conv_pool == 2:
            channels = [16, 32, 64, 128] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3] #各畳込みでのカーネルサイズ
            self.conv1 = nn.Conv2d(n_input_channels, channels[0], kernels[0]) # 46*25*16
            nn.init.kaiming_normal_(self.conv1.weight)
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1]) # 44*23*32
            nn.init.kaiming_normal_(self.conv2.weight)
            # 22*12*32
            self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2]) # 20*10*64
            nn.init.kaiming_normal_(self.conv3.weight)
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernels[3]) # 18*8*128
            nn.init.kaiming_normal_(self.conv4.weight)
            # 9*4*128
            self.img_input = 9*4*128
        
        elif self.conv_pool == 3:
            channels = [16, 32, 64] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3] #各畳込みでのカーネルサイズ
            self.conv1 = nn.Conv2d(n_input_channels, channels[0], kernels[0]) # 46*25*16
            nn.init.kaiming_normal_(self.conv1.weight)
            # 23*13*16
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1]) # 21*11*32
            nn.init.kaiming_normal_(self.conv2.weight)
            # 11*6*32
            self.conv3 = nn.Conv2d(channels[1], channels[2], kernels[2]) # 9*4*64
            nn.init.kaiming_normal_(self.conv3.weight)
            self.img_input = 9*4*64

        elif self.conv_pool == 4:
            channels = [16, 32] #各畳込みでのカーネル枚数
            kernels = [5, 5] #各畳込みでのカーネルサイズ
            self.conv1 = nn.Conv2d(n_input_channels, channels[0], kernels[0]) # 44*23*16
            nn.init.kaiming_normal_(self.conv1.weight)
            # 22*12*16
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1]) # 18*8*32
            nn.init.kaiming_normal_(self.conv2.weight)
            # 9*4*32
            self.img_input = 9*4*32

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
        if self.conv_pool == 1:
            h = self.cnv_act(self.conv1(img))
            h = self.cnv_act(self.conv2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3(h))
            h = self.cnv_act(self.conv4(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv5(h))
            h = self.cnv_act(self.conv6(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_pool == 2:
            h = self.cnv_act(self.conv1(img))
            h = self.cnv_act(self.conv2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3(h))
            h = self.cnv_act(self.conv4(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_pool == 3:
            h = self.cnv_act(self.conv1(img))
            h = self.pool(h)
            h = self.cnv_act(self.conv2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3(h))
            h = h.view(-1, self.img_input)
        elif self.conv_pool == 4:
            h = self.cnv_act(self.conv1(img))
            h = self.pool(h)
            h = self.cnv_act(self.conv2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h, sen), axis=1)

        # 全結合層の構成
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