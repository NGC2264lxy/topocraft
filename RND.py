# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/3 12:18
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : RND.py
# @Software: PyCharm

from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self,device, rnd_input_size, rnd_output_size):
        super(RNDModel, self).__init__()

        self.input_size = rnd_input_size
        self.output_size = rnd_output_size
        self.device = device  # 添加设备参数

        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ).to(device)  # 移动到指定设备

        self.target = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ).to(device)  # 移动到指定设备


        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False
    # next_obs :执行一个动作之后，智能体观测到的新状态
    def forward(self, next_obs):
        # print(f"next_obs shape before view: {next_obs.shape}")
        next_obs=next_obs.to(self.device)
            # 检查张量维度是否是四维或三维
        if next_obs.dim() == 4:  # 形状为 [batch, d1, h, w]
            next_obs = next_obs.view(next_obs.size(0), next_obs.size(1), -1)
        elif next_obs.dim() == 3:  # 形状为 [batch, h, w]
            next_obs = next_obs.view(next_obs.size(0), -1)
        
        # print(f"next_obs shape after view: {next_obs.shape}")
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
    
    def calculate_rnd_loss(self, next_obs):
       # 计算 RND 损失函数，预测网络输出与目标网络输出的误差
        next_obs = next_obs.to(self.device)  # 确保 next_obs 在同一设备上
        predict_feature, target_feature = self.forward(next_obs)
        # Mean Squared Error
        rnd_loss = F.mse_loss(predict_feature, target_feature, reduction='mean')
        return rnd_loss
    
    def update_target(self, tau=0.001):
       # 软更新：用tau更新目标网络的权重
        for target_param, predictor_param in zip(self.target.parameters(), self.predictor.parameters()):
            target_param.data.copy_(tau * predictor_param.data + (1.0 - tau) * target_param.data)
