# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/1 15:01
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from scipy.special import kl_div

# Trick: orthogonal initialization  正交初始化
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


num_heads = 6
gat_hidden_dim = 128
gat_output_dim = 4
state_dim = 256

# 定义Actor/Critic网络的结构
# GATNat类定义了一个使用图注意力网络GAT的神经网络
class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super(GATNet, self).__init__()
        # 输入特征转为隐藏特征
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        # 隐藏特征转为输出特征
        self.gat2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, concat=True)
        
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        #数据首先通过self.gat1进行图卷积，激活函数为 ReLU，
        #然后通过 self.gat2 进行第二次图卷积，返回最终的节点特征。
        return x
    
# Actor类使用GATNet来实现一个actor网络，该网络将节点特征映射到概率分布
class Actor(nn.Module):
    def __init__(self, actor_input_dim):
        super(Actor, self).__init__()
        self.gat_net = GATNet(input_dim=state_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim, heads=num_heads)

    def forward(self, x):
        x = self.gat_net(x)
        prob = torch.softmax(x, dim=-1)
        return prob
    
# Critic 类实现了一个 critic 网络，该网络输出节点的价值估计
class Critic(nn.Module):
    def __init__(self, actor_input_dim):
        super(Critic, self).__init__()
        self.gat_net = GATNet(input_dim=state_dim, hidden_dim=gat_hidden_dim, output_dim=gat_output_dim, heads=num_heads)
        self.fc = nn.Linear(gat_hidden_dim, 1)
        # 一个全连接层，将 GAT 网络的输出映射到一个标量值（即价值）。

    def forward(self, x):
        value = self.fc(x)
        return value

class CentCritic(nn.Module):
    def __init__(self, GATNet, CC_dim):
        super(CentCritic, self).__init__()
        self.GATNet = GATNet
        self.critic = nn.Linear(CC_dim, 1)

    def forward(self, x):
        x = self.GATNet(x)
        value = self.critic(x)
        return value

class TopoAgent(nn.Module):
    def __init__(self):
        super(TopoAgent, self).__init__()

