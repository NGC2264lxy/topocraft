# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/2 13:23
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : reply_buffer.py
# @Software: PyCharm

import numpy as np
import torch

##两个Agent共用的经验池
class ReplyBuffer:
    def __init__(self, args):
        self.N = args.N #节点数
        self.obs_dim = args.obs_dim #观测值维度
        self.state_dim = args.state_dim   #8*8=64
        self.episode_limit = args.episode_limit  #每个训练周期的最大步数 1
        self.batch_size = args.batch_size  #每次训练使用的样本数
        self.limit =16 # 切换批次时使用
        self.episode_num = 0  #训练周期索引
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)
# 字典初始化的配置函数
    def reset_buffer(self):
        self.buffer = {'obs_topo': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       'obs_route': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       
                       # 价值
                       'v_topo': np.empty([self.batch_size, self.episode_limit , self.N]),
                       'v_route': np.empty([self.batch_size, self.episode_limit , self.N]),
                     
                       # 动作
                       'a_topo': np.empty([self.batch_size, self.episode_limit, self.N,4]),  # TopoAgent 的动作 二维数组 每列选四个


                    #    'a_route': np.empty([self.batch_size, self.episode_limit, self.N]),  # RouteAgent 的动作
                       # 动作的概率数组  与策略更新有关
                       'a_logprob_topo': np.empty([self.batch_size, self.episode_limit, self.N,4]),
                        # 动作的概率数组  与策略更新有关
                       'a_logprob_route': np.empty([self.batch_size, self.episode_limit, self.N,self.N]),
                       
                       # 奖励
                       'r_topo': np.empty([self.batch_size, self.episode_limit, self.N]),      #环境奖励
                       'r_route': np.empty([self.batch_size, self.episode_limit, self.N]),      #环境奖励
                       
                       'rnd_r_topo': np.empty([self.batch_size, self.episode_limit, self.N]),  # RND 奖励
                       'rnd_r_route': np.empty([self.batch_size, self.episode_limit, self.N]),  # RND 奖励
                       

                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
                               # 初始化为列表
                       
        self.buffer['a_route'] = [[] for _ in range(self.batch_size)]
        # 字典，其中包含多个键，每个键对应一个 NumPy 数组，
        # 用于存储每个 episode 的不同数据
        self.episode_num = 0
# 数据传入字典的函数
    def store_transition(self, episode_step, obs_topo, obs_route, v_topo,v_route, a_topo,a_route, a_logprob_topo,
                         a_logprob_route, r_topo, r_route, rnd_r_topo, rnd_r_route, done_n):
        self.buffer['obs_topo'][self.episode_num % self.limit][episode_step] = obs_topo
        self.buffer['obs_route'][self.episode_num % self.limit][episode_step] = obs_route

        self.buffer['v_topo'][self.episode_num % self.limit][episode_step] = v_topo.cpu().numpy().squeeze()
        self.buffer['v_route'][self.episode_num % self.limit][episode_step] = v_route.cpu().numpy().squeeze()
        self.buffer['a_topo'][self.episode_num % self.limit][episode_step] = a_topo.cpu().numpy()
        # self.buffer['a_route'][self.episode_num][episode_step] = a_route.cpu().numpy().squeeze()
        # 将 a_route 字典格式转换为张量
        formatted_a_route = torch.full((8, 8), 0, dtype=torch.float)  # 假设有 8 个 POD
        for src in a_route:
            for dst in a_route[src]:
                if a_route[src][dst] is not None:
                    formatted_a_route[src, dst] = a_route[src][dst]


        # 确保为新的 episode 初始化存储结构
        if len(self.buffer['a_route']) <= self.episode_num:
            self.buffer['a_route'].append([])  # 为新的 episode 添加空列表

        # 将转换后的张量存储到 self.buffer 中
        self.buffer['a_route'][self.episode_num % self.limit].append(formatted_a_route)

        self.buffer['a_logprob_topo'][self.episode_num % self.limit][episode_step] = a_logprob_topo.cpu().numpy()
        self.buffer['a_logprob_route'][self.episode_num % self.limit][episode_step] = a_logprob_route.cpu().numpy()
        self.buffer['r_topo'][self.episode_num % self.limit][episode_step] = r_topo
        self.buffer['r_route'][self.episode_num % self.limit][episode_step] = r_route
        self.buffer['rnd_r_topo'][self.episode_num % self.limit][episode_step] = rnd_r_topo
        self.buffer['rnd_r_route'][self.episode_num % self.limit][episode_step] = rnd_r_route
        self.buffer['done_n'][self.episode_num % self.limit][episode_step] = done_n

    # def store_last_value(self, episode_step, v_topo):
    #     self.buffer['v_n'][self.episode_num][episode_step] = v_n
    #     self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_topo'  :
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long).to('cuda')
            elif key == 'a_route' :
                # 仅对非空的 episode 进行 stack
                non_empty_episodes = [torch.stack(episode) for episode in self.buffer['a_route'] if episode]
                max_dim=4
                if non_empty_episodes:
                    for i in range(len(non_empty_episodes)):
                        if non_empty_episodes[i].shape[0] < max_dim:
                           non_empty_episodes[i]=torch.nn.functional.pad(non_empty_episodes[i],(0,0,0,0,0,max_dim-non_empty_episodes[i].shape[0]),value=0)
                        else:
                            max_dim=non_empty_episodes[i].shape[0]
                    batch[key] = torch.stack(non_empty_episodes).to('cuda')
                else:
                    batch[key] = torch.empty((0, 8, 8), dtype=torch.long).to('cuda')  # 根据实际期望的维度设置
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32).to('cuda')
        return batch
        

    
