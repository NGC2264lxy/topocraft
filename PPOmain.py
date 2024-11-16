from argparse import ArgumentParser
import gym
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl
from PPOmodel import parse_args,get_device,training_loop,testing_loop,MyPPO

os.chdir("D:/meteor/PPO")
MODEL_PATH="D:/meteor/PPO/save_model.pth"

if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    print(args)
    # 选取agent
    agent_id=1

    # 获取设备
    device = get_device()

    # 创建环境  倒立摆游戏
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # 创建模型(actor和critic)
    # observation_space.shape：模型的输入维度   action_space：模型的输出维度
    model = MyPPO(in_shape=(1, 1), n_actions=env.action_space.n, rnd_hidden_dim=args.rnd_hidden_dim).to(device)
    
    # model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)

    training_loop(agent_id,env, model,args[""], args["max_iterations"], args["n_actors"], args["horizon"],
                  args["gamma"], args["epsilon"], args["n_epochs"], args["batch_size"],
                  args["lr"], args["c1"], args["c2"], device, env_name=env_name, seed=args["seed"])
    # 加载最佳模型
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)
    # 从 MODEL_PATH 路径加载之前训练好的权重，并将这些权重应用到新创建的模型中。
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 测试
    env = gym.make(env_name, render_mode="human")
    testing_loop(agent_id,env, model, args["n_test_episodes"], device)
    env.close()

#在vscode运行时，只能同一时间打开一个，否则需要在终端运行
#终端运行指令
# C:/Users/antl-pc/.conda/envs/meteor/python.exe d:/meteor/PPO/main.py
