# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/7 22:26
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : main.py
# @Software: PyCharm
import argparse
from envs import HPCEnvironment
import os
import torch
import socket
from Agent import parse_args,get_device,training_loop,testing_loop,MyTopoAgent,MyRouteAgent
from replay_buffer import ReplyBuffer
from RND import RNDModel

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters Setting for GE-PPO in HPC environment")
    parser.add_argument("--max_interact_steps", type=int, default=int(3e6), help="Maximum number of interact steps with OMNeT++")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1, help="Maximum number of steps per episode")
    parser.add_argument("--max_recv_data", type=int, default=8192, help="Maximum number of bytes received from the client")
    parser.add_argument("--num_pods", type=int, default=8, help="The number of pods in network")
    parser.add_argument("--max_path_length", type=int, default=2, help="The maximum length of global paths")
    parser.add_argument("--pod_bandwidth", type=int, default=100, help="The bandwidth of links between pods (Gbps)")
    parser.add_argument("--allocate_interval", type=int, default=1, help="The allocate time interval of traffic")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.5, help="RND reward weight")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")

    parser.add_argument("--max_iterations", type=int, help="训练迭代次数", default=100)
    parser.add_argument("--n_epochs", type=int, help="每次迭代的训练轮数", default=3)
    parser.add_argument("--n_actors", type=int, help="actor数量", default=8)
    parser.add_argument("--horizon", type=int, help="每个actor的时间戳数量", default=1)  # 假定目前一个样本只有一个时间步
    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    
    parser.add_argument("--c1", type=float, help="损失函数价值函数的权重", default=1)
    parser.add_argument("--c2", type=float, help="损失函数熵奖励的权重", default=0.01)
    parser.add_argument("--c3", type=float, help="RND损失的权重", default=0.1) 

    parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=5)

    parser.add_argument("--N", type=int, help="Number of nodes", default=8)
    parser.add_argument("--obs_dim", type=int, help="Observation dimension", default=8)  # 状态都是（8，8）（node，Observation dimension）
    # parser.add_argument("--num_heads", type=int, help="Number of attention heads", default=6)
    parser.add_argument("--gat_hidden_dim", type=int, help="GAT hidden layer dimension", default=128)
    parser.add_argument("--gat_output_dim", type=int, help="GAT output layer dimension", default=4)
    parser.add_argument("--state_dim", type=int, help="State dimension", default=64)
    parser.add_argument("--rnd_hidden_dim", type=int, default=128)  # RND 隐藏层维度

    return parser.parse_args()

os.chdir("D:/meteor/PPO")
MODEL_PATH="D:/meteor/PPO/save_model.pth"


if __name__ == '__main__':
    print(f"well")
    args = parse_args()
    env = HPCEnvironment(args)

    # 获取设备
    device = get_device()

    model_topo = MyTopoAgent(in_shape=(args.num_pods, args.num_pods), 
                        n_actions=env.topology_action_space.shape[0], 
                        ).to(device)

    model_route = MyRouteAgent(in_shape=(args.num_pods, args.num_pods),           
                            ).to(device)

    # 创建经验池
    ReplyBuffer = ReplyBuffer(args)
    

    # 监听
    address = ('127.0.0.1', 8889)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)  # 绑定地址和端口
    server.listen(5)
    print("Waiting for connection...")
    client, addr = server.accept()
    print(f"Accepted connection from {addr}")
    # 创建 Topo 和 Route Agent 模型 


    for cur_step in range(args.max_interact_steps):
        print("cur step: %d / %d " % (cur_step + 1, args.max_interact_steps))
        done = False
        while not done:
            
            data = client.recv(4096).decode()
            print("data:",data)
            # data="False;elsemsg;0,336813,0,0,0,0,0,0;0,0,394526,0,0,0,0,0;0,0,0,366154,0,0,0,0;0,0,0,0,391483,0,0,0;0,0,0,0,0,452172,0,0;0,0,0,0,0,0,408067,0;0,0,0,0,0,0,0,435594;468309,0,0,0,0,0,0,0;0,336813,0,0,0,0,0,0;0,0,394526,0,0,0,0,0;0,0,0,366154,0,0,0,0;0,0,0,0,391483,0,0,0;0,0,0,0,0,452172,0,0;0,0,0,0,0,0,408067,0;0,0,0,0,0,0,0,435594;468309,0,0,0,0,0,0,0;1,0,3,2,5,4,7,6;2,3,0,1,6,7,4,5;3,2,1,0,7,6,5,4;4,5,6,7,0,1,2,3;5,4,7,6,1,0,3,2;6,7,4,5,2,3,0,1;7,6,5,4,3,2,1,0;-1,-1,-1,-1,-1,-1,-1,-1;2.000000"
            env.handle_recvData(data)
            # 训练 Topo-Agent 和 Route-Agent
            # training_loop(env, model_topo, model_route, args, ReplyBuffer,
            #             max_iterations=args.max_iterations, 
            #             n_actors=args.n_actors, 
            #             horizon=args.horizon, 
            #             lamda=args.lamda, 
            #             gamma=args.gamma, 
            #             epsilon=args.epsilon, 
            #             n_epochs=args.n_epochs, 
            #             batch_size=args.batch_size, 
            #             lr=args.lr, 
            #             c1=args.c1, 
            #             c2=args.c2, 
            #             c3=args.c3,
            #             device=device)
            training_loop(env, model_topo, model_route, args, ReplyBuffer, socket_connection=client,
                        max_iterations=args.max_iterations, 
                        n_actors=args.n_actors, 
                        horizon=args.horizon, 
                        lamda=args.lamda, 
                        gamma=args.gamma, 
                        epsilon=args.epsilon, 
                        n_epochs=args.n_epochs, 
                        batch_size=args.batch_size, 
                        lr=args.lr, 
                        c1=args.c1, 
                        c2=args.c2, 
                        c3=args.c3,
                        device=device, 
                        seed=args.seed)

            # 保存最佳模型
            torch.save(model_topo.state_dict(), "best_topo_agent_model.pth")
            torch.save(model_route.state_dict(), "best_route_agent_model.pth")

            # 加载最佳模型进行测试
            model_topo.load_state_dict(torch.load("best_topo_agent_model.pth", map_location=device))
            model_route.load_state_dict(torch.load("best_route_agent_model.pth", map_location=device))

            # 测试模型
            # testing_loop(env, model_topo, model_route, args, args.n_test_episodes, args.horizon, device,socket_connection=client)

            #     server.listen(5)
            #     client, addr = server.accept()
            #     print(f"Accepted connection from {addr}")
            #     print(f"Client socket: {client}")
            
            #    # data = client.recv(args.max_recv_data).decode()
            #     data = client.recv(4096).decode()
            #     print(f"data{data}")
            #      # 将数据传递给环境进行处理
            #     env.update_state(data)  # 这个函数仅用于解析和更新状态
                # topology_data = "0,1,2,3,4,5,6,7;" 
                # client.sendall(topology_data.encode()) 
                # env.handle_recvData(data)
                # env.cal_reward()
                # global_traffic.append(env.cur_traffic)
                # global_configuration.append(env.cur_configuration)
            done = env.get_cur_comFlag()
    client.close()
    server.close()
