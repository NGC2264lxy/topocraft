from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical

from torch_geometric.nn import GATv2Conv
from scipy.special import kl_div
from RND import RNDModel
from replay_buffer import ReplyBuffer

MODEL_PATH="D:/meteor/PPO/save_model.pth"

# Trick: orthogonal initialization  正交初始化
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)



def parse_args():
    """解析参数"""
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16 )
    parser.add_argument("--max_iterations", type=int, help="训练迭代次数", default=100)
    parser.add_argument("--n_epochs", type=int, help="每次迭代的训练轮数", default=3)

    parser.add_argument("--n_actors", type=int, help="actor数量", default=8)
    parser.add_argument("--horizon", type=int, help="每个actor的时间戳数量", default=1) # 假定目前1个样本只有1个时间步

    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    
    parser.add_argument("--lr", type=float, help="学习率", default=2.5 * 1e-4)
    parser.add_argument("--gamma", type=float, help="折扣因子gamma", default=0.99)
    parser.add_argument("--c1", type=float, help="损失函数价值函数的权重", default=1)
    parser.add_argument("--c2", type=float, help="熵奖励的权重", default=0.01)
    parser.add_argument("--c3", type=float, help="RND损失的权重", default=0.1)
    
    parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=5)

    parser.add_argument("--num_pods", type=int, help="Number of pod in system", default=8)

    # parser.add_argument("--num_heads", type=int, help="Number of attention heads", default=6)  # 下面直接赋值了4个头
    parser.add_argument("--gat_hidden_dim", type=int, help="GAT hidden layer dimension", default=128)
    parser.add_argument("--gat_output_dim", type=int, help="GAT output layer dimension", default=4)
    parser.add_argument("--state_dim", type=int, help="State dimension", default=256)

    parser.add_argument("--rnd_hidden_dim", type=int, default=128)  # RND 隐藏层维度

    return vars(parser.parse_args())

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found GPU device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU found: Running on CPU")
    return device

# 装饰器下所有操作不记录梯度
# 把两个Agent放在一起顺序执行
@torch.no_grad()
def run_timestamps(env,model_topo, model_route,args ,ReplyBuffer,socket_connection,horizen, device):

# def run_timestamps(env,model_topo, model_route,args,ReplyBuffer,horizen,batch_size,device):
    # episode_step 是 timestamps 内的一个计数器。
    # render渲染标志  False
    """针对给定数量的时间戳在给定环境中运行给定策略。 
    返回具有状态、动作和奖励的缓冲区。"""
    timestamps=horizen  # 1
    # 重置环境env，并获取初始状态。env.reset()通常返回一个包含状态的元组
    state = env.reset()
    
    state_topo = state[0]  # list  获取 Topo-Agent 的状态
    state_route = state[1]  # list  获取 Route-Agent 的状态
    # edge_index = env.get_edge_index()  # numpy
    # 运行时间戳并收集状态、动作、奖励和终止
    for ts in range(timestamps):  # timestamps=1 ，一个样本目前是一个时间步
         #  将当前状态state转换为PyTorch张量，增加一个维度（批次维度），并将其类型转换为浮点数
        state_tensor_topo = torch.tensor(state_topo, dtype=torch.float32).unsqueeze(0).to(device)
        print("Shape of state_tensor_topo",state_tensor_topo.shape)
        edge_index = env.get_edge_index()  # numpy
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)  #numpy转torch
         #行动概率（logits）
        action_topo, action_probs_topo, value_topo = model_topo(state_tensor_topo, edge_index)
         # 1 是指在环境中选择topo对应的交互    
        # env.selected_matrix = env.step(action_topo.item(),1,socket_connection)#action与environment交互
        action_topo_single = action_topo[0].cpu().numpy()  # 提取批次中的第一个样本的动作
        
        env.selected_matrix = env.step(action_topo_single,1,socket_connection)#action与environment交互
  
        routing_table = env.generate_routing_table()  # 基于 selected_matrix 生成 routing_table
         # 执行 Route-Agent 动作，基于 Topo-Agent 的 selected_matrix 
        state_tensor_route =torch.tensor(state_route,dtype=torch.float32).unsqueeze(0).to(device)
        action_route, action_probs_route, value_route = model_route(state_tensor_route, edge_index,routing_table)
        # Route-Agent 与环境交互，获取新的状态
        # 以下 reward=(lu_reward, path_reward, total_reward)
    
        # new_state, reward ,done = env.step(action_route,2,socket_connection)

        new_state, reward ,done = env.step(action_route,2,socket_connection)
        # 输出 reward 及其各部分
        print("Reward:", reward)
        print("LU Reward:", reward[0])
        print("Path Reward:", reward[1])
         # 计算 RND 奖励（假设两者共享 RND 模型）
        rnd_reward_topo = model_topo.rnd.calculate_rnd_loss(state_tensor_topo).item()
        rnd_reward_route = model_route.rnd.calculate_rnd_loss(state_tensor_route).item()
        # (s, a, r, t)渲染到环境或存储到buffer   后期移动到reply_buffer中
        ReplyBuffer.store_transition(
            episode_step=ts,
            obs_topo=state_topo,  # Topo-Agent 的观测
            obs_route=state_route,  # Route-Agent 的观测
            v_topo=value_topo,  # Topo-Agent 价值
            v_route=value_route,
            a_topo=action_topo,  # Topo-Agent 动作
            a_route=action_route,  # Route-Agent 动作
            a_logprob_topo=action_probs_topo, 
            a_logprob_route=action_probs_route,  # 两个 Agent 的动作概率
            r_topo=reward[0], 
            r_route=reward[1], 
            rnd_r_topo=rnd_reward_topo,
            rnd_r_route=rnd_reward_route,  # 两个 Agent 的 RND 奖励
            done_n=done  # 是否结束标志
        )

        # 更新当前状态
        print("Shape of nmewstate[0]",new_state[0])
        state_topo = new_state[0]
        state_route = new_state[1]

        # 如果episode终止或被截断，则重置环境
        if done:
            state = env.reset()
            state_topo = state[0]  # 获取 Topo-Agent 的状态
            state_route = state[1]  # 获取 Route-Agent 的状态
    ReplyBuffer.episode_num += 1  # 开始新的回合时增加回合编号

    return 
# 链路利用率来表征每个节点的特征，将每个节点的特征定义为它与其他节点之间链路的利用率。
# 链路负载同理
class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super(GATNet, self).__init__()
        # 输入特征转为隐藏特征
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        # 8 -> 4*100
        # 隐藏特征转为输出特征
        self.gat2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, concat=True)
        # 4*100 -> 
    # edge_index：边索引矩阵    
    def forward(self, x, edge_index):
        # x: (batch_size, num_nodes, input_dim) -> (batch_size * num_nodes, input_dim)
        print("Shape of x0:", x.shape) 
        batch_size, num_nodes, input_dim = x.size()
        x = x.view(-1, input_dim)
        print("Shape of x1:", x.shape) 
        x = self.gat1(x, edge_index)
        print("Shape of x2:", x.shape) 
        # x: (batch_size * num_nodes, input_dim) 8*8 -> ((batch_size * num_nodes, hidden_dim * heads)) 8*100
        x = F.leaky_relu(x)
        # x : (batch_size * num_nodes, hidden_dim * heads) 8*400
        x = self.gat2(x, edge_index)
        print("Shape of x3:", x.shape) 
        # x :(batch_size * num_nodes, hidden_dim * heads) 8*400 -> (batch_size * num_nodes, output_dim)
        # 将结果 reshape 回原来的批次形式
        x = x.view(batch_size, num_nodes, -1)
        print("Shape of x4:", x.shape) 
        # x : batch_size,num_nodes, output_dim)
        #数据首先通过self.gat1进行图卷积，激活函数为 ReLU，
        #然后通过 self.gat2 进行第二次图卷积，返回最终的节点特征。
        return x
    
# Actor类使用GATNet来实现一个actor网络，该网络将节点特征映射到概率分布
class Actor(nn.Module):
    def __init__(self, state_dim, gat_hidden_dim, gat_output_dim, num_heads):
        super(Actor, self).__init__()
        self.gat_net = GATNet(
            input_dim=state_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            heads=num_heads
        )

    def forward(self, x, edge_index):
        print("Shape of x5:", x.shape) 
        x = self.gat_net(x, edge_index)
        print("Shape of x6:", x.shape) 
        prob = torch.softmax(x, dim=-1)
        return prob
        # (batch_size, num_nodes, gat_output_dim)。
    
# Critic 类实现了一个 critic 网络，该网络输出节点的价值估计
class Critic(nn.Module):
    def __init__(self,state_dim, gat_hidden_dim, gat_output_dim, num_heads):
        super(Critic, self).__init__()
        self.gat_net = GATNet(
            input_dim=state_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            heads=num_heads
        )
        self.fc = nn.Linear(gat_output_dim, 1)   #将每个节点的 n 个特征变为 1 个标量值
    def forward(self, x, edge_index):
        print("Shape of x7:", x.shape) 
        x = self.gat_net(x, edge_index) 
        print("Shape of x8:", x.shape) 
        # x :(num_nodes, hidden_dim * heads) 8*400 -> (num_nodes, output_dim) 8*100
        value = self.fc(x)
        return value

class MyTopoAgent(nn.Module):
    """
    PPO模型的实现。 
    相同的代码结构即可用于actor，也可用于critic。
    """

    def __init__(self,in_shape, n_actions, hidden_d=100, share_backbone=False, rnd_hidden_dim=128,num_choices=7, num_selected=4):
         # 父类构造函数
        super(MyTopoAgent, self).__init__()

        # 属性
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

        self.num_choices = num_choices  # 每列有7行可供选择
        self.num_selected = num_selected  # 每列选4个
        self.num_columns = 8  # 我们有8列需要操作

        # 共享策略主干和价值函数
        """使用NumPy的np.prod函数计算输入形状in_shape中所有维度的乘积，
        得到输入的总维度in_dim。这通常用于将输入数据展平，以便输入到神经网络中"""

        in_dim = np.prod(in_shape)
    
        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )
        
        #nn.Identity是无操作层
        self.backbone = to_features() if self.share_backbone else nn.Identity()

        # 创建 GAT Actor 和 Critic
        self.actor = Actor(state_dim=in_shape[1], gat_hidden_dim=hidden_d, gat_output_dim=num_choices, num_heads=1)
        self.critic = Critic(state_dim=in_shape[1], gat_hidden_dim=hidden_d, gat_output_dim=hidden_d, num_heads=1)
        # RND
        print("in_dim:",in_dim)

        # 获取设备
        device = get_device()
        self.rnd = RNDModel(device,rnd_input_size=in_dim, rnd_output_size=rnd_hidden_dim)  # Initialize RND



    def forward(self, x, edge_index):
        print("Shape of x9:", x.shape) 
        features = self.backbone(x)
        print("Shape of features:", features.shape)  #1,8,8
        action_probs = self.actor(features, edge_index)
        print("Shape of action_probs:", action_probs.shape)  #1,8,7
        # 对每列使用 multinomial 抽取4个动作（不重复的行选择）
        action = torch.stack([torch.multinomial(action_probs[:, col], self.num_selected, replacement=False) 
                              for col in range(self.num_columns)], dim=1)
        # action : (batch_size, num_columns, num_selected)  (1, 8, 4)
        selected_probs=action_probs.gather(1,action)
        selected_probs=selected_probs.squeeze(-1)
        print("Shape of selected_probs",selected_probs.shape)
         # 获取状态值
        value = self.critic(features, edge_index)
        return action, selected_probs, value
    
class MyRouteAgent(nn.Module):
    """
    PPO模型的实现。 
    相同的代码结构即可用于actor，也可用于critic。
    """

    def __init__(self,in_shape,hidden_d=100, share_backbone=False,rnd_hidden_dim=128,num_choices=2,num_heads=1):
         # 父类构造函数
        super(MyRouteAgent, self).__init__()

        # 属性
        self.in_shape = in_shape
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

        self.num_choices = num_choices  
        self.num_heads = num_heads
        

        # 共享策略主干和价值函数
        """使用NumPy的np.prod函数计算输入形状in_shape中所有维度的乘积，
        得到输入的总维度in_dim。这通常用于将输入数据展平，以便输入到神经网络中"""

        in_dim = np.prod(in_shape)
    
        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )
        
        #nn.Identity是无操作层
        self.backbone = to_features() if self.share_backbone else nn.Identity()

        # 创建 GAT Actor 和 Critic
        self.actor = Actor(state_dim=in_shape, gat_hidden_dim=hidden_d, gat_output_dim=num_choices, num_heads=num_heads)
        self.critic = Critic(state_dim=in_shape, gat_hidden_dim=hidden_d, gat_output_dim=hidden_d, num_heads=num_heads)
        # RND
        # 获取设备
        device = get_device()
        self.rnd = RNDModel(device,rnd_input_size=in_dim, rnd_output_size=rnd_hidden_dim)  # Initialize RND



    def forward(self, x, edge_index,routing_table):
        features = self.backbone(x)
        action_probs = self.actor(features, edge_index)
        # 打印action_probs的形状，确认其维度
        print("Shape of action_probs:", action_probs.shape)
        # 生成每个POD的动作（即流量分配的比例）
        actions = {}
        for src in range(8):  # 假设有8个POD
            actions[src] = {}
            for dst in range(8):
                if src != dst:
                    routes = routing_table[src].get(dst, [])  # 获取src到dst的可用路径
                    if routes:  # 如果有路径
                        route_probs = action_probs[0,src,  :len(routes)]  # 限制选择的路径数量
                        route_probs = torch.softmax(route_probs, dim=-1)  # 归一化概率
                        actions[src][dst] = Categorical(route_probs).sample()  # 根据概率选择路径
                    else:
                        actions[src][dst] = None  # 如果没有可选路径

        value = self.critic(features, edge_index)
        return actions, action_probs, value

def training_loop(env, model_topo, model_route,args,ReplyBuffer,socket_connection, max_iterations, n_actors, 
                   horizon,lamda, gamma, epsilon, n_epochs, batch_size, lr,c1, c2, c3,device,seed):
 
# def training_loop(env, model_topo, model_route,args,ReplyBuffer,max_iterations, n_actors, 
                #   horizon,lamda, gamma, epsilon, n_epochs, batch_size, lr,c1, c2, c3,device,seed):
    """使用最多n个时间戳的多个actor在给定环境中训练模型。"""

    max_reward = float("-inf") # 初始化的奖励值是负无穷
    # 使用 PyTorch 中的 LinearLR 调度器来逐步调整优化器的学习率。
    # 该调度器会将学习率从 1 线性减小到 0，在总共 max_iterations * n_epochs 的训练过程中完成。
        
    optimizer_topo = Adam(model_topo.parameters(), lr=lr)
    optimizer_route = Adam(model_route.parameters(), lr=lr)
    
    scheduler_topo = LinearLR(optimizer_topo, 1, 0, max_iterations * n_epochs)
    scheduler_route = LinearLR(optimizer_route, 1, 0, max_iterations * n_epochs)


    # 训练循环
    for iteration in range(max_iterations):
        ReplyBuffer.reset_buffer()  # 重置经验池
        # 使用当前策略收集所有actor的时间戳
        for actor in range(n_actors):
            # 运行几轮优化 3
            # run_timestamps( env, model_topo, model_route, args, ReplyBuffer,socket_connection, 
            # horizon, render=False, device=device)
            # horizon是1，所以一次run_timestamps 交互出1个样本，即一个批次
            while ReplyBuffer.episode_num < 16:  # 模拟 do while 
                run_timestamps( env, model_topo, model_route, args, ReplyBuffer, socket_connection,
                        horizon, device)
            for epoch in range(n_epochs):
                # 计算累积奖励并刷新缓冲区（函数中规范化了累积奖励）
                # avg_rew_topo,avg_rew_route = compute_cumulative_rewards(ReplyBuffer.buffer, gamma)
                batch_data = ReplyBuffer.get_training_data()
                # np.random.shuffle(batch_data['obs_topo'])
                 
                # 提取当前批次的数据
                topo_state_batch = batch_data['obs_topo']
                route_state_batch = batch_data['obs_route']

                topo_actions_batch = batch_data['a_topo']
                print("Shape!",topo_actions_batch.shape)
                topo_log_probs_batch = batch_data['a_logprob_topo']

                topo_values_batch = batch_data['v_topo']
                topo_rewards_batch = batch_data['r_topo']

                route_actions_batch = batch_data['a_route']
                route_log_probs_batch = batch_data['a_logprob_route']

                route_values_batch = batch_data['v_route']
                route_rewards_batch = batch_data['r_route']

                done_batch = batch_data['done_n']


                ########### Topo-Agent 优化  ###############
                optimizer_topo.zero_grad()
                print("Shape of topo_state_batch",topo_state_batch.shape)
                # 计算 RND 损失
                rnd_loss_topo = model_topo.rnd.calculate_rnd_loss(topo_state_batch)

                # topo_log_prob = Categorical(topo_log_probs_batch).log_prob(topo_actions_batch)
                topo_log_prob = torch.log(topo_actions_batch)
                
                topo_combined_reward = topo_rewards_batch + lamda * rnd_loss_topo
                topo_advantage = topo_combined_reward + (1 - done_batch) * gamma * topo_values_batch - topo_values_batch

            
                # 计算 Topo-Agent 损失
                topo_value_loss = F.mse_loss(topo_values_batch, topo_rewards_batch)
                topo_ratio = (topo_log_prob - topo_log_prob.detach()).exp()
                topo_ratio=torch.mean(topo_ratio,dim=-1)
                topo_surrogate_loss = topo_ratio * topo_advantage.detach()
                ###### 注意下面已经有负号了
                topo_policy_loss = -torch.mean(torch.min(topo_surrogate_loss, torch.clamp(topo_ratio, 1 - epsilon, 1 + epsilon) * topo_advantage.detach()))

                # 总损失
                # 计算熵
                topo_entropy = Categorical(topo_log_probs_batch).entropy().mean()
                topo_loss = topo_policy_loss +c1 * topo_value_loss - c2 *topo_entropy+c3* rnd_loss_topo
                topo_loss.backward()
                optimizer_topo.step()
                

                ########### Route-Agent 优化 ###############
                optimizer_route.zero_grad()

                # 计算 RND 损失
                rnd_loss_route = model_route.rnd.calculate_rnd_loss(route_state_batch)

                # 计算当前策略的动作对数概率
                #  route_log_prob = Categorical(route_log_probs_batch).log_prob(route_actions_batch)
                route_log_prob = torch.log(route_actions_batch)

                # 计算 Route-Agent 的综合奖励（考虑 RND 奖励）
                route_combined_reward = route_rewards_batch + lamda * rnd_loss_route

                # 计算优势函数
                route_advantage = route_combined_reward + (1 - done_batch) * gamma * route_values_batch - route_values_batch

                # 计算 Route-Agent 的损失
                route_value_loss = F.mse_loss(route_values_batch, route_rewards_batch)
                route_ratio = (route_log_prob - route_log_prob.detach()).exp()

                # 计算 PPO 的 surrogate loss（策略损失）
                route_surrogate_loss = route_ratio * route_advantage.detach()
                route_policy_loss = -torch.mean(torch.min(route_surrogate_loss, torch.clamp(route_ratio, 1 - epsilon, 1 + epsilon) * route_advantage.detach()))

                # 总损失
                route_entropy = Categorical(route_log_probs_batch).entropy().mean()
                route_loss = route_policy_loss + c1 * route_value_loss - c2 *route_entropy + c3* rnd_loss_route
                route_loss.backward()
                optimizer_route.step()

                # run一次更新一个buffer内的样本
                run_timestamps( env, model_topo, model_route, args, ReplyBuffer, socket_connection,
                                        horizon, device)
            scheduler_topo.step()
            scheduler_route.step()

        # 保存模型和记录奖励
        # 计算 Topo-Agent 的平均总奖励（外部奖励 + RND 奖励）
        avg_reward_topo = np.mean(topo_combined_reward.detach().cpu().numpy())
        avg_reward_route = np.mean(route_combined_reward.detach().cpu().numpy())

        # 保存 Topo-Agent 模型
        if avg_reward_topo > max_reward:
            max_reward = avg_reward_topo
            torch.save(model_topo.state_dict(), "topo_agent_model.pth")

        # 保存 Route-Agent 模型
        if avg_reward_route > max_reward:
            max_reward = avg_reward_route
            torch.save(model_route.state_dict(), "route_agent_model.pth")

        # Evaluate the model every few iterations
    return model_topo, model_route
                

def testing_loop(env, model_topo, model_route,args,socket_connection, n_episodes,horizon, device):
    # n_episodes=5
    for _ in range(n_episodes):
        run_timestamps( env, model_topo, model_route, args, ReplyBuffer, socket_connection,
                           horizon, device=device)

    
# def compute_cumulative_rewards(buffer, gamma):
#     """
#     给定一个包含状态、策略操作逻辑、奖励和终止的缓冲区，分别计算每个时间步的累积奖励，并将它们代入缓冲区。
#     处理 Topo-Agent 和 Route-Agent 的奖励。
#     """
#     curr_rew_topo = 0.  # Topo-Agent 的累积奖励
#     curr_rew_route = 0.  # Route-Agent 的累积奖励

#     # 反向遍历缓冲区
#     for i in range(len(buffer) - 1, -1, -1):
#         # 获取 Topo-Agent 和 Route-Agent 的奖励及终止标志
#         r_topo, r_route, t = buffer[i][-3], buffer[i][-2], buffer[i][-1]
        
#         # 如果 t=True，说明这一时间步是终止状态，当前累积奖励重置为0
#         if t:
#             curr_rew_topo = 0
#             curr_rew_route = 0
#         else:
#             curr_rew_topo = r_topo + gamma * curr_rew_topo
#             curr_rew_route = r_route + gamma * curr_rew_route

#         # 更新缓冲区中的奖励值
#         buffer[i][-3] = curr_rew_topo  # 更新 Topo-Agent 的累积奖励
#         buffer[i][-2] = curr_rew_route  # 更新 Route-Agent 的累积奖励

#     # 计算平均奖励用于日志记录
#     avg_rew_topo = np.mean([buffer[i][-3] for i in range(len(buffer))])
#     avg_rew_route = np.mean([buffer[i][-2] for i in range(len(buffer))])

#     # 对 Topo-Agent 和 Route-Agent 的累积奖励分别进行规范化
#     mean_topo = np.mean([buffer[i][-3] for i in range(len(buffer))])
#     std_topo = np.std([buffer[i][-3] for i in range(len(buffer))]) + 1e-6

#     mean_route = np.mean([buffer[i][-2] for i in range(len(buffer))])
#     std_route = np.std([buffer[i][-2] for i in range(len(buffer))]) + 1e-6

#     for i in range(len(buffer)):
#         # 规范化 Topo-Agent 的累积奖励
#         buffer[i][-3] = (buffer[i][-3] - mean_topo) / std_topo
#         # 规范化 Route-Agent 的累积奖励
#         buffer[i][-2] = (buffer[i][-2] - mean_route) / std_route

#     return avg_rew_topo, avg_rew_route





