from argparse import ArgumentParser
import gym
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl

MODEL_PATH="D:/meteor/PPO/save_model.pth"
def parse_args():
    """解析参数"""
    parser = ArgumentParser()

    parser.add_argument("--max_iterations", type=int, help="训练迭代次数", default=100)
    parser.add_argument("--n_actors", type=int, help="actor数量", default=8)
    parser.add_argument("--horizon", type=int, help="每个actor的时间戳数量", default=128)
    parser.add_argument("--epsilon", type=float, help="Epsilon", default=0.1)
    parser.add_argument("--n_epochs", type=int, help="每次迭代的训练轮数", default=3)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32 * 8)
    parser.add_argument("--lr", type=float, help="学习率", default=2.5 * 1e-4)
    parser.add_argument("--gamma", type=float, help="折扣因子gamma", default=0.99)
    parser.add_argument("--c1", type=float, help="损失函数价值函数的权重", default=1)
    parser.add_argument("--c2", type=float, help="损失函数熵奖励的权重", default=0.01)
    parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=5)
    parser.add_argument("--seed", type=int, help="随机种子", default=0)

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
@torch.no_grad()
def run_timestamps(env, model, timestamps=128, render=False, device="cpu"):
    # render渲染标志  False
    """针对给定数量的时间戳在给定环境中运行给定策略。 
    返回具有状态、动作和奖励的缓冲区。"""
    buffer = []
    # 重置环境env，并获取初始状态。env.reset()通常返回一个包含状态的元组
    state = env.reset()[0]

    # 运行时间戳并收集状态、动作、奖励和终止
    for ts in range(timestamps):
         # 将当前状态state转换为PyTorch张量，增加一个维度（批次维度），移动到指定的设备（如CPU或GPU），并将其类型转换为浮点数
        model_input = torch.from_numpy(state).unsqueeze(0).to(device).float()
         #行动概率（logits）
        action, action_logits, value = model(model_input)
         # terminated：是否终止 truncated：是否截断
        new_state, reward, terminated, truncated, info = env.step(action.item())#action与environment交互

        # (s, a, r, t)渲染到环境或存储到buffer
        if render:
            env.render()
        else:
            buffer.append([model_input, action, action_logits, value, reward, terminated or truncated])

        # 更新当前状态
        state = new_state

        # 如果episode终止或被截断，则重置环境
        if terminated or truncated:
            state = env.reset()[0]

    return buffer




class MyPPO(nn.Module):
    """
    PPO模型的实现。 
    相同的代码结构即可用于actor，也可用于critic。
    """

    def __init__(self, in_shape, n_actions, hidden_d=100, share_backbone=False):
        # 父类构造函数
        super(MyPPO, self).__init__()

        # 属性
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

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

        # State action function
        self.actor = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value function
        self.critic = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        action = self.actor(features)
        value = self.critic(features)
        # 通过调用 .sample() 方法，从 Categorical 分布中随机抽取一个动作
        # Categorical(action).sample() 创建了一个分类分布，其中包含一个动作（针对每个状态）的动作 logits 和样本。
        return Categorical(action).sample(), action, value

def training_loop(env, model, max_iterations, n_actors, horizon, gamma, epsilon, n_epochs, batch_size, lr,
                  c1, c2, device, env_name=""):
    """使用最多n个时间戳的多个actor在给定环境中训练模型。"""

    # 开始运行新的权重和偏差
    wandb.init(project="Papers Re-implementations",
               entity="peutlefaire",
               name=f"PPO - {env_name}",
               config={
                   "env": str(env),
                   "number of actors": n_actors,#并行actor的数量
                   "horizon": horizon,#每个轨迹的最大步数
                   "gamma": gamma,
                   "epsilon": epsilon,
                   "epochs": n_epochs,
                   "batch size": batch_size,
                   "learning rate": lr,
                   #c1,c2用来控制损失函数的权重
                   "c1": c1,
                   "c2": c2
               })

    # 训练变量
    max_reward = float("-inf") # 初始化的奖励值是负无穷
    optimizer = Adam(model.parameters(), lr=lr, maximize=True)
    # 使用 PyTorch 中的 LinearLR 调度器来逐步调整优化器的学习率。
    # 该调度器会将学习率从 1 线性减小到 0，在总共 max_iterations * n_epochs 的训练过程中完成。
    scheduler = LinearLR(optimizer, 1, 0, max_iterations * n_epochs)
    # 生成一个从 1 到 0 的等间距数组，总共有 max_iterations 个值。
    anneals = np.linspace(1, 0, max_iterations)

    # 训练循环
    for iteration in range(max_iterations):#100
        buffer = []
        # 从数组中获取当前迭代次数对应的值
        annealing = anneals[iteration]

        # 使用当前策略收集所有actor的时间戳
        for actor in range(1, n_actors + 1):
            buffer.extend(run_timestamps(env, model, horizon, False, device))

        # 计算累积奖励并刷新缓冲区
        avg_rew = compute_cumulative_rewards(buffer, gamma)
        np.random.shuffle(buffer)

        # 运行几轮优化
        for epoch in range(n_epochs):
            # 分批次训练
            for batch_idx in range(len(buffer) // batch_size):
                start = batch_size * batch_idx
                end = start + batch_size if start + batch_size < len(buffer) else -1
                batch = buffer[start:end]

                # 归零优化器梯度 清除之前的梯度，防止梯度累加
                optimizer.zero_grad()

                # 获取损失 剪切损失、值函数损失和熵奖励
                l_clip, l_vf, entropy_bonus = get_losses(model, batch, epsilon, annealing, device)

                # 计算总损失并反向传播（计算所有可学习权重的梯度）
                loss = l_clip - c1 * l_vf + c2 * entropy_bonus
                loss.backward()

                # 优化  根据计算得到的梯度更新模型的参数。
                optimizer.step()
            # 更新学习率
            scheduler.step()

        # 记录输出
        curr_loss = loss.item()
        log = f"Iteration {iteration + 1} / {max_iterations}: " \
              f"Average Reward: {avg_rew:.2f}\t" \
              f"Loss: {curr_loss:.3f} " \
              f"(L_CLIP: {l_clip.item():.1f} | L_VF: {l_vf.item():.1f} | L_bonus: {entropy_bonus.item():.1f})"
        if avg_rew > max_reward:
            torch.save(model.state_dict(), MODEL_PATH)
            max_reward = avg_rew
            log += " --> Stored model with highest average reward"
        print(log)

        # 将信息记录到 W&B
        wandb.log({
            "loss (total)": curr_loss,
            "loss (clip)": l_clip.item(),
            "loss (vf)": l_vf.item(),
            "loss (entropy bonus)": entropy_bonus.item(),
            "average reward": avg_rew
        })

    # 完成 W&B 会话
    wandb.finish()

def testing_loop(env, model, n_episodes, device):
    # n_episodes=5
    for _ in range(n_episodes):
        run_timestamps(env, model, timestamps=128, render=True, device=device)

    
def compute_cumulative_rewards(buffer, gamma):
    """
    给定一个包含状态、策略操作逻辑、奖励和终止的缓冲区，计算每个时间的累积奖励并将它们代入缓冲区。
    """
    curr_rew = 0.

    # 反向遍历缓冲区
    for i in range(len(buffer) - 1, -1, -1):
        r, t = buffer[i][-2], buffer[i][-1]

        if t:
            curr_rew = 0
        else:
            curr_rew = r + gamma * curr_rew

        buffer[i][-2] = curr_rew

    # 在规范化之前获得平均奖励（用于日志记录和检查点）
    avg_rew = np.mean([buffer[i][-2] for i in range(len(buffer))])

    # 规范化累积奖励
    mean = np.mean([buffer[i][-2] for i in range(len(buffer))])
    std = np.std([buffer[i][-2] for i in range(len(buffer))]) + 1e-6#1e -6为了防止/0
    for i in range(len(buffer)):
        buffer[i][-2] = (buffer[i][-2] - mean) / std

    return avg_rew

def get_losses(model, batch, epsilon, annealing, device="cpu"):
    """给定模型、给定批次和附加参数返回三个损失项"""
    # 获取旧数据
    n = len(batch)
    states = torch.cat([batch[i][0] for i in range(n)])
    actions = torch.cat([batch[i][1] for i in range(n)]).view(n, 1) #view(n,1) n行1列，即每个action占一行
    logits = torch.cat([batch[i][2] for i in range(n)])
    values = torch.cat([batch[i][3] for i in range(n)])
    cumulative_rewards = torch.tensor([batch[i][-2] for i in range(n)]).view(-1, 1).float().to(device)

    # 使用新模型计算预测
    _, new_logits, new_values = model(states)

    # 状态动作函数损失(L_CLIP)
    advantages = cumulative_rewards - values
    margin = epsilon * annealing
    #从 new_logits 中选择与 actions 对应的列（即选定动作的概率对数），结果是一个张量，包含每个动作的 logits。
    #ratios: 计算新旧策略的比率，表示新策略对特定动作的概率与旧策略的概率之比。这是为了评估策略更新的有效性。
    ratios = new_logits.gather(1, actions) / logits.gather(1, actions)

    l_clip = torch.mean(
        torch.min(
            torch.cat(
                #将比率限制在一个小范围内（1 - margin 到 1 + margin），这样可以防止更新幅度过大，确保新策略不会偏离旧策略太远
                (ratios * advantages,
                 torch.clip(ratios, 1 - margin, 1 + margin) * advantages),
                dim=1),
            dim=1
        ).values
    )

    # 价值函数损失(L_VF)
    l_vf = torch.mean((cumulative_rewards - new_values) ** 2)

    # 熵奖励
    entropy_bonus = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-5)), dim=1))

    return l_clip, l_vf, entropy_bonus




