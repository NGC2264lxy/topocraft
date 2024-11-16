import gym
from gym import spaces
import numpy as np

class TopologyEnv(gym.Env):
    def __init__(self, num_pods):
        super(TopologyEnv, self).__init__()
        
        # 初始化时没有已知的连接矩阵
        self.num_pods = num_pods
        self.matrix_size = num_pods
        
        # 动作空间：每列前7行选择4个不重复的值
        # 每列需要从前7行中选择4个非-1值
        self.action_space = spaces.MultiDiscrete([7] * self.matrix_size)  # 每列前7行选择4个
        
        # 定义状态空间为当前的连接矩阵
        self.observation_space = spaces.Box(low=-1, high=self.matrix_size - 1, shape=(self.matrix_size, self.matrix_size), dtype=np.int)
        
        # 用于存储动态更新的连接矩阵
        self.cur_configuration = None

    def handle_recvData(self, recvData):
        """
        接收并解析字符串，生成traffic_matrix和configuration矩阵.
        """
        parts = recvData.split(';')
        complete_flag = parts[0]
        traffic_matrix = []
        configuration = []
        pathLength = 0
        
        for i in range(1, self.num_pods + 1):
            traffic_matrix.append(list(map(int, parts[i].split(','))))
        for i in range(self.num_pods + 1, 2 * self.num_pods + 1):
            configuration.append(list(map(int, parts[i].split(','))))
        
        self.complete_flag = complete_flag
        self.cur_traffic = traffic_matrix
        self.cur_configuration = configuration  # 动态生成待选择的连接矩阵
        self.cur_PathLength = float(parts[-1])

    def reset(self):
        """
        在重置时，使用 self.cur_configuration 作为初始连接矩阵.
        """
        if self.cur_configuration is None:
            raise ValueError("cur_configuration is not set. Please use handle_recvData to initialize it.")

        # 使用当前接收到的 cur_configuration 初始化连接矩阵
        self.matrix = np.array(self.cur_configuration)
        
        return self.matrix

    def step(self, action):
        """
        处理动作，并根据选择更新连接矩阵.
        """
        selected_matrix = self.matrix.copy()  # 创建一个副本，来根据动作选择调整矩阵
        done = False
        reward = 0
        
        # 遍历每一列，按照动作选择前7行的4个连接
        for col in range(self.matrix_size):  # 遍历所有列
            
            # 提取当前列的前7行中有效的非-1值
            valid_rows = [row for row in range(7) if self.matrix[row, col] != -1]
            
            # 使用传入的action进行选择
            selected_rows = action[col][:4]  # 动作中的前4个选择
            unselected_rows = set(valid_rows) - set(selected_rows)
            
            # 更新selected_matrix中的未选行
            for row in unselected_rows:
                selected_matrix[row, col] = -1  # 未选择的行设置为-1
            
            # 对每列的连接进行反向校正（目标POD连接）
            for row in selected_rows:
                target_pod = self.matrix[row, col]
                if target_pod != -1:
                    selected_matrix[target_pod, col] = row  # 反向更新
        
        # 检查是否满足终止条件
        if self.check_termination():
            done = True
            reward += 100  # 成功完成任务的奖励
            
        return selected_matrix, reward, done, {}

    def check_termination(self):
        """
        检查当前的拓扑是否满足特定的约束，例如一跳可达性。
        """
        # 示例逻辑：可以添加特定的检查规则
        return False

class MyAgent(nn.Module):
    """
    PPO模型的实现。
    Actor 和 Critic 都是基于 GAT 的网络。
    """

    def __init__(self, in_shape, n_actions, hidden_d=100, share_backbone=False, rnd_hidden_dim=128, num_choices=7, num_selected=4):
         # 父类构造函数
        super(MyAgent, self).__init__()

        # 属性
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone
        self.num_choices = num_choices  # 每列有7行可供选择
        self.num_selected = num_selected  # 每列选4个

        # 计算输入的维度
        in_dim = np.prod(in_shape)
    
        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )
        
        # nn.Identity 是无操作层
        self.backbone = to_features() if self.share_backbone else nn.Identity()

        # 创建 GAT Actor 和 Critic
        self.actor = Actor(state_dim=in_dim, gat_hidden_dim=hidden_d, gat_output_dim=num_choices, num_heads=4)
        self.critic = Critic(state_dim=in_dim, gat_hidden_dim=hidden_d, gat_output_dim=hidden_d, num_heads=4)
        # RND
        self.rnd = RND(state_dim=in_dim, hidden_dim=rnd_hidden_dim)  # Initialize RND

    def forward(self, x, edge_index):
        features = self.backbone(x)
        
        # 获取动作的概率分布
        action_probs = self.actor(features, edge_index)

        # 对每列使用 multinomial 抽取4个动作（不重复的行选择）
        action = torch.stack([torch.multinomial(action_probs[:, col], self.num_selected, replacement=False) 
                              for col in range(self.num_choices)], dim=1)
        
        # 获取状态值
        value = self.critic(features, edge_index)

        return action, action_probs, value

class MyRouteAgent(nn.Module):
    def __init__(self, in_shape, hidden_d=100, num_choices=2, num_heads=4):
        super(MyRouteAgent, self).__init__()

        self.in_shape = in_shape
        self.hidden_d = hidden_d
        self.num_choices = num_choices  # 每对POD的可选路径数
        self.num_heads = num_heads

        # 计算输入的维度
        in_dim = np.prod(in_shape)

        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )

        self.backbone = to_features()

        # 创建GAT-based Actor和Critic
        self.actor = Actor(state_dim=in_dim, gat_hidden_dim=hidden_d, gat_output_dim=num_choices, num_heads=num_heads)
        self.critic = Critic(state_dim=in_dim, gat_hidden_dim=hidden_d, gat_output_dim=hidden_d, num_heads=num_heads)

    def forward(self, x, edge_index):
        features = self.backbone(x)
        
        # 获取路径选择的概率分布 (8个POD之间的流量分配)
        action_probs = self.actor(features, edge_index)
        
        # 生成每个POD的动作（即流量分配的比例）
        actions = {}
        for src in range(8):  # 假设有8个POD
            actions[src] = {}
            for dst in range(8):
                if src != dst:
                    # 如果有多个路径可选，则从概率分布中进行采样
                    prob_dist = action_probs[src, dst]
                    # 多路径选择时，给每条路径分配一个概率
                    actions[src][dst] = Categorical(prob_dist).sample()

        # 获取状态价值
        value = self.critic(features, edge_index)

        return actions, action_probs, value
    
class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N  # 节点数
        self.obs_dim_topo = args.obs_dim_topo  # TopoAgent 的观测值维度
        self.obs_dim_route = args.obs_dim_route  # RouteAgent 的观测值维度
        self.state_dim = args.state_dim  # 全局状态维度
        self.episode_limit = args.episode_limit  # 每个训练周期的最大步数
        self.batch_size = args.batch_size  # 批量大小
        self.episode_num = 0  # 当前训练周期索引
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        """
        重置经验池，准备存储新的训练周期数据。
        """
        self.buffer = {
            'obs_topo': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim_topo]),  # TopoAgent 的状态
            'obs_route': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim_route]),  # RouteAgent 的状态
            's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),  # 全局状态
            'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),  # 价值估计
            'a_topo': np.empty([self.batch_size, self.episode_limit, self.N]),  # TopoAgent 的动作
            'a_route': np.empty([self.batch_size, self.episode_limit, self.N]),  # RouteAgent 的动作
            'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),  # 动作的概率
            'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),  # 环境奖励
            'rnd_r_n': np.empty([self.batch_size, self.episode_limit, self.N]),  # RND 奖励
            'done_n': np.empty([self.batch_size, self.episode_limit, self.N])  # 是否结束标志
        }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_topo, obs_route, s, v_n, a_topo, a_route, a_logprob_n, r_n, rnd_r_n, done_n):
        """
        存储当前时间步的数据。
        """
        self.buffer['obs_topo'][self.episode_num][episode_step] = obs_topo
        self.buffer['obs_route'][self.episode_num][episode_step] = obs_route
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_topo'][self.episode_num][episode_step] = a_topo
        self.buffer['a_route'][self.episode_num][episode_step] = a_route
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['rnd_r_n'][self.episode_num][episode_step] = rnd_r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        """
        存储最后时间步的价值。
        """
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.
def run_timestamps(agent_id_topo, agent_id_route, env, model_topo, model_route, args, replay_buffer, timestamps=128, render=False, device="cpu"):
    """
    针对给定数量的时间戳在给定环境中运行两个Agent的策略，Topo-Agent和Route-Agent有先后顺序。
    将数据存储到 ReplayBuffer 中。
    """
    # 重置环境并获取初始状态
    state = env.reset()
    state_topo = state[agent_id_topo - 1]  # 获取 Topo-Agent 的状态
    state_route = state[agent_id_route - 1]  # 获取 Route-Agent 的状态
    edge_index = env.get_edge_index()

    for ts in range(timestamps):
        # --- 执行 Topo-Agent 动作 ---
        state_tensor_topo = torch.tensor(state_topo).unsqueeze(0).to(device).float()
        edge_index_tensor = torch.tensor(env.get_edge_index()).to(device)
        
        # 获取 Topo-Agent 的动作、动作概率和价值估计
        action_topo, action_probs_topo, value_topo = model_topo(state_tensor_topo, edge_index_tensor)

        # Topo-Agent 与环境交互，获取 selected_matrix 和新的状态
        new_state_topo, reward_topo, done_topo = env.step(action_topo.item(), agent_id_topo)

        # --- 执行 Route-Agent 动作，基于 Topo-Agent 的 selected_matrix ---
        state_tensor_route = torch.tensor(state_route).unsqueeze(0).to(device).float()

        # 传递 selected_matrix 给 Route-Agent
        action_route, action_probs_route, value_route = model_route(state_tensor_route, edge_index_tensor, selected_matrix=new_state_topo['selected_matrix'])

        # Route-Agent 与环境交互，获取新的状态
        new_state_route, reward_route, done_route = env.step(action_route.item(), agent_id_route)

        # 计算 RND 奖励（假设两者共享 RND 模型）
        rnd_reward_topo = model_topo.rnd.calculate_rnd_loss(state_tensor_topo).item()
        rnd_reward_route = model_route.rnd.calculate_rnd_loss(state_tensor_route).item()

        # 存储数据到经验池中
        replay_buffer.store_transition(
            episode_step=ts,
            obs_topo=state_topo,  # Topo-Agent 的观测
            obs_route=state_route,  # Route-Agent 的观测
            s=state,  # 全局状态
            v_n=value_topo,  # Topo-Agent 价值
            a_topo=action_topo,  # Topo-Agent 动作
            a_route=action_route,  # Route-Agent 动作
            a_logprob_n=(action_probs_topo, action_probs_route),  # 两个 Agent 的动作概率
            r_n=(reward_topo, reward_route),  # 两个 Agent 的奖励
            rnd_r_n=(rnd_reward_topo, rnd_reward_route),  # 两个 Agent 的 RND 奖励
            done_n=(done_topo, done_route)  # 是否结束标志
        )

        # 更新状态
        state_topo = new_state_topo[agent_id_topo - 1]
        state_route = new_state_route[agent_id_route - 1]

        # 如果 Topo-Agent 或 Route-Agent 终止，则重置环境
        if done_topo or done_route:
            state = env.reset()
            state_topo = state[agent_id_topo - 1]
            state_route = state[agent_id_route - 1]

    return

def training_loop(agent_id_topo, agent_id_route, env, model_topo, model_route, args, replay_buffer, max_iterations, n_actors, horizon, beta, gamma, epsilon, n_epochs, batch_size, lr, c1, c2, device, env_name="", seed=0):
    set_seed(seed)  # Set seed for reproducibility

    """使用最多n个时间戳的多个actor在给定环境中训练Topo-Agent和Route-Agent的模型。"""

    # 开始运行新的权重和偏差
    wandb.init(project="Papers Re-implementations",
               entity="peutlefaire",
               name=f"PPO - {env_name}",
               config={
                   "env": str(env),
                   "number of actors": n_actors,  # 并行actor的数量
                   "horizon": horizon,  # 每个轨迹的最大步数
                   "gamma": gamma,
                   "epsilon": epsilon,
                   "epochs": n_epochs,
                   "batch size": batch_size,
                   "learning rate": lr,
                   "c1": c1,
                   "c2": c2
               })

    max_reward = float("-inf")  # 初始化的奖励值是负无穷

    # 优化器：分别为Topo-Agent和Route-Agent创建优化器
    optimizer_topo = Adam(model_topo.parameters(), lr=lr)
    optimizer_route = Adam(model_route.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler_topo = LinearLR(optimizer_topo, 1, 0, max_iterations * n_epochs)
    scheduler_route = LinearLR(optimizer_route, 1, 0, max_iterations * n_epochs)

    # 训练循环
    for iteration in range(max_iterations):
        replay_buffer.reset_buffer()  # 重置经验池
        # 使用当前策略收集所有actor的时间戳
        for actor in range(n_actors):
            run_timestamps(agent_id_topo, agent_id_route, env, model_topo, model_route, args, replay_buffer, horizon, render=False, device=device)

        # 计算累积奖励并刷新缓冲区
        avg_rew = compute_cumulative_rewards(replay_buffer.buffer, gamma)

        # 随机打乱经验池数据
        batch_data = replay_buffer.get_training_data()
        np.random.shuffle(batch_data['obs_topo'])

        # 优化：对Topo-Agent和Route-Agent分别进行训练
        for epoch in range(n_epochs):
            for batch_idx in range(len(batch_data['obs_topo']) // batch_size):
                start = batch_size * batch_idx
                end = start + batch_size if start + batch_size < len(batch_data['obs_topo']) else -1

                # 提取训练数据
                states_topo = batch_data['obs_topo'][start:end].to(device)
                states_route = batch_data['obs_route'][start:end].to(device)
                actions_topo = batch_data['a_topo'][start:end].to(device)
                actions_route = batch_data['a_route'][start:end].to(device)
                log_probs_topo, log_probs_route = batch_data['a_logprob_n'][start:end]
                values_topo = batch_data['v_n'][start:end].to(device)
                rewards_topo, rewards_route = batch_data['r_n'][start:end]
                rnd_rewards_topo, rnd_rewards_route = batch_data['rnd_r_n'][start:end]

                # Topo-Agent 优化
                optimizer_topo.zero_grad()
                advantage_topo = rewards_topo + (1 - batch_data['done_n'][start:end]) * gamma * values_topo - values_topo
                ratio_topo = (log_probs_topo - log_probs_topo.detach()).exp()
                surrogate_loss_topo = ratio_topo * advantage_topo.detach()
                policy_loss_topo = -torch.mean(torch.min(surrogate_loss_topo, torch.clamp(ratio_topo, 1 - epsilon, 1 + epsilon) * advantage_topo.detach()))
                value_loss_topo = F.mse_loss(values_topo, rewards_topo)
                rnd_loss_topo = torch.mean(rnd_rewards_topo)

                loss_topo = policy_loss_topo + c1 * value_loss_topo - c2 * rnd_loss_topo
                loss_topo.backward()
                optimizer_topo.step()
                scheduler_topo.step()

                # Route-Agent 优化
                optimizer_route.zero_grad()
                advantage_route = rewards_route + (1 - batch_data['done_n'][start:end]) * gamma * values_topo - values_topo
                ratio_route = (log_probs_route - log_probs_route.detach()).exp()
                surrogate_loss_route = ratio_route * advantage_route.detach()
                policy_loss_route = -torch.mean(torch.min(surrogate_loss_route, torch.clamp(ratio_route, 1 - epsilon, 1 + epsilon) * advantage_route.detach()))
                value_loss_route = F.mse_loss(values_topo, rewards_route)
                rnd_loss_route = torch.mean(rnd_rewards_route)

                loss_route = policy_loss_route + c1 * value_loss_route - c2 * rnd_loss_route
                loss_route.backward()
                optimizer_route.step()
                scheduler_route.step()

        avg_reward = np.mean([reward for _, _, _, _, reward, _ in replay_buffer.buffer['r_n']])
        if avg_reward > max_reward:
            max_reward = avg_reward
            torch.save(model_topo.state_dict(), "topo_agent_model.pth")
            torch.save(model_route.state_dict(), "route_agent_model.pth")

        # Evaluate the model every few iterations
        if iteration % 10 == 0:
            test_reward_topo = evaluate_model(env, model_topo)
            test_reward_route = evaluate_model(env, model_route)
            wandb.log({"test_avg_reward_topo": test_reward_topo, "test_avg_reward_route": test_reward_route})

    wandb.finish()
    return model_topo, model_route


def training_loop(agent_id_topo, agent_id_route, env, model_topo, model_route, args, replay_buffer, max_iterations, n_actors, horizon, beta, gamma, epsilon, n_epochs, batch_size, lr, c1, c2, device, env_name="", seed=0):
    set_seed(seed)  # Set seed for reproducibility

    """使用最多n个时间戳的多个actor在给定环境中训练Topo-Agent和Route-Agent的模型。"""

    # 初始化 wandb 日志记录
    wandb.init(project="Papers Re-implementations",
               entity="peutlefaire",
               name=f"PPO - {env_name}",
               config={
                   "env": str(env),
                   "number of actors": n_actors,  # 并行actor的数量
                   "horizon": horizon,  # 每个轨迹的最大步数
                   "gamma": gamma,
                   "epsilon": epsilon,
                   "epochs": n_epochs,
                   "batch size": batch_size,
                   "learning rate": lr,
                   "c1": c1,
                   "c2": c2
               })

    max_reward = float("-inf")  # 初始化的奖励值为负无穷大

    # 优化器：分别为 Topo-Agent 和 Route-Agent 创建优化器
    optimizer_topo = Adam(model_topo.parameters(), lr=lr)
    optimizer_route = Adam(model_route.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler_topo = LinearLR(optimizer_topo, 1, 0, max_iterations * n_epochs)
    scheduler_route = LinearLR(optimizer_route, 1, 0, max_iterations * n_epochs)

    # 训练循环
    for iteration in range(max_iterations):
        replay_buffer.reset_buffer()  # 重置经验池

        # 使用当前策略收集所有 actor 的时间戳
        for actor in range(n_actors):
            run_timestamps(agent_id_topo, agent_id_route, env, model_topo, model_route, args, replay_buffer, horizon, render=False, device=device)

        # 从经验池获取训练数据
        batch_data = replay_buffer.get_training_data()
        np.random.shuffle(batch_data['obs_topo'])

        # 运行几轮优化
        for epoch in range(n_epochs):
            for batch_idx in range(len(batch_data['obs_topo']) // batch_size):
                start = batch_size * batch_idx
                end = start + batch_size if start + batch_size < len(batch_data['obs_topo']) else -1
                batch_topo = batch_data['obs_topo'][start:end]
                batch_route = batch_data['obs_route'][start:end]

                # Topo-Agent 优化
                optimizer_topo.zero_grad()
                topo_states, topo_actions, topo_log_probs, topo_values, topo_rewards, topo_advantages = [], [], [], [], [], []

                for i in range(len(batch_topo)):
                    state_topo, action_topo, action_probs_topo, value_topo, reward_topo, done_topo = batch_topo[i]

                    # 计算 RND 损失
                    rnd_loss_topo = model_topo.rnd.calculate_rnd_loss(state_topo)
                    topo_log_prob = Categorical(action_probs_topo).log_prob(action_topo)
                    topo_combined_reward = reward_topo + beta * rnd_loss_topo
                    topo_advantage = topo_combined_reward + (1 - done_topo) * gamma * value_topo - value_topo

                    topo_values.append(value_topo)
                    topo_log_probs.append(topo_log_prob)
                    topo_advantages.append(topo_advantage)
                    topo_rewards.append(reward_topo)
                    topo_states.append(state_topo)
                    topo_actions.append(action_topo)

                # 将 collected lists 转换为张量
                topo_values = torch.cat(topo_values).view(-1)
                topo_log_probs = torch.cat(topo_log_probs).view(-1)
                topo_advantages = torch.cat(topo_advantages).view(-1)
                topo_rewards = torch.cat(topo_rewards).view(-1)

                # 计算 Topo-Agent 损失
                topo_value_loss = F.mse_loss(topo_values, topo_rewards)
                topo_ratio = (topo_log_probs - topo_log_probs.detach()).exp()
                topo_surrogate_loss = topo_ratio * topo_advantages.detach()
                topo_policy_loss = -torch.mean(torch.min(topo_surrogate_loss, torch.clamp(topo_ratio, 1 - epsilon, 1 + epsilon) * topo_advantages.detach()))

                # 总损失
                topo_loss = topo_policy_loss + c1 * topo_value_loss - c2 * rnd_loss_topo
                topo_loss.backward()
                optimizer_topo.step()
                scheduler_topo.step()

                # Route-Agent 优化
                optimizer_route.zero_grad()
                route_states, route_actions, route_log_probs, route_values, route_rewards, route_advantages = [], [], [], [], [], []

                for i in range(len(batch_route)):
                    state_route, action_route, action_probs_route, value_route, reward_route, done_route = batch_route[i]

                    # 计算 RND 损失
                    rnd_loss_route = model_route.rnd.calculate_rnd_loss(state_route)
                    route_log_prob = Categorical(action_probs_route).log_prob(action_route)
                    route_combined_reward = reward_route + beta * rnd_loss_route
                    route_advantage = route_combined_reward + (1 - done_route) * gamma * value_route - value_route

                    route_values.append(value_route)
                    route_log_probs.append(route_log_prob)
                    route_advantages.append(route_advantage)
                    route_rewards.append(reward_route)
                    route_states.append(state_route)
                    route_actions.append(action_route)

                # 将 collected lists 转换为张量
                route_values = torch.cat(route_values).view(-1)
                route_log_probs = torch.cat(route_log_probs).view(-1)
                route_advantages = torch.cat(route_advantages).view(-1)
                route_rewards = torch.cat(route_rewards).view(-1)

                # 计算 Route-Agent 损失
                route_value_loss = F.mse_loss(route_values, route_rewards)
                route_ratio = (route_log_probs - route_log_probs.detach()).exp()
                route_surrogate_loss = route_ratio * route_advantages.detach()
                route_policy_loss = -torch.mean(torch.min(route_surrogate_loss, torch.clamp(route_ratio, 1 - epsilon, 1 + epsilon) * route_advantages.detach()))

                # 总损失
                route_loss = route_policy_loss + c1 * route_value_loss - c2 * rnd_loss_route
                route_loss.backward()
                optimizer_route.step()
                scheduler_route.step()

        # 保存模型和记录奖励
        avg_reward = np.mean([reward for reward in batch_data['r_n']])
        if avg_reward > max_reward:
            max_reward = avg_reward
            torch.save(model_topo.state_dict(), "topo_agent_model.pth")
            torch.save(model_route.state_dict(), "route_agent_model.pth")

        # Evaluate the model every few iterations
        if iteration % 10 == 0:
            test_reward_topo = evaluate_model(env, model_topo)
            test_reward_route = evaluate_model(env, model_route)
            wandb.log({"test_avg_reward_topo": test_reward_topo, "test_avg_reward_route": test_reward_route})

    wandb.finish()
    return model_topo, model_route

class HPCEnvironment:
    def __init__(self, args, socket_host='127.0.0.1', socket_port=8889):
        self.num_pods = args.num_pods
        self.complete_flag = False
        self.cur_traffic = []        # 当前接收到的流量矩阵
        self.cur_configuration = []  # 当前接收到的MEMS配置
        self.cur_PathLength = 0       # 当前接收到的组间平均路径长度
        self.selected_matrix = []
        self.routing_table = []
        # 初始化socket
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.socket = self._init_socket()

    def _init_socket(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.socket_host, self.socket_port))
        return client_socket

    def update_state(self, data):
        """
        更新环境的状态，解析从 main.py 传入的数据。
        """
        self.handle_recvData(data)

    def get_combined_data(self):
        """
        获取当前的 selected_matrix 和 routing_table，并组合成字符串形式。
        """
        combined_data = self._combine_agent_results()
        return combined_data

    def send_data_to_omnetpp(self, combined_data):
        """
        通过 socket 将拼接好的数据发送给 OMNeT++。
        """
        try:
            self.socket.sendall(combined_data.encode())  # 发送拼接的数据
        except Exception as e:
            print(f"Error while sending data to OMNeT++: {e}")

    # 其余保持不变，包括 _combine_agent_results、handle_recvData 等
            
if __name__ == '__main__':
    args = parser.parse_args()
    env = HPCEnvironment(args)
    global_traffic = []
    global_configuration = []
    
    agent_id = 1
    # 监听
    address = ('127.0.0.1', 8889)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)  # 绑定地址和端口

    server.listen(5)
    print("Waiting for connection...")
    client, addr = server.accept()
    print(f"Accepted connection from {addr}")

    for cur_step in range(args.max_interact_steps):
        print("cur step: %d / %d " % (cur_step + 1, args.max_interact_steps))
        done = False

        while not done:
            # 接收数据
            data = client.recv(4096).decode()  # 接收数据
            print(f"Received data: {data}")
            
            # 将数据传递给环境进行处理
            env.update_state(data)  # 这个函数仅用于解析和更新状态

            # 从环境中获取组合后的数据（包含 selected_matrix 和 routing_table）
            combined_data = env.get_combined_data()

            # 将结果发送回 OMNeT++
            env.send_data_to_omnetpp(combined_data)

            # 检查终止条件
            done = env.get_cur_comFlag()
            print(f"done: {done}")
    
    client.close()
    server.close()


# 初始化模型
topo_model = MyTopoAgent(
    in_shape=(1, 1),  # 假设状态输入形状为(1, 1)，根据你的实际输入调整
    n_actions=env.topology_action_space.shape[0],  # 如果你需要8列，每列7个选项
    hidden_d=args.gat_hidden_dim,
    share_backbone=True,
    rnd_hidden_dim=args.rnd_hidden_dim,
    num_choices=7,
    num_selected=4
).to(device)

route_model = MyRouteAgent(
    in_shape=(1, 1),  # 假设状态输入形状为(1, 1)，根据你的实际输入调整
    hidden_d=args.gat_hidden_dim,
    share_backbone=True,
    rnd_hidden_dim=args.rnd_hidden_dim,
    num_choices=2,  # 对应的路径选择数量
    num_heads=4
).to(device)

# 执行训练
training_loop(
    env=env,
    model_topo=topo_model,
    model_route=route_model,
    args=args,
    replay_buffer=replay_buffer,
    max_iterations=args.max_train_steps,
    n_actors=args.n_actors,
    horizon=args.horizon,
    beta=args.lamda,  # RND reward weight
    gamma=args.gamma,
    epsilon=args.epsilon,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    c1=args.c1,
    c2=args.c2,
    device=device,
    env_name="HPC",
    seed=args.seed
)

# 加载最佳模型
topo_model.load_state_dict(torch.load("topo_agent_model.pth", map_location=device))
route_model.load_state

if __name__ == '__main__':
    args = parser.parse_args()
    env = HPCEnvironment(args)
    global_traffic = []
    global_configuration = []
   

    # 监听
    address = ('127.0.0.1', 8889)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)  # 绑定地址和端口
    for cur_step in range(args.max_interact_steps):
        print("cur step: %d / %d " % (cur_step + 1, args.max_interact_steps))
        done = False
        while not done:
            server.listen(5)
            client, addr = server.accept()
            print(f"Accepted connection from {addr}")
            print(f"Client socket: {client}")
           # data = client.recv(args.max_recv_data).decode()
            data = client.recv(4096).decode()
            print(f"data{data}")
            topology_data = "0,1,2,3,4,5,6,7;" 
            client.sendall(topology_data.encode()) 
            env.handle_recvData(data)
            env.cal_reward()
            # global_traffic.append(env.cur_traffic)
            # global_configuration.append(env.cur_configuration)
            done = env.get_cur_comFlag()
            print(f"done:{done}")
    client.close()

     #######  Socket通信   ######
        self.address = ('127.0.0.1', 8889)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.address)  # 绑定地址和端口
        self.server.listen(5)
        self.client, self.addr = self.server.accept()
        print(f"Accepted connection from {self.addr}")
        print(f"Client socket: {self.client}")
        recvData = self.client.recv(4096).decode()  # 接收反馈
        self.handle_recvData(recvData)
    

    # !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/3 17:14
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : envs.py
# @Software: PyCharm

import numpy as np
import torch
import socket
from gym import spaces
from collections import deque

class HPCEnvironment:
    def __init__(self, args):
        self.num_pods = args.num_pods
        self.complete_flag = False
        self.cur_traffic = []        # 当前接收到的流量矩阵
        self.cur_configuration = []  # 当前接收到的MEMS配置
        self.cur_PathLength = 0       # 当前接收到的组间平均路径长度、
        self.matrix_size=self.num_pods
        self.selected_matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=int)
        self.ports_per_pod=7

        ########  state ############
        self.link_utilization = []
        self.link_load = []
        
        #######   TopoAgent  #####
        # 定义8x8矩阵，代表连接状态
        self.matrix_size=self.num_pods
        # 动作空间：选择前7行中4个非-1值的组合
        # 每列可以从0到7选择4个不重复的值，并且确保有影响其他列的约束
        self.topology_action_space = spaces.MultiDiscrete([7] * self.matrix_size)  # 
        # 定义状态空间为当前的矩阵
        self.topology_observation_space = spaces.Box(low=-1, high=self.matrix_size - 1, shape=(self.matrix_size,self.matrix_size), dtype=np.int32)
        
        ######   RouteAgent   #########
        # 根据selected_matrix生成每个POD的路由表
        self.routing_table = self.generate_routing_table()
        # 动作空间定义：对于每个POD，每对src-dst可能有不同的路径
        # 对每条路径分配流量的概率。假设最多有2条路径（直达和中转一次）。
        self.route_action_space = spaces.Dict({
            src: spaces.Dict({
                dst: spaces.Box(low=0.0, high=1.0, shape=(len(paths),), dtype=np.float32)  # 每条路径的流量分配概率
                for dst, paths in self.routing_table[src].items()
            })
            for src in range(len(self.selected_matrix))
        })
        
        # 状态空间，可以是拓扑和流量需求矩阵的结合
        self.route_observation_space = spaces.Box(low=0, high=1, shape=(8, 8))  # 示例状态空间


        #######  Socket通信   ######
        self.address = ('127.0.0.1', 8889)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.address)  # 绑定地址和端口
        self.server.listen(5)
        self.client, self.addr = self.server.accept()
        print(f"Accepted connection from {self.addr}")
        print(f"Client socket: {self.client}")
        recvData = self.client.recv(4096).decode()  # 接收反馈
        self.handle_recvData(recvData)

    def get_cur_comFlag(self):
        comFlag = self.complete_flag
        return comFlag

    def get_cur_configuration(self):
        configuration = self.cur_configuration
        return configuration

    def get_cur_traffic(self):
        traffic = self.cur_traffic
        return traffic
  
    def get_cur_state(self):
        # 计算链路利用率
        traffic=self.get_cur_traffic()
        link_utilization=[[element * 1000 * 64 * 8 / (100 * (10 ** 9)) for element in row] for row in traffic]#KB
        self.link_utilization=link_utilization
        # 计算流量负载
        link_load=[[element * 1000 * 64 * 8 / (10 * (10 ** -3)) for element in row] for row in traffic]
        self.link_load=link_load
        return  link_utilization,link_load

    def handle_recvData(self, recvData):
        # 解析字符串
        parts = recvData.split(';')
        complete_flag = parts[0]
        self.msgkind = parts[1]  #用以区分数据是omnetpp回传的logicTopo  msgkind=topomsg  还是正常信息 msgkind=elsemsg
        traffic_matrix = []
        configuration = []
        self.traffic_demand = []      #流量需求矩阵
        if self.msgkind=='elsemsg':
            for i in range(2,self.num_pods+2):   #传入时新增流量需求矩阵
                self.traffic_demand.append(list(map(int, parts[i].split(','))))
            for i in range(self.num_pods+2, 2*self.num_pods+2):
                traffic_matrix.append(list(map(int, parts[i].split(','))))
            for i in range(2*self.num_pods+2, 3*self.num_pods+1):
                configuration.append(list(map(int, parts[i].split(','))))
            self.complete_flag = complete_flag
            self.cur_traffic = traffic_matrix
            self.cur_configuration = configuration
            self.cur_PathLength = float(parts[-1])
        elif self.msgkind=='topomsg':
            for i in range(2, self.num_pods+2):
                configuration.append(list(map(int, parts[i].split(','))))
        else:
            print(f"error msgkind from omnetpp")

    def get_cur_topology(self):
        num_groups = len(self.cur_configuration[0])
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_groups, num_groups), dtype=int)
        # 填充邻接矩阵
        for groups in self.cur_configuration:
            for group_id in groups:
                if group_id != -1:
                    for other_group_id in groups:
                        if other_group_id != -1 and other_group_id != group_id:
                            adjacency_matrix[group_id][other_group_id] = 1
        return adjacency_matrix
    
    # topology = self.get_cur_topology()  # 获取邻接矩阵adjacency_matrix
    def get_edge_index(self):
        topology = self.get_cur_topology()  # 获取邻接矩阵 
        edge_index = np.argwhere(topology == 1)  # 找到所有边的索引
        return edge_index.T  # 转置为2xN的形式

    # 
    def cal_cur_reward(self):
        ######################## 计算链路利用率差异度 #########################
        # 剔除链路利用率中的0元素
        lu=self.get_cur_state()[0]  # link_utilization,link_load
        lu=np.array(lu) #list转numpy方便计算
        filtered_matrix = [[value for value in sublist if value != 0.0] for sublist in lu]
        lu_mean = np.mean(filtered_matrix)
        sum_squared_diff = np.sum((filtered_matrix - lu_mean) ** 2)
        num_elements = sum(len(row) for row in filtered_matrix)
        lu_diff = np.sqrt(sum_squared_diff / num_elements)

        ############ 分别计算两个Agent的reward和总reward ######################
        lu_reward = np.exp(-lu_diff)
        path_reward = np.exp(-self.cur_PathLength)
        total_reward = np.exp(-lu_diff - self.cur_PathLength)

        return lu_reward, path_reward, total_reward
    
    def reset(self):
        # 这里可以初始化或重置相关参数
        self.complete_flag = False
        link_utilization,link_load=self.get_cur_state()
         # 将两个状态组合成一个元组
        state = (link_utilization, link_load)
        return state


    
    def step(self, action, agent_id):
        # 根据传入的action更新环境状态
        # 更新MEMS配置、链路分配情况等
        # 动作会给定每一列的选择
        if agent_id==1:
            self.selected_matrix = self.topology_action_space.copy()  # 创建一个副本，来根据动作选择调整矩阵  selected_matrix是numpy
            #Topo_Agent的无效动作识别
            if not self.check_invalid_action_topo(action):
            # 如果动作无效，直接返回负奖励 -100
                return self.selected_matrix,new_state,-100,done
            
            # 遍历每一列，按照动作选择前7行的4个连接
            for col in range(self.matrix_size):  # 遍历所有列
                # 提取当前列的前7行中有效的非-1值
                valid_rows = [row for row in range(7) if self.topology_action_space[row, col] != -1]
                
                # 使用传入的action进行选择
                selected_rows = action[col][:4]  # 动作中的前4个选择
                unselected_rows = set(valid_rows) - set(selected_rows)
                
                # 更新selected_matrix中的未选行
                for row in unselected_rows:
                    self.selected_matrix[row, col] = -1  # 未选择的行设置为-1
                
                # 对每列的连接进行反向校正（目标POD连接）
                for row in selected_rows:
                    target_pod = self.topology_action_space[row, col]
                    if target_pod != -1:
                        self.selected_matrix[target_pod, col] = row  # 反向更新
            topo_str =  self.matrix_to_string(self.selected_matrix)
            topo_str ="topomsg"+ topo_str
            self.send_to_omnetpp(topo_str)
            logicTopo_str = self.server.recv(4096).decode()  # 从 OMNeT++ 接收字符

            logicTopo = [list(map(int, row.split(','))) for row in logicTopo_str.split(';') ] # 将字符串转换为矩阵
            self.selected_matrix=np.array(logicTopo) # list转numpy

            return self.selected_matrix  # selected_matrix是numpy

        elif agent_id==2:
            # 检测无效动作：根据 Algorithm 3 实现无效动作检测逻辑
            if not self.check_invalid_action_route(action):
            # 如果动作无效，直接返回负奖励
                return self.selected_matrix,new_state,-100,done
            # 处理流量分配的逻辑
            # action 是一个嵌套字典，每个POD到目标POD的路由分配
            routing_info = []
            
            for src, dst_action in action.items():
                for dst, route_prob in dst_action.items():   # dst_action中有路由概率
                    # total_demand = self.traffic_demand[src][dst]
                    # 获取路径流量分配比例 使用 zip 将路径和对应的概率结合。
                    for route, prob in zip(self.routing_table[src][dst], route_prob):
                        print(f"流量从 {src} 到 {dst} 分配到路径 {route} 的比例是 {prob}")
                        path_str = '-'.join(map(str, route))  # 将路径转换为字符串，如 "0-1-2"
                        routing_info.append(f"{src},{dst},{path_str},{prob}")  # 将src, dst, 路径和概率组合

                        # transmitted = total_demand * prob  # 分配流量
                        # # transmitted_traffic是一个字典的字典
                        # # 每个 src 键对应一个字典，里面的每个 dst 键对应一个浮点数（或整数），表示从 src 到 dst 的已传输流量
                        # self.transmitted_traffic[src][dst] += transmitted  # 更新已传输流量
        # 将所有路径信息序列化为字符串，并准备发送
            routing_str = ";".join(routing_info)
            # traffic_str = self.traffic_to_string(self.transmitted_traffic)
            routing_str="route;"+routing_str
            self.send_to_omnetpp(routing_str)
        # 拼接两个 Agent 的结果并通过 socket 一起发送到 OMNeT++
        # combined_data = self.combine_agent_results()
        # self.send_combined_data_to_omnetpp(combined_data)

        # 接收 OMNeT++ 的反馈（新的状态）

        new_recvData = self.client.recv(4096).decode()  # 接收反馈
        self.handle_recvData(new_recvData)
        link_utilization,link_load=self.get_cur_state() # 得到链路利用率和流量负载
        new_state=(link_utilization,link_load)   #list

        # 计算奖励
        lu_reward, path_reward, total_reward = self.cal_cur_reward()
        reward=(lu_reward, path_reward, total_reward)
        
        done = self.get_cur_comFlag()
        
        return new_state, reward, done
    #####################################################
    
    def generate_routing_table(self):
        """
        根据selected_matrix生成每个POD的路由表。
        每个POD有到其他所有POD的可选路径，包括直达路径和中转一次的路径。
        返回一个字典，其中key是POD，value是到其他POD的可选路径集合。
        """
        routing_table = {src: {} for src in range(len(self.selected_matrix))}

        for src in range(len(self.selected_matrix)):
            for dst in range(len(self.selected_matrix)):
                if src != dst:
                    routes = []
                    
                    # 检查直达路径
                    if self.selected_matrix[src][dst] != -1:
                        routes.append([src, dst])  # 直达路径

                    # 检查中转一次路径
                    for intermediate in range(len(self.selected_matrix)):
                        if intermediate != src and intermediate != dst:
                            if self.selected_matrix[src][intermediate] != -1 and self.selected_matrix[intermediate][dst] != -1:
                                routes.append([src, intermediate, dst])  # 一次中转路径
                    
                    # 只有存在可达路径时才加入路由表
                    if routes:
                        routing_table[src][dst] = routes
        
        return routing_table

    ################    通信   #############
    #### 因为main中需要调用close
    def close(self):
        """
        关闭 socket 连接。
        """
        self.server.close()
        print("Socket connection closed.")

    # def combine_agent_results(self):
    #     """
    #     将 Agent 1 和 Agent 2 的结果拼接在一起，准备发送到 OMNeT++。
    #     """
    #     # 获取 selected_matrix 的字符串表示
    #     topo_str = self.matrix_to_string(self.selected_matrix)

    #     # 获取流量分配矩阵的字符串表示
    #     traffic_str = self.traffic_to_string(self.transmitted_traffic)

    #     # 拼接两个部分，用分号或其他分隔符区分
    #     combined_str = topo_str + ';' + traffic_str
    #     return combined_str

    def send_to_omnetpp(self, data):
        """
        通过 socket 将拼接好的数据发送给 OMNeT++。
        """
        self.server.sendall(data.encode())  # 发送拼接的数据

    def matrix_to_string(self, matrix):
        """
        将 selected_matrix 转换为字符串格式，用于 socket 传输。
        """
        return ';'.join([','.join(map(str, row)) for row in matrix])

    # def traffic_to_string(self, traffic_matrix):
    #     """
    #     将流量分配矩阵转换为字符串格式，用于 socket 传输。
    #     """
    #     # 使用嵌套字典，遍历源 POD 和目标 POD
    #     return ';'.join([
    #         ','.join(f"{dst}:{traffic_matrix[src][dst]}" for dst in traffic_matrix[src])
    #         for src in traffic_matrix
    #     ])


    ############# 无效动作识别  ############
    def check_invalid_action_topo(self):
        # 初始化 valid 为 True
        valid = True
        for i in range(self.matrix_size):
            # 根据 Algorithm 2，检查端口数限制和孤岛问题
            if sum(self.selected_matrix[i, :]) > self.ports_per_pod or sum(self.selected_matrix[:, i]) > self.ports_per_pod:
                valid = False
                break
            if sum(self.selected_matrix[i, :]) == 0 or sum(self.selected_matrix[:, i]) == 0:
                valid = False
                break
        return valid

    def check_invalid_action_route(self):
        # 初始化 valid 为 True
        valid = True
        Lm = 4  # 最大路径长度限制

        # 检查路径长度限制
        for route in self.routing_table.values():
            if len(route) > Lm:
                valid = False
                break

        # 检查死锁
        if self.has_deadlock():
            valid = False

        return valid

    def has_deadlock(self):
        """
        使用Tarjan算法检测路由图中是否存在环。存在环则表明可能发生网络死锁。
        
        Returns:
            bool: 如果检测到环，则返回True，表示存在死锁。否则返回False。
        """
        graph = self.get_routing_graph()  # 使用邻接表生成路由图

        num_nodes = len(graph)
        index = 0
        indices = [-1] * num_nodes
        lowlink = [-1] * num_nodes
        stack = []
        on_stack = [False] * num_nodes
        has_cycle = [False]  # 标志位，用于判断是否有环

        def tarjan(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            for w in graph[v]:
                if indices[w] == -1:
                    tarjan(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    has_cycle[0] = True

        for v in range(num_nodes):
            if indices[v] == -1:
                tarjan(v)

        return has_cycle[0]  # 如果找到环，返回True


    def get_routing_graph(self):
        """
        根据 selected_matrix 生成POD的邻接表，只用于 Tarjan 算法检测死锁。
        返回的邻接表中每个POD只列出可以直接到达的POD（忽略中转路径）。
        
        Returns:
            dict: POD间的邻接表，key是POD，value是它可以直接到达的POD集合。
        """
        graph = {src: set() for src in range(len(self.selected_matrix))}

        # 遍历 selected_matrix 来构建邻接表
        for src in range(len(self.selected_matrix)):
            for dst in range(len(self.selected_matrix)):
                if src != dst:
                    # 只记录 src 到 dst 的直接路径
                    if self.selected_matrix[src][dst] != -1:
                        graph[src].add(dst)  # 如果 src 可以直达 dst，记录 dst 为 src 的邻居

        return graph


import numpy as np

# 初始化8x8的selected_matrix，初始值为随机示例
self.selected_matrix = np.random.randint(-1, 7, (8, 8))

# 初始化8x8的可更新属性矩阵，最后一行设为不可更新
updatable_matrix = np.ones((8, 8), dtype=bool)
updatable_matrix[7, :] = False  # 最后一行不可更新

# 每列最多更新4个
max_updates_per_col = 4

# 遍历每一列进行更新
for col in range(self.selected_matrix.shape[1]):
    updated_count = np.sum(~updatable_matrix[:, col])  # 已更新的节点数量
    remaining_updates = max_updates_per_col - updated_count  # 还需更新的节点数量

    if remaining_updates > 0:
        # 获取当前列中可更新的节点
        valid_rows = [row for row in range(7) if updatable_matrix[row, col]]

        # 按概率选择剩余需要更新的节点，最多更新remaining_updates个
        selected_rows = np.random.choice(valid_rows, remaining_updates, replace=False)

        for row in selected_rows:
            target_pod = self.selected_matrix[row, col]
            if target_pod != -1:
                # 反向更新selected_matrix
                self.selected_matrix[target_pod, col] = row
                # 将该节点标记为不可更新
                updatable_matrix[row, col] = False
                updatable_matrix[target_pod, col] = False
                updatable_matrix[target_pod, :] = False  # 对应列所有行都不可更新
