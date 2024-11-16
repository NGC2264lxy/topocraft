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
from collections import deque
from gym import spaces
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
        # self.address = ('127.0.0.1', 8889)
        # self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server.bind(self.address)  # 绑定地址和端口
        # self.server.listen(5)
        # self.client, self.addr = self.server.accept()
        # print(f"Accepted connection from {self.addr}")
        # print(f"Client socket: {self.client}")
        # recvData = self.client.recv(4096).decode()  # 接收反馈
        # self.handle_recvData(recvData)

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
                for i in range(2*self.num_pods+2, 3*self.num_pods+2):
                    traffic_matrix.append(list(map(int, parts[i].split(','))))
                for i in range(3*self.num_pods+2, 4*self.num_pods+2):
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
        
    def step(self, action, agent_id,socket_connection):
    # def step(self, action, agent_id):
        done = self.get_cur_comFlag()
        # 根据传入的action更新环境状态
        # 更新MEMS配置、链路分配情况等
        # 动作会给定每一列的选择
        if agent_id==1:
            # topology_action_space 是MultiDiscrete 对象 需要获取一个样本并转换为 NumPy 数组
            # Sample=self.topology_action_space.sample()
            # self.selected_matrix = np.copy(Sample)  # 创建一个副本，来根据动作选择调整矩阵  selected_matrix是numpy
            # print("Shape of selected_matrix",self.selected_matrix.shape)  # 输出 selected_matrix 的形状
            self.selected_matrix=np.copy(self.cur_configuration)
            # print("type1",type(self.selected_matrix))

            # 初始化8x8的可更新属性矩阵，最后一行设为不可更新
            updatable_matrix = np.ones((8, 8), dtype=bool)
            updatable_matrix[7, :] = False  # 最后一行不可更新

            # 每列最多更新4个
            max_updates_per_col = 4

            # 遍历每一列进行更新
            for col in range(self.selected_matrix.shape[1]):
                updated_count = np.sum(~updatable_matrix[:, col]) -1 # 已更新的节点数量(最后一行默认的False不算)
                remaining_updates = max_updates_per_col - updated_count  # 还需更新的节点数量

                if remaining_updates > 0:
                    # 获取当前列中可更新的节点
                    valid_rows = [row for row in range(7) if updatable_matrix[row, col]]

                    # 按概率选择剩余需要更新的节点，最多更新remaining_updates个
                    # selected_rows = np.random.choice(valid_rows, remaining_updates, replace=False)
                    selected_rows = [row for row in action[col] if row in valid_rows][:remaining_updates]

                    for row in selected_rows:
                        target_pod = self.selected_matrix[row, col]
                        if target_pod != -1:
                            # 反向更新selected_matrix
                            # self.selected_matrix[target_pod, col] = row  # 反向重置还是有点问题在的
                            # 将该节点标记为不可更新
                            updatable_matrix[row, col] = False
                            updatable_matrix[target_pod, col] = False
                            updatable_matrix[target_pod, :] = False  # 对应列所有行都不可更新
                for row in valid_rows:
                    if row not in selected_rows:
                        self.selected_matrix[row, col] = -1  # 将未选择的节点设置为 -1
            # # 遍历每一列，按照动作选择前7行的4个连接
            # for col in range(self.matrix_size):  # 遍历所有列
            #     # 提取当前列的前7行中有效的非-1值
            #     valid_rows = [row for row in range(7) if self.selected_matrix[row, col] != -1]
                
            #     # 使用传入的action进行选择
            #     selected_rows = action[col][:4]  # 动作中的前4个选择
            #     unselected_rows = set(valid_rows) - set(selected_rows)
                
            #     # 更新selected_matrix中的未选行
            #     for row in unselected_rows:
            #         self.selected_matrix[row, col] = -1  # 未选择的行设置为-1
                
                
            #     # 对每列的连接进行反向校正（目标POD连接）
            #     for row in selected_rows:
            #         target_pod = self.selected_matrix[row, col]
            #         if target_pod != -1:
            #             # 检查当前列是否已经处理过
                        
            #             # 反向更新，保证同一列不会被重复处理
            #             self.selected_matrix[target_pod, col] = row  # 反向更新
            #                 # 将当前列标记为已处理
            

            #Topo_Agent的无效动作识别
            if not self.check_invalid_action_topo():
            # 如果动作无效，直接返回负奖励 -100
                # print("type2",type(self.selected_matrix))
                return self.selected_matrix
            topo_str =  self.matrix_to_string(self.selected_matrix)
            topo_str ="topomsg;"+ topo_str
            self.send_to_omnetpp(topo_str,socket_connection)
            logicTopo_str = socket_connection.recv(4096).decode()  # 从 OMNeT++ 接收字符
            print("logicTopo_str",logicTopo_str)
            logicTopo = [list(map(int, row.split(','))) for row in logicTopo_str.split(';') ] # 将字符串转换为矩阵
            self.selected_matrix=np.array(logicTopo) # list转numpy

            return self.selected_matrix  # selected_matrix是numpy

        elif agent_id==2:
            
            # 处理流量分配的逻辑
            # action 是一个嵌套字典，每个POD到目标POD的路由分配
            routing_info = []
            
            Lm = 4
            # print(f"action: {action}")
            for src, dst_action in action.items():
                # print(f"Source: {src}")
                for dst, route_prob in dst_action.items():   # dst_action中有路由概率
                    # total_demand = self.traffic_demand[src][dst]
                    # 获取路径流量分配比例 使用 zip 将路径和对应的概率结合。
                    for dst, route_prob in dst_action.items():  # dst_action中有路由概率
                        if route_prob is None:
                            continue
                        else:
                            if route_prob.dim() == 0:  # 处理 0 维张量（标量）
                                # 获取 0 维张量的值并与单一路径结合
                                # 如果是标量，则假设只有一条路径  使用get时，如果不存在则返回空列表
                                temp = self.routing_table.get(src, {}).get(dst, [])
                                if temp:
                                    route = temp[0]  # 只有在 routes 非空时才获取第一个路径
                                else:
                                    # 处理 routes 为空的情况（例如，打印错误或使用默认值）
                                    # print(f"No route found from {src} to {dst}")
                                    continue   #直接跳出
                                if len(route) <= Lm:
                                    path_str = '-'.join(map(str, route))  # 将路径转换为字符串
                                    routing_info.append(f"{src},{dst},{path_str},{route_prob.item()}")
                                    
                            elif route_prob.dim() == 1:  # 处理 1 维张量
                                # 对 1 维张量进行迭代，处理多个路径
                                for route, prob in zip(self.routing_table[src][dst], route_prob):
                                    # print(f"流量从 {src} 到 {dst} 分配到路径 {route} 的比例是 {prob.item()}")
                                    if len(route) <= Lm:  # 过滤掉超过最大路径长度的路径
                                        path_str = '-'.join(map(str, route))  # 将路径转换为字符串
                                        routing_info.append(f"{src},{dst},{path_str},{prob.item()}")

                            else:
                                # 如果是多维张量，可以进一步处理或抛出异常，具体取决于需求
                                raise ValueError(f"route_prob 维度太高，无法处理：{route_prob.dim()}")


                        # transmitted = total_demand * prob  # 分配流量
                        # # transmitted_traffic是一个字典的字典
                        # # 每个 src 键对应一个字典，里面的每个 dst 键对应一个浮点数（或整数），表示从 src 到 dst 的已传输流量
                        # self.transmitted_traffic[src][dst] += transmitted  # 更新已传输流量

            # 检测无效动作：根据 Algorithm 3 实现无效动作检测逻辑
            # if not self.check_invalid_action_route():
            # 如果动作无效，直接返回负奖励
                # return self.selected_matrix,(-100,-100),done
        # 将所有路径信息序列化为字符串，并准备发送
            routing_str = ";".join(routing_info)
            # traffic_str = self.traffic_to_string(self.transmitted_traffic)
            routing_str="routemsg;"+routing_str
            self.send_to_omnetpp(routing_str,socket_connection)
        # 拼接两个 Agent 的结果并通过 socket 一起发送到 OMNeT++
        # combined_data = self.combine_agent_results()
        # self.send_combined_data_to_omnetpp(combined_data)

        # 接收 OMNeT++ 的反馈（新的状态）
        new_recvData=self.receive_from_omnetpp(socket_connection)
        self.handle_recvData(new_recvData)
        link_utilization,link_load=self.get_cur_state() # 得到链路利用率和流量负载
        new_state=(link_utilization,link_load)   #list

        # 计算奖励
        lu_reward, path_reward, total_reward = self.cal_cur_reward()
        # 输出 reward 及其各部分
        print("LU Reward:", lu_reward)
        print("Path Reward:", lu_reward)
        print("Total Reward:", lu_reward)
        reward=(lu_reward, path_reward, total_reward)
        
        
        
        return new_state, reward, done
    #####################################################
    
    def generate_routing_table(self):
        """
        根据selected_matrix生成每个POD的路由表。
        每个POD有到其他所有POD的可选路径，包括直达路径和中转一次的路径。
        返回一个字典，其中key是POD，value是到其他POD的可选路径集合。
        """
        routing_table = {src: {} for src in range(len(self.selected_matrix))}
        
        num_rows, num_cols = self.selected_matrix.shape  # 获取行数和列数

        # 遍历每个源节点(src)列
        for src in range(num_cols):
            # 初始化目的节点的路由
            routing_table[src] = {}

            # 遍历每个中转节点（行）
            for intermediate in range(num_rows):
                dst = self.selected_matrix[intermediate][src]  # 通过中转节点到达的目的节点

                if dst != -1 and src != dst:  # 如果路径有效且目的节点和源节点不相同
                    routes = []
                    # 添加直达路径
                    routes.append([src, dst])  # 直达路径

                    # 查找通过中转节点的路径
                    for next_hop in range(num_cols):
                        if next_hop != src and next_hop != dst:
                            intermediate_dst = self.selected_matrix[intermediate][next_hop]
                            if intermediate_dst != -1:
                                routes.append([src, intermediate, intermediate_dst])  # 一次中转路径

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

    def send_to_omnetpp(self, data,socket_connection):
        """
        通过 socket 将拼接好的数据发送给 OMNeT++。
        """
        socket_connection.sendall(data.encode())  # 发送拼接的数据

    def receive_from_omnetpp(self,socket_connection):
        return socket_connection.recv(4096).decode()  # 接收反馈

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
        visited = [False] * self.num_pods  # 用于标记每个节点是否已访问
        queue = deque([0])  # 从第一个节点开始
        visited[0] = True  # 标记第一个节点为访问过

        while queue:
            src = queue.popleft()
            
            # 遍历 selected_matrix 的 src 列，查找有效的目标节点
            for mid_node in range(self.selected_matrix.shape[0]):  # 遍历中间节点行
                dst = self.selected_matrix[mid_node, src]  # 获取 dst 节点
                if dst != -1 and not visited[dst]:  # 如果存在有效连接，且 dst 未访问过
                    visited[dst] = True
                    queue.append(dst)

        # 检查是否所有节点都被访问
        if all(visited):
            print("The graph is connected.")
            valide = True
        else:
            print("The graph is not connected.")
            valide = False
                            
        return valid

    def check_invalid_action_route(self):
        # 初始化 valid 为 True
        valid = True

        # 检查死锁
        if self.has_deadlock():
            valid = False
        print(valid)
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
        num_rows, num_cols = self.selected_matrix.shape
        graph = {src: set() for src in range(num_cols)}  # 每个源节点(src)作为key

        # 遍历 selected_matrix 来构建邻接表
        for src in range(num_cols):  # 遍历每列
            for intermediate in range(num_rows):  # 遍历每行，找到中转节点
                dst = self.selected_matrix[intermediate][src]  # 通过中转节点找到目的节点
                if dst != -1 and src != dst:  # 如果存在有效路径且源节点和目的节点不同
                    graph[src].add(dst)  # 记录 dst 为 src 的邻居
        return graph


    

