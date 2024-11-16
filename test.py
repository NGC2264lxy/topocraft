# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2024/6/1 15:01
# @Author : LiangQin
# @Email : liangqin@stu.xidian.edu.cn
# @File : test.py
# @Software: PyCharm

import math
import numpy as np
from collections import deque
from scipy.spatial.distance import jensenshannon
from sko.DE import DE
import itertools
import pandas as pd
from openpyxl import load_workbook

file_path = 'F://SJU//idea//topoCrafter_code//data//.logical_topo.xlsx'

string="False;0,42496,2560,1536,256,1280,768,1024;1536,0,44032,768,512,1536,768,768;1280,1280,0,42240,1792,1024,768,2048;1024,768,1280,0,41728,2560,512,1280;1024,768,768,2048,0,41984,1280,2048;1536,1024,2048,1792,1280,0,42496,768;768,768,768,3072,1792,768,0,42240;43008,1024,1280,768,1024,768,1536,0;1,0,3,2,5,4,7,6;2,3,0,1,6,7,4,5;3,2,1,0,7,6,5,4;4,5,6,7,0,1,2,3;5,4,7,6,1,0,3,2;6,7,4,5,2,3,0,1;7,6,5,4,3,2,1,0;-1,-1,-1,-1,-1,-1,-1,-1;8"
# 解析字符串
parts = string.split(';')
complete_flag = parts[0]
traffic_matrix = []
topology = []

B = 100*1e9

pop_size = 100
max_iter = 1000


for i in range(1, 9):
    traffic_matrix.append(list(map(int, parts[i].split(','))))

for i in range(9, 17):
    topology.append(list(map(int, parts[i].split(','))))

def get_cur_topology(data):
    num_groups = len(data[0])
    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((num_groups, num_groups), dtype=int)
    # 填充邻接矩阵
    for groups in data:
        for group_id in groups:
            if group_id != -1:
                for other_group_id in groups:
                    if other_group_id != -1 and other_group_id != group_id:
                        adjacency_matrix[group_id][other_group_id] = 1
    return adjacency_matrix

def find_global_paths(adjacency_matrix, max_length):
    num_groups = len(adjacency_matrix)
    max_length = max_length
    paths_within_groups = []

    for group_id in range(num_groups):
        queue = deque([(group_id, [group_id])])
        while queue:
            (current, path) = queue.popleft()
            if len(path) - 1 > max_length:
                continue
            for neighbor in range(num_groups):
                if adjacency_matrix[current][neighbor] == 1 and neighbor not in path:
                    new_path = path + [neighbor]
                    if len(new_path) - 1 <= max_length:
                        paths_within_groups.append(new_path)
                        queue.append((neighbor, new_path))
    return paths_within_groups

def cal_reward(tm, averageLength):
    # 计算链路利用率
    lu = [[element * 1000 * 64 * 8 / (100*(10**9)) for element in row] for row in tm]
    ######################## 计算链路利用率差异度 #########################
    # 剔除链路利用率中的0元素
    filtered_matrix = [[value for value in sublist if value != 0.0] for sublist in lu]
    lu_mean = np.mean(filtered_matrix)
    sum_squared_diff = np.sum((filtered_matrix - lu_mean) ** 2)
    num_elements = sum(len(row) for row in filtered_matrix)
    lu_diff = np.sqrt(sum_squared_diff / num_elements)

    ############ 分别计算两个Agent的reward和总reward ######################
    lu_reward = np.exp(-lu_diff)
    path_reward = np.exp(-averageLength)
    total_reward = np.exp(-lu_diff-averageLength)

    return lu, path_reward, total_reward

def js_divergence_scipy(P, Q):
    """
    利用SciPy直接计算Jensen-Shannon散度。
    """
    # Ensure the matrices are normalized to represent probability distributions
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if not isinstance(Q, np.ndarray):
        Q = np.array(Q)

    # Flatten the matrices to ensure we handle them as 1D arrays
    P = P.flatten()
    Q = Q.flatten()

    return jensenshannon(P, Q) ** 2

# # 定义适应度函数
# def fitness(individual):
#     lu_diff = calculate_link_utilization_diff(individual, traffic_matrix)
#     return lu_diff
#
#
# # 计算链路利用率差异度的函数
# def calculate_link_utilization_diff(individual, flow_matrix):
#     if not isinstance(individual, np.ndarray):
#         individual = np.array(individual)
#     if not isinstance(flow_matrix, np.ndarray):
#         flow_matrix = np.array(flow_matrix)
#
#     # Flatten the matrices to ensure we handle them as 1D arrays
#     P = individual.flatten()
#     Q = flow_matrix.flatten()
#     return jensenshannon(P, Q) ** 2
#
# def constraint(individual):
#     topology = individual.reshape((N, N))
#     link_count = np.sum(topology, axis=1)
#     total_links = np.sum(topology)
#     constraint1 = np.max(link_count) - N
#     constraint2 = total_links - 16
#     return np.array([constraint1, constraint2])
#
# # 初始化种群
# def initialize_population(pop_size, N, M, initial_topo):
#     population = []
#     for _ in range(pop_size):
#         individual = initial_topo.copy().flatten()
#         available_indices = np.where(individual == 0)[0]
#         if len(available_indices) < M:
#             raise ValueError("Not enough available indices to initialize population")
#         chosen_indices = np.random.choice(available_indices, M, replace=False)
#         individual[chosen_indices] = 1
#         population.append(individual)
#     return np.array(population)

def generate_possible_topologies(initial_topology, additional_links):
    size = initial_topology.shape[0]
    possible_topologies = []

    # Get all pairs of nodes
    all_pairs = list(itertools.product(range(size), repeat=2))

    # Generate all possible ways to distribute additional links among these pairs
    for added_links in itertools.product(range(additional_links + 1), repeat=len(all_pairs)):
        if sum(added_links) != additional_links:
            continue

        new_topology = initial_topology.copy()
        links_added = 0

        for count, (i, j) in zip(added_links, all_pairs):
            if new_topology[i][j] + count <= 2:
                new_topology[i][j] += count
                links_added += count

        if links_added == additional_links:
            if np.all(np.sum(new_topology, axis=0) <= size) and np.all(np.sum(new_topology, axis=1) <= size):
                possible_topologies.append(new_topology)

    return possible_topologies


def gen_topo_action_space(tmx):
    '''
    # step 1: 初始化链路配置，每个Pod与至少1个Pod有至少1条链路, 构成一个环形；
    step 2: 根据流量矩阵的需求分配链路（双向配置）；
    stpe 3: 将剩下的链路分配至各个节点；
    :param tmx: 流量矩阵
    '''
    dim_tmx = len(tmx)
    total_link = int(0.5 * dim_tmx * (dim_tmx - 1))
    num_globalLink = int(0.5 * dim_tmx * (dim_tmx - 1))
    #################### step 1 ######################
    # cycle = []
    # for i in range(dim_tmx):
    #     tmp = []
    #     for j in range(dim_tmx):
    #         if (abs(i-j)==1 or abs(i-j)==dim_tmx-1):
    #             tmp.append(1)
    #         else:
    #             tmp.append(0)
    #     cycle.append(tmp)
    # num_iniLink = cycle
    ################### step 2 ######################
    num_iniLink = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(tmx)):
        for j in range(len(tmx)):
            tmp_value = max(tmx[i][j], tmx[j][i])   # 双向配置
            num_pod_link = math.floor(tmp_value * 1000 * 64 * 8 * 10 / B)
            if (num_pod_link==0):
                continue
            else:
                num_iniLink[i][j] = num_pod_link
                num_iniLink[j][i] = num_pod_link
    num_iniLink = np.array(num_iniLink)
    link_count = np.sum(num_iniLink, axis=1)
    print(link_count)      # 统计初始拓扑的链路数量
    num_iniLink = np.array(num_iniLink, dtype=int)
    print(num_iniLink)
    # ################### step 3 #####################
    possible_topologies = generate_possible_topologies(num_iniLink, 32)
    # List to store JS divergence values and topologies
    topologies_js_divergence = []

    # Calculate JS divergence for each topology
    for topology in possible_topologies:
        js_div = js_divergence_scipy(topology, traffic_matrix)
        topologies_js_divergence.append((js_div, topology))

    # Sort the topologies by JS divergence in descending order and select top 500
    topologies_js_divergence.sort(key=lambda x: x[0], reverse=True)
    top_500_topologies = topologies_js_divergence[:500]

    file_path2 = 'F://SJU//idea//topoCrafter_code//data//.logical_topo.xlsx'

    # Prepare data for saving to CSV
    data_raw = []
    for js_div, topology in topologies_js_divergence:
        data_raw.append([js_div] + topology.flatten().tolist())

    df = pd.DataFrame(data_raw)
    df.to_excel(file_path2, index=False)



    # initial_population = initialize_population(pop_size, dim_tmx, 12, num_iniLink)
    # print(initial_population)
    # de = DE(func=fitness, n_dim=dim_tmx*dim_tmx, size_pop=pop_size, max_iter=max_iter, lb=0, ub=1, constraint_eq=[constraint])
    # de.Chrom = initial_population  # 使用初始化的种群
    #
    # # 运行差分进化算法
    # best_x, best_y = de.run()
    #
    # # 将优化结果转换为拓扑结构
    # best_topology = best_x.reshape((dim_tmx, dim_tmx)).astype(int)
    # print("Optimal Topology:\n", best_topology)
    # print("Best Fitness:\n", best_y)

gen_topo_action_space(traffic_matrix)




# data = pd.read_excel(file_path, header=None, skiprows=1)
# topologies_js_divergence = []
# file_path2 = 'G://topoCrafter_code//data//JS_logical_topo.xlsx'
#
# for i in range(0, len(data)):
#     tmp_topo = np.array(data.iloc[i]).reshape(8, 8)
#     js_value = js_divergence_scipy(tmp_topo, traffic_matrix)
#     topologies_js_divergence.append((js_value, tmp_topo))

# # Sort the topologies by JS divergence in descending order
# topologies_js_divergence.sort(key=lambda x: x[0], reverse=True)

