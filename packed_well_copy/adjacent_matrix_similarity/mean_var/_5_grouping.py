import numpy as np
from collections import defaultdict

# 假设 compute_adjacency_matrix 已经在另一个文件中定义，可以导入
from graph2 import compute_adjacency_matrix

def group_heads(adj_matrix):
    """
    根据邻接矩阵将头分成若干组。每次选择边权重最大的两个节点分为一组，直到所有节点分组完毕。

    参数:
    - adj_matrix (np.ndarray): 已归一化的邻接矩阵。

    返回:
    - groups (list): 包含每组头的列表。
    """
    num_nodes = adj_matrix.shape[0]
    grouped = set()  # 用于记录已经分组的节点
    groups = []      # 用于存储分组结果

    # 开始分组过程
    while len(grouped) < num_nodes:
        max_weight = -1
        max_pair = None

        # 找到未分组节点中边权重最大的两个节点
        for i in range(num_nodes):
            if i in grouped:
                continue
            for j in range(i + 1, num_nodes):
                if j in grouped:
                    continue
                weight = adj_matrix[i, j]
                if weight > max_weight:
                    max_weight = weight
                    max_pair = (i, j)

        # 如果找到最大边，则将对应的节点加入分组
        if max_pair:
            grouped.update(max_pair)
            groups.append(max_pair)

        # 如果节点数是奇数，则最后剩一个节点单独成组
        if len(grouped) < num_nodes and num_nodes - len(grouped) == 1:
            remaining_node = [i for i in range(num_nodes) if i not in grouped][0]
            groups.append((remaining_node,))
            grouped.add(remaining_node)

    return groups

# 示例调用
file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/calculate/group_all_aline.txt"
adj_matrix = compute_adjacency_matrix(file_path, num_nodes=12)
grouped_heads = group_heads(adj_matrix)
# print(adj_matrix)

print("分组结果:", grouped_heads)
