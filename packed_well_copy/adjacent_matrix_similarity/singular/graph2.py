import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from collections import defaultdict
import re
import math
import numpy as np

def compute_adjacency_matrix(file_path, num_nodes=12):
    """
    计算给定文件中的分组方案所对应的邻接矩阵，并将邻接矩阵的均值缩放为 1。

    参数:
    - file_path (str): 包含分组方案和准确度的文件路径。
    - num_nodes (int): 节点数量，默认为 12。

    返回:
    - adj_matrix (np.ndarray): 归一化后的邻接矩阵。
    """
    # 初始化边权重计数
    edge_weights = defaultdict(float)
    count_pairs = defaultdict(int)

    # 正则表达式匹配分组方案中的每对头
    pattern = re.compile(r"\((\d+), (\d+)\)")

    # 读取文件并处理数据
    with open(file_path, "r") as file:
        for line in file:
            try:
                # 分割行，提取分组方案和准确度部分
                parts = line.strip().rsplit(", ", 2)
                groups_part = parts[0]  # 分组方案部分
                test_acc = float(parts[1])  # 解析准确度
                # print(test_acc)

                # 使用指数函数对 test_acc 进行放大
                adjusted_acc = math.exp(200 * test_acc)

                # 使用正则表达式查找分组对
                groups = pattern.findall(groups_part)
                if not groups:
                    continue  # 跳过没有匹配到分组的行

                # 累加每个头对的权重（按放大后的准确度）和分组次数
                for pair in groups:
                    head1, head2 = int(pair[0]), int(pair[1])
                    edge = tuple(sorted((head1, head2)))  # 确保无向边
                    edge_weights[edge] += adjusted_acc
                    count_pairs[edge] += 1  # 仅在头被分在一起时计数

            except (ValueError, IndexError, SyntaxError) as e:
                print(f"Skipping line due to error: {e}")

    # 初始化邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # 填充邻接矩阵，除以每对头被分在一起的次数以取平均
    for edge, total_weight in edge_weights.items():
        node1, node2 = edge
        if count_pairs[edge] > 0:  # 确保至少有一个分组包含这对头
            avg_weight = total_weight / count_pairs[edge]
            # print(avg_weight)
            adj_matrix[node1, node2] = avg_weight
            adj_matrix[node2, node1] = avg_weight  # 对称填充

    # 归一化处理，使得邻接矩阵的均值为 1
    matrix_mean = adj_matrix.mean()
    if matrix_mean != 0:
        adj_matrix /= matrix_mean

    return adj_matrix



def plot_adjacency_matrix(adj_matrix, save_path="/data/yjzhang/desktop/try/key-driven-gqa/figure/_head_grouping_graph.png"):
    """
    根据邻接矩阵绘制图并保存为图像文件。

    参数:
    - adj_matrix (np.ndarray): 邻接矩阵。
    - save_path (str): 保存图像的路径。
    """
    # 从邻接矩阵生成图
    G = nx.from_numpy_array(adj_matrix)

    # 设置图布局
    num_nodes = adj_matrix.shape[0]
    radius = 1.0
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    pos = {i: (radius * np.cos(angle), radius * np.sin(angle)) for i, angle in enumerate(angles)}

    # 设置边的颜色映射
    edges = G.edges(data=True)
    weights = [attr['weight'] for _, _, attr in edges]

    # 使用 PowerNorm 设置非线性颜色条
    norm = colors.PowerNorm(gamma=3, vmin=min(weights), vmax=max(weights))
    edge_colors = cm.Blues(norm(weights))

    # 创建图像和颜色条
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

    # 添加颜色条到图的轴
    sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Grouping Frequency Weight")

    # 保存图像
    plt.savefig(save_path, format="PNG")
    plt.show()


# 允许其他模块调用的接口
def generate_and_plot_adjacency_matrix():
    """
    从文件生成邻接矩阵并绘制图。

    参数:
    - file_path (str): 包含分组方案和准确度的文件路径。
    - save_path (str): 保存图像的路径。
    """
    file_path = "/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt"
    save_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/_head_grouping_graph.png"
    adj_matrix = compute_adjacency_matrix(file_path)
    # print(adj_matrix)
    plot_adjacency_matrix(adj_matrix, save_path)




# 调用函数生成并绘制邻接矩阵图
generate_and_plot_adjacency_matrix()