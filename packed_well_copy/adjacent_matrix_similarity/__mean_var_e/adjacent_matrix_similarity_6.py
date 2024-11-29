
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm, colors
from collections import defaultdict
from _1_reading_ckpt import YourTransformerModel
from graph import compute_adjacency_matrix
from graph_imp_sim import generate_adjacency_matrices
from tools import save_to_txt

# 确保目录存在
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# 计算矩阵相似度
def similarity_score_matrix(A, B, K=10):
    """
    使用对齐最大权重边的余弦相似性计算矩阵 A 和 B 的相似性。
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    if isinstance(B, torch.Tensor):
        B = B.cpu().detach().numpy()

    # 获取 A 和 B 中权重最高的前 K 条边的权重向量
    top_k_weights_A = np.sort(A[np.triu_indices_from(A, k=1)])[-K:]
    top_k_weights_B = np.sort(B[np.triu_indices_from(B, k=1)])[-K:]

    # 计算余弦相似性
    dot_product = np.dot(top_k_weights_A, top_k_weights_B)
    norm_A = np.linalg.norm(top_k_weights_A)
    norm_B = np.linalg.norm(top_k_weights_B)
    similarity_score = dot_product / (norm_A * norm_B) if norm_A != 0 and norm_B != 0 else 0

    return similarity_score




# 绘制并保存邻接矩阵图
def plot_adjacency_matrix_graph(adj_matrix, combination_name, iteration, output_dir):
    G = nx.Graph()
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    pos = nx.circular_layout(G)
    edges = G.edges(data=True)
    weights = [attr['weight'] for _, _, attr in edges]
    norm = colors.PowerNorm(gamma=4, vmin=min(weights), vmax=max(weights))
    edge_colors = cm.Blues(norm(weights))

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Edge Weight (Adjacency Matrix Value)")

    # 保存图片
    filename = os.path.join(output_dir, f"{combination_name}_iter_{iteration}.png")
    plt.savefig(filename, format="PNG")
    plt.close(fig)
    print(f"Saved adjacency graph for {combination_name} (Iteration {iteration}) at: {filename}")

# 多启动蒙特卡洛优化函数
def multi_start_monte_carlo_search(objective_func, bounds, num_starts=5, num_iterations=3):
    best_score = -np.inf
    best_params = None

    for _ in range(num_starts):
        params = [np.random.uniform(low, high) for low, high in bounds]

        for _ in range(num_iterations):
            print("1")
            score = objective_func(params)
            if score > best_score:
                best_score = score
                best_params = params

            params = [np.clip(param + np.random.normal(scale=0.1), low, high)
                      for param, (low, high) in zip(params, bounds)]

    return best_params, best_score

# 搜索空间
bounds = [(-1, 1), (-1, 1), (-1, 1)]

# 目标函数
def objective(params):
    weight_mean, weight_var, weight_similarity = params
    combined_adj_matrices = generate_adjacency_matrices(
        model,
        weight_mean=weight_mean,
        weight_var=weight_var,
        weight_similarity=weight_similarity,
        importance_mean_key=importance_mean_key,
        importance_var_key=importance_var_key,
        similarity_key=sim_key
    )
    return similarity_score_matrix(torch.tensor(adj_matrix_1), combined_adj_matrices)

# 初始化模型
model = YourTransformerModel(num_heads=12, dim=768)
model.load_pretrained_qkv_weights()

# 第一个文件生成的邻接矩阵
file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/calculate/group_all_aline.txt"
adj_matrix_1 = compute_adjacency_matrix(file_path)
plot_adjacency_matrix_graph(adj_matrix_1, "adj_matrix_1", iteration="initial", output_dir="/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_6/figure")

# 定义五种绑定组合和五种相似性方式
importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
similarity_keys = [
    'K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix',
    'KxQ_similarity_matrix', 'KxQxV_similarity_matrix'
]

# 设置保存文件的基础目录
output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_6/files'
figure_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_6/figure'

# 遍历所有 25 种组合并进行多启动蒙特卡洛优化
for imp_key in importance_keys:
    importance_mean_key = f"{imp_key}_mean"
    importance_var_key = f"{imp_key}_var"
    
    for sim_key in similarity_keys:
        combination_name = f"{importance_mean_key}_{importance_var_key}_{sim_key}"
        combination_dir = os.path.join(output_dir, combination_name)
        ensure_directory_exists(combination_dir)

        # 多启动蒙特卡洛搜索最佳权重组合
        best_params, best_score = multi_start_monte_carlo_search(objective, bounds)

        # 使用最佳参数生成邻接矩阵
        best_adj_matrix = generate_adjacency_matrices(
            model,
            weight_mean=best_params[0],
            weight_var=best_params[1],
            weight_similarity=best_params[2],
            importance_mean_key=importance_mean_key,
            importance_var_key=importance_var_key,
            similarity_key=sim_key
        )

        # 保存邻接矩阵到 txt 文件
        adj_matrix_path = os.path.join(combination_dir, "adjacent.txt")
        save_to_txt(adj_matrix_path, np.array2string(best_adj_matrix.numpy()))

        # 保存最佳参数和相似度得分到另一个 txt 文件
        params_path = os.path.join(combination_dir, "params.txt")
        params_content = (
            f"best_params:\n{best_params}\n"
            f"best_score:\n{best_score}"
        )
        save_to_txt(params_path, params_content)

        # 绘制并保存邻接矩阵图
        plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=best_score, output_dir=combination_dir)
        plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=best_score, output_dir=figure_dir)

        # 打印当前方案的结果
        print(f"{combination_name} 的最佳参数组合:", best_params)
        print(f"{combination_name} 的最高相似度分数: {best_score}")
