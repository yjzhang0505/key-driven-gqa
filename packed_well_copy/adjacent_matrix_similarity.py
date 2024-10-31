import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm, colors
from skopt import gp_minimize
from skopt.space import Real
from collections import defaultdict
from _1_reading_ckpt import YourTransformerModel
from graph import compute_adjacency_matrix
from graph_imp_sim import generate_adjacency_matrices
from tools import save_to_txt

# 第一个文件生成的邻接矩阵
file_path = "/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt"
adj_matrix_1 = compute_adjacency_matrix(file_path)

# def similarity_score_matrix(A, B):
#     # 确保 A 和 B 是 numpy 数组
#     if isinstance(A, torch.Tensor):
#         A = A.cpu().detach().numpy()
#     if isinstance(B, torch.Tensor):
#         B = B.cpu().detach().numpy()
    
#     # 将矩阵 A 和 B 展平为一维向量
#     A_flat = A.flatten()
#     B_flat = B.flatten()

#     # 计算余弦相似度
#     dot_product = np.dot(A_flat, B_flat)
#     norm_A = np.linalg.norm(A_flat)
#     norm_B = np.linalg.norm(B_flat)
    
#     if norm_A == 0 or norm_B == 0:
#         return 0  # 若模长为零，返回0以避免除零错误
    
#     return dot_product / (norm_A * norm_B)


def similarity_score_matrix(A, B):
    # 确保 A 和 B 是 numpy 数组
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    if isinstance(B, torch.Tensor):
        B = B.cpu().detach().numpy()
    
    n = A.shape[0]  # 假设 A 和 B 都是 n x m 的矩阵
    
    # 计算 sim(A, B)
    sim_AB = 0
    for i in range(n):
        cosine_line = cosine_similarity(A[i], B[i])
        cosine_row = cosine_similarity(A[:, i], B[:, i])
        sim_AB += cosine_line + cosine_row

    sim_AB = sim_AB / (2 * n)
    return sim_AB

def cosine_similarity(u, v):
    # 计算余弦相似度
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0  # 若模长为零，返回0
    return dot_product / (norm_u * norm_v)

def plot_adjacency_matrix_graph(adj_matrix, combination_name, iteration):
    G = nx.Graph()
    n = adj_matrix.shape[0]
    
    # 创建边并设置权重
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj_matrix[i, j]
            if weight > 0:  # 仅绘制有权重的边
                G.add_edge(i, j, weight=weight)

    # 设置布局为圆形
    pos = nx.circular_layout(G)

    # 获取边的权重
    edges = G.edges(data=True)
    weights = [attr['weight'] for _, _, attr in edges]

    # 使用 PowerNorm 设置非线性颜色条
    norm = colors.PowerNorm(gamma=1, vmin=min(weights), vmax=max(weights))
    edge_colors = cm.Blues(norm(weights))

    # 绘制图像并保存
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)
    
    # 添加颜色条到图的轴
    sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Edge Weight (Adjacency Matrix Value)")

    # 保存图像
    filename = f"/data/yjzhang/desktop/try/key-driven-gqa/figure/adjacent_matrix/figure/{combination_name}_iter_{iteration}.png"
    plt.savefig(filename, format="PNG")
    plt.close(fig)
    print(f"Saved adjacency graph for {combination_name} (Iteration {iteration}) at: {filename}")

# 初始化模型
model = YourTransformerModel(num_heads=12, dim=768)
model.load_pretrained_qkv_weights()
# 绘制并保存 adj_matrix_1 的图
# plot_adjacency_matrix_graph(torch.tensor(adj_matrix_1), "adj_matrix_1", iteration="initial")

# adj_matrix_2 = generate_adjacency_matrices(model, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4,
#                                 importance_mean_key='K_mean', importance_var_key='K_var', similarity_key='K_similarity_matrix')
# a = similarity_score_matrix(adj_matrix_1, adj_matrix_2)
# print(a)


# # 定义五种绑定组合和五种相似性方式
importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
similarity_keys = [
    'K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix',
    'KxQ_similarity_matrix', 'KxQxV_similarity_matrix'
]

# 贝叶斯优化搜索空间
space = [
    Real(-1, 1, name="weight_mean"),
    Real(-1, 1, name="weight_var"),
    Real(-1, 1, name="weight_similarity")
]

# 记录最佳结果
results = {}
best_overall_score = -1e6
best_overall_combination = None
best_overall_params = None

# 遍历所有25种组合并进行贝叶斯优化
for imp_key in importance_keys:
    importance_mean_key = f"{imp_key}_mean"
    importance_var_key = f"{imp_key}_var"
    
    for sim_key in similarity_keys:
        combination_name = f"{importance_mean_key}_{importance_var_key}_{sim_key}"
        
        # 定义优化目标函数
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
            similarity_score = similarity_score_matrix(torch.tensor(adj_matrix_1), combined_adj_matrices)
            return -similarity_score  # 使用负值，因为 gp_minimize 是最小化

        # 使用贝叶斯优化搜索最佳权重组合
        result = gp_minimize(objective, space, n_calls=20, n_initial_points=10)

        # 获取最佳参数和相似度分数
        best_params = result.x
        best_score = -result.fun  # 因为返回的是负相似度，所以需要取负值还原
        results[combination_name] = {
            "best_params": best_params,
            "best_score": best_score
        }

        # 绘制并保存邻接矩阵图
        best_adj_matrix = generate_adjacency_matrices(
            model,
            weight_mean=best_params[0],
            weight_var=best_params[1],
            weight_similarity=best_params[2],
            importance_mean_key=importance_mean_key,
            importance_var_key=importance_var_key,
            similarity_key=sim_key
        )
        # print(best_adj_matrix)
        Output_dir = '/data/yjzhang/desktop/try/key-driven-gqa/figure/adjacent_matrix/files'
        output_dir = os.path.join(Output_dir,combination_name)
        output_adjacent_path = os.path.join(output_dir, "adjacent.txt")
        save_to_txt(output_adjacent_path, np.array2string(best_adj_matrix.numpy()))
        output_params_path = os.path.join(output_dir, "params.txt")
        output_content = (
            "best_params:\n" +
            str(best_params) + "\n" +
            "best_score:\n" +
            str(best_score)
        )

        # 一次性写入
        save_to_txt(output_params_path, output_content)

        plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=result.fun)

        print(f"{combination_name} 的最佳参数组合:", best_params)
        print(f"{combination_name} 的最高相似度分数: {best_score}")

        # 检查是否为当前最高分数
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_combination = combination_name
            best_overall_params = best_params

# # 打印总体最佳结果
# print("\n总体最高相似度分数的组合:")
# print(f"组合: {best_overall_combination}")
# print(f"最高相似度分数: {best_overall_score}")
# print(f"最佳参数: {best_overall_params}")
