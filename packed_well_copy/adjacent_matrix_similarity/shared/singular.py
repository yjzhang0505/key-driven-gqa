import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm, colors
from collections import defaultdict
from _1_reading_ckpt import YourTransformerModel
from graph import compute_adjacency_matrix
from graph_imp_sim import generate_adjacency_matrices, generate_adjacency_matrices_2
from tools import save_to_txt
from scipy.optimize import minimize

# # 确保目录存在
# def ensure_directory_exists(directory_path):
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)

# # 计算矩阵相似度
# def normalize_matrix(matrix):
#     # 对每个元素进行归一化，使用 e^(x - max) 的方式
#     max_element = np.max(matrix)
#     normalized_matrix = np.exp(matrix - max_element)
#     return normalized_matrix

# def modified_similarity(vec1, vec2):
#     # 计算向量差
#     difference = vec1 - vec2

#     # 计算每个元素差的平方后乘以vec1的对应位置元素
#     weighted_difference = difference ** 2 * np.exp(vec1)

#     # 计算加权差的模
#     norm_difference = np.linalg.norm(weighted_difference)

#     # 计算原始向量的模
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)

#     # 计算结果
#     if norm_vec1 == 0 or norm_vec2 == 0:
#         return 0.0  # 避免除以零
#     return 1 - norm_difference / (norm_vec1 * norm_vec2)

# def similarity_score_matrix(A, B):
#     # 确保输入为numpy数组
#     A = np.array(A)
#     B = np.array(B)

#     # 归一化矩阵
#     norm_A = normalize_matrix(A)
#     norm_B = normalize_matrix(B)

#     # 计算整体相似性
#     sim_AB = modified_similarity(norm_A.flatten(), norm_B.flatten())  # 展平后计算相似性
#     return sim_AB


# # 绘制并保存邻接矩阵图
# def plot_adjacency_matrix_graph(adj_matrix, combination_name, iteration, output_dir):
#     G = nx.Graph()
#     n = adj_matrix.shape[0]
#     for i in range(n):
#         for j in range(i + 1, n):
#             weight = adj_matrix[i, j]
#             if weight > 0:
#                 G.add_edge(i, j, weight=weight)

#     pos = nx.circular_layout(G)
#     edges = G.edges(data=True)
#     weights = [attr['weight'] for _, _, attr in edges]

#     # 设置非线性规范，调整gamma值控制非线性程度
#     gamma_value = 4  # 根据需要调整gamma值
#     norm = colors.PowerNorm(gamma=gamma_value, vmin=min(weights), vmax=max(weights))
#     edge_colors = cm.Blues(norm(weights))

#     fig, ax = plt.subplots(figsize=(8, 8))
#     nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
#     nx.draw_networkx_labels(G, pos, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

#     # 设置非线性色条
#     sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label="Edge Weight (Adjacency Matrix Value)")

#     # 保存图片
#     filename = os.path.join(output_dir, f"{combination_name}_iter_{iteration}.png")
#     plt.savefig(filename, format="PNG")
#     plt.close(fig)
#     print(f"Saved adjacency graph for {combination_name} (Iteration {iteration}) at: {filename}")

# # 目标函数
# def objective(params):
#     weight_importance, weight_similarity = params
#     combined_adj_matrices = generate_adjacency_matrices_2(
#         model,
#         weight_importance=weight_importance,
#         weight_similarity=weight_similarity,
#         singularity_key='K_singular', 
#         similarity_key='V_cosine'
#     )
#     return similarity_score_matrix(torch.tensor(adj_matrix_1), combined_adj_matrices)

# # 初始化模型
model = YourTransformerModel(num_heads=12, dim=768)
model.load_pretrained_qkv_weights()


# # 第一个文件生成的邻接矩阵
# file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/calculate/group_all_aline.txt"
# adj_matrix_1 = compute_adjacency_matrix(file_path)
# plot_adjacency_matrix_graph(adj_matrix_1, "adj_matrix_1", iteration="initial", output_dir="/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/dustbin/try2/figure")

# # 定义五种绑定组合和五种相似性方式
# importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
# similarity_keys = ['V_cosine']

# # 设置保存文件的基础目录
# output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/dustbin/try2/files'
# figure_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/dustbin/try2/figure'

# # 遍历所有 25 种组合并进行优化
# for imp_key in importance_keys:
#     importance_singular_key = f"{imp_key}_singular"
    
#     for sim_key in similarity_keys:
#         combination_name = f"{importance_singular_key}_{sim_key}"
#         combination_dir = os.path.join(output_dir, combination_name)
#         ensure_directory_exists(combination_dir)

#         # 使用 scipy.optimize.minimize 进行优化
#         bounds = [(-1, 1), (-1, 1)]  # 权重范围
#         initial_guess = [0.0, 0.0]  # 初始猜测
#         print("begin")
#         result = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 30, 'xatol': 1e-4})
#         print("end")
#         # 获取最佳参数和最佳得分
#         best_params = result.x
#         best_score = -result.fun  # 因为我们最小化的是相似度的负值

#         # 使用最佳参数生成邻接矩阵
#         best_adj_matrix = generate_adjacency_matrices_2(
#             model,
#             weight_importance=best_params[0],
#             weight_similarity=best_params[1],
#             singularity_key='K_singular', 
#             similarity_key='V_cosine'
#         )

#         # 保存邻接矩阵到 txt 文件
#         adj_matrix_path = os.path.join(combination_dir, "adjacent.txt")
#         save_to_txt(adj_matrix_path, np.array2string(best_adj_matrix.numpy()))

#         # 保存最佳参数和相似度得分到另一个 txt 文件
#         params_path = os.path.join(combination_dir, "params.txt")
#         params_content = (
#             f"best_params:\n{best_params}\n"
#             f"best_score:\n{best_score}"
#         )
#         save_to_txt(params_path, params_content)

#         # 绘制并保存邻接矩阵图
#         plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=best_score, output_dir=combination_dir)
#         plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=best_score, output_dir=figure_dir)

#         # 打印当前方案的结果
#         print(f"{combination_name} 的最佳参数组合:", best_params)
#         print(f"{combination_name} 的最高相似度分数: {best_score}")
