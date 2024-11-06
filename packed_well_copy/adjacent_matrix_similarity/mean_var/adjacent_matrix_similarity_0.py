# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib import cm, colors
# from skopt import gp_minimize
# from skopt.space import Real
# from collections import defaultdict
# from _1_reading_ckpt import YourTransformerModel
# from graph import compute_adjacency_matrix
# from graph_imp_sim import generate_adjacency_matrices
# from tools import save_to_txt

# # 第一个文件生成的邻接矩阵
# file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/calculate/group_all_aline.txt"
# adj_matrix_1 = compute_adjacency_matrix(file_path)

# # def similarity_score_matrix(A, B):
# #     # 确保 A 和 B 是 numpy 数组
# #     if isinstance(A, torch.Tensor):
# #         A = A.cpu().detach().numpy()
# #     if isinstance(B, torch.Tensor):
# #         B = B.cpu().detach().numpy()
    
# #     # 将矩阵 A 和 B 展平为一维向量
# #     A_flat = A.flatten()
# #     B_flat = B.flatten()

# #     # 计算余弦相似度
# #     dot_product = np.dot(A_flat, B_flat)
# #     norm_A = np.linalg.norm(A_flat)
# #     norm_B = np.linalg.norm(B_flat)
    
# #     if norm_A == 0 or norm_B == 0:
# #         return 0  # 若模长为零，返回0以避免除零错误
    
# #     return dot_product / (norm_A * norm_B)


# def similarity_score_matrix(A, B):
#     # 确保 A 和 B 是 numpy 数组
#     if isinstance(A, torch.Tensor):
#         A = A.cpu().detach().numpy()
#     if isinstance(B, torch.Tensor):
#         B = B.cpu().detach().numpy()
    
#     n = A.shape[0]  # 假设 A 和 B 都是 n x m 的矩阵
    
#     # 计算 sim(A, B)
#     sim_AB = 0
#     for i in range(n):
#         cosine_line = cosine_similarity(A[i], B[i])
#         cosine_row = cosine_similarity(A[:, i], B[:, i])
#         sim_AB += cosine_line + cosine_row

#     sim_AB = sim_AB / (2 * n)
#     return sim_AB

# def cosine_similarity(u, v):
#     # 计算余弦相似度
#     dot_product = np.dot(u, v)
#     norm_u = np.linalg.norm(u)
#     norm_v = np.linalg.norm(v)
#     if norm_u == 0 or norm_v == 0:
#         return 0  # 若模长为零，返回0
#     return dot_product / (norm_u * norm_v)

# def plot_adjacency_matrix_graph(adj_matrix, combination_name, iteration):
#     G = nx.Graph()
#     n = adj_matrix.shape[0]
    
#     # 创建边并设置权重
#     for i in range(n):
#         for j in range(i + 1, n):
#             weight = adj_matrix[i, j]
#             if weight > 0:  # 仅绘制有权重的边
#                 G.add_edge(i, j, weight=weight)

#     # 设置布局为圆形
#     pos = nx.circular_layout(G)

#     # 获取边的权重
#     edges = G.edges(data=True)
#     weights = [attr['weight'] for _, _, attr in edges]

#     # 使用 PowerNorm 设置非线性颜色条
#     norm = colors.PowerNorm(gamma=1, vmin=min(weights), vmax=max(weights))
#     edge_colors = cm.Blues(norm(weights))

#     # 绘制图像并保存
#     fig, ax = plt.subplots(figsize=(8, 8))
#     nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
#     nx.draw_networkx_labels(G, pos, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)
    
#     # 添加颜色条到图的轴
#     sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label="Edge Weight (Adjacency Matrix Value)")

#     # 保存图像
#     filename = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix/figure/{combination_name}_iter_{iteration}.png"
#     plt.savefig(filename, format="PNG")
#     plt.close(fig)
#     print(f"Saved adjacency graph for {combination_name} (Iteration {iteration}) at: {filename}")

# # 初始化模型
# model = YourTransformerModel(num_heads=12, dim=768)
# model.load_pretrained_qkv_weights()
# # 绘制并保存 adj_matrix_1 的图
# plot_adjacency_matrix_graph(torch.tensor(adj_matrix_1), "adj_matrix_1", iteration="initial")

# # adj_matrix_2 = generate_adjacency_matrices(model, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4,
# #                                 importance_mean_key='K_mean', importance_var_key='K_var', similarity_key='K_similarity_matrix')
# # a = similarity_score_matrix(adj_matrix_1, adj_matrix_2)
# # print(a)


# # # 定义五种绑定组合和五种相似性方式
# importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
# similarity_keys = [
#     'K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix',
#     'KxQ_similarity_matrix', 'KxQxV_similarity_matrix'
# ]

# # 贝叶斯优化搜索空间
# space = [
#     Real(-1, 1, name="weight_mean"),
#     Real(-1, 1, name="weight_var"),
#     Real(-1, 1, name="weight_similarity")
# ]

# # 记录最佳结果
# results = {}
# best_overall_score = -1e6
# best_overall_combination = None
# best_overall_params = None

# # 遍历所有25种组合并进行贝叶斯优化
# for imp_key in importance_keys:
#     importance_mean_key = f"{imp_key}_mean"
#     importance_var_key = f"{imp_key}_var"
    
#     for sim_key in similarity_keys:
#         combination_name = f"{importance_mean_key}_{importance_var_key}_{sim_key}"
        
#         # 定义优化目标函数
#         def objective(params):
#             weight_mean, weight_var, weight_similarity = params
#             combined_adj_matrices = generate_adjacency_matrices(
#                 model,
#                 weight_mean=weight_mean,
#                 weight_var=weight_var,
#                 weight_similarity=weight_similarity,
#                 importance_mean_key=importance_mean_key,
#                 importance_var_key=importance_var_key,
#                 similarity_key=sim_key
#             )
#             similarity_score = similarity_score_matrix(torch.tensor(adj_matrix_1), combined_adj_matrices)
#             return -similarity_score  # 使用负值，因为 gp_minimize 是最小化

#         # 使用贝叶斯优化搜索最佳权重组合
#         result = gp_minimize(objective, space, n_calls=200, n_initial_points=10)
#         print(result)

#         # 获取最佳参数和相似度分数
#         best_params = result.x
#         best_score = -result.fun  # 因为返回的是负相似度，所以需要取负值还原
#         results[combination_name] = {
#             "best_params": best_params,
#             "best_score": best_score
#         }

#         # 绘制并保存邻接矩阵图
#         best_adj_matrix = generate_adjacency_matrices(
#             model,
#             weight_mean=best_params[0],
#             weight_var=best_params[1],
#             weight_similarity=best_params[2],
#             importance_mean_key=importance_mean_key,
#             importance_var_key=importance_var_key,
#             similarity_key=sim_key
#         )
#         # print(best_adj_matrix)
#         Output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix/files'
#         output_dir = os.path.join(Output_dir,combination_name)
#         output_adjacent_path = os.path.join(output_dir, "adjacent.txt")
#         save_to_txt(output_adjacent_path, np.array2string(best_adj_matrix.numpy()))
#         output_params_path = os.path.join(output_dir, "params.txt")
#         output_content = (
#             "best_params:\n" +
#             str(best_params) + "\n" +
#             "best_score:\n" +
#             str(best_score)
#         )

#         # 一次性写入
#         save_to_txt(output_params_path, output_content)

#         plot_adjacency_matrix_graph(best_adj_matrix, combination_name, iteration=result.fun)

#         print(f"{combination_name} 的最佳参数组合:", best_params)
#         print(f"{combination_name} 的最高相似度分数: {best_score}")

#         # 检查是否为当前最高分数
#         if best_score > best_overall_score:
#             best_overall_score = best_score
#             best_overall_combination = combination_name
#             best_overall_params = best_params

# # # 打印总体最佳结果
# # print("\n总体最高相似度分数的组合:")
# # print(f"组合: {best_overall_combination}")
# # print(f"最高相似度分数: {best_overall_score}")
# # print(f"最佳参数: {best_overall_params}")


###################################

# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib import cm, colors
# from skopt import gp_minimize
# from skopt.space import Real
# from collections import defaultdict
# from _1_reading_ckpt import YourTransformerModel
# from graph import compute_adjacency_matrix
# from graph_imp_sim import generate_adjacency_matrices
# from tools import save_to_txt

# # 第一个文件生成的邻接矩阵
# file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/calculate/group_all_aline.txt"
# adj_matrix_1 = compute_adjacency_matrix(file_path)

# # 定义相似度计算函数
# def similarity_score_matrix(A, B):
#     # 确保 A 和 B 是 numpy 数组
#     if isinstance(A, torch.Tensor):
#         A = A.cpu().detach().numpy()
#     if isinstance(B, torch.Tensor):
#         B = B.cpu().detach().numpy()
    
#     n = A.shape[0]  # 假设 A 和 B 都是 n x m 的矩阵
#     sim_AB = 0
#     for i in range(n):
#         cosine_line = cosine_similarity(A[i], B[i])
#         cosine_row = cosine_similarity(A[:, i], B[:, i])
#         sim_AB += cosine_line + cosine_row
#     sim_AB = sim_AB / (2 * n)
#     return sim_AB

# def cosine_similarity(u, v):
#     dot_product = np.dot(u, v)
#     norm_u = np.linalg.norm(u)
#     norm_v = np.linalg.norm(v)
#     if norm_u == 0 or norm_v == 0:
#         return 0
#     return dot_product / (norm_u * norm_v)

# def plot_adjacency_matrix_graph(adj_matrix, combination_name, iteration):
#     G = nx.Graph()
#     n = adj_matrix.shape[0]
    
#     # 创建边并设置权重
#     for i in range(n):
#         for j in range(i + 1, n):
#             weight = adj_matrix[i, j]
#             if weight > 0:  # 仅绘制有权重的边
#                 G.add_edge(i, j, weight=weight)

#     # 设置布局为圆形
#     pos = nx.circular_layout(G)

#     # 获取边的权重
#     edges = G.edges(data=True)
#     weights = [attr['weight'] for _, _, attr in edges]

#     # 使用 PowerNorm 设置非线性颜色条
#     norm = colors.PowerNorm(gamma=1, vmin=min(weights), vmax=max(weights))
#     edge_colors = cm.Blues(norm(weights))

#     # 绘制图像并保存
#     fig, ax = plt.subplots(figsize=(8, 8))
#     nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
#     nx.draw_networkx_labels(G, pos, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)
    
#     # 添加颜色条到图的轴
#     sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label="Edge Weight (Adjacency Matrix Value)")

#     # 保存图像
#     filename = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix/figure/{combination_name}_iter_{iteration}.png"
#     plt.savefig(filename, format="PNG")
#     plt.close(fig)
#     print(f"Saved adjacency graph for {combination_name} (Iteration {iteration}) at: {filename}")


# # 多启动蒙特卡洛优化函数
# def multi_start_monte_carlo_search(objective_func, bounds, num_starts=10, num_iterations=50):
#     best_score = -np.inf
#     best_params = None

#     for _ in range(num_starts):
#         # 随机初始化参数
#         params = [np.random.uniform(low, high) for low, high in bounds]

#         for _ in range(num_iterations):
#             # 计算相似度得分
#             score = objective_func(params)
#             if score > best_score:
#                 best_score = score
#                 best_params = params

#             # 随机调整参数（可以加入更复杂的局部优化策略）
#             params = [np.clip(param + np.random.normal(scale=0.1), low, high)
#                       for param, (low, high) in zip(params, bounds)]

#     return best_params, best_score

# # 搜索空间
# bounds = [(-1, 1), (-1, 1), (-1, 1)]  # weight_mean, weight_var, weight_similarity

# # 目标函数
# def objective(params):
#     weight_mean, weight_var, weight_similarity = params
#     combined_adj_matrices = generate_adjacency_matrices(
#         model,
#         weight_mean=weight_mean,
#         weight_var=weight_var,
#         weight_similarity=weight_similarity,
#         importance_mean_key=importance_mean_key,
#         importance_var_key=importance_var_key,
#         similarity_key=sim_key
#     )
#     return similarity_score_matrix(torch.tensor(adj_matrix_1), combined_adj_matrices)

# # 初始化模型
# model = YourTransformerModel(num_heads=12, dim=768)
# model.load_pretrained_qkv_weights()

# # 定义五种绑定组合和五种相似性方式
# importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
# similarity_keys = [
#     'K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix',
#     'KxQ_similarity_matrix', 'KxQxV_similarity_matrix'
# ]

# # 记录最佳结果
# results = {}
# best_overall_score = -1e6
# best_overall_combination = None
# best_overall_params = None

# # 遍历所有 25 种组合并进行多启动蒙特卡洛优化
# for imp_key in importance_keys:
#     importance_mean_key = f"{imp_key}_mean"
#     importance_var_key = f"{imp_key}_var"
    
#     for sim_key in similarity_keys:
#         combination_name = f"{importance_mean_key}_{importance_var_key}_{sim_key}"

#         # 多启动蒙特卡洛搜索最佳权重组合
#         best_params, best_score = multi_start_monte_carlo_search(objective, bounds)

#         results[combination_name] = {
#             "best_params": best_params,
#             "best_score": best_score
#         }

#         # 检查是否为当前最高分数
#         if best_score > best_overall_score:
#             best_overall_score = best_score
#             best_overall_combination = combination_name
#             best_overall_params = best_params

#         # 打印结果
#         print(f"{combination_name} 的最佳参数组合:", best_params)
#         print(f"{combination_name} 的最高相似度分数: {best_score}")

# # 打印总体最佳结果
# print("\n总体最高相似度分数的组合:")
# print(f"组合: {best_overall_combination}")
# print(f"最高相似度分数: {best_overall_score}")
# print(f"最佳参数: {best_overall_params}")

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
def similarity_score_matrix(A, B):
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    if isinstance(B, torch.Tensor):
        B = B.cpu().detach().numpy()
    
    n = A.shape[0]
    sim_AB = 0
    for i in range(n):
        cosine_line = cosine_similarity(A[i], B[i])
        cosine_row = cosine_similarity(A[:, i], B[:, i])
        sim_AB += cosine_line + cosine_row
    sim_AB /= (2 * n)
    return sim_AB

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot_product / (norm_u * norm_v) if norm_u != 0 and norm_v != 0 else 0

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

    # 设置非线性规范，调整gamma值控制非线性程度
    gamma_value = 4  # 根据需要调整gamma值
    norm = colors.PowerNorm(gamma=gamma_value, vmin=min(weights), vmax=max(weights))
    edge_colors = cm.Blues(norm(weights))

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

    # 设置非线性色条
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
plot_adjacency_matrix_graph(adj_matrix_1, "adj_matrix_1", iteration="initial", output_dir="/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_0/figure")

# 定义五种绑定组合和五种相似性方式
importance_keys = ['K', 'Q', 'V', 'KxQ', 'KxQxV']
similarity_keys = [
    'K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix',
    'KxQ_similarity_matrix', 'KxQxV_similarity_matrix'
]

# 设置保存文件的基础目录
output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_0/files'
figure_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix_0/figure'

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
