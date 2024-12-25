# import torch
# import os
# import numpy as np
# from _1_reading_ckpt import YourTransformerModel
# from _4_ordering import load_stats_from_model, load_singular_values_from_model
# from _3_similarity import calculate_similarity
# from tools import save_to_txt
# # import argparse
# # from sklearn.cluster import SpectralClustering


# import torch

# def group_heads_by_importance_and_similarity(group_sizes, similarity_matrix):
#     """
#     贪心算法将头按照相似性分组，使每组内的相似性和最大化。
    
#     参数:
#     - similarity_matrix: torch.Tensor，相似性矩阵，形状为 (num_heads, num_heads)。
#     - group_sizes: list，每组头的数量。

#     返回:
#     - grouped_heads: list，分组结果，每个元素是一个组，包含头的索引。
#     """
    
#     num_heads = similarity_matrix.size(0)
#     grouped_heads = []  # 存储分组结果

#     # 克隆相似性矩阵，避免修改原矩阵
#     similarity_matrix = similarity_matrix.clone()
#     for i in range(num_heads):
#         similarity_matrix[i, i] = -float('inf')  # 避免头与自己分组

#     # 记录每个头的状态，True 表示未分组，False 表示已分组
#     active_heads = [True] * num_heads

#     # 遍历每个组的大小
#     for group_size in group_sizes:
#         group = []  # 当前组的头索引

#         # 贪心选择头，直到当前组的大小满足要求
#         while len(group) < group_size:
#             # 检查是否还有足够的活跃头
#             if sum(active_heads) < group_size - len(group):
#                 raise ValueError("组大小超过了剩余活跃头的数量，请调整 group_sizes 或检查代码逻辑。")
            
#             if not group:  # 如果当前组为空，选择相似性最大的一对头
#                 # 筛选活跃头
#                 active_indices = torch.tensor(active_heads, dtype=torch.bool)
#                 active_similarity = similarity_matrix[active_indices][:, active_indices]
                
#                 # 找到相似性矩阵中的最大值
#                 max_sim_value, indices = torch.topk(active_similarity.view(-1), 1)
#                 head1, head2 = divmod(indices.item(), active_similarity.size(0))  # 转换为活跃头索引
                
#                 # 获取实际头的索引
#                 head1 = torch.arange(num_heads)[active_indices][head1].item()
#                 head2 = torch.arange(num_heads)[active_indices][head2].item()

#                 # 将头加入组并失活
#                 group.append(head1)
#                 group.append(head2)
#                 active_heads[head1] = False
#                 active_heads[head2] = False

#                 # 失活这两个头的行和列
#                 # similarity_matrix[head1, :] = -float('inf')
#                 # similarity_matrix[:, head1] = -float('inf')
#                 # similarity_matrix[head2, :] = -float('inf')
#                 # similarity_matrix[:, head2] = -float('inf')
#             else:  # 如果组非空，选择与组中头相似性贡献最大的头
#                 # 筛选活跃头索引
#                 active_indices = torch.tensor(active_heads, dtype=torch.bool)

#                 # 计算每个活跃头与当前组内头的相似性之和
#                 active_to_group_similarities = similarity_matrix[active_indices][:, group]
#                 current_sum = active_to_group_similarities.sum(dim=1)  # 每个活跃头的总贡献

#                 # 选择贡献最大的头
#                 max_index = torch.argmax(current_sum).item()  # 获取最大相似性贡献的索引
#                 head = torch.arange(num_heads)[active_indices][max_index].item()  # 映射回原始索引

#                 # 将选中的头加入当前组
#                 group.append(head)
#                 active_heads[head] = False  # 失活该头

#                 # 失活相似性矩阵中的行和列
#                 # similarity_matrix[head, :] = -float('inf')
#                 # similarity_matrix[:, head] = -float('inf')


#         # 当前组分配完成，加入分组列表
#         grouped_heads.append(group)

#     return grouped_heads


#     # # for i in range(num_heads):
#     # #     similarity_matrix[i, i] = -float('inf')    
#     # num_groups = 6

#     # # 使用谱聚类进行分组
#     # spectral_clustering = SpectralClustering(n_clusters=num_groups, affinity='precomputed', random_state=42)

#     # min_val = similarity_matrix.min()
#     # max_val = similarity_matrix.max()
    
#     # # 归一化公式: (x - min) / (max - min)
#     # normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    
#     # # 将对角线元素设置为 0
#     # eye = torch.eye(normalized_matrix.size(0), dtype=normalized_matrix.dtype)
#     # normalized_matrix = normalized_matrix * (1 - eye)

#     # print(similarity_matrix)

#     # # 将相似性矩阵作为输入，并进行聚类
#     # labels = spectral_clustering.fit_predict(normalized_matrix)

#     # # 按照聚类标签分组
#     # grouped_heads = [[] for _ in range(num_groups)]
#     # for head_idx, label in enumerate(labels):
#     #     grouped_heads[label].append(head_idx)

#     # return grouped_heads    



# def process_importance_matrix(importance_matrix):
#     """
#     确保重要性矩阵是一个一维向量。
#     """
#     if importance_matrix.ndimension() > 1:
#         importance_matrix = importance_matrix.view(-1)
#     return importance_matrix


# def group_heads_for_layer(model, similartiry_type, importance_type, group_sizes, layer_idx):
#     """
#     输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
#     """
#     # 获取相似性矩阵和重要性矩阵
#     similarity_matrices = calculate_similarity(model)
#     importance_matrix = load_stats_from_model(model, layer_idx).get(importance_type)
#     print(importance_matrix)
#     # importance_matrix = load_stats_from_model(model, layer_idx).get('K_var')

#     # 获取指定层的相似性矩阵
#     layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
#     similarity_matrix = layer_similarity.get(similartiry_type)
#     # similarity_matrix = layer_similarity.get('K_similarity_matrix')

#     if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
#         similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

#     # 调用分组函数
#     grouped_heads = group_heads_by_importance_and_similarity(group_sizes, similarity_matrix)
#     return grouped_heads

# def group_heads_singular(model, similartiry_type, importance_type, group_sizes, layer_idx):
#     """
#     输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
#     """
#     # 获取相似性矩阵和重要性矩阵
#     similarity_matrices = calculate_similarity(model)
#     # importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
#     # importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
#     # print(importance_matrix)
    

#     # 获取指定层的相似性矩阵
#     layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
#     similarity_matrix = layer_similarity.get(similartiry_type)
#     # similarity_matrix = layer_similarity.get('K_similarity_matrix')
#     print(similarity_matrix)
#     if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
#         similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

#     # 调用分组函数
#     grouped_heads = group_heads_by_importance_and_similarity(group_sizes, similarity_matrix)
#     return grouped_heads


# # parser = argparse.ArgumentParser(description='put in filepath.')
# # parser.add_argument('--group', type=str, help='222222')
# # args = parser.parse_args()

# # # 提取--group后的数字并转为group_sizes
# # group_str = args.group
# # group_sizes = []

# # # 遍历group_str，按字符顺序添加数字
# # for char in group_str:
# #     group_sizes.append(int(char))
# group_sizes = [6, 6]
# # group_sizes = [2, 2, 2, 2, 2, 2]

# # print(group_sizes)

# # 示例调用
# # 初始化模型
# model = YourTransformerModel(num_heads=12, dim=768)
# model.load_pretrained_qkv_weights()

# # 定义所有的相似性矩阵和重要性矩阵的组合
# similarity_keys = ['V_cosine']
# importance_keys = ['V_singular']
# # similarity_keys = ['K_cosine', 'V_cosine']
# # importance_keys = ['K_singular', 'Q_singular', 'V_singular', 'KxQ_singular']

# # 遍历相似性矩阵和重要性矩阵的组合
# for similarity_key in similarity_keys:
#     for importance_key in importance_keys:
#         # 创建存储结果的字典
#         group_schemes = {}
        
#         # 遍历 12 层
#         for layer_idx in range(12):
#             # 假设 group_heads_for_layer 是根据相似性矩阵和重要性矩阵对头部分组的函数
#             # grouped_heads = group_heads_for_layer(model, similarity_key, importance_key, group_sizes, layer_idx)
#             grouped_heads = group_heads_singular(model, similarity_key, importance_key, group_sizes, layer_idx)
#             group_schemes[layer_idx] = grouped_heads

#         # 格式化输出的字符串
#         output_str = ""
#         for layer_idx, grouped_heads in group_schemes.items():
#             output_str += f"{grouped_heads},\n"

#         # 创建输出文件夹路径（使用组合名称）
#         output_dir = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/group_by_similarity"
#         os.makedirs(output_dir, exist_ok=True)

#         # 定义输出文件路径
#         output_path = os.path.join(output_dir, f'groups=2_euclidean.txt')
#         # output_path = os.path.join(output_dir, 'group_112244.txt')

#         # 保存到文件
#         save_to_txt(output_path, output_str)

#         print(f"Saved group scheme for {similarity_key} and {importance_key} to {output_path}")

import torch
import os
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _4_ordering import load_stats_from_model, load_singular_values_from_model, load_nuclear_norm_from_model
from _3_similarity import calculate_similarity
from tools import save_to_txt
# import argparse
# from sklearn.cluster import SpectralClustering

import numpy as np

import numpy as np

# def kmeans_with_similarity_matrix(similarity_matrix, K, max_iters=100):
#     N = similarity_matrix.shape[0]  # 头的数量
#     centroids = np.random.choice(N, K, replace=False)  # 随机选择 K 个簇中心
#     prev_centroids = centroids.copy()  # 记录前一轮的簇中心
#     labels = np.zeros(N)  # 每个头的簇标签

#     for _ in range(max_iters):
#         # 步骤 2: 分配每个头到最近的簇中心
#         for i in range(N):
#             # 获取每个簇中心的距离，确保 centroids 是整数
#             distances = [similarity_matrix[i, int(cent)] for cent in centroids]  # 强制类型转换为整数
#             labels[i] = np.argmin(distances)  # 将头分配给最小距离的簇

#         # 步骤 3: 更新簇中心
#         new_centroids = []
#         for k in range(K):
#             # 获取所有被分配到簇 k 的头部
#             cluster_heads = np.where(labels == k)[0]
            
#             # 计算每个头与簇内所有其他头的距离和
#             min_distance_sum = float('inf')
#             best_head = None
            
#             for head in cluster_heads:
#                 # 计算当前头与其他所有头的距离和
#                 distance_sum = 0
#                 for other_head in cluster_heads:
#                     if head != other_head:  # 不计算与自身的距离
#                         distance_sum += similarity_matrix[head, other_head]
                
#                 # 选择距离和最小的头作为新的簇中心
#                 if distance_sum < min_distance_sum:
#                     min_distance_sum = distance_sum
#                     best_head = head
            
#             # 将选择的头作为新的簇中心
#             new_centroids.append(best_head)
        
#         # 更新簇中心
#         centroids = np.array(new_centroids)
    
#     # 返回最终的簇分配结果
#     clusters = [[] for _ in range(K)]  # 创建一个空的簇列表

#     for i in range(N):
#         clusters[int(labels[i])].append(i)  # 将头部索引添加到对应簇中

#     return clusters  # 返回每个簇的成员


import numpy as np

import numpy as np

def kmeans_with_similarity_matrix(similarity_matrix, importance_matrix, K, max_iters=20, distance_penalty=0.1):
    N = similarity_matrix.shape[0]  # 头的数量
    centroids = []
    
    # 步骤 1: 找到相似度矩阵中最大值的位置，并将其作为前两个簇中心
    # 选择最大相似度值对应的行列索引作为前两个簇中心
    max_similarity_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

    # 获取最大值对应的行列索引作为前两个簇中心
    centroids = list(max_similarity_idx)

    # 步骤 2: 从第3个簇中心到第K个簇中心，选择与当前集合内所有簇中心的距离之和最远的点
    for _ in range(2, K):  # 选择第3到第K个簇中心
        # 计算每个点到当前簇中心集合内所有簇中心的距离之和
        distance_sum = np.zeros(N)
        for i in range(N):
            # 距离之和是该点到所有簇中心的相似度的负值
            if i in centroids:  # 跳过已经在簇中心集合中的点
                continue
            distance_sum[i] = np.sum(similarity_matrix[i, centroids])

        # 选择距离之和最大的点作为下一个簇中心
        new_centroid = np.argmax(distance_sum)
        centroids.append(new_centroid)

    importance_matrix = importance_matrix.reshape(-1)
    # 步骤 1: 获取 centroids 对应的 importance_matrix 中的值
    importance_values = [importance_matrix[c] for c in centroids]

    # 步骤 2: 根据 importance_matrix 中的值对 centroids 进行排序
    sorted_centroids = [x for _, x in sorted(zip(importance_values, centroids), reverse=True)]

    centroids = sorted_centroids

    labels = np.zeros(N)  # 每个头的簇标签

    group_sizes = [4, 4, 4]  # 每个簇内的元素个数要求：簇1有2个，簇2有4个，簇3有6个

    
    labels = np.zeros(N, dtype=int)  # 每个点的簇标签
    assigned_counts = np.ones(K, dtype=int)  # 记录每个簇已经分配的元素个数

    # # 3
    # # 用来记录每个簇中已分配点的索引
    # cluster_points = [[] for _ in range(K)]
    # for i in range(K):
    #     cluster_points[i].append(centroids[i].item())  # 将头部索引添加到对应簇中

    # unassigned_points = list(set(range(N)) - set(centroids))  # 记录所有未分配的点
    # # 步骤 2: 聚类分配，控制每个簇内的元素个数
    # for k in range(K):  # 遍历每个簇
    #     # 对于每个簇，选择未分配点中与簇内点距离之和最小的点
        
    #     while assigned_counts[k] < group_sizes[k]:  # 当前簇还没有达到预定大小
    #         min_distance_sum = float('inf')  # 最小距离和初始化为无穷大
    #         best_point = -1  # 记录当前簇中距离最近的点

    #         # 遍历所有未分配的点
    #         for i in unassigned_points:
    #             # 计算点 i 到当前簇内所有已分配点的距离之和
    #             distance_sum = np.sum([similarity_matrix[i, p] for p in cluster_points[k]])

    #             # 找到距离最小的点
    #             if distance_sum < min_distance_sum:
    #                 min_distance_sum = distance_sum
    #                 best_point = i

    #         # 将最小距离的点分配给当前簇
    #         labels[best_point] = k
    #         assigned_counts[k] += 1
    #         cluster_points[k].append(best_point)

    #         best_point = int(best_point)  # 如果 best_point 是 np.int64 或其他非 int 类型
    #         unassigned_points = [int(p) for p in unassigned_points]  # 如果 unassigned_points 中的元素是其他类型

    #         # 从未分配点中移除该点
    #         unassigned_points.remove(best_point)


    # 2
    # 步骤 1: 计算每个点到各个簇中心的相似度
    distances = np.zeros((N, K))  # 用于存储每个点到每个簇中心的相似度

    for i in range(N):
        for j in range(K):
            distances[i, j] = similarity_matrix[i, centroids[j]]

    # 步骤 2: 聚类分配，控制每个簇内的元素个数
    labels = np.zeros(N, dtype=int)  # 每个点的簇标签
    assigned_counts = np.zeros(K, dtype=int)  # 记录每个簇已经分配的元素个数

    # 创建0到11的序号数组
    indices = np.arange(12)

    # 按照importance_matrix的值从大到小重新排序序号
    sorted_indices = indices[np.argsort(-importance_matrix)]  # -importance_matrix是按降序排序

    cluster_points = [[] for _ in range(K)]

    for i in sorted_indices:
        # 获取当前点到各个簇中心的相似度，并找到相似度最高的簇
        possible_centroids = np.argsort(distances[i])  # 按相似度升序排序簇中心
        for c in possible_centroids:
            if assigned_counts[c] < group_sizes[c]:  # 如果当前簇还没有达到预定大小
                labels[i] = c  # 将点分配给该簇
                cluster_points[c].append(int(i))
                assigned_counts[c] += 1  # 更新该簇已分配的元素个数
                break  # 一旦分配到一个簇，就跳出循环

    # 4
    # distances = np.zeros((N, K))  # 用于存储每个点到每个簇中心的相似度

    # for i in range(N):
    #     for j in range(K):
    #         distances[i, j] = similarity_matrix[i, centroids[j]]

    # for i in centroids:
    #     distances[centroids, :] = -np.inf  # 设置整行

    # unassigned_points = list(set(range(N)) - set(centroids))  # 记录所有未分配的点

    # cluster_points = [[] for _ in range(K)]
    # for i in range(K):
    #     cluster_points[i].append(centroids[i])
    #     labels[centroids[i]] = i
    # # distance_matrix = similarity_matrix.copy()  # 克隆相似性矩阵
    
  
    # # 循环直到所有簇都分配完
    # while 1 < 100:
    #     # 查找相似性矩阵中的最大值
    #     max_similarity_idx = np.unravel_index(np.argmax(distances), distances.shape)
    #     max_similarity_point = max_similarity_idx[0]  # 最大相似度对应的点
        
    #     if assigned_counts[max_similarity_idx[1]] < group_sizes[max_similarity_idx[1]]:
    #         labels[max_similarity_point] = max_similarity_idx[1]  # 将当前点分配给簇
    #         assigned_counts[max_similarity_idx[1]] += 1  # 更新该簇已分配的元素个数
    #         cluster_points[max_similarity_idx[1]].append(max_similarity_point)  # 将该点作为簇中心加入
        
    #         # 将已分配的点从待分配点中移除
    #         unassigned_points.remove(max_similarity_point)
            
    #         # 更新相似性矩阵，已分配的点的相似度设为负无穷，防止重复选择

    #         distances[max_similarity_point, :] = -np.inf  # 设置整行

    #     else:
    #         distances[max_similarity_idx[0], max_similarity_idx[1]] = -np.inf
            
    #     if not unassigned_points:
    #         break  # 退出循环




    # 1
    # for _ in range(max_iters):
    #     # 步骤 2: 分配每个头到最近的簇中心
    #     for i in range(N):
    #         # 获取每个簇中心的距离，确保 centroids 是整数
    #         distances = [similarity_matrix[i, int(cent)] for cent in centroids]  # 强制类型转换为整数
    #         labels[i] = np.argmin(distances)  # 将头分配给最小距离的簇

    #     # 步骤 3: 更新簇中心
    #     new_centroids = []
    #     for k in range(K):
    #         # 获取所有被分配到簇 k 的头部
    #         cluster_heads = np.where(labels == k)[0]
            
    #         # 如果该簇没有点，则跳过
    #         if len(cluster_heads) == 0:
    #             continue

    #         # 计算当前簇内所有点到各簇中心的距离和
    #         # max_cluster_intra_distance = float('-inf')
    #         # best_head = None
    #         min_intercluster_distance = float('inf')
    #         for head in cluster_heads:
    #             # 计算当前头与簇内所有其他头的距离和
    #             distance_sum = 0
    #             for other_head in cluster_heads:
    #                 if head != other_head:  # 不计算与自身的距离
    #                     distance_sum += similarity_matrix[head, other_head]

    #             # 选择与组内其他点距离和最小的头
    #             # if distance_sum < min_cluster_intra_distance:
    #             #     min_cluster_intra_distance = distance_sum
    #             #     best_head = head

    #         # 步骤 4: 选择一个与其他组簇心距离更远的点作为新的簇中心
    #         # if best_head is not None:
    #             # 计算当前簇中心与其他簇中心的距离和
                
    #             # best_candidate = None

    #         # for head in cluster_heads:
    #             intercluster_distance = 0
    #             for existing_centroid in centroids:
    #                 # 计算簇间的距离和
    #                 if existing_centroid != head:  # 不计算与自身的距离
    #                     intercluster_distance += similarity_matrix[head, int(existing_centroid)]
                
    #             a = 0.2+ 0.5*_/max_iters
    #             b = 0.8- 0.5*_/max_iters

    #             xx = a * distance_sum - b * intercluster_distance

    #             if xx < min_intercluster_distance:
    #                 min_intercluster_distance = xx
    #                 best_candidate = head
                
    #         # 选择与其他簇心距离更远的候选点
    #         new_centroids.append(best_candidate)

    #     # 更新簇中心
    #     centroids = np.array(new_centroids)



    # # 返回最终的簇分配结果
    # clusters = [[] for _ in range(K)]  # 创建一个空的簇列表

    # for i in range(N):
    #     clusters[int(labels[i])].append(i)  # 将头部索引添加到对应簇中

    clusters = cluster_points
        
    return clusters  # 返回每个簇的成员



# 假设 similarity_matrix 是你已有的欧几里得距离矩阵，K 是预定的簇数
# similarity_matrix = np.random.rand(10, 10)  # 示例，假设 10 个头
# K = 3  # 假设我们想分成 3 个簇

# 调用 K-means 聚类
# labels, centroids = kmeans_with_similarity_matrix(similarity_matrix, K)

# # 打印聚类结果
# print("头的簇分配:", labels)
# print("簇中心:", centroids)


def group_heads_by_importance_and_similarity(model, importance_matrix, group_sizes, similarity_matrix):
    """
    按照重要性对头进行不均匀分组，优先选择最重要的头并与最相似的头进行配对。
    每次配对后失活已分组的头部，行和列同时失活，且避免与自己分组。
    每组的头部数量根据 group_sizes 来分配。
    """
    # group_sizes = [1, 1, 1, 1, 4, 4]
    num_heads = model.num_heads
    grouped_heads = []
    # 在加载相似性矩阵后，将对角线设置为负无穷，避免与自己分组
    for i in range(num_heads):
        similarity_matrix[i, i] = -float('inf')

    # 初始化头的活动状态（所有头都为活动状态）
    active_heads = [True] * num_heads
    
    # 遍历每个组的大小
    group_idx = 0
    while group_idx < len(group_sizes):
        group_size = group_sizes[group_idx]  # 当前组的目标大小
        group = []

        # 选择最相似的一对头并加入当前组
        while len(group) < group_size:
            # 找到相似性矩阵中最大的相似度
            max_sim_value = np.max(similarity_matrix)  # 获取最大值
            max_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)  # 获取最大值的索引 (行, 列)
            
            # 获取最大相似度对应的行列索引
            # head1, head2 = divmod(indices.item(), num_heads)  # 转换为行列索引
            head1 = max_index[0]
            head2 = max_index[1]
            
            # 如果这两个头不是活动的，跳过
            if not active_heads[head1] or not active_heads[head2]:
                # 清空这个位置的相似性，防止重复选择
                similarity_matrix[head1, head2] = similarity_matrix[head2, head1] = -float('inf')
                continue
            
            # 将这两个头加入当前组
            group.append(head1)
            group.append(head2)
            
            # 失活这两个头
            active_heads[head1] = False
            active_heads[head2] = False
            
            # 在相似性矩阵中失活这两个头的行和列
            similarity_matrix[head1, :] = -float('inf')
            similarity_matrix[:, head1] = -float('inf')
            similarity_matrix[head2, :] = -float('inf')
            similarity_matrix[:, head2] = -float('inf')
        
        grouped_heads.append(group)
        group_idx += 1  # 切换到下一个组

    return grouped_heads

    # # for i in range(num_heads):
    # #     similarity_matrix[i, i] = -float('inf')    
    # num_groups = 6

    # # 使用谱聚类进行分组
    # spectral_clustering = SpectralClustering(n_clusters=num_groups, affinity='precomputed', random_state=42)

    # min_val = similarity_matrix.min()
    # max_val = similarity_matrix.max()
    
    # # 归一化公式: (x - min) / (max - min)
    # normalized_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    
    # # 将对角线元素设置为 0
    # eye = torch.eye(normalized_matrix.size(0), dtype=normalized_matrix.dtype)
    # normalized_matrix = normalized_matrix * (1 - eye)

    # print(similarity_matrix)

    # # 将相似性矩阵作为输入，并进行聚类
    # labels = spectral_clustering.fit_predict(normalized_matrix)

    # # 按照聚类标签分组
    # grouped_heads = [[] for _ in range(num_groups)]
    # for head_idx, label in enumerate(labels):
    #     grouped_heads[label].append(head_idx)

    # return grouped_heads    



def process_importance_matrix(importance_matrix):
    """
    确保重要性矩阵是一个一维向量。
    """
    if importance_matrix.ndimension() > 1:
        importance_matrix = importance_matrix.view(-1)
    return importance_matrix


def group_heads_for_layer(model, similartiry_type, importance_type, group_sizes, layer_idx):
    """
    输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
    """
    # 获取相似性矩阵和重要性矩阵
    similarity_matrices = calculate_similarity(model)
    # importance_matrix = load_stats_from_model(model, layer_idx).get(importance_type)
    # print(importance_matrix)
    importance_matrix=1
    print("1")
    # importance_matrix = load_stats_from_model(model, layer_idx).get('K_var')

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)
    # similarity_matrix = layer_similarity.get('K_similarity_matrix')

    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 调用分组函数
    grouped_heads = group_heads_by_importance_and_similarity(model, importance_matrix, group_sizes, similarity_matrix)
    return grouped_heads

def group_heads_singular(model, similartiry_type, importance_type, group_sizes, layer_idx):
    """
    输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
    """
    # 获取相似性矩阵和重要性矩阵
    similarity_matrices = calculate_similarity(model)
    # importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
    # importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
    # print(importance_matrix)
    importance_matrix=1
    # print(1)

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)
    # similarity_matrix = layer_similarity.get('K_similarity_matrix')

    # if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
    #     similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 调用分组函数
    grouped_heads = group_heads_by_importance_and_similarity(model, importance_matrix, group_sizes, similarity_matrix)
    return grouped_heads

def group_heads_K_means(model, similartiry_type, importance_type, group_sizes, layer_idx):
    """
    输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
    """
    # 获取相似性矩阵和重要性矩阵
    similarity_matrices = calculate_similarity(model)

    K = 3

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)

    importance_matrix = load_nuclear_norm_from_model(model, layer_idx).get(importance_type) 
    print(importance_matrix)

    importance_matrix = importance_matrix + 0.01
    
    # 将 importance_matrix 转换为 12x12 的矩阵
    importance_matrix_2d = importance_matrix.reshape(-1, 1) * importance_matrix  # 广播机制，得到一个 12x12 的矩阵

    importance_matrix_2d = importance_matrix_2d.numpy()
    importance_matrix_2d = importance_matrix_2d * 0.1 

    # 逐元素乘法：相似性矩阵和重要性矩阵的乘积
    # similarity_matrix = similarity_matrix * importance_matrix_2d


    clusters = kmeans_with_similarity_matrix(similarity_matrix, importance_matrix, K)

    # 打印聚类结果
    print("簇分配结果:")
    for i, cluster in enumerate(clusters):
        print(f"簇 {i}: {cluster}")
    return clusters

# parser = argparse.ArgumentParser(description='put in filepath.')
# parser.add_argument('--group', type=str, help='222222')
# args = parser.parse_args()

# # 提取--group后的数字并转为group_sizes
# group_str = args.group
# group_sizes = []

# # 遍历group_str，按字符顺序添加数字
# for char in group_str:
#     group_sizes.append(int(char))
group_sizes = [3, 3, 3, 3]
# group_sizes = [2, 2, 2, 2, 2, 2]

# print(group_sizes)

# 示例调用
# 初始化模型
model = YourTransformerModel(num_heads=12, dim=768)
model.load_pretrained_qkv_weights()

# 定义所有的相似性矩阵和重要性矩阵的组合
similarity_keys = ['V_cosine']
importance_keys = ['V_singular']
# similarity_keys = ['K_cosine', 'V_cosine']
# importance_keys = ['K_singular', 'Q_singular', 'V_singular', 'KxQ_singular']

# 遍历相似性矩阵和重要性矩阵的组合
for similarity_key in similarity_keys:
    for importance_key in importance_keys:
        # 创建存储结果的字典
        group_schemes = {}
        
        # 遍历 12 层
        for layer_idx in range(12):
            # 假设 group_heads_for_layer 是根据相似性矩阵和重要性矩阵对头部分组的函数
            # grouped_heads = group_heads_for_layer(model, similarity_key, importance_key, group_sizes, layer_idx)
            # grouped_heads = group_heads_singular(model, similarity_key, importance_key, group_sizes, layer_idx)
            grouped_heads = group_heads_K_means(model, similarity_key, importance_key, group_sizes, layer_idx)
            group_schemes[layer_idx] = grouped_heads

        # 格式化输出的字符串
        # print(group_schemes)
        output_str = ""
        for key, tensor in group_schemes.items():
            # 将 tensor 转换为 numpy 数组，并转换为列表
            grouped_heads = tensor
            output_str += f"{grouped_heads},\n"

        # output_str = ""
        # for layer_idx, grouped_heads in group_schemes.items():
        #     output_str += f"{grouped_heads},\n"

        # 创建输出文件夹路径（使用组合名称）
        output_dir = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/group_by_K_means"
        os.makedirs(output_dir, exist_ok=True)

        # 定义输出文件路径
        output_path = os.path.join(output_dir, f'groups=3.txt')
        # output_path = os.path.join(output_dir, 'group_112244.txt')

        # 保存到文件
        save_to_txt(output_path, output_str)

        print(f"Saved group scheme for {similarity_key} and {importance_key} to {output_path}")