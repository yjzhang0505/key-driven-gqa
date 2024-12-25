import torch
import os
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _4_ordering import load_stats_from_model, load_singular_values_from_model, load_nuclear_norm_from_model
from _3_similarity import calculate_similarity
from tools import save_to_txt
import argparse


def group_heads_by_importance_and_similarity(model, importance_matrix, group_sizes, similarity_matrix):
    """
    按照重要性对头进行不均匀分组，优先选择最重要的头并与最相似的头进行配对。
    每次配对后失活已分组的头部，行和列同时失活，且避免与自己分组。
    每组的头部数量根据 group_sizes 来分配。
    """
    # group_sizes = [1, 1, 1, 1, 4, 4]
    num_heads = model.num_heads
    grouped_heads = []
    
    # 标记哪些头部是活动的
    active_heads = np.ones(num_heads, dtype=bool)  # 每个头部是否活跃

    # 按重要性排序，重要性高的头排前面
    importance_matrix = process_importance_matrix(importance_matrix)
    sorted_heads_by_importance = torch.argsort(importance_matrix, descending=True).tolist()
    print(sorted_heads_by_importance)

    # 在加载相似性矩阵后，将对角线设置为负无穷，避免与自己分组
    # similarity_matrix = similarity_matrix.clone()  # 保证不修改原矩阵
    for i in range(num_heads):
        similarity_matrix[i, i] = -float('inf')

    # 遍历每个组的大小
    group_idx = 0
    while group_idx < len(group_sizes):
        group_size = group_sizes[group_idx]  # 当前组的目标大小
        group = []

        # 选择最重要的未分组头，并加入当前组
        while len(group) < group_size:
            # 选择最重要的未分组头
            head_idx = sorted_heads_by_importance.pop(0)
            
            # 如果该头已经失活，跳过它
            if not active_heads[head_idx]:
                continue
            
            # 将当前头添加到组内
            group.append(head_idx)

            # 找到与当前组内头相似度之和最大的新头
            while len(group) < group_size:
                max_similarity_sum = -float('inf')
                best_head_idx = -1

                # 遍历所有活动的头，计算每个头与当前组内所有头的相似度之和
                for candidate_idx in range(num_heads):
                    if active_heads[candidate_idx] and candidate_idx not in group:
                        # 计算与当前组内所有头的相似度之和
                        similarity_sum = sum([similarity_matrix[candidate_idx, h] for h in group])
                        
                        if similarity_sum > max_similarity_sum:
                            max_similarity_sum = similarity_sum
                            best_head_idx = candidate_idx
                
                # 将相似度和最大的头加入组内
                if best_head_idx != -1:
                    group.append(best_head_idx)

                # 将这些头失活
                for idx in group:
                    active_heads[idx] = False

            # 在相似性矩阵中同时失活这些头的行和列
            for idx in group:
                similarity_matrix[idx, :] = 0  # 清空该行
                similarity_matrix[:, idx] = 0  # 清空该列
            
        grouped_heads.append(group)
        group_idx += 1  # 切换到下一个组

    return grouped_heads



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
    importance_matrix = load_stats_from_model(model, layer_idx).get(importance_type)
    print(importance_matrix)
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

    ###################
    importance_matrix = load_nuclear_norm_from_model(model, layer_idx).get(importance_type)   
    # importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
    ##################

    print(importance_matrix)

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)
    # similarity_matrix = layer_similarity.get('K_similarity_matrix')

    # if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
    #     similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 调用分组函数
    grouped_heads = group_heads_by_importance_and_similarity(model, importance_matrix, group_sizes, similarity_matrix)
    return grouped_heads


parser = argparse.ArgumentParser(description='put in filepath.')
parser.add_argument('--group', type=str, help='222222')
args = parser.parse_args()

# 提取--group后的数字并转为group_sizes
# args.group = "66"
group_str = args.group
group_sizes = []

# 遍历group_str，按字符顺序添加数字
for char in group_str:
    group_sizes.append(int(char))

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
            grouped_heads = group_heads_singular(model, similarity_key, importance_key, group_sizes, layer_idx)
            group_schemes[layer_idx] = grouped_heads

        # 格式化输出的字符串
        output_str = ""
        for layer_idx, grouped_heads in group_schemes.items():
            output_str += f"{grouped_heads},\n"

        # 创建输出文件夹路径（使用组合名称）
        output_dir = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/group_by_similarity"
        os.makedirs(output_dir, exist_ok=True)

        # 定义输出文件路径
        # output_path = os.path.join(output_dir, f'group_{args.group}.txt')
        output_path = os.path.join(output_dir, 'euclidean_nucnorm_66.txt')

        # 保存到文件
        save_to_txt(output_path, output_str)

        print(f"Saved group scheme for {similarity_key} and {importance_key} to {output_path}")