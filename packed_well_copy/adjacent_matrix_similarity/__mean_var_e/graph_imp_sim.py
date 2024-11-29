import torch
import os
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _4_ordering import load_stats_from_model, load_singular_values_from_model
from _3_similarity import calculate_similarity
from tools import save_to_txt

import numpy as np

import numpy as np
import torch

import torch

def combine_matrices(importance_mean, importance_var, similarity_matrix, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4):
    # 将 importance_mean 和 importance_var 转换为一维向量
    importance_mean = importance_mean.flatten()
    importance_var = importance_var.flatten()
    
    # 生成方阵并转换为 torch.Tensor
    mean_matrix = torch.tensor(np.outer(importance_mean, importance_mean) * 3)
    var_matrix = torch.tensor(np.outer(importance_var, importance_var) * 3)
    
    # 加权组合
    combined_matrix = (weight_mean * mean_matrix +
                       weight_var * var_matrix +
                       weight_similarity * similarity_matrix)
    
    return combined_matrix



import os
import torch
import numpy as np

def generate_adjacency_matrices(model, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4,
                                importance_mean_key='K_mean', importance_var_key='K_var', similarity_key='K_similarity_matrix'):
    # 计算相似性和重要性矩阵
    similarity_matrices = calculate_similarity(model)
    layer_index = 0
    importance_matrix = load_stats_from_model(model, layer_index)

    # 从第0层提取指定的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_index}', {})
    similarity_matrix = layer_similarity.get(similarity_key)

    # 设置输出目录
    Output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix/files'
    folder_name = f"{importance_mean_key}_{importance_var_key}_{similarity_key}"
    output_dir = os.path.join(Output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # 设置各文件路径
    output_similarity_path = os.path.join(output_dir, f"{similarity_key}.txt")
    output_mean_path = os.path.join(output_dir, f"{importance_mean_key}.txt")
    output_var_path = os.path.join(output_dir, f"{importance_var_key}.txt")

    # 确保相似性矩阵和重要性矩阵是有效的张量
    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        print(f"Warning: Similarity matrix '{similarity_key}' not found or not a tensor. Using a zero matrix.")
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 提取指定的重要性矩阵
    importance_mean = importance_matrix.get(importance_mean_key)
    importance_var = importance_matrix.get(importance_var_key)

    if importance_mean is None or not isinstance(importance_mean, torch.Tensor):
        print(f"Warning: Importance matrix '{importance_mean_key}' not found or not a tensor. Using a zero matrix.")
        importance_mean = torch.zeros((model.num_heads, model.num_heads))

    if importance_var is None or not isinstance(importance_var, torch.Tensor):
        print(f"Warning: Importance matrix '{importance_var_key}' not found or not a tensor. Using a zero matrix.")
        importance_var = torch.zeros((model.num_heads, model.num_heads))

    # 生成融合后的矩阵
    combined_matrix = combine_matrices(importance_mean, importance_var, similarity_matrix, weight_mean, weight_var, weight_similarity)

    # 归一化融合后的矩阵到0-1之间
    max_val = combined_matrix.max()
    normalized_combined_matrix = np.exp(combined_matrix - max_val)
    
    return normalized_combined_matrix


import os
import torch
import numpy as np

def combine_matrices_2(importance, similarity_matrix, weight_importance=0.6, weight_similarity=0.4):
    # 将 importance 转换为张量
    importance = torch.tensor(importance).view(-1, 1)  # 转换为列向量

    # 计算外积，生成方阵
    importance_matrix = torch.mm(importance, importance.t()) * 2  # 外积并乘以2

    # 加权组合
    combined_matrix = (weight_importance * importance_matrix +
                       weight_similarity * similarity_matrix)
    
    return combined_matrix


def generate_adjacency_matrices_2(model, weight_importance=0.6, weight_similarity=0.4,
                                singularity_key='K_singular', similarity_key='K_similarity_matrix'):
    # 计算相似性矩阵
    similarity_matrices = calculate_similarity(model)
    layer_index = 0
    importance_matrix = load_singular_values_from_model(model, layer_index)

    # 从第0层提取指定的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_index}', {})
    similarity_matrix = layer_similarity.get(similarity_key)

    # 设置输出目录
    # Output_dir = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/figure/adjacent_matrix/files'
    # folder_name = f"{singularity_key}_{similarity_key}"
    # output_dir = os.path.join(Output_dir, folder_name)
    # os.makedirs(output_dir, exist_ok=True)

    # # 设置各文件路径
    # output_similarity_path = os.path.join(output_dir, f"{similarity_key}.txt")
    # output_singular_path = os.path.join(output_dir, f"{singularity_key}.txt")

    # 确保相似性矩阵是有效的张量
    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        print(f"Warning: Similarity matrix '{similarity_key}' not found or not a tensor. Using a zero matrix.")
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 提取指定的奇异值矩阵
    importance = importance_matrix.get(singularity_key)

    if importance is None or not isinstance(importance, torch.Tensor):
        print(f"Warning: Importance matrix '{singularity_key}' not found or not a tensor. Using a zero matrix.")
        importance = torch.zeros((model.num_heads, model.num_heads))

    # 生成融合后的矩阵
    combined_matrix = combine_matrices_2(importance, similarity_matrix, weight_importance, weight_similarity)

    # 归一化融合后的矩阵到0-1之间
    min_val = combined_matrix.min()
    max_val = combined_matrix.max()
    normalized_combined_matrix = (combined_matrix - min_val) / (max_val - min_val)

    return normalized_combined_matrix
