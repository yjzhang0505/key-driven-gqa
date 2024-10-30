import torch
import os
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _4_ordering import load_stats_from_model
from _3_similarity import calculate_similarity
from tools import save_to_txt

def combine_matrices(importance_mean, importance_var, similarity_matrix, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4):
    """
    将加权后的 mean 和 var 重要性矩阵与相似性矩阵加权相加，生成融合后的矩阵。
    """
    combined_matrix = weight_mean * importance_mean + weight_var * importance_var + weight_similarity * similarity_matrix
    return combined_matrix

def generate_adjacency_matrices(model, weight_mean=0.3, weight_var=0.3, weight_similarity=0.4,
                                importance_mean_key='K_mean', importance_var_key='K_var', similarity_key='K_similarity_matrix'):
    # 计算相似性和重要性矩阵
    similarity_matrices = calculate_similarity(model)  
    importance_matrix = load_stats_from_model(model)
    # layer_0_importance = importance_matrix.get('Layer_0', {})

    # 从第0层提取指定的相似性矩阵
    layer_0_similarity = similarity_matrices.get('Layer_0', {})
    similarity_matrix = layer_0_similarity.get(similarity_key)

    Output_dir = '/data/yjzhang/desktop/try/key-driven-gqa/figure/adjacent_matrix/files'
    folder_name = f"{importance_mean_key}_{importance_var_key}_{similarity_key}"
    output_dir = os.path.join(Output_dir,folder_name)
    output_similarity_path = os.path.join(output_dir,f"{similarity_key}.txt")
    output_mean_path = os.path.join(output_dir,f"{importance_mean_key}.txt")
    output_var_path = os.path.join(output_dir,f"{importance_var_key}.txt")

    save_to_txt(output_similarity_path,np.array2string(similarity_matrix.numpy()))
    # print(similarity_matrix)

    # 提取指定的importance矩阵
    importance_mean = importance_matrix.get(importance_mean_key)
    importance_var = importance_matrix.get(importance_var_key)

    save_to_txt(output_mean_path, np.array2string(importance_mean.numpy()))
    save_to_txt(output_var_path, np.array2string(importance_var.numpy()))
    # print(importance_mean)
    # print(importance_var)

    # 确保矩阵为张量，若缺失则填充为零矩阵
    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        print(f"Warning: Similarity matrix '{similarity_key}' not found or not a tensor. Using a zero matrix.")
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))
    
    if importance_mean is None or not isinstance(importance_mean, torch.Tensor):
        print(f"Warning: Importance matrix '{importance_mean_key}' not found or not a tensor. Using a zero matrix.")
        importance_mean = torch.zeros((model.num_heads, model.num_heads))
    
    if importance_var is None or not isinstance(importance_var, torch.Tensor):
        print(f"Warning: Importance matrix '{importance_var_key}' not found or not a tensor. Using a zero matrix.")
        importance_var = torch.zeros((model.num_heads, model.num_heads))

    # 生成融合后的矩阵
    combined_matrix = combine_matrices(importance_mean, importance_var, similarity_matrix, weight_mean, weight_var, weight_similarity)

    # 返回生成的融合矩阵
    return combined_matrix

