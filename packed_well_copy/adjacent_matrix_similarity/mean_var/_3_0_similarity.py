
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from _1_reading_ckpt import YourTransformerModel


# 示例：加载模型并使用预训练权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained: 
        # 加载每层的预训练权重
        model.load_pretrained_qkv_weights( )

        # 调用相似度计算函数
        all_similarity_matrices = calculate_similarity(model)
        save_similarity_matrices_to_excel(all_similarity_matrices)

    return model

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(A):
    """计算矩阵 A 和 B 之间每个头之间的余弦相似性矩阵 (n x n)，
    其中每个头的相似性是基于每个维度展平后的平均余弦相似度。
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    # if isinstance(B, torch.Tensor):
    #     B = B.cpu().detach().numpy()

    n = A.shape[0]  # 假设 A 和 B 都是 n x m 的矩阵
    similarity_matrix = np.zeros((n, n))  # 初始化 n x n 的相似性矩阵

    for i in range(n):
        for j in range(n):
            # 计算头 i 和头 j 的各个维度的展平后余弦相似度的平均
            # m = A.shape[1]  # 假设 A 和 B 的形状为 (n, m, ...)
            # dim_similarities = [cosine_similarity(A[i, d].flatten(), B[j, d].flatten()) for d in range(m)]
            similarity_matrix[i, j] = cosine_similarity(A[i], A[j])
            # similarity_matrix[i, j] = euclidean_similarity(A[i], A[j])
            # # 将各维度的相似度平均作为头 i 和头 j 的相似度
            # similarity_matrix[i, j] = sum(dim_similarities)

    # 创建一个与原矩阵相同的副本
    # similarity_matrix_no_diag = similarity_matrix.copy()

    # # 将对角线元素替换为 -inf，这样在计算最大值和最小值时会忽略对角线元素
    # np.fill_diagonal(similarity_matrix_no_diag, -np.inf)

    # # 计算非对角线元素的最大值和最小值
    # max_value = np.max(similarity_matrix_no_diag)
    # min_value = np.min(similarity_matrix_no_diag[similarity_matrix_no_diag != -np.inf])
   
    # # 归一化
    # normalized_matrix = (similarity_matrix - min_value) / (max_value - min_value)
    return similarity_matrix

def cosine_similarity(u, v):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0  # 若模长为零，返回0
    return 2 - dot_product / (norm_u * norm_v)   #越大越相似




def euclidean_similarity(u, v):
    """计算两个向量之间的欧几里得距离"""

    euclidean = np.linalg.norm(u - v) # euclidean

    return euclidean  #越小越相似