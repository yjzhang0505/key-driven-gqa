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

def cosine_similarity_matrix(A, B):
    """ 计算矩阵 A 和矩阵 B 之间每个头之间的余弦相似性矩阵 (12x12) """
    n = A.shape[0]
    similarity_matrix = torch.zeros(n, n)
        
    for i in range(n):
        for j in range(n):
            # 计算头 i 和头 j 之间的余弦相似度
            cos_sim_Ai_Bj = F.cosine_similarity(A[i], B[j], dim=-1)  # 计算每个头向量的相似度
            similarity_matrix[i, j] = cos_sim_Ai_Bj.mean().item()  # 平均后得到标量，填入相似性矩阵中

    # 归一化处理，使得相似性矩阵的均值为 1/3
    matrix_mean = similarity_matrix.mean().item()
    if matrix_mean != 0:
        similarity_matrix /= (3*matrix_mean)
        
    return similarity_matrix

def calculate_similarity(model):
    """
    计算 Q、K、V 之间的相似性矩阵，并保存结果
    """
    all_similarity_matrices = {}

    for block_idx in range(model.num_layers):
        # 从模型的 q_layers、k_layers 和 v_layers 中提取已经加载好的权重
        q_weight = model.q_layers[block_idx].weight.data
        k_weight = model.k_layers[block_idx].weight.data
        v_weight = model.v_layers[block_idx].weight.data

        # 将 Q、K、V 权重 reshape 为 (num_heads, dim_per_head, dim) 形状
        dim_per_head = model.dim // model.num_heads
        q_weight_heads = q_weight.view(model.num_heads, dim_per_head, model.dim)
        k_weight_heads = k_weight.view(model.num_heads, dim_per_head, model.dim)
        v_weight_heads = v_weight.view(model.num_heads, dim_per_head, model.dim)

        # 初始化字典用于保存头部之间的相似性矩阵
        similarity_matrices = {
            'K_similarity_matrix': None,
            'Q_similarity_matrix': None,
            'V_similarity_matrix': None,
            'KxQ_similarity_matrix': None,
            'KxQxV_similarity_matrix': None
        }

        # 计算每个头部之间的余弦相似性矩阵 (12x12)
        similarity_matrices['K_similarity_matrix'] = cosine_similarity_matrix(k_weight_heads, k_weight_heads)
        similarity_matrices['Q_similarity_matrix'] = cosine_similarity_matrix(q_weight_heads, q_weight_heads)
        similarity_matrices['V_similarity_matrix'] = cosine_similarity_matrix(v_weight_heads, v_weight_heads)

        # 计算 Q * K^T 的相似性
        kq_heads = torch.matmul(k_weight_heads, q_weight_heads.transpose(-2, -1))
        similarity_matrices['KxQ_similarity_matrix'] = cosine_similarity_matrix(kq_heads, kq_heads)

        # 计算 Q * K^T * V 的相似性
        kqv_heads = torch.matmul(kq_heads, v_weight_heads)
        similarity_matrices['KxQxV_similarity_matrix'] = cosine_similarity_matrix(kqv_heads, kqv_heads)

        # 保存该层的相似性矩阵
        all_similarity_matrices[f'Layer_{block_idx}'] = similarity_matrices

    return all_similarity_matrices


# def cosine_similarity_matrix(A, B):
#     """ 计算矩阵A和矩阵B之间每个头之间的余弦相似性矩阵 (12x12) """
#     n = A.shape[0]
#     similarity_matrix = torch.zeros(n, n)
        
#     for i in range(n):
#         for j in range(n):
#             # 计算头i和头j之间的余弦相似度
#             cos_sim_Ai_Bj = F.cosine_similarity(A[i], B[j], dim=-1)  # 计算每个头向量的相似度
#             similarity_matrix[i, j] = cos_sim_Ai_Bj.mean().item()  # 平均后得到标量，填入相似性矩阵中
                
#     return similarity_matrix

# def calculate_similarity(model):
#     """
#     计算 Q、K、V 之间的相似性矩阵，并保存结果
#     """
#     all_similarity_matrices = {}

#     for block_idx in range(model.num_layers):
#         # 从模型的 q_layers、k_layers 和 v_layers 中提取已经加载好的权重
#         q_weight = model.q_layers[block_idx].weight.data
#         k_weight = model.k_layers[block_idx].weight.data
#         v_weight = model.v_layers[block_idx].weight.data

#         # 将 Q、K、V 权重 reshape 为 (num_heads, dim_per_head, dim) 形状
#         dim_per_head = model.dim // model.num_heads
#         q_weight_heads = q_weight.view(model.num_heads, dim_per_head, model.dim)
#         k_weight_heads = k_weight.view(model.num_heads, dim_per_head, model.dim)
#         v_weight_heads = v_weight.view(model.num_heads, dim_per_head, model.dim)

#         # 初始化字典用于保存头部之间的相似性矩阵
#         similarity_matrices = {
#             'K_similarity_matrix': None,
#             'Q_similarity_matrix': None,
#             'V_similarity_matrix': None,
#             'KxQ_similarity_matrix': None,
#             'KxQxV_similarity_matrix': None
#         }

#         # 计算每个头部之间的余弦相似性矩阵 (12x12)
#         similarity_matrices['K_similarity_matrix'] = cosine_similarity_matrix(k_weight_heads, k_weight_heads)
#         similarity_matrices['Q_similarity_matrix'] = cosine_similarity_matrix(q_weight_heads, q_weight_heads)
#         similarity_matrices['V_similarity_matrix'] = cosine_similarity_matrix(v_weight_heads, v_weight_heads)

#         # 计算 Q * K^T 的相似性
#         kq_heads = torch.matmul(k_weight_heads, q_weight_heads.transpose(-2, -1))
#         similarity_matrices['KxQ_similarity_matrix'] = cosine_similarity_matrix(kq_heads, kq_heads)

#         # 计算 Q * K^T * V 的相似性
#         kqv_heads = torch.matmul(kq_heads, v_weight_heads)
#         similarity_matrices['KxQxV_similarity_matrix'] = cosine_similarity_matrix(kqv_heads, kqv_heads)

#         # 保存该层的相似性矩阵
#         all_similarity_matrices[f'Layer_{block_idx}'] = similarity_matrices

#     return all_similarity_matrices


def save_similarity_matrices_to_excel(all_similarity_matrices):
    """
    将所有层的相似性矩阵保存到 Excel 文件
    """
    output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/_similarity_matrices_all_layers.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file) as writer:
        for layer, matrices in all_similarity_matrices.items():
            for key, matrix in matrices.items():
                df = pd.DataFrame(matrix.numpy())
                # 添加两行空白行（全为 NaN）
                empty_rows = pd.DataFrame(np.nan, index=range(2), columns=df.columns)
                df_with_empty = pd.concat([df, empty_rows], ignore_index=True)
                df_with_empty.to_excel(writer, sheet_name=f"{layer}_{key}", index=False, header=False)

    print(f"所有层的相似性矩阵已保存到 {output_file}")

# 调用模型并加载预训练权重
# model = vit_small_patch16_224(pretrained=True)
