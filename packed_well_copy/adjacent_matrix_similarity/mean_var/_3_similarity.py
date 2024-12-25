
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _3_0_similarity import cosine_similarity_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def do_PCA(X, n):
    # 1. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 初始化PCA，指定要降到n维
    pca = PCA(n_components=n)

    # 3. 拟合PCA模型并转换数据
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_sum = sum(pca.explained_variance_ratio_[:n])

    # print(f'前 {n} 个主成分的方差解释比例之和: {explained_variance_sum}')

    return X_pca

def calculate_similarity(model):
    """
    计算 Q、K、V 之间的相似性矩阵，并保存结果
    """
    all_similarity_matrices = {}

    for block_idx in range(model.num_layers):
        # q_weight = model.q_layers[block_idx].weight.data
        # k_weight = model.k_layers[block_idx].weight.data
        v_weight = model.v_layers[block_idx].weight.data

        # 将 Q、K、V 权重 reshape 为 (num_heads, dim_per_head, dim) 形状
        dim_per_head = model.dim // model.num_heads
        # q_weight_heads = q_weight.view(model.num_heads, dim_per_head, model.dim)
        # k_weight_heads = k_weight.view(model.num_heads, dim_per_head, model.dim)
        v_weight_heads = v_weight.view(model.dim, -1)

        n = 100
        v_weight_heads = do_PCA(v_weight_heads, n)
        v_weight_heads = v_weight.view(model.num_heads, -1)
        

        similarity_matrices = {
            'K_cosine': 0,
            'Q_cosine': 0,
            'V_cosine': cosine_similarity_matrix(v_weight_heads),
            'KxQ_cosine': 0,
            'KxQxV_cosine': 0
            
        }

        all_similarity_matrices[f'Layer_{block_idx}'] = similarity_matrices
        # print("Available similarity matrices keys:", similarity_matrices.keys())
        # print("K_similarity_matrix:", similarity_matrices.get('K_similarity_matrix'))


    return all_similarity_matrices


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
