import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def cosine_similarity_matrix(self, A, B):
        """ 计算矩阵A和矩阵B之间每个头之间的余弦相似性矩阵 (12x12) """
        n = A.shape[0]
        similarity_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                # 计算头i和头j之间的余弦相似度
                cos_sim_Ai_Bj = F.cosine_similarity(A[i], B[j], dim=-1)  # 计算每个头向量的相似度
                similarity_matrix[i, j] = cos_sim_Ai_Bj.mean().item()  # 平均后得到标量，填入相似性矩阵中
                
        return similarity_matrix

    def load_pretrained_qkv_weights(self, state_dict, num_layers):
        """
        遍历每一层的 Q, K, V 权重，计算每一层的相似性矩阵
        """
        # 初始化一个字典来保存所有层的相似性矩阵
        all_similarity_matrices = {}

        for block_idx in range(num_layers):
            # 加载 Q, K, V 的权重
            qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']

            # 分割 Q、K、V 权重
            qkv_dim = qkv_weight.shape[0] // 3
            q_weight = qkv_weight[0:qkv_dim]  # Q矩阵部分
            k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]  # K矩阵部分
            v_weight = qkv_weight[2 * qkv_dim:]  # V矩阵部分

            # 将 Q、K、V 重塑为 (num_heads, dim_per_head, dim) 形状
            dim_per_head = self.dim // self.num_heads
            q_weight_heads = q_weight.view(self.num_heads, dim_per_head, self.dim)
            k_weight_heads = k_weight.view(self.num_heads, dim_per_head, self.dim)
            v_weight_heads = v_weight.view(self.num_heads, dim_per_head, self.dim)

            # 初始化字典用于保存头部之间的相似性矩阵
            similarity_matrices = {
                'K_similarity_matrix': None,
                'Q_similarity_matrix': None,
                'V_similarity_matrix': None,
                'KxQ_similarity_matrix': None,
                'KxQxV_similarity_matrix': None
            }

            # 计算每个头部之间的余弦相似性矩阵 (12x12)
            similarity_matrices['K_similarity_matrix'] = self.cosine_similarity_matrix(k_weight_heads, k_weight_heads)
            similarity_matrices['Q_similarity_matrix'] = self.cosine_similarity_matrix(q_weight_heads, q_weight_heads)
            similarity_matrices['V_similarity_matrix'] = self.cosine_similarity_matrix(v_weight_heads, v_weight_heads)

            # 计算 Q * K^T 的相似性
            kq_heads = torch.matmul(k_weight_heads, q_weight_heads.transpose(-2, -1))
            similarity_matrices['KxQ_similarity_matrix'] = self.cosine_similarity_matrix(kq_heads, kq_heads)

            # 计算 Q * K^T * V 的相似性
            kqv_heads = torch.matmul(kq_heads, v_weight_heads)
            similarity_matrices['KxQxV_similarity_matrix'] = self.cosine_similarity_matrix(kqv_heads, kqv_heads)

            # 保存该层的相似性矩阵
            all_similarity_matrices[f'Layer_{block_idx}'] = similarity_matrices

        # 将所有层的结果保存到 Excel 文件
        output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/not_shared_similarity_matrices_all_layers.xlsx"
        # output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/similarity_matrices_all_layers.xlsx"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 使用 pandas 写入 Excel
        with pd.ExcelWriter(output_file) as writer:
            for layer, matrices in all_similarity_matrices.items():
                for key, matrix in matrices.items():
                    df = pd.DataFrame(matrix.numpy())
                    # 添加两行空白行（全为 NaN）
                    empty_rows = pd.DataFrame(np.nan, index=range(2), columns=df.columns)
                    df_with_empty = pd.concat([df, empty_rows], ignore_index=True)
                    df_with_empty.to_excel(writer, sheet_name=f"{layer}_{key}", index=False, header=False)

        print(f"所有层的相似性矩阵已保存到 {output_file}")

        # 将最后一个 block 的权重赋值到模型的 K, Q, V 层
        self.k.weight.data.copy_(k_weight)
        self.q.weight.data.copy_(q_weight)
        self.v.weight.data.copy_(v_weight)

# 示例：加载模型并使用权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载预训练的 checkpoint
        # checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/share/best.pth'
        checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/mhsa/config/best.pth'
        checkpoint = torch.load(checkpoint_path)

        # 加载每层的预训练权重并计算相似性矩阵
        model.load_pretrained_qkv_weights(checkpoint, num_layers=12)

    return model

# 调用模型
model = vit_small_patch16_224(pretrained=True)
