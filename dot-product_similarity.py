import torch
import os
import pandas as pd

class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def load_pretrained_qkv_weights(self, state_dict, block_idx):
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

        # 初始化字典用于保存头部之间的相似性
        similarity_matrices = {
            'K_similarity': torch.zeros(self.num_heads, self.num_heads),
            'Q_similarity': torch.zeros(self.num_heads, self.num_heads),
            'V_similarity': torch.zeros(self.num_heads, self.num_heads),
            'KxQ_similarity': torch.zeros(self.num_heads, self.num_heads),
            'KxQxV_similarity': torch.zeros(self.num_heads, self.num_heads)
        }

        # 计算每个头部之间的相似性矩阵
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                # 计算 K、Q、V 之间的点积相似性
                similarity_matrices['K_similarity'][i, j] = torch.matmul(k_weight_heads[i], k_weight_heads[j].transpose(-2, -1)).mean().item()
                similarity_matrices['Q_similarity'][i, j] = torch.matmul(q_weight_heads[i], q_weight_heads[j].transpose(-2, -1)).mean().item()
                similarity_matrices['V_similarity'][i, j] = torch.matmul(v_weight_heads[i], v_weight_heads[j].transpose(-2, -1)).mean().item()

                # 计算 Q * K^T 的相似性
                kq = torch.matmul(k_weight_heads[i], q_weight_heads[j].transpose(-2, -1))
                similarity_matrices['KxQ_similarity'][i, j] = kq.mean().item()

                # 计算 Q * K^T * V 的相似性
                kqv = torch.matmul(kq, v_weight_heads[j])
                similarity_matrices['KxQxV_similarity'][i, j] = kqv.mean().item()

        # 对每个相似性矩阵进行归一化 (最小值-最大值归一化)
        for key, matrix in similarity_matrices.items():
            min_val = matrix.min()
            max_val = matrix.max()
            similarity_matrices[key] = (matrix - min_val) / (max_val - min_val)

        # 将结果保存到 Excel 文件
        output_file = f"./output/similarity_matrices_normalized.xlsx"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 创建一个 pandas ExcelWriter 对象
        with pd.ExcelWriter(output_file) as writer:
            for key, matrix in similarity_matrices.items():
                # 将每个矩阵转换为 DataFrame
                df = pd.DataFrame(matrix.numpy())
                # 写入 Excel，标题为相似性矩阵名称
                df.to_excel(writer, sheet_name=key, index=False, header=False)

        print(f"归一化相似性矩阵已保存到 {output_file}")

        # 将权重赋值到模型的 K, Q, V 层
        self.k.weight.data.copy_(k_weight)
        self.q.weight.data.copy_(q_weight)
        self.v.weight.data.copy_(v_weight)

# 示例：加载模型并使用权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载预训练的 checkpoint
        checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/share/best.pth'
        checkpoint = torch.load(checkpoint_path)

        # 加载预训练权重到模型的某个 block (例如第0层)
        model.load_pretrained_qkv_weights(checkpoint, block_idx=0)

    return model

# 调用模型
model = vit_small_patch16_224(pretrained=True)
