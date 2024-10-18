import torch
import os

class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def load_pretrained_qkv_weights(self, state_dict, block_idx):
        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']

        # 提取 Q、K、V 权重
        qkv_dim = qkv_weight.shape[0] // 3
        q_weight = qkv_weight[0:qkv_dim]  # 提取 Q 矩阵部分
        k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]  # 提取 K 矩阵部分
        v_weight = qkv_weight[2 * qkv_dim:]  # 提取 V 矩阵部分

        # 将 Q、K、V 权重视为 (num_heads, dim_per_head, dim) 形状
        dim_per_head = self.dim // self.num_heads
        q_weight_heads = q_weight.view(self.num_heads, dim_per_head, self.dim)
        k_weight_heads = k_weight.view(self.num_heads, dim_per_head, self.dim)
        v_weight_heads = v_weight.view(self.num_heads, dim_per_head, self.dim)

        # 记录L2范数
        l2_norms = {'K_L2': [], 'Q_L2': [], 'V_L2': [], 'KxQ_L2': [], 'KxQxV_L2': []}

        # 计算每个头的 Q、K、V 权重以及 K * Q 和 K * Q * V 的 L2 范数
        for i in range(self.num_heads):
            k_l2_norm = torch.norm(k_weight_heads[i], p=2).item()
            q_l2_norm = torch.norm(q_weight_heads[i], p=2).item()
            v_l2_norm = torch.norm(v_weight_heads[i], p=2).item()

            # 计算 K * Q
            kq = torch.matmul(k_weight_heads[i], q_weight_heads[i].transpose(-2, -1))
            kq_l2_norm = torch.norm(kq, p=2).item()

            # 计算 K * Q * V
            kqv = torch.matmul(kq, v_weight_heads[i])
            kqv_l2_norm = torch.norm(kqv, p=2).item()

            # 保存到字典中
            l2_norms['K_L2'].append(k_l2_norm)
            l2_norms['Q_L2'].append(q_l2_norm)
            l2_norms['V_L2'].append(v_l2_norm)
            l2_norms['KxQ_L2'].append(kq_l2_norm)
            l2_norms['KxQxV_L2'].append(kqv_l2_norm)

        # 将结果保存到 TXT 文件中
        output_file = f"./output/arbitrary/share/l2_norms.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("K_L2: " + ",".join([str(v) for v in l2_norms['K_L2']]) + "\n")
            f.write("Q_L2: " + ",".join([str(v) for v in l2_norms['Q_L2']]) + "\n")
            f.write("V_L2: " + ",".join([str(v) for v in l2_norms['V_L2']]) + "\n")
            f.write("KxQ_L2: " + ",".join([str(v) for v in l2_norms['KxQ_L2']]) + "\n")
            f.write("KxQxV_L2: " + ",".join([str(v) for v in l2_norms['KxQxV_L2']]) + "\n")

        print(f"L2 范数已保存到 {output_file}")

        # 将权重赋值到模型的 K, Q, V 层
        self.k.weight.data.copy_(k_weight)
        self.q.weight.data.copy_(q_weight)
        self.v.weight.data.copy_(v_weight)

def assign_check(tensor, new_tensor):
    '''
    Helper function to check and assign weights
    '''
    assert tensor.shape == new_tensor.shape, f"Shape mismatch: {tensor.shape} vs {new_tensor.shape}"
    tensor.data.copy_(new_tensor)
    return tensor

# 示例：加载模型并使用权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载预训练的 checkpoint
        # /data/yjzhang/desktop/try/key-driven-gqa/output/share/best.pth!
        checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/share/best.pth'
        # checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/mhsa_2/config/best.pth'
        checkpoint = torch.load(checkpoint_path)

        # 加载预训练权重到模型的某个 block (例如第0层)
        model.load_pretrained_qkv_weights(checkpoint, block_idx=0)

    return model

# 调用模型
model = vit_small_patch16_224(pretrained=True)
