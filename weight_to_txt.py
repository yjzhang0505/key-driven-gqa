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
        """
        从预训练的 state_dict 中加载第 block_idx 层的 Q, K, V 权重。
        并将 Q, K, V 权重分别保存在三个 txt 文件中。
        """
        # 从 state_dict 提取 QKV 权重
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']

        # 提取 Q, K, V 权重
        qkv_dim = qkv_weight.shape[0] // 3
        q_weight = qkv_weight[0:qkv_dim]
        k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]
        v_weight = qkv_weight[2 * qkv_dim:]

        # 将权重保存到三个不同的txt文件
        q_file = f"/data/yjzhang/desktop/try/key-driven-gqa/data/finetuned/q_weights.txt"
        k_file = f"/data/yjzhang/desktop/try/key-driven-gqa/data/finetuned/k_weights.txt"
        v_file = f"/data/yjzhang/desktop/try/key-driven-gqa/data/finetuned/v_weights.txt"

        os.makedirs(os.path.dirname(q_file), exist_ok=True)

        # 保存 Q 权重
        with open(q_file, 'w') as f:
            for val in q_weight.flatten().tolist():
                f.write(f"{val}\n")
        
        # 保存 K 权重
        with open(k_file, 'w') as f:
            for val in k_weight.flatten().tolist():
                f.write(f"{val}\n")

        # 保存 V 权重
        with open(v_file, 'w') as f:
            for val in v_weight.flatten().tolist():
                f.write(f"{val}\n")

        print(f"Q, K, V 权重已分别保存到 {q_file}, {k_file}, {v_file}")

        # 将提取的 Q、K、V 权重赋值回模型中
        self.k.weight.data.copy_(k_weight)
        self.q.weight.data.copy_(q_weight)
        self.v.weight.data.copy_(v_weight)

    def load_qkv_weights_from_files(self, q_file, k_file, v_file):
        """
        从保存的 txt 文件中重新加载 Q, K, V 权重，并恢复到模型的层中。
        """
        # 读取 Q 权重
        with open(q_file, 'r') as f:
            q_weight_flat = [float(line.strip()) for line in f]
        q_weight = torch.tensor(q_weight_flat).view(self.num_heads * self.dim // self.num_heads, self.dim)

        # 读取 K 权重
        with open(k_file, 'r') as f:
            k_weight_flat = [float(line.strip()) for line in f]
        k_weight = torch.tensor(k_weight_flat).view(self.num_heads * self.dim // self.num_heads, self.dim)

        # 读取 V 权重
        with open(v_file, 'r') as f:
            v_weight_flat = [float(line.strip()) for line in f]
        v_weight = torch.tensor(v_weight_flat).view(self.num_heads * self.dim // self.num_heads, self.dim)

        # 将权重加载回模型
        self.q.weight.data.copy_(q_weight)
        self.k.weight.data.copy_(k_weight)
        self.v.weight.data.copy_(v_weight)

        print("Q, K, V 权重已从文件恢复")

def assign_check(tensor, new_tensor):
    """
    Helper function to check and assign weights
    """
    assert tensor.shape == new_tensor.shape, f"Shape mismatch: {tensor.shape} vs {new_tensor.shape}"
    tensor.data.copy_(new_tensor)
    return tensor

# 示例：加载模型并保存 QKV 权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载预训练的 checkpoint
        checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/mhsa/config/best.pth'
        checkpoint = torch.load(checkpoint_path)

        # 加载预训练权重到模型的第 0 层 block，并保存到文件
        model.load_pretrained_qkv_weights(checkpoint, block_idx=0)

    return model

# 调用模型并保存 Q, K, V 权重
model = vit_small_patch16_224(pretrained=True)

# # 恢复 Q, K, V 权重的示例调用
# model.load_qkv_weights_from_files(
#     q_file='/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/q_weights.txt',
#     k_file='/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/k_weights.txt',
#     v_file='/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/v_weights.txt'
# )
