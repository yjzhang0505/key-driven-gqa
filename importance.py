import torch
import timm

class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        self.k = torch.nn.Linear(dim, dim)

    def load_pretrained_k_weights(self, state_dict, block_idx):
        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']

        # 提取 K 权重
        qkv_dim = qkv_weight.shape[0] // 3
        k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]  # 提取 K 矩阵部分

        # 将 K 权重视为 (num_heads, dim_per_head, dim) 形状
        dim_per_head = self.dim // self.num_heads
        k_weight_heads = k_weight.view(self.num_heads, dim_per_head, self.dim)

        # 对每个头的 K 权重计算 L2 范数
        for i in range(self.num_heads):
            k_l2_norm = torch.norm(k_weight_heads[i], p=2)  # 计算每个头的 L2 范数
            print(f"第 {i+1} 个头的 K 权重 L2 范数: {k_l2_norm.item()}")

        # 将权重赋值到模型的 K 层
        self.k.weight.data.copy_(k_weight)
        self.k.bias.data.copy_(qkv_bias[qkv_dim: 2 * qkv_dim])

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
        checkpoint_path = '/data/yjzhang/desktop/try/local_checkpoint/finetuned_vit_base_patch16_224.pth'
        checkpoint = torch.load(checkpoint_path)

        # 加载预训练权重到模型的某个 block (例如第0层)
        model.load_pretrained_k_weights(checkpoint, block_idx=0)

    return model

# 调用模型
model = vit_small_patch16_224(pretrained=True)
# import torch

# # 加载 .pth 文件
# checkpoint_path = '/data/yjzhang/desktop/try/local_checkpoint/finetuned_vit_base_patch16_224.pth'
# checkpoint = torch.load(checkpoint_path)

# # 提取 state_dict
# state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# # 假设 attention 层的权重名称是 'blocks.0.attn.qkv.weight'
# # 这通常是多头注意力的 QKV 权重的命名规则
# qkv_weight = state_dict['blocks.0.attn.qkv.weight']

# # 获取 QKV 权重的形状
# qkv_shape = qkv_weight.shape
# print(f"QKV 权重的形状: {qkv_shape}")

# # QKV 权重通常是形状 [3 * embed_dim, embed_dim]
# embed_dim = qkv_shape[1]  # 嵌入维度
# qkv_dim = qkv_shape[0]    # 3 * embed_dim（QKV 合并）

# 计算头的数量
# num_heads = qkv_dim // (3 * embed_dim)
# print(f"多头注意力中的头数: {num_heads}")
