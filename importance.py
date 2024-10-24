import torch
import os
class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def load_pretrained_qkv_weights(self, state_dict, block_idx, stats):
        """
        从预训练的 state_dict 中加载第 block_idx 层的 Q, K, V 权重。
        并计算每个头的 Q、K、V 权重及其组合 (K*Q 和 K*Q*V) 的均值和方差。
        """
        # 从 state_dict 提取 QKV 权重
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']

        # 提取 Q, K, V 权重
        qkv_dim = qkv_weight.shape[0] // 3
        q_weight = qkv_weight[0:qkv_dim]
        k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]
        v_weight = qkv_weight[2 * qkv_dim:]

        # 将 Q、K、V 权重 reshape 为 (num_heads, dim_per_head, dim)
        dim_per_head = self.dim // self.num_heads
        q_weight_heads = q_weight.view(self.num_heads, dim_per_head, self.dim)
        k_weight_heads = k_weight.view(self.num_heads, dim_per_head, self.dim)
        v_weight_heads = v_weight.view(self.num_heads, dim_per_head, self.dim)

        # 计算每个头的 Q、K、V 权重以及 K * Q 和 K * Q * V 的均值和方差
        for i in range(self.num_heads):
            k_mean, k_var = k_weight_heads[i].mean().item(), k_weight_heads[i].var().item()
            q_mean, q_var = q_weight_heads[i].mean().item(), q_weight_heads[i].var().item()
            v_mean, v_var = v_weight_heads[i].mean().item(), v_weight_heads[i].var().item()

            # 计算 K * Q
            kq = torch.matmul(k_weight_heads[i], q_weight_heads[i].transpose(-2, -1))
            kq_mean, kq_var = kq.mean().item(), kq.var().item()

            # 计算 K * Q * V
            kqv = torch.matmul(kq, v_weight_heads[i])
            kqv_mean, kqv_var = kqv.mean().item(), kqv.var().item()

            # 保存每个头的均值和方差
            stats['K'].append([k_mean, k_var])
            stats['Q'].append([q_mean, q_var])
            stats['V'].append([v_mean, v_var])
            stats['KxQ'].append([kq_mean, kq_var])
            stats['KxQxV'].append([kqv_mean, kqv_var])

        # 将提取的 Q、K、V 权重赋值回模型中
        self.k.weight.data.copy_(k_weight)
        self.q.weight.data.copy_(q_weight)
        self.v.weight.data.copy_(v_weight)

# 示例：加载模型并使用预训练权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载预训练的 checkpoint
        checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/mhsa/config/best.pth'
        checkpoint = torch.load(checkpoint_path)

        # 记录所有层的均值和方差
        all_stats = {'K': [], 'Q': [], 'V': [], 'KxQ': [], 'KxQxV': []}

        # 加载每一层的预训练权重
        for block_idx in range(12):
            print(f"Processing block {block_idx}")
            model.load_pretrained_qkv_weights(checkpoint, block_idx, all_stats)

        # 将每层的12个头的均值和方差结果保存到 TXT 文件
        output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            for layer in range(12):
                f.write(f"Layer {layer}:\n")
                for head in range(12):
                    # 输出每个头的所有均值和方差，逗号分隔
                    f.write(f"Head {head}: ")
                    f.write(f"K_mean: {all_stats['K'][layer * 12 + head][0]}, K_var: {all_stats['K'][layer * 12 + head][1]}, ")
                    f.write(f"Q_mean: {all_stats['Q'][layer * 12 + head][0]}, Q_var: {all_stats['Q'][layer * 12 + head][1]}, ")
                    f.write(f"V_mean: {all_stats['V'][layer * 12 + head][0]}, V_var: {all_stats['V'][layer * 12 + head][1]}, ")
                    f.write(f"KxQ_mean: {all_stats['KxQ'][layer * 12 + head][0]}, KxQ_var: {all_stats['KxQ'][layer * 12 + head][1]}, ")
                    f.write(f"KxQxV_mean: {all_stats['KxQxV'][layer * 12 + head][0]}, KxQxV_var: {all_stats['KxQxV'][layer * 12 + head][1]}\n")
                # 每层之间空一行
                f.write("\n")

        print(f"所有层的均值和方差已保存到 {output_file}")

    return model

# 调用模型
model = vit_small_patch16_224(pretrained=True)
