import torch
import os
from _1_reading_ckpt import YourTransformerModel

def calculate_stats(model):
    """
    计算每层每个头的 Q、K、V 权重的均值和方差，以及 K * Q 和 K * Q * V 的均值和方差
    """
    all_stats = {'K': [], 'Q': [], 'V': [], 'KxQ': [], 'KxQxV': []}

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

        # 计算每个头的 Q、K、V 权重以及 K * Q 和 K * Q * V 的均值和方差
        for i in range(model.num_heads):
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
            all_stats['K'].append([k_mean, k_var])
            all_stats['Q'].append([q_mean, q_var])
            all_stats['V'].append([v_mean, v_var])
            all_stats['KxQ'].append([kq_mean, kq_var])
            all_stats['KxQxV'].append([kqv_mean, kqv_var])

    return all_stats

# 示例：加载模型并使用预训练权重
def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
    model = YourTransformerModel(num_heads=12, dim=768)

    if pretrained:
        # 加载 Q, K, V 权重到模型中
        model.load_pretrained_qkv_weights( )

        # 调用统计函数，计算权重参数的分布情况
        all_stats = calculate_stats(model)

        # 将统计结果保存到文件
        output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/_mean_var.txt"
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

# 调用模型并加载预训练权重
# model = vit_small_patch16_224(pretrained=True)
