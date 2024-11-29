import torch
import numpy as np
import os
from _1_reading_ckpt import YourTransformerModel

import torch
import numpy as np

def calculate_stats(model, layer_idx):
    """
    计算指定层的每个头的 Q、K、V 权重的均值和方差，并对均值和方差矩阵进行归一化，使其均值为 1/3。
    """
    all_stats = {'K': [], 'Q': [], 'V': [], 'KxQ': [], 'KxQxV': []}

    # 从模型的 q_layers、k_layers 和 v_layers 中提取指定层的权重
    q_weight = model.q_layers[layer_idx].weight.data
    k_weight = model.k_layers[layer_idx].weight.data
    v_weight = model.v_layers[layer_idx].weight.data

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

    # 对每种类型的 mean 和 var 进行归一化处理，使每种类型的均值为 1/3
    for key in all_stats:
        means = np.array([item[0] for item in all_stats[key]])
        vars = np.array([item[1] for item in all_stats[key]])

        # 归一化 mean
        mean_mean = np.mean(means)
        normalized_means = means / (3 * mean_mean)  # 缩放到均值为 1/3
        for i, val in enumerate(normalized_means):
            all_stats[key][i][0] = val

        # 归一化 var
        var_mean = np.mean(vars)
        normalized_vars = vars / (3 * var_mean)  # 缩放到均值为 1/3
        for i, val in enumerate(normalized_vars):
            all_stats[key][i][1] = val

    return all_stats


def calculate_singular_values(model, layer_idx, alpha=0.5):
    """
    计算指定层每个头的 Q、K、V 权重的奇异值，以及 K * Q 和 K * Q * V 的奇异值，并计算前五个奇异值的指数平滑值。
    """
    all_singulars = {
        'K': [],
        'Q': [],
        'V': [],
        'KxQ': [],
        'KxQxV': []
    }

    # 从模型的 q_layers、k_layers 和 v_layers 中提取指定层的权重
    q_weight = model.q_layers[layer_idx].weight.data
    k_weight = model.k_layers[layer_idx].weight.data
    v_weight = model.v_layers[layer_idx].weight.data

    # 将 Q、K、V 权重 reshape 为 (num_heads, dim_per_head, dim) 形状
    dim_per_head = model.dim // model.num_heads
    q_weight_heads = q_weight.view(model.num_heads, dim_per_head, model.dim)
    k_weight_heads = k_weight.view(model.num_heads, dim_per_head, model.dim)
    v_weight_heads = v_weight.view(model.num_heads, dim_per_head, model.dim)

    # 定义计算奇异值的函数
    def compute_singular_values(matrix):
        _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
        return s  # 返回奇异值张量

    # 计算每个头的奇异值，并保存
    for i in range(model.num_heads):
        k_singular_values = compute_singular_values(k_weight_heads[i])
        q_singular_values = compute_singular_values(q_weight_heads[i])
        v_singular_values = compute_singular_values(v_weight_heads[i])

        # 计算 K * Q
        kq = torch.matmul(k_weight_heads[i], q_weight_heads[i].transpose(-2, -1))
        kq_singular_values = compute_singular_values(kq)

        # 计算 K * Q * V
        kqv = torch.matmul(kq, v_weight_heads[i])
        kqv_singular_values = compute_singular_values(kqv)

        # 保存每个头的奇异值
        all_singulars['K'].append(k_singular_values)
        all_singulars['Q'].append(q_singular_values)
        all_singulars['V'].append(v_singular_values)
        all_singulars['KxQ'].append(kq_singular_values)
        all_singulars['KxQxV'].append(kqv_singular_values)

    # 计算每种重要性标准的前五个奇异值的指数平滑值，并保存
    smoothed_singulars = {
        'K': [],
        'Q': [],
        'V': [],
        'KxQ': [],
        'KxQxV': []
    }

    for key in all_singulars:
        for i in range(model.num_heads):
            singular_values = all_singulars[key][i][:5]  # 获取前五个奇异值
            smoothed_value = singular_values[0]  # 初始平滑值为第一个奇异值
            for value in singular_values[1:]:
                smoothed_value = alpha * value + (1 - alpha) * smoothed_value
            smoothed_singulars[key].append(smoothed_value)

    return smoothed_singulars  # 返回每种标准的平滑奇异值，包含12个值





# def calculate_singular_values(model):
#     """
#     计算每层每个头的 Q、K、V 权重的奇异值，以及 K * Q 和 K * Q * V 的奇异值
#     """
#     all_singular_values = {'K': [], 'Q': [], 'V': [], 'KxQ': [], 'KxQxV': []}

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

#         # 计算每个头的 Q、K、V 奇异值
#         for i in range(model.num_heads):
#             def compute_singular_values(matrix):
#                 # 计算矩阵的奇异值
#                 u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
#                 return s.tolist()  # 返回奇异值列表

#             # 计算 Q、K、V 的奇异值
#             k_singular_values = compute_singular_values(k_weight_heads[i])
#             q_singular_values = compute_singular_values(q_weight_heads[i])
#             v_singular_values = compute_singular_values(v_weight_heads[i])

#             # 计算 K * Q 的奇异值
#             kq = torch.matmul(k_weight_heads[i], q_weight_heads[i].transpose(-2, -1))
#             kq_singular_values = compute_singular_values(kq)

#             # 计算 K * Q * V 的奇异值
#             kqv = torch.matmul(kq, v_weight_heads[i])
#             kqv_singular_values = compute_singular_values(kqv)

#             # 保存每个头的奇异值
#             all_singular_values['K'].append(k_singular_values)
#             all_singular_values['Q'].append(q_singular_values)
#             all_singular_values['V'].append(v_singular_values)
#             all_singular_values['KxQ'].append(kq_singular_values)
#             all_singular_values['KxQxV'].append(kqv_singular_values)

#     return all_singular_values

# # 示例调用：计算模型各层奇异值
# model = YourTransformerModel(num_heads=12, dim=768)
# model.load_pretrained_qkv_weights()
# singular_values = calculate_singular_values(model)

# # 保存奇异值信息到文件
# output_file = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/_singular_values.txt"
# with open(output_file, 'w') as f:
#     for layer in range(model.num_layers):
#         f.write(f"Layer {layer}:\n")
#         for head in range(model.num_heads):
#             f.write(f"Head {head}: ")
#             f.write(f"K_singular_values: {singular_values['K'][layer * model.num_heads + head]}, ")
#             f.write(f"Q_singular_values: {singular_values['Q'][layer * model.num_heads + head]}, ")
#             f.write(f"V_singular_values: {singular_values['V'][layer * model.num_heads + head]}, ")
#             f.write(f"KxQ_singular_values: {singular_values['KxQ'][layer * model.num_heads + head]}, ")
#             f.write(f"KxQxV_singular_values: {singular_values['KxQxV'][layer * model.num_heads + head]}\n")
#         f.write("\n")

# print(f"所有层的奇异值已保存到 {output_file}")



# 示例：加载模型并使用预训练权重
# def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
#     model = YourTransformerModel(num_heads=12, dim=768)

#     if pretrained:
#         # 加载 Q, K, V 权重到模型中
#         model.load_pretrained_qkv_weights( )

#         # 调用统计函数，计算权重参数的分布情况
#         all_stats = calculate_stats(model)

#         # 将统计结果保存到文件
#         output_file = f"/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/_mean_var.txt"
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)

#         with open(output_file, 'w') as f:
#             for layer in range(12):
#                 f.write(f"Layer {layer}:\n")
#                 for head in range(12):
#                     # 输出每个头的所有均值和方差，逗号分隔
#                     f.write(f"Head {head}: ")
#                     f.write(f"K_mean: {all_stats['K'][layer * 12 + head][0]}, K_var: {all_stats['K'][layer * 12 + head][1]}, ")
#                     f.write(f"Q_mean: {all_stats['Q'][layer * 12 + head][0]}, Q_var: {all_stats['Q'][layer * 12 + head][1]}, ")
#                     f.write(f"V_mean: {all_stats['V'][layer * 12 + head][0]}, V_var: {all_stats['V'][layer * 12 + head][1]}, ")
#                     f.write(f"KxQ_mean: {all_stats['KxQ'][layer * 12 + head][0]}, KxQ_var: {all_stats['KxQ'][layer * 12 + head][1]}, ")
#                     f.write(f"KxQxV_mean: {all_stats['KxQxV'][layer * 12 + head][0]}, KxQxV_var: {all_stats['KxQxV'][layer * 12 + head][1]}\n")
#                 # 每层之间空一行
#                 f.write("\n")

#         print(f"所有层的均值和方差已保存到 {output_file}")

#     return model

# 调用模型并加载预训练权重
# model = vit_small_patch16_224(pretrained=True)
