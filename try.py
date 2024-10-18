# # import torch

# # # 创建一个 3x4x5 的输入张量（6: batch size, 8: num_heads, 32: head_dim）
# # # kv_head =4, num_head=8
# # x1 = torch.randn(6, 8, 32)
# # w1 = torch.randn(32, 32)
# # w1 = w1.view(8, 4, 32)

# # permuted_indices = torch.arange(8 - 1, -1, -1) # 生成倒序
# # print(f"permuted_indices: {self.permuted_indices}")
# # w2 = w1[permuted_indices, :, :]
# # w1 = torch.split(w1,4,dim=0)
# # w1 = [w1[i].mean(dim=0) for i in range(2)]
# # w1 = torch.cat(w1, dim=0)
# # w2 = torch.split(w2,4,dim=0)
# # w2 = [w2[i].mean(dim=0) for i in range(2)]
# # w2 = torch.cat(w2, dim=0)

# # o1 = x1 @ w1.T
# # # print(o1)
# # x2 = x1[:, permuted_indices, :]
# # o2 = x2 @ w2.T
# # inverse_indices = torch.empty_like(permuted_indices)
# # inverse_indices[permuted_indices] = torch.arange(len(permuted_indices))
# # o2 = o2[:, inverse_indices, :]
# # # print(o2)

# # assert torch.equal(o1, o2), "Restored input does not match the original!"
# # print("\nRestoration successful, input matches the original!")
# import torch

# # Parameters
# batch_size = 1
# num_head = 6
# dim = 12
# kv_head = 2
# group_size = 3
# # group_size = num_head // kv_head

# # Creating input tensors
# x1 = torch.randn(batch_size, num_head, dim)
# x11 = x1
# w1 = torch.randn(dim, dim)
# w11 = w1
# w1 = w1.view(num_head, dim//num_head, dim)
# w12 = w1

# # 自定义一个顺序
# custom_permuted_indices = [5,4,3,2,1,0]  # 这是你自己定义的顺序 （保证分到一组的头不变则结果不变）
# # 将其转换为张量
# permuted_indices = torch.tensor(custom_permuted_indices)
# # permuted_indices = torch.arange(num_head - 1, -1, -1)  # Generating reverse order
# w13 = permuted_indices
# w2 = w1[permuted_indices, :, :]
# w14 = w2

# w3 = torch.split(w1, group_size, dim=0)
# w15 = w3
# w1 = [w3[i].mean(dim=0) for i in range(kv_head)]
# w16 = w1
# w1 = torch.cat(w1, dim=0)
# w17 = w1

# w2 = torch.split(w2,group_size,dim=0)
# w25=w2
# w2 = [w2[i].mean(dim=0) for i in range(kv_head)]
# w26=w2
# w2 = torch.cat(w2, dim=0)
# w27=w2

# o1 = x1 @ w1.T
# x2 = x1[:, permuted_indices, :]
# x22=x2
# o2 = x2 @ w2.T
# inverse_indices = torch.empty_like(permuted_indices)
# inverse_indices[permuted_indices] = torch.arange(len(permuted_indices))
# o2 = o2[:, inverse_indices, :]
# # o2 = o2[:, :, inverse_indices]
# # print(o2)

# # Saving the tensors in a txt file with spacing between them
# with open('try.txt', 'w') as f:
#     f.write(f"x11:\n{x11}\n\n")
#     f.write(f"x22:\n{x22}\n\n")
#     f.write(f"w11:\n{w11}\n\n")
#     f.write(f"w12:\n{w12}\n\n")
#     f.write(f"w13:\n{w13}\n\n")
#     f.write(f"w14:\n{w14}\n\n")
#     f.write(f"w15:\n{w15}\n\n")
#     f.write(f"w16:\n{w16}\n\n")
#     f.write(f"w17:\n{w17}\n\n")
#     f.write(f"w25:\n{w25}\n\n")
#     f.write(f"w26:\n{w26}\n\n")
#     f.write(f"w27:\n{w27}\n\n")
#     f.write(f"o1:\n{o1}\n\n")
#     f.write(f"o2:\n{o2}\n\n")


# # Output file path
# output_file_path = '/mnt/data/tensors_output.txt'
# output_file_path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Optional

from einops import rearrange, einsum

from utils import assign_check

import os

def save_to_file(filename, data):
    """Helper function to save data to a file."""
    # 获取文件所在的目录路径
    directory = os.path.dirname(filename)
    
    # 如果目录不存在，创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 打开文件并写入数据
    with open(filename, 'w') as f:
        for line in data:
            f.write(line + '\n')

def shuffle_heads_once(x: torch.Tensor, num_heads: int, group_size: int, exp_num: int, load = True, save_groups: bool = True) -> torch.Tensor:
    
    B, P, C = x.shape
    head_dim = C // num_heads  # 每个头的维度

    # 如果permuted_indices不是None，则读取组文件并恢复permuted_indices
    filename_with_exp = f"./output/arbitrary/proxy/111/group.txt"

    with open(filename_with_exp, 'r') as f:
        group_lines = f.readlines()

        groups = [list(map(int, line.strip().split(','))) for line in group_lines]

        # 根据组重新生成 permuted_indices
        permuted_indices = torch.cat([torch.tensor(group) for group in groups])

    x = x.view(B, P, num_heads, head_dim)
    x = x[:, :, permuted_indices, :]
    x = x.view(B, P, C)

    return x, permuted_indices

def append_to_file(filename, data):
    """Helper function to append data to a file."""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'a') as f:  # 以 'a' 模式打开文件，追加写入
        f.write(data + '\n')
        
class GQA(nn.Module):
    def __init__(self, exp_num: int, dim: int, num_heads: int = 2, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0., num_kv_heads: int = 1) -> None:
        super().__init__()
        self.exp_num = exp_num
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else (num_heads // 2)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化并应用权重转换
        self.apply_weight_conversion()

    def apply_weight_conversion(self):
        # 对q, k, v的权重和偏置进行转换
        qkv_weights = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        qkv_biases = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0)
        
        converted = self.att_weight_conversion(qkv_weights)
        converted_bias = self.att_weight_conversion(qkv_biases, is_bias=True)

        # 更新q, k, v的权重和偏置
        self.q.weight.data = assign_check(self.q.weight, converted['q'])
        self.k.weight.data = assign_check(self.k.weight, converted['k'])
        self.v.weight.data = assign_check(self.v.weight, converted['v'])
        
        self.q.bias.data = assign_check(self.q.bias, converted_bias['q'])
        self.k.bias.data = assign_check(self.k.bias, converted_bias['k'])
        self.v.bias.data = assign_check(self.v.bias, converted_bias['v'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads  # 总共的 heads 数量
        group_size = self.num_heads // self.num_kv_heads
        result_file = "./output/arbitrary/proxy/111/forward_results.txt"  # 结果文件的路径

        append_to_file(result_file, f"Initial Input Shape: {x.shape}")

        x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, group_size, self.exp_num, load=True)
        append_to_file(result_file, f"Shuffled Input Shape: {x_shuffled.shape}, Permuted Indices: {self.permuted_indices.tolist()}")

        inverse_indices = torch.empty_like(self.permuted_indices)
        inverse_indices[self.permuted_indices] = torch.arange(len(self.permuted_indices))

        # 获取 query, key, value 并为每个 head 计算
        q = self.q(x).view(B, P, H, -1).transpose(1, 2)  # (B, H, P, head_size)
        append_to_file(result_file, f"Query Shape: {q.shape}")
        
        k = self.k(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
        append_to_file(result_file, f"Key Shape: {k.shape}")
        
        v = self.v(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
        append_to_file(result_file, f"Value Shape: {v.shape}")
        
        # 缩放 query
        q = q * self.scale
        append_to_file(result_file, f"Scaled Query Shape: {q.shape}")

        # q_heads 是一个包含 H 个元素的列表，每个元素是 (B, P, head_size)
        q_heads = torch.split(q, 1, dim=1)  # 拆分为 (B, 1, P, head_size)
        k_heads = torch.split(k, 1, dim=1)  # 拆分为 (B, 1, P, head_size)
        v_heads = torch.split(v, 1, dim=1)  # 拆分为 (B, 1, P, head_size)

        # 根据 permuted_indices 将每个 q 分配到对应的 k 和 v
        q_groups = [tuple(self.permuted_indices[i:i + group_size].tolist()) for i in range(0, len(self.permuted_indices), group_size)]
        append_to_file(result_file, f"q_groups: {q_groups}")

        head_outputs = [None] * H

        # 遍历每个组
        for i, q_group in enumerate(q_groups):
            # kv 索引是 i，获取相同的 kv 头
            curr_k = k_heads[i]  # 对应的 k head (B, P, head_size)
            curr_v = v_heads[i]  # 对应的 v head (B, P, head_size)
            append_to_file(result_file, f"Group {i}: Processing q_group {q_group}")

            # 对每个 q_idx 进行遍历
            for q_idx in q_group:
                curr_q = q_heads[q_idx]  # 获取当前 q head (B, P, head_size)

                # 计算注意力分数
                attn_scores = torch.matmul(curr_q, curr_k.transpose(-2, -1))  # (B, P, P)
                attn_weights = F.softmax(attn_scores, dim=-1)  # 归一化注意力分数
                attn_weights = self.attn_drop(attn_weights)  # 注意力 dropout
                append_to_file(result_file, f"Attention Weights Shape (q_idx={q_idx}): {attn_weights.shape}")

                # 计算当前 q 的注意力输出
                curr_att = torch.matmul(attn_weights, curr_v)  # (B, P, head_size)
                head_outputs[q_idx] = curr_att.squeeze(1)  # 去除多余的维度

        # 合并所有 head 的输出，保持原始顺序
        x = torch.stack(head_outputs, dim=1)  # 在 q 的维度上拼接 (B, H, P, head_size)
        append_to_file(result_file, f"head_outputs: {head_outputs}")
        x = x.transpose(1, 2).contiguous().view(B, P, C)  # 恢复原来的形状 (B, P, C)
        append_to_file(result_file, f"Final Output Shape: {x.shape}")

        x = self.proj(x)  # 线性映射
        x = self.proj_drop(x)  # dropout

        return x

    def att_weight_conversion(self, qkv_params, is_bias=False):
        """
        参考你提供的att_weight_conversion逻辑，按需进行权重和偏置的转换。
        """
        q, k, v = torch.split(qkv_params, qkv_params.shape[0] // 3, dim=0)

        # 使用shuffle_heads_once打乱头的顺序
        _, self.permuted_indices = shuffle_heads_once(torch.empty(1, 1, self.dim), self.num_heads, self.num_heads // self.num_kv_heads, self.exp_num, load=True, save_groups=True)

        def convert_weight(param):
            x = param.clone()
            x = x.view(self.dim, self.num_heads, self.dim // self.num_heads)
            x = x[:, self.permuted_indices, :]
            x = x.view(self.num_heads, self.dim // self.num_heads, self.dim)
            x = x[self.permuted_indices, :, :]
            xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]
            return torch.cat(xs, dim=0)

        def convert_bias(param):
            x = param.clone()
            x = x.view(self.num_heads, self.dim // self.num_heads)
            x = x[self.permuted_indices, :]
            xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]
            return torch.cat(xs, dim=0)

        return {
            "q": q,
            "k": convert_weight(k) if not is_bias else convert_bias(k),
            "v": convert_weight(v) if not is_bias else convert_bias(v)
        }




# 初始化函数
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier 初始化
        if m.bias is not None:
            init.zeros_(m.bias)  # 将偏置初始化为 0
    elif isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming 初始化
        if m.bias is not None:
            init.zeros_(m.bias)

# 初始化模型并应用随机初始化
model = GQA(exp_num=1, dim=12, num_heads=6)
model.apply(initialize_weights)

# 查看模型的参数是否已经被随机初始化
for name, param in model.named_parameters():
    print(f"Parameter {name}: mean {param.mean()}, std {param.std()}")

# 输入张量示例
input_tensor = torch.tensor([[[ 1.5288, -0.8545, -0.1863, -1.0856, -0.7266,  1.2592, -0.2336,
                               -0.0513, -1.9333, -0.0741, -0.3703, -0.1896],
                              [ 0.3367, -0.5519,  1.1299,  2.3674, -0.4457,  0.3300,  0.0580,
                                1.9037,  1.7153, -1.7961,  1.6131, -0.0464],
                              [ 0.6445,  1.2819, -0.2745, -0.0162,  1.3366,  0.6605, -1.3033,
                                0.0700,  0.8757,  0.4360,  1.6960, -0.1203],
                              [ 0.5167, -1.8086,  1.0891, -0.4061,  0.8611, -1.3879, -0.3916,
                                1.1669, -1.1804,  0.3100, -0.4933,  0.8879],
                              [-0.7666,  1.1264, -0.0279,  1.0127, -2.3311, -0.6831,  0.1196,
                                0.6979,  1.6160,  0.2208, -1.5156, -1.0752]]])
output = model(input_tensor)

# 结果会自动写入到 ./output/forward_results.txt 文件中
