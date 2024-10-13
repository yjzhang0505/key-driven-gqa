import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # if load is True and permuted_indices is None:
    # exp_num = args.exp_num 
    file_path = f"./output/arbitrary/gqa_finetuned/{exp_num}/group.txt"
    if not os.path.exists(file_path):
        print(load)
        # 创建一个局部生成器，不使用全局随机数种子
        g = torch.Generator()
        g.manual_seed(torch.seed() + int(torch.initial_seed() % (2**32)))  # 生成一个新的种子

        # 使用局部生成器生成打乱顺序
        permuted_indices = torch.randperm(num_heads, generator=g)

        if save_groups:
            # group_size = num_heads // 2
            groups = [permuted_indices[i:i+group_size].cpu().numpy() for i in range(0, num_heads, group_size)]
            group_lines = [','.join(map(str, group)) for group in groups]
            filename_with_exp = f"./output/arbitrary/gqa_finetuned/{exp_num}/group.txt"
            save_to_file(filename_with_exp, group_lines)
    else:
        # 如果permuted_indices不是None，则读取组文件并恢复permuted_indices
        filename_with_exp = f"./output/arbitrary/gqa_finetuned/{exp_num}/group.txt"
        if os.path.exists(filename_with_exp):
            with open(filename_with_exp, 'r') as f:
                group_lines = f.readlines()

            groups = [list(map(int, line.strip().split(','))) for line in group_lines]

            # 根据组重新生成 permuted_indices
            permuted_indices = torch.cat([torch.tensor(group) for group in groups])

    x = x.view(B, P, num_heads, head_dim)
    x = x[:, :, permuted_indices, :]
    x = x.view(B, P, C)
    # print(permuted_indices)

    return x, permuted_indices



class GQA(nn.Module):

    def __init__(
            self,
            exp_num: int,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            num_kv_heads: Optional[int] = None,          
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.exp_num = exp_num
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else (num_heads // 2) # have at least two heads in each group

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.num_kv_heads*self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.num_kv_heads*self.head_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # print("gqa")

        # 保存打乱后的头顺序索引
        # self.permuted_indices = None

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # exp_num = 1
    #     B, P, C = x.shape
    #     H = self.num_heads
    #     group_size = self.num_heads // self.num_kv_heads

    #     # 只第一次打乱顺序，后续保持相同顺序
    #     x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, group_size, self.exp_num, load=False)
    #     # inverse_indices = torch.empty_like(self.permuted_indices)
    #     # inverse_indices[self.permuted_indices] = torch.arange(len(self.permuted_indices))

    #     q = self.q(x).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
    #     # k = self.k(x).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
    #     # v = self.v(x).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
    #     # q = self.q(x_shuffled).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
    #     k = self.k(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
    #     v = self.v(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
        
    #     q = q * self.scale

    #     q_grps = [q[:, i] for i in range(H)]  # 每个 head 的 query (B, P, head_size)
    #     k_grps = [k[:, i % self.num_kv_heads] for i in range(H)]  # 每个 head 的 key (B, P, head_size)
    #     v_grps = [v[:, i % self.num_kv_heads] for i in range(H)]  # 每个 head 的 value (B, P, head_size)

        
    #     # q_grps = torch.split(q, group_size, dim=1)
    #     # k_grps = torch.split(k, 1, dim=1) 
    #     # v_grps = torch.split(v, 1, dim=1)

    #     outputs = [None] * len(k_grps)
    #     for i in range(len(k_grps)):
                
    #         # Collect items (note q has a larger head axis)
    #         curr_q = q_grps[i]  # (B, num_heads//num_kv_heads, num_patches, head_size)
    #         curr_k = k_grps[i]  # (B, 1, num_patches, head_size)
    #         curr_v = v_grps[i]  # (B, 1, num_patches, head_size)
            
    #         scores = (curr_q @ curr_k.transpose(-2, -1))
    #         weights = F.softmax(scores, dim=-1) # (B, num_heads//num_kv_heads, num_patches, num_patches)
    #         weights = self.attn_drop(weights)
    #         curr_att = weights @ curr_v # (B, num_heads//num_kv_heads, num_patches, head_size)
    #         outputs[i] = curr_att

    #     x = torch.cat(outputs, dim=1) # (B, num_heads, num_patches, head_size)
    #     x = x.transpose(1, 2).contiguous().view(B, P, C) # (B, num_patches, emb_dim)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x




    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     B, P, C = x.shape
    #     H = self.num_heads  # 总共的 heads 数量
    #     group_size = self.num_heads // self.num_kv_heads

    #     x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, group_size, self.exp_num, load=False)
    #     inverse_indices = torch.empty_like(self.permuted_indices)
    #     inverse_indices[self.permuted_indices] = torch.arange(len(self.permuted_indices))

    #     # 获取 query, key, value 并为每个 head 计算
    #     q = self.q(x).view(B, P, H, -1).transpose(1, 2)  # (B, H, P, head_size)
    #     k = self.k(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
    #     v = self.v(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
        
    #     # 缩放 query
    #     q = q * self.scale

    #     # q = q[:, self.permuted_indices, :, :]

    #     # 存储每个 head 的 q, k, v
    #     q_heads = torch.split(q, 1, dim=1)
    #     # [q[:, i] for i in range(H)]  # 每个 head 的 query (B, P, head_size)
    #     k_heads = torch.split(k, 1, dim=1) # 每个 head 的 key (B, P, head_size)
    #     v_heads = torch.split(v, 1, dim=1)  # 每个 head 的 value (B, P, head_size)

    #     # 根据 permuted_indices 将每个 q 分配到对应的 k 和 v
    #     q_groups = [tuple(self.permuted_indices[i:i + group_size].tolist()) for i in range(0, len(self.permuted_indices), group_size)]
    #     # print(q_groups)

    #     head_outputs = [None] * H

    #     # 构建一个映射，表示每个 q 头应该对应哪个 k 和 v
    #     for i, q_group in enumerate(q_groups):
    #         for q_idx in q_group:
    #             curr_q = q_heads[q_idx]  # 获取当前 q head (B, P, head_size)
                
    #             # kv 索引是 i，获取 kv 头
    #             curr_k = k_heads[i]  # 对应的 k head (B, P, head_size)
    #             curr_v = v_heads[i]  # 对应的 v head (B, P, head_size)
    #             # print(q_idx,i)

    #             # 计算注意力分数
    #             attn_scores = torch.matmul(curr_q, curr_k.transpose(-2, -1))  # (B, P, P)
    #             attn_weights = F.softmax(attn_scores, dim=-1)  # 归一化注意力分数
    #             attn_weights = self.attn_drop(attn_weights)  # 注意力 dropout

    #             # 计算当前 q 的注意力输出
    #             curr_att = torch.matmul(attn_weights, curr_v)  # (B, P, head_size)

    #             # 将输出按 q 的顺序存储
    #             head_outputs[q_idx] = curr_att

    #     # 合并所有 head 的输出，保持原始顺序
    #     x = torch.stack(head_outputs, dim=1)  # 在 q 的维度上拼接 (B, H, P, head_size)
    #     # print(x.shape)
    #     # x = x[:, inverse_indices, :,:]
    #     x = x.transpose(1, 2).contiguous().view(B, P, C)  # 恢复原来的形状 (B, P, C)
    #     x = self.proj(x)  # 线性映射
    #     x = self.proj_drop(x)  # dropout

    #     return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads  # 总共的 heads 数量
        group_size = self.num_heads // self.num_kv_heads

        x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, group_size, self.exp_num, load=False)
        inverse_indices = torch.empty_like(self.permuted_indices)
        inverse_indices[self.permuted_indices] = torch.arange(len(self.permuted_indices))

        # 获取 query, key, value 并为每个 head 计算
        q = self.q(x).view(B, P, H, -1).transpose(1, 2)  # (B, H, P, head_size)
        k = self.k(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
        v = self.v(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2)  # (B, num_kv_heads, P, head_size)
        
        # 缩放 query
        q = q * self.scale

        # q_heads 是一个包含 H 个元素的列表，每个元素是 (B, P, head_size)
        q_heads = torch.split(q, 1, dim=1)  # 拆分为 (B, 1, P, head_size)
        k_heads = torch.split(k, 1, dim=1)  # 拆分为 (B, 1, P, head_size)
        v_heads = torch.split(v, 1, dim=1)  # 拆分为 (B, 1, P, head_size)

        # 根据 permuted_indices 将每个 q 分配到对应的 k 和 v
        q_groups = [tuple(self.permuted_indices[i:i + group_size].tolist()) for i in range(0, len(self.permuted_indices), group_size)]

        head_outputs = [None] * H

        # 遍历每个组
        for i, q_group in enumerate(q_groups):
            # kv 索引是 i，获取相同的 kv 头
            curr_k = k_heads[i]  # 对应的 k head (B, P, head_size)
            curr_v = v_heads[i]  # 对应的 v head (B, P, head_size)

            # 对每个 q_idx 进行遍历
            for q_idx in q_group:
                curr_q = q_heads[q_idx]  # 获取当前 q head (B, P, head_size)

                # 计算注意力分数
                attn_scores = torch.matmul(curr_q, curr_k.transpose(-2, -1))  # (B, P, P)
                attn_weights = F.softmax(attn_scores, dim=-1)  # 归一化注意力分数
                attn_weights = self.attn_drop(attn_weights)  # 注意力 dropout

                # 计算当前 q 的注意力输出
                curr_att = torch.matmul(attn_weights, curr_v)  # (B, P, head_size)

                # 将输出按 q 的顺序存储
                head_outputs[q_idx] = curr_att.squeeze(1)  # 去除多余的维度
                # print(q_idx)

        # 合并所有 head 的输出，保持原始顺序
        x = torch.stack(head_outputs, dim=1)  # 在 q 的维度上拼接 (B, H, P, head_size)
        x = x.transpose(1, 2).contiguous().view(B, P, C)  # 恢复原来的形状 (B, P, C)
        x = self.proj(x)  # 线性映射
        x = self.proj_drop(x)  # dropout

        return x


        
    def att_weight_conversion(self, qkv_params, is_bias=False):
        '''
        Split and convert the QKV parameters from ViT checkpoints for the GQA implementation
        '''
        q, k, v = torch.split(qkv_params, qkv_params.shape[0] // 3, dim=0)

        # 使用shuffle_heads_once打乱头的顺序，并保存打乱后的顺序
        _, self.permuted_indices = shuffle_heads_once(torch.empty(1, 1, self.dim), self.num_heads, self.num_heads // self.num_kv_heads, self.exp_num, load = True, save_groups=True)

        # 基于打乱后的头顺序进行池化
        def convert_weight(param):
            x = param.clone()  # (dim, dim)

            x = x.view(self.dim, self.num_heads, self.dim // self.num_heads)
            x = x[:, self.permuted_indices, :]  # 按照打乱后的顺序重新排列
            x = x.view(self.dim, self.dim)

            # 将权重视为 (num_heads, dim//num_heads, dim)
            x = x.view(self.num_heads, self.dim // self.num_heads, self.dim)

            # 使用打乱后的顺序进行分组
            x = x[self.permuted_indices,:,:]  # 按照打乱后的顺序重新排列
            xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)  # 按打乱后的分组进行分割
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]  # 平均池化
            x = torch.cat(xs, dim=0)

            expected_shape = (self.num_kv_heads * self.dim // self.num_heads, self.dim)
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
            return x

        def convert_weight_q(param):
            x = param.clone()  # (dim, dim)

            # 将权重视为 (num_heads, dim//num_heads, dim)
            x = x.view(self.num_heads, self.dim // self.num_heads, self.dim)

            # 使用打乱后的顺序进行分组
            # x = x[self.permuted_indices,:,:]  # 按照打乱后的顺序重新排列
            # xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)  # 按打乱后的分组进行分割
            # xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]  # 平均池化
            # x = torch.cat(xs, dim=0)
            x = x.view(self.dim, self.dim)

            expected_shape = (self.dim, self.dim)
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
            return x

        def convert_bias_q(param):
            x = param.clone()
            # 假设 param 的形状为 (dim,)
            
            # 将 bias 参数的形状调整为 (num_heads, dim//num_heads)
            x = x.view(self.num_heads, self.dim // self.num_heads)
            
            # 按照 heads 的数量分割成多个部分
            xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)
            
            # 将所有片段合并为一个张量
            x = torch.cat(xs, dim=0)
            
            # 将结果 reshape 成 (dim,)
            x = x.view(self.dim)

            # 断言输出的形状是否符合预期
            expected_shape = torch.Size([self.dim])  # 确保 expected_shape 和 x 的形状匹配
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'

            return x

        def convert_bias(param):
            x = param.clone()
            x = x.view(self.num_heads, self.dim // self.num_heads)

            # 使用打乱后的头顺序
            
            x = x[self.permuted_indices,:]
            # print(f"permuted_indices: {self.permuted_indices}")
            # print(x)
            xs = torch.split(x, self.num_heads // self.num_kv_heads, dim=0)
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]
            x = torch.cat(xs, dim=0)

            expected_shape = (self.num_kv_heads * self.dim // self.num_heads,)
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
            return x

        return {
            "q": q,
            "k": convert_weight(k) if not is_bias else convert_bias(k),
            "v": convert_weight(v) if not is_bias else convert_bias(v)
        }

    def proj_conversion(self, params, is_bias=False):
        '''
        Split and convert the QKV parameters from ViT checkpoints for the GQA implementation
        '''

        # 使用shuffle_heads_once打乱头的顺序，并保存打乱后的顺序
        _, self.permuted_indices = shuffle_heads_once(torch.empty(1, 1, self.dim), self.num_heads, self.num_heads // self.num_kv_heads, self.exp_num, load = True, save_groups=True)

        # 基于打乱后的头顺序进行池化
        # def convert_weight(param):
        x = params.clone()  # (dim, dim)

        # 将权重视为 (num_heads, dim//num_heads, dim)
        x = x.view(self.num_heads, self.dim // self.num_heads, self.dim)

        # 使用打乱后的顺序进行分组
        x = x[self.permuted_indices,:,:]  # 按照打乱后的顺序重新排列

        x = x.reshape(self.dim, self.dim)

        # expected_shape = (self.num_kv_heads * self.dim // self.num_heads, self.dim)
        # assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
        return x

        # def convert_bias(param):
        #     x = param.clone()
        #     head_dim = x.shape[0] // self.num_heads
        #     x = x.view(num_heads, head_dim)


        #     # 使用打乱后的头顺序
            
        #     x = x[self.permuted_indices,:]
        #     # print(f"permuted_indices: {self.permuted_indices}")
        #     # print(x)
        #     x = x.reshape(self.dim)

        #     # expected_shape = (self.num_kv_heads * self.dim // self.num_heads,)
        #     # assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
        #     return x

        # return {
        #     "proj": convert_weight(params) if not is_bias else convert_bias(params),
        # }

    def load_pretrained_weights(self, state_dict, block_idx):

        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']
        proj_weight = state_dict[f'blocks.{block_idx}.attn.proj.weight']
        proj_bias = state_dict[f'blocks.{block_idx}.attn.proj.bias']

        wdict = self.att_weight_conversion(qkv_weight)
        bdict = self.att_weight_conversion(qkv_bias, is_bias=True)

        # wproj = self.proj_conversion(proj_weight)
        # bproj = self.proj_conversion(proj_weight, is_bias=True)

        self.q.weight = assign_check(self.q.weight, wdict['q'])
        self.q.bias = assign_check(self.q.bias, bdict['q'])

        self.k.weight = assign_check(self.k.weight, wdict['k'])
        self.k.bias = assign_check(self.k.bias, bdict['k'])
        
        self.v.weight = assign_check(self.v.weight, wdict['v'])
        self.v.bias = assign_check(self.v.bias, bdict['v'])

        # Load in parameters for the output projection
        self.proj.weight = assign_check(self.proj.weight, state_dict[f'blocks.{block_idx}.attn.proj.weight'])
        self.proj.bias = assign_check(self.proj.bias, state_dict[f'blocks.{block_idx}.attn.proj.bias'])
        # print(f"self.proj.weight shape: {self.proj.weight.shape}")
        # print(f"wproj['proj'] shape: {wproj.shape}")

        # self.proj.weight = assign_check(self.proj.weight,wproj)
        # self.proj.bias = assign_check(self.proj.bias, bproj['proj'])