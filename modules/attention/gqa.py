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

def shuffle_heads_once(x: torch.Tensor, num_heads: int, group_size: int, exp_num: int, permuted_indices: Optional[torch.Tensor] = None, save_groups: bool = True) -> torch.Tensor:
    B, P, C = x.shape
    head_dim = C // num_heads  # 每个头的维度

    if permuted_indices is None:
        # 创建一个局部生成器，不使用全局随机数种子
        g = torch.Generator()
        g.manual_seed(torch.seed() + int(torch.initial_seed() % (2**32)))  # 生成一个新的种子

        # 使用局部生成器生成打乱顺序
        permuted_indices = torch.randperm(num_heads, generator=g)

        if save_groups:
            # group_size = num_heads // 2
            groups = [permuted_indices[i:i+group_size].cpu().numpy() for i in range(0, num_heads, group_size)]
            group_lines = [','.join(map(str, group)) for group in groups]
            filename_with_exp = f"/data/yjzhang/desktop/key-driven-gqa_new_kv/output/arbitrary/{exp_num}/group.txt"
            save_to_file(filename_with_exp, group_lines)

    x = x.view(B, P, num_heads, head_dim)
    x = x[:, :, permuted_indices, :]
    x = x.view(B, P, C)

    return x, permuted_indices



class GQA(nn.Module):

    def __init__(
            self,
            dim: int,
            num_kv_heads: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.           
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
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
        self.permuted_indices = None

    def forward(self, x: torch.Tensor, exp_num: int) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads
        group_size = self.num_heads // self.num_kv_heads

        # 只第一次打乱顺序，后续保持相同顺序
        # print('111', flush=True)
        x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, group_size, exp_num, permuted_indices=self.permuted_indices)
        # print('333')

        # q = self.q(x).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
        # k = self.k(x).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
        # v = self.v(x).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
        q = self.q(x_shuffled).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
        k = self.k(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
        v = self.v(x_shuffled).view(B, P, self.num_kv_heads, -1).transpose(1, 2) # (B, num_kv_heads, P, head_size)
        
        q = q * self.scale

        # print(self.num_kv_heads)
        
        q_grps = torch.split(q, group_size, dim=1)
        k_grps = torch.split(k, 1, dim=1) 
        v_grps = torch.split(v, 1, dim=1)

        outputs = [None] * len(k_grps)
        for i in range(len(k_grps)):
                
            # Collect items (note q has a larger head axis)
            curr_q = q_grps[i]  # (B, num_heads//num_kv_heads, num_patches, head_size)
            curr_k = k_grps[i]  # (B, 1, num_patches, head_size)
            curr_v = v_grps[i]  # (B, 1, num_patches, head_size)
            
            scores = (curr_q @ curr_k.transpose(-2, -1))
            weights = F.softmax(scores, dim=-1) # (B, num_heads//num_kv_heads, num_patches, num_patches)
            weights = self.attn_drop(weights)
            curr_att = weights @ curr_v # (B, num_heads//num_kv_heads, num_patches, head_size)
            outputs[i] = curr_att

        x = torch.cat(outputs, dim=1) # (B, num_heads, num_patches, head_size)
        x = x.transpose(1, 2).contiguous().view(B, P, C) # (B, num_patches, emb_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def att_weight_conversion(self, qkv_params, is_bias=False):
        '''
        Split and convert the QKV parameters from ViT checkpoints for the GQA implementation
        '''
        q, k, v = torch.split(qkv_params, qkv_params.shape[0] // 3, dim=0)

        group_size = self.num_heads // self.num_kv_heads

        def convert_weight(param):
            x = param.clone() # (dim, dim)

            # This totally breaks if you reshape as (dim, H, dim/H) and split across dim=1
            # You have to shape it as (H, dim/H, dim) and split the across dim=0
            x = x.view(self.num_heads, self.dim//self.num_heads, self.dim)
           
            xs = torch.split(x, group_size, dim=0) # split across head axis
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]
            x = torch.cat(xs, dim=0)

            expected_shape = (self.num_kv_heads*self.dim//self.num_heads, self.dim)
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
            return x
        
        def convert_bias(param):
            x = param.clone()
            x = x.view(self.num_heads, self.dim//self.num_heads)
            xs = torch.split(x, group_size, dim=0) # split across head axis
            xs = [xs[i].mean(dim=0) for i in range(self.num_kv_heads)]
            x = torch.cat(xs, dim=0)

            expected_shape = (self.num_kv_heads*self.dim//self.num_heads,)
            assert x.shape == expected_shape, f'Expected {expected_shape}, got {x.shape}'
            return x
        
        return {
            "q": q,
            "k": convert_weight(k) if not is_bias else convert_bias(k),
            "v": convert_weight(v) if not is_bias else convert_bias(v)
        }
    
    def load_pretrained_weights(self, state_dict, block_idx):

        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']

        wdict = self.att_weight_conversion(qkv_weight)
        bdict = self.att_weight_conversion(qkv_bias, is_bias=True)

        self.q.weight = assign_check(self.q.weight, wdict['q'])
        self.q.bias = assign_check(self.q.bias, bdict['q'])

        self.k.weight = assign_check(self.k.weight, wdict['k'])
        self.k.bias = assign_check(self.k.bias, bdict['k'])
        
        self.v.weight = assign_check(self.v.weight, wdict['v'])
        self.v.bias = assign_check(self.v.bias, bdict['v'])

        # Load in parameters for the output projection
        self.proj.weight = assign_check(self.proj.weight, state_dict[f'blocks.{block_idx}.attn.proj.weight'])
        self.proj.bias = assign_check(self.proj.bias, state_dict[f'blocks.{block_idx}.attn.proj.bias'])