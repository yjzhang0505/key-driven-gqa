from typing import Optional
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import math
from typing import Optional
import re
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import assign_check
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum

from utils import assign_check

import os

def load_group_schemes_from_txt(file_path: str):
    """
    从文本文件中加载分组方案，适用于每行表示一个层级的分组方案，并且每行最外层有 [] 包围。
    """
    group_schemes = {}
    
    with open(file_path, 'r') as f:
        for layer_index, line in enumerate(f):
            # 去掉最外层的括号，保留中间的内容
            line_content = line.strip()[1:-2]
            
            # 使用 ast.literal_eval 将字符串形式的分组数据转换为实际的列表
            groups = ast.literal_eval(f"[{line_content}]")  # 添加[]，使其变成有效的列表格式
            
            # 将分组方案存入字典
            group_schemes[layer_index] = groups

    return group_schemes



def shuffle_heads_once( x: torch.Tensor, num_heads: int, layer_index: int, file_path: str, load=True, save_groups: bool = True) -> torch.Tensor:
    """
    打乱头部的顺序，根据读取的分组方案进行排列
    """
    # file_path = "/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/V_cosine_V_singular"
    B, P, C = x.shape
    head_dim = C // num_heads  # 每个头的维度

    # 从txt文件加载group_schemes
    file_path1 = os.path.join(file_path, 'group.txt')
    group_schemes = load_group_schemes_from_txt(file_path1)

    # 根据layer_index选择相应的分组方案
    if layer_index in group_schemes:
        groups = group_schemes[layer_index]
        permuted_indices = torch.cat([torch.tensor(group) for group in groups])
    else:
        raise ValueError(f"Invalid layer_index: {layer_index}")

    x = x.view(B, P, num_heads, head_dim)
    x = x[:, :, permuted_indices, :]  # 使用 permuted_indices 对头进行打乱
    x = x.view(B, P, C)

    return x, permuted_indices

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,   
            layer_index: int = 13,
            file_path: str = "1",       
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.layer_index=layer_index
        self.file_path=file_path
        # print(layer_index)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_kv_heads = num_heads // 2 # have at least two heads in each group

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.num_kv_heads*self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.num_kv_heads*self.head_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # print("gqa")

        # 保存打乱后的头顺序索引
        # self.permuted_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads  # 总共的 heads 数量 
        group_size = self.num_heads // self.num_kv_heads

        x_shuffled, self.permuted_indices = shuffle_heads_once(x, H, self.layer_index, self.file_path, load=False)
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
        # print(f"q_groups: {q_groups}")

        head_outputs = [None] * H

        # 遍历每个组
        for i, q_group in enumerate(q_groups):
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

        # 合并所有 head 的输出，保持原始顺序
        x = torch.stack(head_outputs, dim=1)  # 在 q 的维度上拼接 (B, H, P, head_size)
        
        # 保存 head_outputs 到文件中 (仅在第一次调用时)
        # x_str = "\n".join([str(batch.tolist()) for batch in x])
        # output_file = f"./output/arbitrary/proxy/{self.exp_num}/head_outputs2.txt"
        # save_to_file_once(output_file, x_str)

        x = x.transpose(1, 2).contiguous().view(B, P, C)  # 恢复原来的形状 (B, P, C)
        x = self.proj(x)  # 线性映射
        x = self.proj_drop(x)  # dropout

        return x


        
    def att_weight_conversion(self, qkv_params, block_idx, is_bias=False):
        '''
        Split and convert the QKV parameters from ViT checkpoints for the GQA implementation
        '''
        q, k, v = torch.split(qkv_params, qkv_params.shape[0] // 3, dim=0)

        # 使用shuffle_heads_once打乱头的顺序，并保存打乱后的顺序
        _, self.permuted_indices = shuffle_heads_once(torch.empty(1, 1, self.dim), self.num_heads, block_idx, self.file_path, load = True, save_groups=True)

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

    def load_pretrained_weights(self, state_dict, block_idx):

        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']
        proj_weight = state_dict[f'blocks.{block_idx}.attn.proj.weight']
        proj_bias = state_dict[f'blocks.{block_idx}.attn.proj.bias']

        wdict = self.att_weight_conversion(qkv_weight,block_idx)
        bdict = self.att_weight_conversion(qkv_bias, block_idx, is_bias=True)

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


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # 激活函数，默认为 GELU
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)  # dropout 层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # 全连接层1
        x = self.act(x)   # 激活函数
        x = self.fc2(x)   # 全连接层2
        x = self.drop(x)  # dropout
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            layer_index: int = 13,
            file_path: str="1",
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.file_path=file_path
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            layer_index=layer_index,
            file_path=self.file_path
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def load_pretrained_weights(self, state_dict, block_idx):

        # print("1")
        self.attn.load_pretrained_weights(state_dict, block_idx)

        self.norm1.weight = assign_check(self.norm1.weight, state_dict[f'blocks.{block_idx}.norm1.weight'])
        self.norm1.bias = assign_check(self.norm1.bias, state_dict[f'blocks.{block_idx}.norm1.bias'])
        
        self.norm2.weight = assign_check(self.norm2.weight, state_dict[f'blocks.{block_idx}.norm2.weight'])
        self.norm2.bias = assign_check(self.norm2.bias, state_dict[f'blocks.{block_idx}.norm2.bias'])

        self.mlp.fc1.weight = assign_check(self.mlp.fc1.weight, state_dict[f'blocks.{block_idx}.mlp.fc1.weight'])
        self.mlp.fc1.bias = assign_check(self.mlp.fc1.bias, state_dict[f'blocks.{block_idx}.mlp.fc1.bias'])
        self.mlp.fc2.weight = assign_check(self.mlp.fc2.weight, state_dict[f'blocks.{block_idx}.mlp.fc2.weight'])
        self.mlp.fc2.bias = assign_check(self.mlp.fc2.bias, state_dict[f'blocks.{block_idx}.mlp.fc2.bias'])



class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        # 保存输入图像大小、patch大小、输入通道数、嵌入维度
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 计算输出的 patch 数量 (num_patches)
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积操作将图像分割为 patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入尺寸: (batch_size, channels, height, width)
        # 输出: (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        
        # 展平操作：将 H 和 W 展平为一个维度
        x = x.flatten(2)  # 输出尺寸: (batch_size, embed_dim, num_patches)
        
        # 转置以便将 patch 数量放到第二维
        x = x.transpose(1, 2)  # 输出尺寸: (batch_size, num_patches, embed_dim)
        
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndimension() - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()  # Random binary mask
        output = x / keep_prob * binary_mask
        return output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 学习的缩放系数
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 学习的偏置
        self.eps = eps  # 防止除以零的一个小常数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta



class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        init_values=None,
        representation_size=None,
        file_path="1"
    ):
        super(VisionTransformer, self).__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Class token
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks (Encoder layers)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                layer_index = i,
                file_path=file_path,
            ) for i in range(depth)
        ])

        # Layer normalization before the final classification head
        self.norm = norm_layer(embed_dim)

        # MLP head for classification
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

        # Optionally add a representation size for embedding layer before classification
        self.representation_size = representation_size
        if self.representation_size and self.head is not nn.Identity:
            self.head = nn.Linear(embed_dim, self.representation_size)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Patch embedding and class token
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        # Add class token and positional embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Dropout after position embedding
        x = self.pos_drop(x)

        # Pass through Transformer blocks (Encoder)
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.norm(x)

        # Extract class token (first token) for classification
        x = x[:, 0]  # cls token

        # Classification head
        x = self.head(x)

        return x

    def load_pretrained_weights(self, state_dict):
        print("Loading in weights...")
        
        for b, block in enumerate(self.blocks):
            block.load_pretrained_weights(state_dict, b)
        print(f"Finished with {b+1} blocks...")

        self.patch_embed.proj.weight = assign_check(self.patch_embed.proj.weight, state_dict['patch_embed.proj.weight'])
        self.patch_embed.proj.bias = assign_check(self.patch_embed.proj.bias, state_dict['patch_embed.proj.bias'])
        self.cls_token = assign_check(self.cls_token, state_dict['cls_token'])
        self.pos_embed = assign_check(self.pos_embed, state_dict['pos_embed'])

        print("Success!")