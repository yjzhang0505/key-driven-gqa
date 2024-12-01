from typing import Optional
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import assign_check

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads
        q = self.q(x).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
        k = self.k(x).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
        v = self.v(x).view(B, P, H, -1).transpose(1, 2) # (B, H, P, head_size)
        
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, P, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def att_weight_conversion(self, qkv_params):
        '''
        Split and convert the QKV parameters from ViT checkpoints for the GQA implementation
        '''
        q, k, v = torch.split(qkv_params, qkv_params.shape[0] // 3, dim=0)
        
        return {
            "q": q,
            "k": k,
            "v": v
        }
    
    def load_pretrained_weights(self, state_dict, block_idx):

        # Load in parameters for the Query Key Value layers
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']

        wdict = self.att_weight_conversion(qkv_weight)
        bdict = self.att_weight_conversion(qkv_bias)

        self.q.weight = assign_check(self.q.weight, wdict['q'])
        self.q.bias = assign_check(self.q.bias, bdict['q'])

        self.k.weight = assign_check(self.k.weight, wdict['k'])
        self.k.bias = assign_check(self.k.bias, bdict['k'])
        
        self.v.weight = assign_check(self.v.weight, wdict['v'])
        self.v.bias = assign_check(self.v.bias, bdict['v'])

        # Load in parameters for the output projection
        self.proj.weight = assign_check(self.proj.weight, state_dict[f'blocks.{block_idx}.attn.proj.weight'])
        self.proj.bias = assign_check(self.proj.bias, state_dict[f'blocks.{block_idx}.attn.proj.bias'])
        
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
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
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
                norm_layer=norm_layer
            ) for _ in range(depth)
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