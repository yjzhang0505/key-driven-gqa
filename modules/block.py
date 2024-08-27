from typing import Optional

import torch
import torch.nn as nn
from timm.layers import Mlp

from .attention import *
from utils import assign_check
    
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            num_kv_heads: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            att_scheme: str = 'mhsa',
            window_size: int = 1
    ) -> None:
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        if att_scheme == 'mhsa':
            self.attn = MHSA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
        elif att_scheme == 'gqa':
            self.attn = GQA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_kv_heads=num_kv_heads
            )
        elif att_scheme == 'dgqa_ema':
            self.attn = DGQA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_kv_heads=num_kv_heads,
                kind='ema',
                window_size=window_size
            )
        elif att_scheme == 'dgqa_diff':
            self.attn = DGQA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_kv_heads=num_kv_heads,
                kind='diff',
                window_size=window_size
            )
        elif att_scheme == 'kdgqa':
            self.attn = KDGQA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_kv_heads=num_kv_heads,
            )
        elif att_scheme == 'pgqa':
            self.attn = PGQA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                num_kv_heads=num_kv_heads,
            )
        else:
            raise ValueError('Invalid attention scheme - please look into block.py')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    def load_pretrained_weights(self, state_dict, block_idx):

        self.attn.load_pretrained_weights(state_dict, block_idx)

        self.norm1.weight = assign_check(self.norm1.weight, state_dict[f'blocks.{block_idx}.norm1.weight'])
        self.norm1.bias = assign_check(self.norm1.bias, state_dict[f'blocks.{block_idx}.norm1.bias'])
        
        self.norm2.weight = assign_check(self.norm2.weight, state_dict[f'blocks.{block_idx}.norm2.weight'])
        self.norm2.bias = assign_check(self.norm2.bias, state_dict[f'blocks.{block_idx}.norm2.bias'])

        self.mlp.fc1.weight = assign_check(self.mlp.fc1.weight, state_dict[f'blocks.{block_idx}.mlp.fc1.weight'])
        self.mlp.fc1.bias = assign_check(self.mlp.fc1.bias, state_dict[f'blocks.{block_idx}.mlp.fc1.bias'])
        self.mlp.fc2.weight = assign_check(self.mlp.fc2.weight, state_dict[f'blocks.{block_idx}.mlp.fc2.weight'])
        self.mlp.fc2.bias = assign_check(self.mlp.fc2.bias, state_dict[f'blocks.{block_idx}.mlp.fc2.bias'])