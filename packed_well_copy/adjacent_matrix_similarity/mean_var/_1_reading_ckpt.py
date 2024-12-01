import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

class YourTransformerModel:
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim
        num_layers = 12
        self.num_layers = num_layers
        
        # 初始化每一层的 Q, K, V
        self.q_layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)])
        self.k_layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)])
        self.v_layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)])

    def load_pretrained_qkv_weights(self):
        """
        遍历每一层的 Q, K, V 权重，并为模型的每一层分别设置权重
        """
        checkpoint_path = '/data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth'
        state_dict = torch.load(checkpoint_path)

        for block_idx in range(self.num_layers):
            # 加载 Q, K, V 的权重
            qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']

            # 分割 Q、K、V 权重
            qkv_dim = qkv_weight.shape[0] // 3
            q_weight = qkv_weight[0:qkv_dim]  # Q矩阵部分
            k_weight = qkv_weight[qkv_dim: 2 * qkv_dim]  # K矩阵部分
            v_weight = qkv_weight[2 * qkv_dim:]  # V矩阵部分

            # 将 Q、K、V 重塑为 (num_heads, dim_per_head, dim) 形状
            dim_per_head = self.dim // self.num_heads
            q_weight_heads = q_weight.view(self.num_heads, dim_per_head, self.dim)
            k_weight_heads = k_weight.view(self.num_heads, dim_per_head, self.dim)
            v_weight_heads = v_weight.view(self.num_heads, dim_per_head, self.dim)

            # 为当前层的 Q、K、V 层赋值
            self.q_layers[block_idx].weight.data.copy_(q_weight)
            self.k_layers[block_idx].weight.data.copy_(k_weight)
            self.v_layers[block_idx].weight.data.copy_(v_weight)

# # 示例：加载模型并使用权重
# def vit_small_patch16_224(num_classes=10, pretrained=False, in_chans=3):
#     model = YourTransformerModel(num_heads=12, dim=768)

#     if pretrained:
#         # 加载预训练的 checkpoint
#         checkpoint_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/mhsa/config/best.pth'
#         checkpoint = torch.load(checkpoint_path)

#         # 加载每层的预训练权重
#         model.load_pretrained_qkv_weights( )

#     return model

# # 调用模型
# model = vit_small_patch16_224(pretrained=True)
