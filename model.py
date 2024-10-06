from functools import partial
from typing import Optional, Union, Tuple, Callable, Type

import torch
import torch.nn as nn
import timm
from timm.layers import PatchEmbed, Mlp, LayerType

from modules import Block
from utils import assign_check

class VisionTransformer(nn.Module):

    def __init__(
            self,
            exp_num: int,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            num_kv_heads: Optional[int] = None,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            att_scheme: str = 'mhsa',
            window_size: int = 1,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification heads
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer is None else norm_layer
        act_layer = nn.GELU if act_layer is None else act_layer

        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=False
        )

        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        self.norm = norm_layer(embed_dim)

        self.blocks = nn.Sequential(*[
            block_fn(
                exp_num=exp_num,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                num_kv_heads=num_kv_heads,
                mlp_layer=mlp_layer,
                att_scheme=att_scheme,
                window_size=window_size
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)
        ]

        # Classifier Head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        x = x + pos_embed

        return self.pos_drop(x)
    
    def pool(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
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

def split_qkv_weights(state_dict):
    """
    将合并的 qkv 参数拆分为 q, k, v。
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        # 如果是qkv层的权重
        if 'attn.qkv.weight' in key:
            # 例如: 'blocks.0.attn.qkv.weight'
            block_num = key.split('.')[1]  # 获取block编号
            dim = value.shape[0] // 3  # 计算每个部分的维度
            
            # 拆分 q, k, v
            q_weight, k_weight, v_weight = value[:dim], value[dim:dim*2], value[dim*2:]
            
            # 新键名
            new_state_dict[f'blocks.{block_num}.attn.q.weight'] = q_weight
            new_state_dict[f'blocks.{block_num}.attn.k.weight'] = k_weight
            new_state_dict[f'blocks.{block_num}.attn.v.weight'] = v_weight

        # 如果是qkv层的偏置
        elif 'attn.qkv.bias' in key:
            block_num = key.split('.')[1]
            dim = value.shape[0] // 3  # 计算每个部分的维度

            # 拆分 q, k, v 偏置
            q_bias, k_bias, v_bias = value[:dim], value[dim:dim*2], value[dim*2:]

            # 新键名
            new_state_dict[f'blocks.{block_num}.attn.q.bias'] = q_bias
            new_state_dict[f'blocks.{block_num}.attn.k.bias'] = k_bias
            new_state_dict[f'blocks.{block_num}.attn.v.bias'] = v_bias

        else:
            # 其他不需要处理的键值对，直接加入
            new_state_dict[key] = value

    return new_state_dict


def vit_small_patch16_224(
        num_classes: int = 10,
        pretrained: bool = False,
        att_scheme: str = 'mhsa',
        window_size: int = 10,
        num_kv_heads: int = 3,
        in_chans: int = 3,
        drop_rate: float = 0,
        pos_drop_rate: float = 0,
        attn_drop_rate: float = 0,
        proj_drop_rate: float = 0.
        ):
    
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=384,
        num_heads=6,
        depth=12,
        num_kv_heads=num_kv_heads,
        att_scheme=att_scheme,
        window_size=window_size,
        drop_rate=drop_rate,
        pos_drop_rate=pos_drop_rate,
        attn_drop_rate=attn_drop_rate,
        proj_drop_rate=proj_drop_rate
    )

    if pretrained:
        ckpt = 'vit_small_patch16_224'
        if in_chans != 3:
            raise ValueError(f"Cannot load in checkpoint with {in_chans=}")
        print(f'Using checkpoint {ckpt}...')
        hf_model = timm.create_model(ckpt, pretrained=True)
        model.load_pretrained_weights(hf_model.state_dict())
    
    return model

def vit_base_patch16_224(
        exp_num: int = 1,
        num_classes: int = 10,
        pretrained: bool = False,
        att_scheme: str = 'mhsa',
        window_size: int = 10,
        num_kv_heads: int = 3,
        in_chans: int = 3
        ):
    
    model = VisionTransformer(
        exp_num=exp_num,
        img_size=224,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        depth=12,
        num_kv_heads=num_kv_heads,
        att_scheme=att_scheme,
        window_size=window_size
    )
    # print(att_scheme)

    if pretrained:
        ckpt = 'vit_base_patch16_224'
        
        print(f'Using checkpoint {ckpt}...')
        checkpoint_path = '/data/yjzhang/desktop/key-driven-gqa_new_kv/output/pretrained/config_pretrained/final.pth'

        # 加载检查点
        checkpoint = torch.load(checkpoint_path)

        # 获取 state_dict
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 调用你定义的 split_qkv_weights 函数来拆分 qkv 权重
        # state_dict = split_qkv_weights(state_dict)

        # 将拆分后的权重加载到模型中
        model.load_pretrained_weights(state_dict)
        print("Checkpoint loaded successfully!")


    # if pretrained:
    #     ckpt = 'vit_base_patch16_224'
    #     if in_chans != 3:
    #         raise ValueError(f"Cannot load in checkpoint with {in_chans=}")
    #     print(f'Using checkpoint {ckpt}...')
    #     # hf_model = timm.create_model(ckpt, pretrained=True)
    #     # model.load_pretrained_weights(hf_model.state_dict())
    #     # checkpoint_path = '/data/yjzhang/desktop/key-driven-gqa_new_kv/output/pretrained/config_pretrained/final.pth'
    #     checkpoint_path = '/data/yjzhang/desktop/try/local_checkpoint/finetuned_vit_base_patch16_224.pth'
    #     checkpoint = torch.load(checkpoint_path)
    #     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    #     # 过滤掉不匹配的层，比如分类头
    #     # filtered_state_dict = {k: v for k, v in state_dict.items() if 'head' not in k and 'mlp.fc' not in k}

    #     model.load_state_dict(state_dict)
    
    return model

def vit_large_patch16_224(
        num_classes: int = 10,
        pretrained: bool = False,
        att_scheme: str = 'mhsa',
        window_size: int = 10,
        num_kv_heads: int = 4,
        in_chans: int = 3
        ):
    
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=1024,
        num_heads=16,
        depth=24,
        num_kv_heads=num_kv_heads,
        att_scheme=att_scheme,
        window_size=window_size
    )

    if pretrained:
        ckpt = 'vit_large_patch16_224'
        if in_chans != 3:
            raise ValueError(f"Cannot load in checkpoint with {in_chans=}")
        print(f'Using checkpoint {ckpt}...')
        hf_model = timm.create_model(ckpt, pretrained=True)
        model.load_pretrained_weights(hf_model.state_dict())
    
    return model