# import torch

# # 文件路径
# checkpoint_path = '/data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16.pth'  # 你的 pth 文件路径
# output_checkpoint_path = '/data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16_renamed.pth'  # 输出的 checkpoint 文件路径

# # 1. 加载原始 checkpoint 文件
# checkpoint = torch.load(checkpoint_path)

# # 2. 定义键名的映射关系
# key_replacements = {
#     "cls_token": "class_token",
#     "pos_embed": "positional_embedding.pos_embedding",
#     "patch_embed.proj.weight": "patch_embedding.weight",
#     "patch_embed.proj.bias": "patch_embedding.bias",
#     "head.weight": "fc.weight",
#     "head.bias": "fc.bias",
#     # Blocks and layers
# }

# # 从 duiying.txt 文件中提取的键映射，自动处理 transformer blocks
# for layer in range(12):
#     key_replacements[f"blocks.{layer}.norm1.weight"] = f"transformer.blocks.{layer}.norm1.weight"
#     key_replacements[f"blocks.{layer}.norm1.bias"] = f"transformer.blocks.{layer}.norm1.bias"
#     key_replacements[f"blocks.{layer}.norm2.weight"] = f"transformer.blocks.{layer}.norm2.weight"
#     key_replacements[f"blocks.{layer}.norm2.bias"] = f"transformer.blocks.{layer}.norm2.bias"
#     key_replacements[f"blocks.{layer}.mlp.fc1.weight"] = f"transformer.blocks.{layer}.pwff.fc1.weight"
#     key_replacements[f"blocks.{layer}.mlp.fc1.bias"] = f"transformer.blocks.{layer}.pwff.fc1.bias"
#     key_replacements[f"blocks.{layer}.mlp.fc2.weight"] = f"transformer.blocks.{layer}.pwff.fc2.weight"
#     key_replacements[f"blocks.{layer}.mlp.fc2.bias"] = f"transformer.blocks.{layer}.pwff.fc2.bias"
#     key_replacements[f"blocks.{layer}.attn.proj.weight"] = f"transformer.blocks.{layer}.proj.weight"
#     key_replacements[f"blocks.{layer}.attn.proj.bias"] = f"transformer.blocks.{layer}.proj.bias"
    
#     # 合并 qkv 键
#     key_replacements[f"blocks.{layer}.attn.qkv.weight"] = [
#         f"transformer.blocks.{layer}.attn.proj_q.weight",
#         f"transformer.blocks.{layer}.attn.proj_k.weight",
#         f"transformer.blocks.{layer}.attn.proj_v.weight"
#     ]
#     key_replacements[f"blocks.{layer}.attn.qkv.bias"] = [
#         f"transformer.blocks.{layer}.attn.proj_q.bias",
#         f"transformer.blocks.{layer}.attn.proj_k.bias",
#         f"transformer.blocks.{layer}.attn.proj_v.bias"
#     ]

# # 3. 创建一个新的 state_dict，并处理名称
# new_state_dict = {}

# # 处理 qkv 合并的逻辑
# for target_name, pth_names in key_replacements.items():
#     if isinstance(pth_names, list):  # 如果是列表，说明需要合并
#         if all(pth_name in checkpoint for pth_name in pth_names):
#             # 获取三个权重张量
#             tensors = [checkpoint[pth_name] for pth_name in pth_names]
#             # 合并这些张量 (沿第 0 维合并)
#             merged_tensor = torch.cat(tensors, dim=0)
#             # 保存到新的键名中
#             new_state_dict[target_name] = merged_tensor
#             # 删除旧的权重
#             for pth_name in pth_names:
#                 del checkpoint[pth_name]
#         else:
#             print(f"警告: 某些键在检查点中找不到，跳过合并: {pth_names}")
#     else:
#         # 直接替换名称
#         if pth_names in checkpoint:
#             new_state_dict[target_name] = checkpoint[pth_names]
#             del checkpoint[pth_names]

# # 4. 保留其他未被修改的权重
# for key, value in checkpoint.items():
#     if key not in new_state_dict:
#         new_state_dict[key] = value

# # 5. 保存处理后的 checkpoint
# torch.save(new_state_dict, output_checkpoint_path)
# print(f"修改后的检查点已保存至: {output_checkpoint_path}")



###### 不合并
import torch

# 文件路径
checkpoint_path = '/data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16.pth'  # 你的 pth 文件路径
output_checkpoint_path = '/data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16_renamed_no_merge.pth'  # 输出的 checkpoint 文件路径

# 1. 加载原始 checkpoint 文件
checkpoint = torch.load(checkpoint_path)

# 2. 定义键名的映射关系
key_replacements = {
    "cls_token": "class_token",
    "pos_embed": "positional_embedding.pos_embedding",
    "patch_embed.proj.weight": "patch_embedding.weight",
    "patch_embed.proj.bias": "patch_embedding.bias",
    # 不需要加载 head，因为它的大小与任务不匹配
    # 忽略 head.weight 和 head.bias
}

# 从 duiying.txt 文件中提取的键映射，自动处理 transformer blocks
for layer in range(12):
    key_replacements[f"blocks.{layer}.norm1.weight"] = f"transformer.blocks.{layer}.norm1.weight"
    key_replacements[f"blocks.{layer}.norm1.bias"] = f"transformer.blocks.{layer}.norm1.bias"
    key_replacements[f"blocks.{layer}.norm2.weight"] = f"transformer.blocks.{layer}.norm2.weight"
    key_replacements[f"blocks.{layer}.norm2.bias"] = f"transformer.blocks.{layer}.norm2.bias"
    key_replacements[f"blocks.{layer}.mlp.fc1.weight"] = f"transformer.blocks.{layer}.pwff.fc1.weight"
    key_replacements[f"blocks.{layer}.mlp.fc1.bias"] = f"transformer.blocks.{layer}.pwff.fc1.bias"
    key_replacements[f"blocks.{layer}.mlp.fc2.weight"] = f"transformer.blocks.{layer}.pwff.fc2.weight"
    key_replacements[f"blocks.{layer}.mlp.fc2.bias"] = f"transformer.blocks.{layer}.pwff.fc2.bias"
    key_replacements[f"blocks.{layer}.attn.proj.weight"] = f"transformer.blocks.{layer}.proj.weight"
    key_replacements[f"blocks.{layer}.attn.proj.bias"] = f"transformer.blocks.{layer}.proj.bias"

    # 不再合并 qkv，直接替换 q, k, v 键名
    key_replacements[f"blocks.{layer}.attn.q.weight"] = f"transformer.blocks.{layer}.attn.proj_q.weight"
    key_replacements[f"blocks.{layer}.attn.q.bias"] = f"transformer.blocks.{layer}.attn.proj_q.bias"
    key_replacements[f"blocks.{layer}.attn.k.weight"] = f"transformer.blocks.{layer}.attn.proj_k.weight"
    key_replacements[f"blocks.{layer}.attn.k.bias"] = f"transformer.blocks.{layer}.attn.proj_k.bias"
    key_replacements[f"blocks.{layer}.attn.v.weight"] = f"transformer.blocks.{layer}.attn.proj_v.weight"
    key_replacements[f"blocks.{layer}.attn.v.bias"] = f"transformer.blocks.{layer}.attn.proj_v.bias"

# 3. 创建一个新的 state_dict，并处理名称
new_state_dict = {}

# 遍历键名，直接替换名称，不合并
for target_name, pth_name in key_replacements.items():
    if pth_name in checkpoint:
        # 直接替换名称
        new_state_dict[target_name] = checkpoint[pth_name]
        del checkpoint[pth_name]
    else:
        print(f"警告: 键 {pth_name} 在检查点中找不到，跳过该键。")

# 4. 保留其他未被修改的权重
for key, value in checkpoint.items():
    if key not in new_state_dict and 'head' not in key:  # 忽略 head.weight 和 head.bias
        new_state_dict[key] = value

# 5. 保存处理后的 checkpoint
torch.save(new_state_dict, output_checkpoint_path)
print(f"修改后的检查点已保存至: {output_checkpoint_path}")
