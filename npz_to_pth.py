import numpy as np
import torch
from model import VisionTransformer  # 假设 VisionTransformer 定义在 vision_transformer.py 中
import os

# 定义键名转换函数
def convert_key(npz_key):
    # 根据 ViT 模型定义调整键名
    npz_key = npz_key.replace('Transformer/encoderblock_', 'blocks.')
    npz_key = npz_key.replace('MultiHeadDotProductAttention_1/query/kernel', 'attn.qkv.weight')
    npz_key = npz_key.replace('MultiHeadDotProductAttention_1/query/bias', 'attn.qkv.bias')
    npz_key = npz_key.replace('MlpBlock_3/Dense_0/kernel', 'mlp.fc1.weight')
    npz_key = npz_key.replace('MlpBlock_3/Dense_0/bias', 'mlp.fc1.bias')
    npz_key = npz_key.replace('MlpBlock_3/Dense_1/kernel', 'mlp.fc2.weight')
    npz_key = npz_key.replace('MlpBlock_3/Dense_1/bias', 'mlp.fc2.bias')
    # 继续根据需要转换更多键名
    return npz_key

# 加载 npz 文件
npz_file = '/data/yjzhang/desktop/try/not_share/sam_ViT-B_16.npz'  # 修改为你的 npz 文件路径
npz_weights = np.load(npz_file)

# 创建 VisionTransformer 模型实例，标准的 ViT-B 配置
vit_model = VisionTransformer(
    file_path="1",
    exp_num=1,               # 这是你模型的额外参数，按你需求定义
    img_size=224,            # 输入图像大小 224x224
    patch_size=16,           # patch 大小为 16x16
    in_chans=3,              # 输入通道数，通常是 RGB 图像的 3 通道
    num_classes=1000,        # 分类数目 (通常为 1000 个类，ImageNet 预训练模型)
    embed_dim=768,           # 嵌入维度
    depth=12,                # transformer block 的层数
    num_heads=12,            # 注意力头的数量
    mlp_ratio=4.0,           # MLP 扩展比例
    qkv_bias=True,           # 是否使用 bias
    drop_rate=0.0,           # dropout 率
    pos_drop_rate=0.0,       # 位置嵌入的 dropout
    proj_drop_rate=0.0,      # 投影的 dropout
    attn_drop_rate=0.0       # 注意力 dropout
)

# 将 npz 权重加载到 PyTorch 模型中
state_dict = vit_model.state_dict()

# 遍历 npz 文件的键并转换为 PyTorch 格式
for npz_key in npz_weights:
    pytorch_key = convert_key(npz_key)  # 转换键名
    if pytorch_key in state_dict:
        state_dict[pytorch_key] = torch.tensor(npz_weights[npz_key])
    else:
        print(f"警告: {pytorch_key} 不在模型的 state_dict 中，可能不需要该权重")

# 加载转换后的权重到模型
vit_model.load_state_dict(state_dict)

# 保存转换后的 PyTorch 模型为 .pth 文件
save_path = '/data/yjzhang/desktop/try/not_share/sam_ViT-B_16.pth'
torch.save(vit_model.state_dict(), save_path)

print(f"转换完成并保存为 {save_path}")
