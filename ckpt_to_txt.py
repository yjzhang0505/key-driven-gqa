import torch

# 加载检查点
checkpoint_path = '/data/yjzhang/desktop/PyTorch-Pretrained-ViT/jax_to_pytorch/weights/B_16_renamed_no_merge-2.pth'  # 修改为你的检查点路径
state_dict = torch.load(checkpoint_path)

# 指定输出 txt 文件路径
output_txt_path = '/data/yjzhang/desktop/checkpoint/B_16_renamed_no_merge-2.txt'

# 打开一个文件来写入检查点的结构
with open(output_txt_path, 'w') as f:
    for key, value in state_dict.items():
        # 将每个权重的名称和形状写入文件
        f.write(f'{key}: {value.shape}\n')

print(f'Checkpoint structure saved to {output_txt_path}')
