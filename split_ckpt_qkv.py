import torch

def split_qkv_and_save_checkpoint(input_pth, output_pth, num_blocks):
    # 加载原始检查点
    state_dict = torch.load(input_pth, map_location='cpu')
    
    # 处理每一层的 qkv 权重
    for block_idx in range(num_blocks):
        # 获取原始 qkv 权重
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']
        
        # 拆分 qkv 权重
        q, k, v = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

        # 将拆分后的权重更新到新的键中
        state_dict[f'blocks.{block_idx}.attn.q.weight'] = q
        state_dict[f'blocks.{block_idx}.attn.k.weight'] = k
        state_dict[f'blocks.{block_idx}.attn.v.weight'] = v
        state_dict[f'blocks.{block_idx}.attn.q.bias'] = q_bias
        state_dict[f'blocks.{block_idx}.attn.k.bias'] = k_bias
        state_dict[f'blocks.{block_idx}.attn.v.bias'] = v_bias

        # 删除原来的 qkv 权重
        del state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        del state_dict[f'blocks.{block_idx}.attn.qkv.bias']

    # 保存修改后的检查点
    torch.save(state_dict, output_pth)
    print(f'Checkpoint saved to {output_pth}')

# 输入的检查点文件路径
input_pth = '/data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth'
# 输出的新检查点文件路径
output_pth = '/data/yjzhang/desktop/try/ckpt/split_qkv.pth'
# Vision Transformer 的层数，假设是12
num_blocks = 12

# 调用函数进行 qkv 权重拆分并保存新检查点
split_qkv_and_save_checkpoint(input_pth, output_pth, num_blocks)
