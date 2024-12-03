import torch

def swap_qkv_and_save_checkpoint(input_pth, output_pth, num_blocks, num_heads, head_swap_orders):
    # 加载原始检查点
    state_dict = torch.load(input_pth, map_location='cpu')
    
    # 处理每一层的 qkv 权重和 bias
    for block_idx in range(num_blocks):
        # 获取原始 qkv 权重和 bias
        qkv_weight = state_dict[f'blocks.{block_idx}.attn.qkv.weight']
        qkv_bias = state_dict[f'blocks.{block_idx}.attn.qkv.bias']
        
        # 拆分 qkv 权重为 q, k, v
        q, k, v = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
        
        # 确保拆分后每个 q, k, v 都有 12 个头
        q = q.view(num_heads, -1, q.size(-1))  # 重塑为 [num_heads, embed_dim, embed_dim]
        k = k.view(num_heads, -1, k.size(-1))
        v = v.view(num_heads, -1, v.size(-1))

        # 根据每层对应的交换顺序重新排列 q, k, v 中的头
        head_swap_order = head_swap_orders[block_idx]
        q = q[head_swap_order]  # 交换 q 中的头
        k = k[head_swap_order]  # 交换 k 中的头
        v = v[head_swap_order]  # 交换 v 中的头

        # 对 bias 也做相同的顺序调整
        q_bias = q_bias[head_swap_order]  # 交换 q_bias
        k_bias = k_bias[head_swap_order]  # 交换 k_bias
        v_bias = v_bias[head_swap_order]  # 交换 v_bias

        # 拼接回原来的形状 [num_heads * embed_dim, embed_dim]
        q = q.view(-1, q.size(-1))
        k = k.view(-1, k.size(-1))
        v = v.view(-1, v.size(-1))

        # 将新的 qkv 权重和 bias 重新拼接成合并形式
        new_qkv_weight = torch.cat([q, k, v], dim=0)
        new_qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        # 用新的 qkv 替换旧的 qkv 权重和 bias
        state_dict[f'blocks.{block_idx}.attn.qkv.weight'] = new_qkv_weight
        state_dict[f'blocks.{block_idx}.attn.qkv.bias'] = new_qkv_bias

    # 保存修改后的检查点
    torch.save(state_dict, output_pth)
    print(f'Checkpoint saved to {output_pth}')

# 输入的检查点文件路径
input_pth = '/home/yjzhang/desktop/try/ckpt/cifar100/4/model.pth'
# 输出的新检查点文件路径
output_pth = '/home/yjzhang/desktop/try/ckpt/split_qkv.pth'
# Vision Transformer 的层数，假设是12
num_blocks = 12
# 每个 block 中的头数
num_heads = 12
# 每层不同的头交换顺序
head_swap_orders = [
    [0, 10, 4, 7, 9, 1, 2, 3, 11, 8, 6, 5],
    [10, 1, 7, 0, 2, 3, 9, 4, 6, 5, 8, 11],
    [7, 2, 5, 8, 9, 10, 11, 3, 1, 0, 4, 6],
    [8, 6, 2, 5, 7, 10, 11, 4, 1, 3, 0, 9],
    [8, 7, 0, 10, 11, 4, 1, 6, 2, 5, 9, 3],
    [2, 3, 9, 11, 4, 0, 10, 6, 7, 1, 8, 5],
    [6, 7, 11, 1, 10, 4, 3, 8, 0, 2, 9, 5],
    [9, 6, 11, 5, 0, 8, 10, 2, 4, 7, 3, 1],
    [5, 6, 4, 7, 10, 9, 8, 11, 3, 2, 1, 0],
    [11, 7, 10, 9, 4, 6, 1, 8, 3, 5, 2, 0],
    [10, 9, 5, 6, 0, 7, 4, 1, 11, 3, 8, 2],
    [11, 0, 6, 7, 3, 5, 4, 9, 2, 8, 10, 1]
]

# 调用函数进行 qkv 权重和 bias 拆分并保存新检查点
swap_qkv_and_save_checkpoint(input_pth, output_pth, num_blocks, num_heads, head_swap_orders)
