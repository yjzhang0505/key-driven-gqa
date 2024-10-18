# import torch

# def check_shared_params(state_dict):
#     """
#     检查所有层是否共用相同的参数
#     """
#     param_refs = {}
#     shared_params = {}

#     for name, param in state_dict.items():
#         # 将参数的内存地址作为唯一标识
#         param_id = id(param)

#         if param_id not in param_refs:
#             param_refs[param_id] = name
#         else:
#             # 记录共用相同参数的层
#             if param_id not in shared_params:
#                 shared_params[param_id] = [param_refs[param_id]]
#             shared_params[param_id].append(name)

#     if shared_params:
#         print("以下层共用相同的参数:")
#         for param_id, layer_names in shared_params.items():
#             print(f"参数ID {param_id} 共用的层: {layer_names}")
#     else:
#         print("没有发现层共用相同的参数。")

# if __name__ == "__main__":
#     # 加载.pth文件
#     checkpoint_path = './output/checkpoint/final.pth'
#     checkpoint = torch.load(checkpoint_path)

#     # 如果是带有 "state_dict" 的checkpoint
#     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

#     # 检查是否有层共用相同的参数
#     check_shared_params(state_dict)

import torch

# 加载保存的模型权重
checkpoint = torch.load('/data/yjzhang/desktop/try/key-driven-gqa/output/share/final.pth')

# 检查所有层的权重是否是同一个对象
for i in range(1, 12):
    is_equal = torch.equal(checkpoint[f'blocks.0.attn.qkv.weight'], checkpoint[f'blocks.{i}.attn.qkv.weight'])
    print(f'Blocks 0 and {i} qkv weight are the same: {is_equal}')