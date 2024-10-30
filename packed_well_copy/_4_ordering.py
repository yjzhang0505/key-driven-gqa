import numpy as np
import pandas as pd
from _1_reading_ckpt import YourTransformerModel
from _2_importance import calculate_stats
import torch
import numpy as np
import pandas as pd
import re

def load_stats_from_model(model):
    """
    从模型中计算 Q、K、V 的统计信息，并将其转换为张量。
    """
    # 使用 calculate_stats 函数从模型中获取统计信息
    all_stats = calculate_stats(model)
    # print(all_stats)  # 打印 all_stats，确认其包含键 'K'

    
    # 创建存储张量的字典，与原始函数返回的格式保持一致
    stats_tensor = {
        'K_mean': [], 'K_var': [],
        'Q_mean': [], 'Q_var': [],
        'V_mean': [], 'V_var': [],
        'KxQ_mean': [], 'KxQ_var': [],
        'KxQxV_mean': [], 'KxQxV_var': []
    }


    # 从 all_stats 中提取每层每个头的均值和方差，并保存在对应的列表中
    for layer in range(len(all_stats['K']) // 12):  # 假设每层有12个头
        
        for head in range(12):
            stats_tensor['K_mean'].append(abs(all_stats['K'][layer * 12 + head][0]))  # 取绝对值
            stats_tensor['K_var'].append(abs(all_stats['K'][layer * 12 + head][1]))
            stats_tensor['Q_mean'].append(abs(all_stats['Q'][layer * 12 + head][0]))
            stats_tensor['Q_var'].append(abs(all_stats['Q'][layer * 12 + head][1]))
            stats_tensor['V_mean'].append(abs(all_stats['V'][layer * 12 + head][0]))
            stats_tensor['V_var'].append(abs(all_stats['V'][layer * 12 + head][1]))
            stats_tensor['KxQ_mean'].append(abs(all_stats['KxQ'][layer * 12 + head][0]))
            stats_tensor['KxQ_var'].append(abs(all_stats['KxQ'][layer * 12 + head][1]))
            stats_tensor['KxQxV_mean'].append(abs(all_stats['KxQxV'][layer * 12 + head][0]))
            stats_tensor['KxQxV_var'].append(abs(all_stats['KxQxV'][layer * 12 + head][1]))

    # 将列表转换为张量，并 reshape 成每层有 12 个头的形式
    stats_tensor = {key: torch.tensor(value).view(-1, 12) for key, value in stats_tensor.items()}

    return stats_tensor



def get_importance_orders(stats_tensor, layer_idx):
    """
    对指定层的每个头的张量进行排序并生成动态排序序列。
    - 对于 mean，按值从大到小排序。
    - 对于 var，按值从小到大排序。
    
    返回一个包含重要性顺序的字典。
    """
    sorted_indices = {}

    for stat_type, tensor in stats_tensor.items():
        if 'mean' in stat_type:  # mean 从大到小排序
            sort_descending = True
        elif 'var' in stat_type:  # var 从小到大排序
            sort_descending = False
        else:
            raise ValueError(f"Unrecognized stat_type: {stat_type}")
        
        # 对指定层进行排序
        layer_values = tensor[layer_idx]
        sorted_idx = torch.argsort(layer_values, descending=sort_descending).tolist()
        sorted_indices[stat_type] = sorted_idx

    # 生成重要性顺序的字典
    importance_orders = {}
    labels = [
        "K_mean", "K_var",
        "Q_mean", "Q_var",
        "V_mean", "V_var",
        "KxQ_mean", "KxQ_var",
        "KxQxV_mean", "KxQxV_var",
    ]
    for label in labels:
        importance_orders[label] = sorted_indices[label]

    return importance_orders


# 文件路径
file_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt"  # TXT 文件路径
excel_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/not_shared_similarity_matrices_all_layers.xlsx'  # Excel 文件路径
output_txt_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/dustbin/_grouping_mean_var.txt'  # 输出txt文件路径
groupings_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_aline.txt'  # 分组方案文件路径

# 加载统计数据并进行排序
# model = YourTransformerModel(num_heads=12, dim=768)
# model.load_pretrained_qkv_weights( )
# stats_tensor = load_stats_from_model(model)


