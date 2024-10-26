import torch
import numpy as np
import pandas as pd
import re

# 保存所有输出到txt文件
def save_to_txt(output_path, content):
    with open(output_path, 'a') as f:
        f.write(content + '\n')


def load_stats_from_model(model):
    """
    从模型中计算 Q、K、V 的统计信息，并将其转换为张量。
    """
    # 使用 calculate_stats 函数从模型中获取统计信息
    all_stats = calculate_stats(model)
    
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