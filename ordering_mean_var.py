import torch
import re

def load_stats_from_txt(file_path):
    """
    从TXT文件加载统计信息，并转换为张量。
    """
    stats = {
        'K_mean': [], 'K_var': [],
        'Q_mean': [], 'Q_var': [],
        'V_mean': [], 'V_var': [],
        'KxQ_mean': [], 'KxQ_var': [],
        'KxQxV_mean': [], 'KxQxV_var': []
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    layer_idx = -1
    for line in lines:
        line = line.strip()
        
        # 检查是否是新的Layer
        if line.startswith("Layer"):
            layer_idx += 1
            continue
        
        # 如果是Head行，提取数值
        if line.startswith("Head"):
            numbers = re.findall(r'-?\d+\.\d+e[+-]?\d+|-?\d+\.\d+', line)
            numbers = [float(num) for num in numbers]  # 转换为浮点数
            
            if len(numbers) == 10:
                stats['K_mean'].append(abs(numbers[0]))  # 存储绝对值
                stats['K_var'].append(abs(numbers[1]))   # 存储绝对值
                stats['Q_mean'].append(abs(numbers[2]))  # 存储绝对值
                stats['Q_var'].append(abs(numbers[3]))   # 存储绝对值
                stats['V_mean'].append(abs(numbers[4]))  # 存储绝对值
                stats['V_var'].append(abs(numbers[5]))   # 存储绝对值
                stats['KxQ_mean'].append(abs(numbers[6]))  # 存储绝对值
                stats['KxQ_var'].append(abs(numbers[7]))   # 存储绝对值
                stats['KxQxV_mean'].append(abs(numbers[8]))  # 存储绝对值
                stats['KxQxV_var'].append(abs(numbers[9]))   # 存储绝对值

    # 将统计量转换为张量
    stats_tensor = {key: torch.tensor(value).view(-1, 12) for key, value in stats.items()}  # 每个层有12个头
    return stats_tensor

# 加载TXT文件并转换为张量
file_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var_share.txt"
stats_tensor = load_stats_from_txt(file_path)

def sort_head_by_abs(stats_tensor):
    """
    对每层每个头的张量进行排序。
    - 对于 mean，按值从大到小排序。
    - 对于 var，按值从小到大排序。
    """
    sorted_indices = {}
    
    for stat_type, tensor in stats_tensor.items():
        sorted_indices[stat_type] = []
        
        # 判断排序类型
        if 'mean' in stat_type:  # mean从大到小排序
            sort_descending = True
        elif 'var' in stat_type:  # var从小到大排序
            sort_descending = False
        else:
            raise ValueError(f"Unrecognized stat_type: {stat_type}")
        
        # 对每一层（行）进行排序
        for layer_idx in range(tensor.shape[0]):
            # 每一层的12个头的值
            layer_values = tensor[layer_idx]
            
            # 按照指定的顺序进行排序，返回排序后的序号
            sorted_idx = torch.argsort(layer_values, descending=sort_descending).tolist()
            
            # 将每一层的排序结果保存
            sorted_indices[stat_type].append(sorted_idx)
    
    return sorted_indices

# 对每个统计类型的张量进行排序，并获取排序后的序号
sorted_indices = sort_head_by_abs(stats_tensor)

# 输出排序结果
for stat_type, sorted_list in sorted_indices.items():
    print(f"{stat_type} sorted indices:")
    for layer_idx, indices in enumerate(sorted_list):
        print(f"Layer {layer_idx}: {indices}")
