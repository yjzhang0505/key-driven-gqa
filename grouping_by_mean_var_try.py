import torch
import numpy as np
import pandas as pd
import re

# 读取文件中的分组方案
def read_groupings(file_path):
    """
    读取文件中的分组情况，并将其转换为与生成的分组格式一致的列表格式。
    返回分组和相应的路径地址。
    """
    all_groupings = []
    paths = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if '<--' in line:
            # 获取 <-- 前的分组部分
            group_str = line.split('<--')[0].strip()
            # 获取 <-- 后的路径部分
            path_str = line.split('<--')[1].strip()
            
            # 将字符串格式的分组转换为列表形式
            # e.g., "(0, 1) (2, 4)" -> [(0, 1), (2, 4)]
            group_list = re.findall(r'\((\d+), (\d+)\)', group_str)
            group_list = [(int(a), int(b)) for a, b in group_list]
            
            all_groupings.append(group_list)
            paths.append(path_str)
    
    return all_groupings, paths


# 查找分组是否在已有分组列表中
def check_groupings(groups, all_groupings, paths):
    """
    检查 groups 中的每一行分组是否在 all_groupings 中，并返回匹配的路径地址。
    """
    for idx, group_list in enumerate(all_groupings):
        if groups == group_list or groups == [tuple(reversed(g)) for g in group_list]:
            return paths[idx]  # 返回对应的路径地址
    return None


# 载入并排序函数
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

    # 将统计量转换为张量，每个层有12个头
    stats_tensor = {key: torch.tensor(value).view(-1, 12) for key, value in stats.items()}  
    return stats_tensor



def sort_head_by_abs(stats_tensor, layer_idx):
    """
    对指定层的每个头的张量进行排序。
    - 对于 mean，按值从大到小排序。
    - 对于 var，按值从小到大排序。
    """
    sorted_indices = {}
    
    for stat_type, tensor in stats_tensor.items():
        # 判断排序类型
        if 'mean' in stat_type:  # mean从大到小排序
            sort_descending = True
        elif 'var' in stat_type:  # var从小到大排序
            sort_descending = False
        else:
            raise ValueError(f"Unrecognized stat_type: {stat_type}")
        
        # 对指定层进行排序
        layer_values = tensor[layer_idx]
        sorted_idx = torch.argsort(layer_values, descending=sort_descending).tolist()
        sorted_indices[stat_type] = sorted_idx
    
    return sorted_indices

# 替换后的 load_mean_var 函数
def load_mean_var(sorted_indices):
    """
    返回由 sorted_indices 生成的动态排序序列。
    """
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

# 分组函数
def importance_prioritized(similarity_matrix, importance_order):
    """
    第一种分组方法：给定重要性的顺序，从最重要的行开始分组，不能和自己分为一组。
    确保每个头只能被分到一个组。
    """
    n = similarity_matrix.shape[0]
    groups = []
    active_rows = set(importance_order)  # 按重要性排序的行的集合
    active_cols = set(range(n))  # 所有列都最开始是活跃的

    for row in importance_order:
        if row not in active_rows:
            continue
        
        # 如果没有更多活跃的列，则跳过
        available_cols = [c for c in active_cols if c != row]
        if not available_cols:
            continue

        # 找到该行中相似性最高的列，且该列不能是自己
        col = max(available_cols, key=lambda c: similarity_matrix[row, c])
        
        # 添加组：row 和相似性最高的列 col
        groups.append((row, col))

        # 失活：row 和 col 不再参与后续的分组
        if row in active_rows:
            active_rows.remove(row)
        if col in active_rows:
            active_rows.remove(col)
        if row in active_cols:
            active_cols.remove(row)
        if col in active_cols:
            active_cols.remove(col)
    
    return groups

def sort_groups(groups):
    """
    对分组进行排序：
    - 组内从小到大排列
    - 组间按第一个数从小到大排列
    """
    # 组内从小到大
    sorted_groups = [tuple(sorted(group)) for group in groups]
    # 组间按第一个数从小到大排序
    sorted_groups.sort(key=lambda x: x[0])
    return sorted_groups

def highest_similarity(similarity_matrix):
    """
    第二种分组方法：遍历矩阵，找出当前相似性最高的两个头分为一组，不能和自己分为一组。
    """
    n = similarity_matrix.shape[0]
    groups = []
    active = set(range(n))  # 活跃的行和列集合

    while len(active) > 1:
        # 找到相似性最大的两个头，且不能是自己
        max_sim = -1
        max_pair = (-1, -1)
        
        for i in active:
            for j in active:
                if i != j and similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    max_pair = (i, j)

        # 添加组：相似性最大的两个头 i 和 j
        i, j = max_pair
        groups.append((i, j))

        # 失活：i 和 j 的行和列不再参与分组
        active.remove(i)
        active.remove(j)

    return groups

# 从 Excel 文件中读取 12x12 相似性矩阵
def load_similarity_matrix_from_excel(file_path):
    """
    遍历 Excel 文件中的所有子表格，逐个读取相似性矩阵。
    """
    # 使用 pandas 的 ExcelFile 类来加载多个子表格
    xl = pd.ExcelFile(file_path)
    similarity_matrices = {}

    # 遍历所有子表格
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, header=None)
        similarity_matrices[sheet_name] = df.values  # 将 DataFrame 转换为 numpy 数组

    return similarity_matrices

# 保存所有输出到txt文件
def save_to_txt(output_path, content):
    with open(output_path, 'a') as f:
        f.write(content + '\n')

# 对12层进行分组计算并保存结果
def process_and_save_all_layers(stats_tensor, similarity_matrices, output_txt_path, all_groupings, paths):
    """
    对12层都进行分组计算，并将结果保存到txt文件中。每层使用相应的相似性矩阵。
    """
    # 预定义每个层使用的相似性矩阵标签
    matrix_types = ['K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix', 'KxQ_similarity_matrix', 'KxQxV_similarity_matrix']

    # 对所有12层分别执行操作
    for layer_idx in range(12):
        save_to_txt(output_txt_path, f"\nProcessing Layer {layer_idx}")

        # 根据当前层的数据进行排序
        sorted_indices = sort_head_by_abs(stats_tensor, layer_idx)

        # 获取当前层的排序顺序
        importance_orders = load_mean_var(sorted_indices)

        # 遍历5个相似性矩阵类型，分别处理
        for i, matrix_type in enumerate(matrix_types):
            # 选择当前层的相应相似性矩阵，假设每个层有5个表格，顺序保存
            sheet_name = f"Layer_{layer_idx}_{matrix_type}"
            similarity_matrix = similarity_matrices[sheet_name]

            save_to_txt(output_txt_path, f"\nProcessing {matrix_type} for Layer {layer_idx}")

            # 按重要性顺序分组
            for label, importance_order in importance_orders.items():
                # 使用重要性顺序进行分组
                groups_importance = importance_prioritized(similarity_matrix, importance_order)

                # 排序分组
                sorted_groups = sort_groups(groups_importance)

                # 检查分组是否已经在已有分组中出现过
                path_found = check_groupings(sorted_groups, all_groupings, paths)
                if path_found:
                    save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} Importance-prioritized sorted groups for {label}: {sorted_groups} - Found at: {path_found}")
                else:
                    save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} Importance-prioritized sorted groups for {label}: {sorted_groups} - Not found")

            # 按最高相似性分组
            groups_highest_similarity = highest_similarity(similarity_matrix)
            sorted_groups_similarity = sort_groups(groups_highest_similarity)

            # 检查分组是否已经在已有分组中出现过
            path_found = check_groupings(sorted_groups_similarity, all_groupings, paths)
            if path_found:
                save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} Highest similarity sorted groups: {sorted_groups_similarity} - Found at: {path_found}")
            else:
                save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} Highest similarity sorted groups: {sorted_groups_similarity} - Not found")

    save_to_txt(output_txt_path, "\nAll processing complete.")



# 文件路径
file_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt"  # TXT 文件路径
excel_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/not_shared_similarity_matrices_all_layers.xlsx'  # Excel 文件路径
output_txt_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/dustbin/grouping_mean_var.txt'  # 输出txt文件路径
groupings_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_aline.txt'  # 分组方案文件路径

# 加载统计数据并进行排序
stats_tensor = load_stats_from_txt(file_path)

# 加载所有子表格中的相似性矩阵
similarity_matrices = load_similarity_matrix_from_excel(excel_file_path)

# 加载已有的分组方案和路径
all_groupings, paths = read_groupings(groupings_file_path)

# 对12层进行分组计算并保存结果
process_and_save_all_layers(stats_tensor, similarity_matrices, output_txt_path, all_groupings, paths)