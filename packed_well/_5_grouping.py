import torch
import numpy as np
import pandas as pd
from _4_ordering import load_stats_from_model, get_importance_orders
from _1_reading_ckpt import YourTransformerModel
from _3_similarity import calculate_similarity
from _5_1_reading_groups import read_groupings, check_groupings
from _5_2_matching_groups import check_and_log_grouping

# 文件路径
file_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt"  # TXT 文件路径
# excel_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/dustbin/not_shared_similarity_matrices_all_layers.xlsx'  # Excel 文件路径
output_txt_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/dustbin/_grouping_mean_var.txt'  # 输出txt文件路径
groupings_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt'  # 分组方案文件路径

# 加载统计数据并进行排序
model = YourTransformerModel(num_heads=12, dim=768)  # 假设您在 _1_reading_ckpt 中定义了 YourTransformerModel
model.load_pretrained_qkv_weights()
stats_tensor = load_stats_from_model(model)

similarity_matrices = calculate_similarity(model)

# 加载所有子表格中的相似性矩阵
# similarity_matrices = load_similarity_matrix_from_excel(excel_file_path)

# 保存所有输出到txt文件
def save_to_txt(output_path, content):
    with open(output_path, 'a') as f:
        f.write(content + '\n')

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


# 对12层进行分组计算并保存结果
def process_and_save_all_layers(stats_tensor, similarity_matrices, output_txt_path, all_groupings, paths):
    """
    对12层都进行分组计算，并将结果保存到txt文件中。每层使用相应的相似性矩阵。
    """
    matrix_types = ['K_similarity_matrix', 'Q_similarity_matrix', 'V_similarity_matrix', 'KxQ_similarity_matrix', 'KxQxV_similarity_matrix']

    for layer_idx in range(12):
        save_to_txt(output_txt_path, f"\nProcessing Layer {layer_idx}")
        
        # 获取重要性排序
        importance_orders = get_importance_orders(stats_tensor, layer_idx)

        for matrix_type in matrix_types:
            layer_key = f'Layer_{layer_idx}'
            similarity_matrix = similarity_matrices.get(layer_key, {}).get(matrix_type)

            if similarity_matrix is None:
                print(f"Warning: Similarity matrix for {layer_key} and {matrix_type} not found.")
                continue

            save_to_txt(output_txt_path, f"\nProcessing {matrix_type} for Layer {layer_idx}")

            for label, importance_order in importance_orders.items():
                # 使用重要性顺序进行分组
                groups_importance = importance_prioritized(similarity_matrix, importance_order)
                sorted_groups = sort_groups(groups_importance)

                # 调用分组检查和记录函数，并传入相似矩阵
                check_and_log_grouping(output_txt_path, layer_idx, matrix_type, label, sorted_groups, all_groupings, paths, "Importance-prioritized")


            # 使用最高相似性分组方法
            groups_highest_similarity = highest_similarity(similarity_matrix)
            sorted_groups_similarity = sort_groups(groups_highest_similarity)

            # 调用分组检查和记录函数，并传入相似矩阵
            check_and_log_grouping(output_txt_path, layer_idx, matrix_type, "Highest similarity", sorted_groups_similarity, all_groupings, paths, "Highest similarity")

    save_to_txt(output_txt_path, "\nAll processing complete.")



all_groupings, paths = read_groupings(groupings_file_path)

# 运行分组计算并保存
process_and_save_all_layers(stats_tensor, similarity_matrices, output_txt_path, all_groupings, paths)

