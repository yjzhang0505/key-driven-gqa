# import torch
# import numpy as np
# import pandas as pd
# import re
# from _5_1_reading_groups import save_to_txt

# def calculate_group_score(groups, group_list, similarity_matrix):
#     """
#     计算每个组的匹配得分。
#     - 完全匹配的组得分为1/6；
#     - 部分匹配的组得分为1/6 * similarity_matrix中的相似度；
#     - 完全不匹配的组得分为0。
#     """
#     group_score = 1 / len(groups)  # 每个组的满分为1/6
#     total_score = 0

#     # 跟踪已匹配的组索引，避免重复计算
#     matched_indices = set()

#     # 1. 完全匹配优先
#     for g in groups:
#         if g in group_list or tuple(reversed(g)) in group_list:
#             # 完全匹配
#             total_score += group_score
#             matched_indices.add(group_list.index(g) if g in group_list else group_list.index(tuple(reversed(g))))

#     # 2. 部分匹配，优先寻找类似组
#     for i, g in enumerate(groups):
#         if i in matched_indices:
#             continue  # 跳过已完全匹配的组

#         best_partial_score = 0
#         best_partial_index = None

#         for j, og in enumerate(group_list):
#             if j in matched_indices:
#                 continue  # 跳过已完全匹配的组

#             if set(g) & set(og):  # 部分匹配
#                 # 从相似矩阵中获取相似度分值
#                 idx1, idx2 = g[0], g[1]
#                 jdx1, jdx2 = og[0], og[1]
#                 similarity_value = similarity_matrix[idx1, jdx1].item() if idx1 == jdx1 or idx2 == jdx2 else similarity_matrix[idx1, jdx2].item()
#                 partial_score = group_score * similarity_value
                
#                 # 更新最佳部分匹配的分数和索引
#                 if partial_score > best_partial_score:
#                     best_partial_score = partial_score
#                     best_partial_index = j

#         # 如果找到最佳部分匹配，则添加得分并标记为已匹配
#         if best_partial_index is not None:
#             total_score += best_partial_score
#             matched_indices.add(best_partial_index)

#     return total_score

# def check_groupings(groups, all_groupings, paths, similarity_matrix):
#     """
#     检查 groups 是否在 all_groupings 中，返回与其相似度最高的分组方案的路径及相似得分。
#     """
#     best_match_path = None
#     highest_score = 0

#     for idx, group_list in enumerate(all_groupings):
#         if groups == group_list or groups == [tuple(reversed(g)) for g in group_list]:
#             # 完全匹配时返回路径和满分
#             return paths[idx], 1.0
        
#         # 计算当前组的相似得分
#         score = calculate_group_score(groups, group_list, similarity_matrix)

#         # 若得分更高，或与最高得分相同但位置更靠前，更新最高得分的分组方案
#         if score > highest_score or (score == highest_score and best_match_path is None):
#             highest_score = score
#             best_match_path = paths[idx]

#     return best_match_path, highest_score

# def check_and_log_grouping(output_txt_path, layer_idx, matrix_type, label, sorted_groups, all_groupings, paths, method, similarity_matrix):
#     """
#     检查分组是否在 all_groupings 中并保存结果。
#     - 若找到完全匹配的方案，记录路径地址；
#     - 若找不到完全匹配方案，计算分组得分，返回得分最高的方案的地址和得分。
#     """
#     path_found, score = check_groupings(sorted_groups, all_groupings, paths, similarity_matrix)
#     if score == 1.0:
#         # 完全匹配的情况
#         save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} {method} sorted groups for {label}: {sorted_groups} - Found at: {path_found}")
#     else:
#         # 不完全匹配的情况，记录最高得分的方案
#         save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} {method} sorted groups for {label}: {sorted_groups} - Closest match at: {path_found} with score: {score:.2f}")
import torch
import numpy as np
import pandas as pd
import re
from _5_1_reading_groups import save_to_txt

def find_most_similar_group(groups, all_groupings, paths):
    """
    按顺序逐步筛选出与 groups 最相似的分组方案。
    - 逐步检查每一个分组，筛选包含该分组的方案，逐步缩小候选范围。
    - 如果没有找到完全匹配的项，则不筛选，继续检查下一组。
    - 最后若有多个候选方案，返回第一个方案。
    """
    candidates = list(zip(all_groupings, paths))  # 初始候选为所有方案

    # 按顺序逐步筛选
    for g in groups:
        new_candidates = []
        for group_list, path in candidates:
            # 如果当前分组 g 存在于 group_list 中，则保留该方案为新的候选
            if g in group_list or tuple(reversed(g)) in group_list:
                new_candidates.append((group_list, path))

        # 如果找到匹配项，则更新候选列表
        if new_candidates:
            candidates = new_candidates
        # 如果没有找到匹配项，则不做筛选，继续检查下一组

        # 如果候选数量减少到1，则提前退出循环
        if len(candidates) == 1:
            break

    # 返回候选方案中最前的一个，作为最相似的方案
    if candidates:
        best_match_group, best_match_path = candidates[0]
        return best_match_path
    else:
        # 如果没有找到候选方案，则返回空路径
        return None

def check_and_log_grouping(output_txt_path, layer_idx, matrix_type, label, sorted_groups, all_groupings, paths, method):
    """
    检查分组是否在 all_groupings 中并保存结果。
    - 若找到完全匹配的方案，记录路径地址；
    - 若找不到完全匹配方案，通过逐步筛选找到最相似的方案。
    """
    # 查找最相似的分组方案
    path_found = find_most_similar_group(sorted_groups, all_groupings, paths)
    if path_found:
        # 如果找到最相似的分组方案，记录路径
        save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} {method} sorted groups for {label}: {sorted_groups} - Closest match at: {path_found}")
    else:
        # 如果找不到最相似的方案
        save_to_txt(output_txt_path, f"Layer {layer_idx} - {matrix_type} {method} sorted groups for {label}: {sorted_groups} - No similar grouping found")

