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

def check_groupings(groups, all_groupings, paths):
    """
    检查 groups 中的每一行分组是否在 all_groupings 中，并返回匹配的路径地址。
    """
    for idx, group_list in enumerate(all_groupings):
        if groups == group_list or groups == [tuple(reversed(g)) for g in group_list]:
            return paths[idx]  # 返回对应的路径地址
    return None

# 保存所有输出到txt文件
def save_to_txt(output_path, content):
    with open(output_path, 'a') as f:
        f.write(content + '\n')



# 文件路径
groupings_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_aline.txt'  # 分组方案文件路径


# 加载已有的分组方案和路径
all_groupings, paths = read_groupings(groupings_file_path)

# print(all_groupings)