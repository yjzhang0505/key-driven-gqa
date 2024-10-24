import numpy as np
import re

def read_groupings(file_path):
    groupings_list = []
    
    # 打开文件并读取每一行
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取分组情况，即(数字, 数字)部分
            groupings = re.findall(r'\((\d+),\s*(\d+)\)', line)
            # 将分组转换为整数并存入列表
            groupings = [(int(x), int(y)) for x, y in groupings]
            groupings_list.append(groupings)
    
    # 将所有分组转换为张量形式（NumPy数组）
    tensor = np.array(groupings_list)
    
    return tensor

# 示例用法
file_path = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt'
tensor = read_groupings(file_path)

# 打印张量
print(tensor)
