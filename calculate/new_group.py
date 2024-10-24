import os
import random

# 要生成的100个新文件夹路径
output_base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/more'

# 定义要生成的文件夹范围 101-200
folders = list(range(101, 1001))

# 读取已存在的分组情况（已生成的txt文件路径）
grouping_txt = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_aline.txt'
existing_groups = set()

# 读取现有分组的文件并存储为集合，避免重复
with open(grouping_txt, 'r') as f:
    for line in f:
        # 提取每一行的分组部分，移除括号等符号
        group_part = line.split('<--')[0].strip()
        # 移除可能的格式符号，只保留数字部分
        existing_groups.add(group_part.replace('(', '').replace(')', '').replace(',', '').replace(' ', ''))

# 生成新的分组，确保与现有的不重复
def generate_unique_group(existing_groups):
    while True:
        # 生成0-11的数字列表并打乱顺序
        numbers = list(range(12))
        random.shuffle(numbers)
        
        # 将12个数字分成6组，每组两个数字
        new_pairs = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]
        
        # 确保每组内数字小的在前
        new_pairs = [(min(a, b), max(a, b)) for a, b in new_pairs]
        
        # 将组按第一个数字排序
        new_pairs.sort(key=lambda x: (x[0], x[1]))
        
        # 将生成的组格式化为字符串，不包含任何符号
        formatted_group = ' '.join([f'{a} {b}' for a, b in new_pairs])
        
        # 确保生成的分组不与现有的重复
        if formatted_group not in existing_groups:
            return formatted_group

# 创建新文件夹并生成group.txt文件
for folder in folders:
    folder_path = os.path.join(output_base_path, str(folder))
    
    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)
    
    # 生成新的、不重复的分组
    new_group = generate_unique_group(existing_groups)
    
    # 打印生成的分组，方便调试
    print(f"Generated group for folder {folder}: {new_group}")
    
    # 将新的分组加入已存在分组集合，避免重复
    existing_groups.add(new_group)
    
    # 将分组写入group.txt，每两个数字为一行，逗号分隔
    group_file_path = os.path.join(folder_path, 'group.txt')
    with open(group_file_path, 'w') as f:
        new_group_numbers = new_group.split()
        for i in range(0, len(new_group_numbers), 2):
            f.write(f'{new_group_numbers[i]}, {new_group_numbers[i+1]}\n')

print(f"100 unique groupings have been generated and saved to folders {folders[0]}-{folders[-1]}.")
