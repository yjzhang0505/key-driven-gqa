import os

# 定义要处理的路径
folders_proxy = list(range(1, 102)) + list(range(200, 204))  # 包含1到101，和200到203的文件夹
folders_concrete_proxy = list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)) + list(range(51, 57))
folders_concrete_mean_var = list(range(101, 116)) + list(range(201, 216)) + list(range(301, 316)) + list(range(401, 416)) + list(range(501, 516))
folders_more = list(range(1, 1000))

base_paths = {
    '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy': folders_proxy,
    '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/proxy': folders_concrete_proxy,
    '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/mean_var': folders_concrete_mean_var,
    '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/more': folders_more
}

output_file = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt'

# 函数用于读取group.txt并格式化内容
def read_group_file(group_file_path):
    with open(group_file_path, 'r') as f:
        groups = f.readlines()
    # 解析并格式化每行，确保每对数字中的小的在左边，大的在右边，并按第一个数排序
    formatted_groups = []
    for line in groups:
        num1, num2 = map(int, line.strip().split(','))
        # 确保组内较小的数字在前
        if num1 > num2:
            num1, num2 = num2, num1
        formatted_groups.append((num1, num2))
    
    # 对行内组按照组内第一个数排序，如果相同按第二个数排序
    formatted_groups.sort(key=lambda x: (x[0], x[1]))
    
    return formatted_groups

# 自定义行排序规则：将每一行的组展开成一个排序键，依次比较各组的各个数
def custom_sort_key(group_line):
    return [item for group in group_line for item in group]

# 打开目标输出文件
with open(output_file, 'w') as outfile:
    all_group_lines = []
    paths = []  # 保存路径
    for base_path, folders in base_paths.items():
        for folder in folders:
            group_file_path = os.path.join(base_path, str(folder), 'group.txt')
            if os.path.exists(group_file_path):
                formatted_group = read_group_file(group_file_path)
                all_group_lines.append(formatted_group)
                # 记录group.txt的上一级路径
                paths.append(os.path.dirname(group_file_path))
    
    # 对所有行进行全局排序，按照自定义排序规则
    all_group_lines_with_paths = sorted(zip(all_group_lines, paths), key=lambda x: custom_sort_key(x[0]))
    
    # 用于存储已经出现过的完整分组内容及其路径
    seen_groups = {}

    # 将排序后的结果写入目标文件，每行格式化为 (num1, num2) 并加上路径
    for group_line, group_path in all_group_lines_with_paths:
        # 使用 tuple(group_line) 作为唯一标识符
        group_line_key = tuple(group_line)  # 将分组方案作为唯一键
        
        # 格式化组成为 (num1, num2)
        formatted_line = ' '.join([f'({num1}, {num2})' for num1, num2 in group_line])
        
        # 如果该分组方案已经存在，追加路径
        if group_line_key in seen_groups:
            seen_groups[group_line_key].append(group_path)
        else:
            # 否则记录该分组和其路径
            seen_groups[group_line_key] = [group_path]
    
    # 将结果写入文件
    for group_line_key, group_paths in seen_groups.items():
        # 格式化输出分组行
        formatted_line = ' '.join([f'({num1}, {num2})' for num1, num2 in group_line_key])
        # 将所有路径用逗号连接
        outfile.write(f'{formatted_line} <-- {", ".join(group_paths)}\n')

print(f'Grouping information has been written to {output_file}')
