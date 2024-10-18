import os

def save_group_to_folder(base_path, folder_name, groups):
    """
    将给定的分组信息保存到指定路径中的 group.txt 文件。
    """
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, "group.txt")
    with open(file_path, 'w') as f:
        for group in groups:
            group_str = ",".join(map(str, group))  # 将每个组用逗号分隔，保存为字符串
            f.write(group_str + '\n')


def get_folder_number(sheet_name, label, is_highest_similarity=False):
    """
    根据 sheet_name 和 label 生成文件夹编号。
    sheet_name 决定前缀，label 决定具体的文件夹编号。
    """
    sheet_prefix_mapping = {
        'K_similarity_matrix': 10,
        'Q_similarity_matrix': 20,
        'V_similarity_matrix': 30,
        'KxQ_similarity_matrix': 40,
        'KxQxV_similarity_matrix': 50
    }
    
    label_mapping = {
        'K_L2': 1,
        'Q_L2': 2,
        'V_L2': 3,
        'KxQ_L2': 4,
        'KxQxV_L2': 5
    }
    
    # 获取 sheet 的前缀
    prefix = sheet_prefix_mapping.get(sheet_name, 0)
    
    if is_highest_similarity:
        return f"{prefix + 6}"  # Highest similarity always goes to the folder ending with '6'

    # 获取文件夹编号，第一组是 11-16，第二组是 21-26，以此类推
    folder_num = label_mapping.get(label, 0)
    return f"{prefix + folder_num}"  # 文件夹编号为 11-16, 21-26, 31-36, 依次类推


def parse_group_line(group_line):
    """
    从 'Importance-prioritized groups for K_L2: [(7, 6), (11, 3), ...]' 格式的行中提取出分组信息。
    返回 [(7, 6), (11, 3), ...] 的列表。
    """
    start_idx = group_line.find("[")  # 找到列表开始的位置
    groups_str = group_line[start_idx:].strip()  # 提取出 '[...]' 部分
    groups = eval(groups_str)  # 将字符串转换为列表对象
    return groups


def process_file(input_file, output_base_path):
    """
    处理输入文件中的分组信息，并将每种分组信息保存到指定的文件夹中。
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    current_sheet = None
    current_group_data = {key: [] for key in ['K_L2', 'Q_L2', 'V_L2', 'KxQ_L2', 'KxQxV_L2']}  # 用于临时存储每种分组信息
    current_group_data['Highest similarity groups'] = []  # 添加最高相似性组的键名

    for line in lines:
        line = line.strip()
        
        # 检测到新 sheet 的处理
        if line.startswith("Processing sheet:"):
            # 如果之前有分组信息，保存到对应的文件夹
            if current_sheet:
                for label, group_data in current_group_data.items():
                    if label != "Highest similarity groups":  # 单独处理最高相似性组
                        folder_name = get_folder_number(current_sheet, label)
                        # 将分组数据提取并保存
                        for group_line in group_data:
                            groups = parse_group_line(group_line)
                            save_group_to_folder(output_base_path, folder_name, groups)

                # 处理最高相似性的情况
                folder_name = get_folder_number(current_sheet, None, is_highest_similarity=True)
                highest_similarity_data = current_group_data.get('Highest similarity groups', [])
                for group_line in highest_similarity_data:
                    groups = parse_group_line(group_line)
                    save_group_to_folder(output_base_path, folder_name, groups)

            # 清空当前分组数据
            current_group_data = {key: [] for key in ['K_L2', 'Q_L2', 'V_L2', 'KxQ_L2', 'KxQxV_L2']}
            current_group_data['Highest similarity groups'] = []  # 重新初始化最高相似性组
            current_sheet = line.replace("Processing sheet: ", "")
        
        # 处理不同的分组情况并添加到对应的字典项
        elif line.startswith("Importance-prioritized groups for "):
            for label in current_group_data.keys():
                if f"Importance-prioritized groups for {label}" in line:
                    current_group_data[label].append(line)
        
        # 处理最高相似性的情况
        elif line.startswith("Highest similarity groups:"):
            current_group_data['Highest similarity groups'].append(line)
    
    # 保存最后一个 sheet 的分组信息
    if current_sheet:
        for label, group_data in current_group_data.items():
            if label != "Highest similarity groups":
                folder_name = get_folder_number(current_sheet, label)
                for group_line in group_data:
                    groups = parse_group_line(group_line)
                    save_group_to_folder(output_base_path, folder_name, groups)

        folder_name = get_folder_number(current_sheet, None, is_highest_similarity=True)
        highest_similarity_data = current_group_data.get('Highest similarity groups', [])
        for group_line in highest_similarity_data:
            groups = parse_group_line(group_line)
            save_group_to_folder(output_base_path, folder_name, groups)


# 文件路径配置
input_file = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/grouping_results.txt'
output_base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete'

# 运行处理
process_file(input_file, output_base_path)
