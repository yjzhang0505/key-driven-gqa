import os
import shutil

# 已生成的txt文件路径
grouping_txt = '/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_aline.txt'

# 目标路径
output_base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/more'

# 读取文件并复制前100行的第一个路径对应的文件夹
with open(grouping_txt, 'r') as f:
    for line_number, line in enumerate(f, start=1):
        if line_number > 100:
            break  # 只处理前100行
        
        # 获取第一个路径（以逗号分隔）
        path_part = line.split('<--')[-1].strip()
        first_path = path_part.split(',')[0].strip()  # 获取第一个路径

        # 确保第一个路径存在
        if os.path.exists(first_path):
            # 构建目标文件夹路径，以行号命名
            target_folder = os.path.join(output_base_path, str(line_number))
            
            # 复制文件夹并重命名为行号
            shutil.copytree(first_path, target_folder)
            print(f"Copied folder {first_path} to {target_folder}")
        else:
            print(f"Path {first_path} does not exist for line {line_number}")

print("Copying complete for the first 100 lines.")
