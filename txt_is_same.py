def are_files_identical(file1_path, file2_path):
    """判断两个文件的内容是否相同"""
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            # 逐行比较两个文件
            for line1, line2 in zip(file1, file2):
                if line1 != line2:
                    return False

            # 检查文件是否有剩余内容
            for remaining_line in file1:
                return False
            for remaining_line in file2:
                return False

        return True  # 所有内容都相同

    except FileNotFoundError:
        print(f"其中一个文件没有找到：{file1_path} 或 {file2_path}")
        return False

# 使用示例
file1_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy/203/head_outputs2.txt"
file2_path = "/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy/203/head_outputs.txt"

if are_files_identical(file1_path, file2_path):
    print("两个文件内容相同。")
else:
    print("两个文件内容不同。")
