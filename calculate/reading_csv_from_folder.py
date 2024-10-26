import os
import pandas as pd

# 定义要处理的路径和输出文件路径
# folders_proxy = list(range(1, 102)) + list(range(200, 204))
# folders_concrete_proxy = list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)) + list(range(51, 57))
# folders_concrete_mean_var = list(range(101, 116)) + list(range(201, 216)) + list(range(301, 316)) + list(range(401, 416)) + list(range(501, 516))
folders_more = list(range(1, 1001))

base_paths = {
    # '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy': folders_proxy,
    # '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/proxy': folders_concrete_proxy,
    # '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/concrete/mean_var': folders_concrete_mean_var,
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
        if num1 > num2:
            num1, num2 = num2, num1
        formatted_groups.append((num1, num2))
    formatted_groups.sort(key=lambda x: (x[0], x[1]))
    return formatted_groups

# 函数用于读取 run.csv 文件中的 test_acc
def read_test_acc(run_csv_path):
    try:
        df = pd.read_csv(run_csv_path, skiprows=1, header=None, names=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        if not df.empty:
            return df['test_acc'].iloc[-1]  # 获取最后一个epoch的test_acc
        else:
            print(f"文件 {run_csv_path} 是空的。")
            return None
    except pd.errors.EmptyDataError:
        print(f"文件 {run_csv_path} 没有有效数据。")
        return None
    except Exception as e:
        print(f"读取文件 {run_csv_path} 时出错: {e}")
        return None

# 存储所有方案的分组、路径和test_acc
results = []

# 遍历所有路径，读取分组和test_acc
for base_path, folders in base_paths.items():
    for folder in folders:
        group_file_path = os.path.join(base_path, str(folder), 'group.txt')
        run_csv_path = os.path.join(base_path, str(folder), 'run.csv')
        
        if os.path.exists(group_file_path) and os.path.exists(run_csv_path):
            formatted_group = read_group_file(group_file_path)
            test_acc = read_test_acc(run_csv_path)
            if test_acc is not None:
                results.append({
                    'group': formatted_group,
                    'path': os.path.dirname(group_file_path),
                    'test_acc': test_acc
                })

# 按test_acc从大到小排序
results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

# 保存到输出文件，每行格式为“分组, test_acc, 路径”
with open(output_file, 'w') as outfile:
    for result in results:
        formatted_group = ' '.join([f'({num1}, {num2})' for num1, num2 in result['group']])
        outfile.write(f"{formatted_group}, {result['test_acc']}, {result['path']}\n")

print(f'分组、路径和 test_acc 已按 test_acc 从大到小排序并保存到 {output_file}')
