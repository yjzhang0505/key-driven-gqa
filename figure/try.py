import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 设置根路径
base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'
base_path2 = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share'
l2_norms_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/l2_norms.txt'

# 初始化字典存储每组(a, b)及其所有的test_acc
group_acc_dict = defaultdict(list)
# 初始化字典存储L2范数
group_l2_norms_dict = {}

# 遍历1到10和51到64的文件夹
for folder_num in list(range(1, 50)) + list(range(51, 101)):
    folder_path2 = os.path.join(base_path2, str(folder_num))
    folder_path = os.path.join(base_path, str(folder_num))
    
    # 获取run.csv的最后一行中的test_acc
    run_csv_path = os.path.join(folder_path2, 'run.csv')
    if os.path.exists(run_csv_path):
        run_df = pd.read_csv(run_csv_path)
        test_acc = run_df.iloc[-1]['test_acc']  # 读取最后一行的test_acc
        test_acc = np.exp(700 * test_acc) / np.array(1e187)
    else:
        print(f"文件不存在: {run_csv_path}")
        continue

    # 获取group.txt的分组情况
    group_txt_path = os.path.join(folder_path, 'group.txt')
    if os.path.exists(group_txt_path):
        with open(group_txt_path, 'r') as f:
            for line in f:
                group = list(map(int, line.strip().split(',')))  # 将每一行读取为一组
                if len(group) == 2:
                    a, b = group
                    group_acc_dict[(a, b)].append(test_acc)
                    group_acc_dict[(b, a)].append(test_acc)  # 对称存储
    else:
        print(f"文件不存在: {group_txt_path}")
        continue

# 读取L2范数文件
if os.path.exists(l2_norms_path):
    with open(l2_norms_path, 'r') as f:
        for line in f:
            key, values = line.strip().split(':')
            l2_norms = list(map(float, values.split(',')))  # 将字符串转换为浮点数列表
            group_l2_norms_dict[key.strip()] = l2_norms  # 存储L2范数向量

# 创建矩阵形式表示
unique_groups = sorted(set([a for a, _ in group_acc_dict.keys()]))  # 提取唯一的a,b组合
matrix_size = len(unique_groups)
acc_matrix = np.zeros((matrix_size, matrix_size))

# 填充矩阵
for (a, b), acc_list in group_acc_dict.items():
    avg_acc = np.mean(acc_list)  # 计算每个组合的平均test_acc
    acc_matrix[a-1, b-1] = avg_acc  # 将(a,b)的平均test_acc放入矩阵中

# 打印Test Accuracy矩阵
print("Test Accuracy Matrix:")
print(pd.DataFrame(acc_matrix, index=unique_groups, columns=unique_groups))

# 初始化字典以存储每个 L2 范数的重新排序矩阵
reordered_matrices = {}

# 初始化一个列表以存储每个 L2 范数的 axis
axis_list = []

# 打印每组的L2范数并生成矩阵
for key, l2_norm in group_l2_norms_dict.items():
    # 生成第一个维度：0到11
    first_dimension = np.arange(12)
    
    # 第二个维度：L2范数
    second_dimension = np.array(l2_norm)
    
    # 第三个维度：根据L2范数的大小排序
    sorted_indices = np.argsort(second_dimension)[::-1]  # 从大到小排序
    
    # 生成axis向量，Rank位置赋值为Index的值
    axis = np.zeros(12, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        axis[rank] = first_dimension[idx]  # 在Rank的位置上填充Index值
    
    # 将当前的 axis 保存到列表中
    axis_list.append(axis)
    
    # 创建重新排序后的acc_matrix
    reordered_acc_matrix = acc_matrix[axis][:, axis]
    
    # 存储重新排序的矩阵到字典
    reordered_matrices[key] = reordered_acc_matrix
    
    print(f"Group: {key}")
    print("Reordered Test Accuracy Matrix:")
    print(pd.DataFrame(reordered_acc_matrix, index=axis, columns=axis))
    print("\n")

# 设置路径
output_path = '/data/yjzhang/desktop/try/key-driven-gqa/figure'
os.makedirs(output_path, exist_ok=True)  # 创建文件夹（如果不存在）

# 创建画布
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2行3列的子图

# 绘制热图，第一张是原始的 test_acc
sns.heatmap(acc_matrix, annot=False, fmt=".2f", cmap='Greens', 
            xticklabels=unique_groups, yticklabels=unique_groups, ax=axes[0, 0])
axes[0, 0].set_title('Initial Test Acc')
axes[0, 0].set_xlabel('Group B')
axes[0, 0].set_ylabel('Group A')
axes[0, 0].invert_yaxis()  # 反转y轴

# 绘制剩余的五个热图
titles = list(reordered_matrices.keys())
for ax, title, axis in zip(axes.flat[1:], titles, axis_list):
    sns.heatmap(reordered_matrices[title], annot=False, fmt=".2f", cmap='Greens', 
                xticklabels=axis, yticklabels=axis, ax=ax)  # 使用对应的 axis
    ax.set_title(f'{title} Test Acc')
    ax.set_xlabel('Group B')
    ax.set_ylabel('Group A')
    ax.invert_yaxis()  # 反转y轴

# 调整布局
plt.tight_layout()

# 保存图片
image_path = os.path.join(output_path, 'test_acc_multiple.png')
plt.savefig(image_path, bbox_inches='tight')
plt.close()  # 关闭图形，以节省内存

print(f"热图副本已保存到: {image_path}")
