import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置根路径
base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'
base_path2 = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share'
l2_norms_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/l2_norms.txt'

# 初始化字典存储每组(a, b)及其所有的test_acc
group_acc_dict = defaultdict(list)

# 遍历1到10和51到64的文件夹
for folder_num in list(range(1, 50)) + list(range(51, 101)):
    folder_path2 = os.path.join(base_path2, str(folder_num))
    folder_path = os.path.join(base_path, str(folder_num))
    
    # 获取run.csv的最后一行中的test_acc
    run_csv_path = os.path.join(folder_path2, 'run.csv')
    if os.path.exists(run_csv_path):
        run_df = pd.read_csv(run_csv_path)
        test_acc = run_df.iloc[-1]['test_acc']  # 读取最后一行的test_acc
        test_acc = np.exp(500*test_acc)/np.array(1e187)
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

# 提取横纵坐标和颜色值的原始图
x_original = []
y_original = []
colors_original = []

for (a, b), acc_list in group_acc_dict.items():
    avg_acc = np.mean(acc_list)  # 计算每个组合的平均test_acc
    x_original.append(a)
    y_original.append(b)
    colors_original.append(avg_acc)

# 将test_acc归一化到0-1范围内，以便用于颜色映射
norm = plt.Normalize(vmin=min(colors_original), vmax=max(colors_original))

# 读取L2范数文件
l2_norms_dict = {}
if os.path.exists(l2_norms_path):
    with open(l2_norms_path, 'r') as f:
        for line in f:
            key, values = line.strip().split(':')
            l2_norms_dict[key.strip()] = list(map(float, values.split(',')))

# 获取排序数列
sorted_indices = {}
for key, values in l2_norms_dict.items():
    sorted_indices[key] = np.argsort(values)[::-1] + 1  # 返回排序后的索引，并将最大值对应为1

# 现在统一排序顺序：我们对每种L2范数生成相同的排序
# 将第一个L2范数的排序作为基准
reference_sorted_indices = sorted_indices[next(iter(sorted_indices))]

# 创建2行3列的子图
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.ravel()

# 绘制原始图像
scatter = axs[0].scatter(x_original, y_original, c=colors_original, cmap='Greens', norm=norm, s=100)
axs[0].set_title('Original Group-wise Pairings')
axs[0].set_xlabel('Group A (Original Head Order)')
axs[0].set_ylabel('Group B (Original Head Order)')
axs[0].set_xticks(list(set(x_original)))
axs[0].set_yticks(list(set(y_original)))
axs[0].set_xticklabels(list(set(x_original)))
axs[0].set_yticklabels(list(set(y_original)))
cbar = plt.colorbar(scatter, ax=axs[0])
cbar.set_label('Average Test Accuracy')

# 绘制根据不同L2范数排序后的图，并统一横纵坐标排序
for i, (key, indices) in enumerate(sorted_indices.items(), start=1):
    # 使用基准排序统一x和y的顺序
    sorted_x = [reference_sorted_indices[a - 1] for a in x_original]
    sorted_y = [reference_sorted_indices[b - 1] for b in y_original]

    # 绘制子图
    scatter = axs[i].scatter(sorted_x, sorted_y, c=colors_original, cmap='Greens', norm=norm, s=100)
    axs[i].set_title(f'Sorted by {key}')  # 使用L2范数的名称作为标题
    axs[i].set_xlabel('Group A (Sorted by L2 Norm)')
    axs[i].set_ylabel('Group B (Sorted by L2 Norm)')
    
    # 统一设置排序后的坐标标签，横纵坐标使用同样的排序
    sorted_labels = [str(idx) for idx in reference_sorted_indices]
    axs[i].set_xticks(list(set(sorted_x)))
    axs[i].set_yticks(list(set(sorted_y)))
    axs[i].set_xticklabels(sorted_labels, rotation=90)  # 横坐标旋转90度
    axs[i].set_yticklabels(sorted_labels)

    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label('Average Test Accuracy')

# 调整子图之间的布局
plt.tight_layout()

# 保存图像到文件 (保存为png格式)
output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/l2_norm.png'
plt.savefig(output_image_path)

# 显示图像
plt.show()

