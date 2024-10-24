# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # 设置根路径
# base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'
# base_path2 = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share'
# l2_norms_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt'

# # 初始化字典存储每组(a, b)及其所有的test_acc
# group_acc_dict = defaultdict(list)

# # 遍历1到10和51到64的文件夹
# for folder_num in list(range(1, 10)) + list(range(51, 101)):
#     folder_path2 = os.path.join(base_path2, str(folder_num))
#     folder_path = os.path.join(base_path, str(folder_num))
    
#     # 获取run.csv的最后一行中的test_acc
#     run_csv_path = os.path.join(folder_path2, 'run.csv')
#     if os.path.exists(run_csv_path):
#         run_df = pd.read_csv(run_csv_path)
#         test_acc = run_df.iloc[-1]['test_acc']  # 读取最后一行的test_acc
#         test_acc = np.exp(800*test_acc)/np.array(1e187)
#     else:
#         print(f"文件不存在: {run_csv_path}")
#         continue

#     # 获取group.txt的分组情况
#     group_txt_path = os.path.join(folder_path, 'group.txt')
#     if os.path.exists(group_txt_path):
#         with open(group_txt_path, 'r') as f:
#             for line in f:
#                 group = list(map(int, line.strip().split(',')))  # 将每一行读取为一组
#                 if len(group) == 2:
#                     a, b = group
#                     group_acc_dict[(a, b)].append(test_acc)
#                     group_acc_dict[(b, a)].append(test_acc)  # 对称存储
#     else:
#         print(f"文件不存在: {group_txt_path}")
#         continue

# # 提取横纵坐标和颜色值的原始图
# x_original = []
# y_original = []
# colors_original = []

# for (a, b), acc_list in group_acc_dict.items():
#     avg_acc = np.mean(acc_list)  # 计算每个组合的平均test_acc
#     x_original.append(a)
#     y_original.append(b)
#     colors_original.append(avg_acc)

# # 将test_acc归一化到0-1范围内，以便用于颜色映射
# norm = plt.Normalize(vmin=min(colors_original), vmax=max(colors_original))

# # 读取方差文件
# var_dict = {}
# mean_var_dict = {}
# if os.path.exists(l2_norms_path):
#     with open(l2_norms_path, 'r') as f:
#         lines = f.readlines()  # 读取所有行
#         for i in range(0, len(lines), 2):  # 每次读取两行
#             key_mean, mean_values = lines[i].strip().split(':')  # 奇数行，均值
#             key_var, var_values = lines[i + 1].strip().split(':')  # 偶数行，方差
            
#             # 提取均值和方差
#             mean_values = list(map(float, mean_values.split(',')))
#             var_values = list(map(float, var_values.split(',')))
            
#             # 将均值和方差分别存储在两个字典中
#             mean_var_dict[key_mean.strip()] = mean_values
#             var_dict[key_var.strip()] = var_values


# # 获取 K、Q、V、KQ、KQV 的方差，并按升序排序
# sorted_indices = {}
# for key in ['K_var', 'Q_var', 'V_var', 'KxQ_var', 'KxQxV_var']:
#     sorted_indices[key] = np.argsort(var_dict[key]) + 1  # 方差越小，排得越靠前

# # 将第一个方差的排序作为基准
# reference_sorted_indices = sorted_indices['K_var']

# # 创建2行3列的子图
# fig, axs = plt.subplots(2, 3, figsize=(18, 12))
# axs = axs.ravel()

# # 绘制原始图像
# scatter = axs[0].scatter(x_original, y_original, c=colors_original, cmap='Greens', norm=norm, s=100)
# axs[0].set_title('Original Group-wise Pairings')
# axs[0].set_xlabel('Group A (Original Head Order)')
# axs[0].set_ylabel('Group B (Original Head Order)')
# axs[0].set_xticks(list(set(x_original)))
# axs[0].set_yticks(list(set(y_original)))
# axs[0].set_xticklabels(list(set(x_original)))
# axs[0].set_yticklabels(list(set(y_original)))
# cbar = plt.colorbar(scatter, ax=axs[0])
# cbar.set_label('Average Test Accuracy')

# # 绘制根据不同方差排序后的图，并统一横纵坐标排序
# for i, (key, indices) in enumerate(sorted_indices.items(), start=1):
#     # 使用基准排序统一x和y的顺序
#     sorted_x = [reference_sorted_indices[a - 1] for a in x_original]
#     sorted_y = [reference_sorted_indices[b - 1] for b in y_original]

#     # 绘制子图
#     scatter = axs[i].scatter(sorted_x, sorted_y, c=colors_original, cmap='Greens', norm=norm, s=100)
#     axs[i].set_title(f'Sorted by {key}')  # 使用方差的名称作为标题
#     axs[i].set_xlabel('Group A (Sorted by Variance)')
#     axs[i].set_ylabel('Group B (Sorted by Variance)')
    
#     # 统一设置排序后的坐标标签，横纵坐标使用同样的排序
#     sorted_labels = [str(idx) for idx in reference_sorted_indices]
#     axs[i].set_xticks(list(set(sorted_x)))
#     axs[i].set_yticks(list(set(sorted_y)))
#     axs[i].set_xticklabels(sorted_labels, rotation=90)  # 横坐标旋转90度
#     axs[i].set_yticklabels(sorted_labels)

#     cbar = plt.colorbar(scatter, ax=axs[i])
#     cbar.set_label('Average Test Accuracy')

# # 调整子图之间的布局
# plt.tight_layout()

# # 保存图像到文件 (保存为png格式)
# output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.png'
# plt.savefig(output_image_path)

# # 显示图像
# plt.show()





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置根路径
base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'
base_path2 = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share'
l2_norms_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt'

# 初始化字典存储每组(a, b)及其所有的test_acc
group_acc_dict = defaultdict(list)

# 遍历1到10和51到64的文件夹
for folder_num in list(range(1, 10)) + list(range(51, 101)):
    folder_path2 = os.path.join(base_path2, str(folder_num))
    folder_path = os.path.join(base_path, str(folder_num))
    
    # 获取run.csv的最后一行中的test_acc
    run_csv_path = os.path.join(folder_path2, 'run.csv')
    if os.path.exists(run_csv_path):
        run_df = pd.read_csv(run_csv_path)
        test_acc = run_df.iloc[-1]['test_acc']  # 读取最后一行的test_acc
        test_acc = np.exp(800*test_acc)/np.array(1e187)
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
color_mapping = {}  # 创建一个颜色映射字典

for (a, b), acc_list in group_acc_dict.items():
    total_acc = sum(acc_list)  # 累加每个组合的test_acc
    x_original.append(a)
    y_original.append(b)
    colors_original.append(total_acc)
    color_mapping[(a, b)] = total_acc  # 在字典中记录每个(a, b)的累加test_acc

# 去掉test_acc的归一化，直接使用累加后的test_acc进行颜色映射
norm = None

# 读取方差文件
var_dict = {}
mean_var_dict = {}
if os.path.exists(l2_norms_path):
    with open(l2_norms_path, 'r') as f:
        lines = f.readlines()  # 读取所有行
        for i in range(0, len(lines), 2):  # 每次读取两行
            key_mean, mean_values = lines[i].strip().split(':')  # 奇数行，均值
            key_var, var_values = lines[i + 1].strip().split(':')  # 偶数行，方差
            
            # 提取均值和方差
            mean_values = list(map(float, mean_values.split(',')))
            var_values = list(map(float, var_values.split(',')))
            
            # 将均值和方差分别存储在两个字典中
            mean_var_dict[key_mean.strip()] = mean_values
            var_dict[key_var.strip()] = var_values


# 获取 K、Q、V、KQ、KQV 的方差，并按升序排序
sorted_indices = {}
for key in ['K_var', 'Q_var', 'V_var', 'KxQ_var', 'KxQxV_var']:
    sorted_indices[key] = np.argsort(var_dict[key]) # 方差越小，排得越靠前

# 将第一个方差的排序作为基准
reference_sorted_indices = sorted_indices['K_var']

# 创建2行3列的子图
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.ravel()

# 绘制原始图像
scatter = axs[0].scatter(x_original, y_original, c=colors_original, cmap='Greens', s=100)
axs[0].set_title('Original Group-wise Pairings')
axs[0].set_xlabel('Group A (Original Head Order)')
axs[0].set_ylabel('Group B (Original Head Order)')
axs[0].set_xticks(list(set(x_original)))
axs[0].set_yticks(list(set(y_original)))
axs[0].set_xticklabels(list(set(x_original)))
axs[0].set_yticklabels(list(set(y_original)))
cbar = plt.colorbar(scatter, ax=axs[0])
cbar.set_label('Total Test Accuracy')

# 绘制根据不同方差排序后的图，并分别调整x和y的顺序
for i, (key, indices) in enumerate(sorted_indices.items(), start=1):
    # 根据每个key的排序对x和y坐标分别进行重新排序
    sorted_x = [sorted_indices[key][a] for a in x_original]
    sorted_y = [sorted_indices[key][b] for b in y_original]

    # 打印调试信息
    # print(f"sorted_x: {sorted_x}")
    # print(f"sorted_y: {sorted_y}")
    # print(f"color_mapping keys: {list(color_mapping.keys())}")

    # 创建新的颜色映射，确保每个排序后的(a, b)仍然是原来的test_acc值
    sorted_colors = [color_mapping.get((int(a), int(b)), color_mapping.get((int(b), int(a)), 0)) for a, b in zip(sorted_x, sorted_y)]


    # 绘制子图，颜色依然使用原始(a, b)组合对应的test_acc
    scatter = axs[i].scatter(sorted_x, sorted_y, c=sorted_colors, cmap='Greens', s=100)
    axs[i].set_title(f'Sorted by {key}')  # 使用方差的名称作为标题
    axs[i].set_xlabel('Group A (Sorted by Variance)')
    axs[i].set_ylabel('Group B (Sorted by Variance)')
    
    # 统一设置排序后的坐标标签
    sorted_labels = [str(idx) for idx in sorted_indices[key]]
    axs[i].set_xticks(list(set(sorted_x)))
    axs[i].set_yticks(list(set(sorted_y)))
    axs[i].set_xticklabels(sorted_labels, rotation=90)  # 横坐标旋转90度
    axs[i].set_yticklabels(sorted_labels)

    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label('Total Test Accuracy')



# 调整子图之间的布局
plt.tight_layout()

# 保存图像到文件 (保存为png格式)
output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/total_acc.png'
plt.savefig(output_image_path)

# 显示图像
plt.show()





# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.preprocessing import MinMaxScaler  # 用于归一化

# # 设置根路径
# base_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/proxy'
# base_path2 = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share'
# l2_norms_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var.txt'

# # 初始化字典存储每组(a, b)及其所有的test_acc
# group_acc_dict = defaultdict(list)

# # 遍历1到10和51到64的文件夹
# for folder_num in list(range(1, 10)) + list(range(51, 101)):
#     folder_path2 = os.path.join(base_path2, str(folder_num))
#     folder_path = os.path.join(base_path, str(folder_num))
    
#     # 获取run.csv的最后一行中的test_acc
#     run_csv_path = os.path.join(folder_path2, 'run.csv')
#     if os.path.exists(run_csv_path):
#         run_df = pd.read_csv(run_csv_path)
#         test_acc = run_df.iloc[-1]['test_acc']  # 读取最后一行的test_acc
#         test_acc = np.exp(800*test_acc)/np.array(1e187)
#     else:
#         print(f"文件不存在: {run_csv_path}")
#         continue

#     # 获取group.txt的分组情况
#     group_txt_path = os.path.join(folder_path, 'group.txt')
#     if os.path.exists(group_txt_path):
#         with open(group_txt_path, 'r') as f:
#             for line in f:
#                 group = list(map(int, line.strip().split(',')))  # 将每一行读取为一组
#                 if len(group) == 2:
#                     a, b = group
#                     group_acc_dict[(a, b)].append(test_acc)
#                     group_acc_dict[(b, a)].append(test_acc)  # 对称存储
#     else:
#         print(f"文件不存在: {group_txt_path}")
#         continue

# # 提取横纵坐标和颜色值的原始图
# x_original = []
# y_original = []
# colors_original = []

# for (a, b), acc_list in group_acc_dict.items():
#     avg_acc = np.mean(acc_list)  # 计算每个组合的平均test_acc
#     x_original.append(a)
#     y_original.append(b)
#     colors_original.append(avg_acc)

# # 将test_acc归一化到0-1范围内，以便用于颜色映射
# norm = plt.Normalize(vmin=min(colors_original), vmax=max(colors_original))

# # 读取方差文件
# var_dict = {}
# mean_var_dict = {}
# if os.path.exists(l2_norms_path):
#     with open(l2_norms_path, 'r') as f:
#         lines = f.readlines()  # 读取所有行
#         for i in range(0, len(lines), 2):  # 每次读取两行
#             key_mean, mean_values = lines[i].strip().split(':')  # 奇数行，均值
#             key_var, var_values = lines[i + 1].strip().split(':')  # 偶数行，方差
            
#             # 提取均值和方差
#             mean_values = list(map(float, mean_values.split(',')))
#             var_values = list(map(float, var_values.split(',')))
            
#             # 将均值和方差分别存储在两个字典中
#             mean_var_dict[key_mean.strip()] = mean_values
#             var_dict[key_var.strip()] = var_values

# # 初始化一个MinMaxScaler，用于归一化均值和方差
# scaler = MinMaxScaler()

# # 计算综合评分并排序
# sorted_scores = {}
# for key in ['K_var', 'Q_var', 'V_var', 'KxQ_var', 'KxQxV_var']:
#     # 取出对应的均值和方差
#     means = np.array(mean_var_dict[key.replace('var', 'mean')])
#     variances = np.array(var_dict[key])
    
#     # 分别对均值和方差进行归一化
#     means_normalized = scaler.fit_transform(means.reshape(-1, 1)).flatten()
#     variances_normalized = scaler.fit_transform(variances.reshape(-1, 1)).flatten()
    
#     # 计算综合评分：归一化均值减去归一化方差
#     scores = means_normalized - variances_normalized
    
#     # 按综合评分从大到小排序，返回排序后的索引（+1 以便与头的编号对齐）
#     sorted_scores[key] = np.argsort(scores)[::-1] + 1

# # 将第一个综合评分排序作为基准
# reference_sorted_indices = sorted_scores['K_var']

# # 创建2行3列的子图
# fig, axs = plt.subplots(2, 3, figsize=(18, 12))
# axs = axs.ravel()

# # 绘制原始图像
# scatter = axs[0].scatter(x_original, y_original, c=colors_original, cmap='Greens', norm=norm, s=100)
# axs[0].set_title('Original Group-wise Pairings')
# axs[0].set_xlabel('Group A (Original Head Order)')
# axs[0].set_ylabel('Group B (Original Head Order)')
# axs[0].set_xticks(list(set(x_original)))
# axs[0].set_yticks(list(set(y_original)))
# axs[0].set_xticklabels(list(set(x_original)))
# axs[0].set_yticklabels(list(set(y_original)))
# cbar = plt.colorbar(scatter, ax=axs[0])
# cbar.set_label('Average Test Accuracy')

# # 绘制根据不同方差排序后的图，并分别对每个key进行坐标映射
# for i, (key, indices) in enumerate(sorted_scores.items(), start=1):
#     # 使用每个key的排序分别对x和y进行重新排序
#     sorted_x = [sorted_scores[key][a - 1] for a in x_original]
#     sorted_y = [sorted_scores[key][b - 1] for b in y_original]

#     # 绘制子图
#     scatter = axs[i].scatter(sorted_x, sorted_y, c=colors_original, cmap='Greens', norm=norm, s=100)
#     axs[i].set_title(f'Sorted by {key}')  # 使用方差的名称作为标题
#     axs[i].set_xlabel('Group A (Sorted by Composite Score)')
#     axs[i].set_ylabel('Group B (Sorted by Composite Score)')
    
#     # 统一设置排序后的坐标标签，横纵坐标使用当前key的排序
#     sorted_labels = [str(idx) for idx in sorted_scores[key]]
#     axs[i].set_xticks(list(set(sorted_x)))
#     axs[i].set_yticks(list(set(sorted_y)))
#     axs[i].set_xticklabels(sorted_labels, rotation=90)  # 横坐标旋转90度
#     axs[i].set_yticklabels(sorted_labels)

#     cbar = plt.colorbar(scatter, ax=axs[i])
#     cbar.set_label('Average Test Accuracy')


# # 调整子图之间的布局
# plt.tight_layout()

# # 保存图像到文件 (保存为png格式)
# output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/mean_var_composite_score.png'
# plt.savefig(output_image_path)

# # 显示图像
# plt.show()
