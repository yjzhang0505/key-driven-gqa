import pandas as pd
import matplotlib.pyplot as plt

# 加载生成的Excel文件
file_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/acc_results.xlsx'
acc_df = pd.read_excel(file_path)

# 生成箱线图
plt.figure(figsize=(8, 6))

# 生成箱线图，设置箱子的主体颜色为浅灰色 #F2F2F2
box = plt.boxplot([acc_df['train_acc'], acc_df['test_acc']], labels=['train_acc', 'test_acc'], patch_artist=True)

# 设置箱体颜色为 #F2F2F2
for patch in box['boxes']:
    patch.set_facecolor('#F2F2F2')

# 去除网格线
plt.grid(False)

# 添加标题和标签
plt.title('Boxplot of Train and Test Accuracy')
plt.ylabel('Accuracy')

# 提取箱线图的统计值（四分位数、最小值、中位数、最大值）
for i, data in enumerate([acc_df['train_acc'], acc_df['test_acc']], start=1):
    # 获取统计值：最小值 (min)，Q1（第一个四分位数），中位数 (median)，Q3（第三个四分位数），最大值 (max)
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)
    mean_val = data.mean()

    # 调整标注位置，避免重叠，并向右移动部分标注
    offset = 0.02  # 位置偏移量
    
    # 标注最小值
    plt.text(i + offset, min_val, f'{min_val:.3f}', ha='left', va='top', fontsize=8)
    # 标注Q1
    plt.text(i + offset, q1, f'{q1:.3f}', ha='left', va='top', fontsize=8)
    # 标注中位数，微调位置避免重叠
    plt.text(i - offset, median, f'{median:.3f}', ha='right', va='bottom', fontsize=8, color='blue')
    # 标注Q3
    plt.text(i + offset, q3, f'{q3:.3f}', ha='left', va='bottom', fontsize=8)
    # 标注最大值
    plt.text(i + offset, max_val, f'{max_val:.3f}', ha='left', va='bottom', fontsize=8)
    # 标注平均值并用 * 标识
    plt.text(i, mean_val, f'*{mean_val:.3f}', ha='center', va='bottom', fontsize=8, color='red')

# 保存图表到指定路径
output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/boxplot_train_test_acc.png'
plt.savefig(output_image_path)

print(f"Boxplot saved to {output_image_path}")
