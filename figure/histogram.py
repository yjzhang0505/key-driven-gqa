import torch
import matplotlib.pyplot as plt

def plot_k_weights_histograms(k_file, num_heads, dim, output_image_path):
    """
    从保存的 txt 文件中读取 K 权重，并绘制每个 head 的权重值的分布直方图。
    将所有直方图放在同一张画布上并保存图片。
    """
    # 读取 K 权重
    with open(k_file, 'r') as f:
        k_weight_flat = [float(line.strip()) for line in f]
    
    # 将平铺数据转换为张量并 reshape 成 (num_heads, dim_per_head, dim)
    k_weight = torch.tensor(k_weight_flat).view(num_heads, dim // num_heads, dim)

    # 创建一个画布，指定子图的排列方式（3行 x 4列 = 12个子图）
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for i in range(num_heads):
        ax = axes[i // 4, i % 4]  # 计算子图位置

        # 将权重矩阵展开为一维，计算分布并绘制直方图
        head_data = k_weight[i].flatten().cpu().detach().numpy()

        # 绘制直方图，bins可以设置为合适的数值以显示数值范围
        ax.hist(head_data, bins=50, color='green', alpha=0.7)
        ax.set_title(f"Head {i + 1} Weight Distribution")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count")

    # 调整布局，防止子图重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_image_path)
    plt.close()

    print(f"K权重的12个头的权重分布图已保存到 {output_image_path}")

# 使用示例，假设 K 文件路径和模型的参数
k_file = '/data/yjzhang/desktop/try/key-driven-gqa/data/finetuned/k_weights.txt'
# k_file = '/data/yjzhang/desktop/try/key-driven-gqa/data/share/k_weights.txt'
num_heads = 12  # 假设有12个头
dim = 768       # 假设每个头的维度为768

# 输出图片路径
output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/data/finetuned/k_weights_histogram.png'
# output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/data/share/k_weights_histogram.png'

# 绘制直方图并保存
plot_k_weights_histograms(k_file, num_heads, dim, output_image_path)
