import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_k_weights_heatmaps(k_file, num_heads, dim, output_image_path, target_sum=1.0):
    """
    从保存的 txt 文件中读取 K 权重，先按比例缩放每个头的数据，使得每个头的总和相等，然后对数据做指数处理。
    所有热图共享一个统一的色标，数值越大颜色越深（绿色）。
    将所有热图放在同一张画布上并保存图片。
    """
    # 读取 K 权重
    with open(k_file, 'r') as f:
        k_weight_flat = [float(line.strip()) for line in f]
    
    # 将平铺数据转换为张量并 reshape 成 (num_heads, dim_per_head, dim)
    k_weight = torch.tensor(k_weight_flat).view(num_heads, dim // num_heads, dim)

    # 对每个 head 进行缩放，使总和相同，然后进行指数处理
    scaled_k_weight = []
    for i in range(num_heads):
        head_data = k_weight[i]
        head_sum = torch.sum(head_data)
        
        # 缩放每个 head，使其总和等于 target_sum
        if head_sum != 0:
            scaled_head_data = head_data * (target_sum / head_sum)
        else:
            scaled_head_data = head_data  # 如果总和为0，保持不变

        # 对缩放后的数据应用指数函数 e^x
        exp_head_data = torch.exp(scaled_head_data)*700
        scaled_k_weight.append(exp_head_data)
    
    scaled_k_weight = torch.stack(scaled_k_weight)  # 将列表转为张量

    # 获取所有 head 的全局最小值和最大值，确保热图共享一个色标
    global_min = torch.min(scaled_k_weight)
    global_max = torch.max(scaled_k_weight)

    # 创建一个画布，指定子图的排列方式（3行 x 4列 = 12个子图）
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for i in range(num_heads):
        ax = axes[i // 4, i % 4]  # 计算子图位置

        # 使用绿色渐变的 cmap（"Greens"），数值越大颜色越深
        sns.heatmap(scaled_k_weight[i].cpu().detach().numpy(), ax=ax, cmap="Greens", cbar=True, vmin=global_min, vmax=global_max)
        ax.set_title(f"Head {i + 1} (Exp)")

    # 调整布局，防止子图重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_image_path)
    plt.close()

    print(f"K权重的12个头的绿色热图已保存到 {output_image_path}，并且所有子图共享相同的色标。")

# 使用示例，假设 K 文件路径和模型的参数
k_file = '/data/yjzhang/desktop/try/key-driven-gqa/data/share/k_weights.txt'
num_heads = 12  # 假设有12个头
dim = 768       # 假设每个头的维度为768

# 输出图片路径
output_image_path = '/data/yjzhang/desktop/try/key-driven-gqa/data/share/k_weights_colorbar_heatmap.png'

# 绘制热图并保存，每个头的数据总和相同并做指数处理，且所有热图共享同一个色标
plot_k_weights_heatmaps(k_file, num_heads, dim, output_image_path)

