import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from collections import defaultdict
import re
import math
import numpy as np

# 初始化图和边权重计数
G = nx.Graph()
edge_weights = defaultdict(float)
count_pairs = defaultdict(int)

# 正则表达式匹配分组方案中的每对头
pattern = re.compile(r"\((\d+), (\d+)\)")

# 读取文件并处理数据
with open("/data/yjzhang/desktop/try/key-driven-gqa/calculate/group_all_aline.txt", "r") as file:
    for line in file:
        try:
            # 分割行，提取分组方案和准确度部分
            parts = line.strip().rsplit(", ", 2)
            groups_part = parts[0]  # 分组方案部分
            test_acc = float(parts[1])  # 解析准确度

            # 使用指数函数对test_acc进行放大
            # adjusted_acc = test_acc
            adjusted_acc = math.exp(45*test_acc)

            # 使用正则表达式查找分组对
            groups = pattern.findall(groups_part)
            if not groups:
                continue  # 跳过没有匹配到分组的行
            
            # 累加每个头对的权重（按指数放大后的准确度）
            for pair in groups:
                head1, head2 = int(pair[0]), int(pair[1])
                edge = tuple(sorted((head1, head2)))  # 确保无向边
                edge_weights[edge] += adjusted_acc
                count_pairs[edge] += 1

        except (ValueError, IndexError, SyntaxError) as e:
            print(f"Skipping line due to error: {e}")

# 检查是否有有效的边
if edge_weights:
    # 计算每条边的平均权重
    for edge in edge_weights:
        edge_weights[edge] /= count_pairs[edge]

    # 添加节点和边到图
    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    # 手动定义圆形布局
    num_nodes = 12  # 假设有12个节点
    radius = 1.0
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    pos = {i: (radius * np.cos(angle), radius * np.sin(angle)) for i, angle in enumerate(angles)}

    # 设置边的颜色映射
    edges = G.edges(data=True)
    weights = [attr['weight'] for _, _, attr in edges]

    # 使用 PowerNorm 设置非线性颜色条
    norm = colors.PowerNorm(gamma=1, vmin=min(weights), vmax=max(weights))  # 调整gamma以改变分布
    edge_colors = cm.Blues(norm(weights))

    # 创建图像和颜色条
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)
    
    # 添加颜色条到图的轴
    sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Grouping Frequency Weight")

    # 保存图像
    plt.savefig("/data/yjzhang/desktop/try/key-driven-gqa/figure/head_grouping_graph.png", format="PNG")
    plt.show()
else:
    print("No valid edges found to plot.")
