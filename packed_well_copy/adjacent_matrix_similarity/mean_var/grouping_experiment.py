import torch
import os
import numpy as np
from _1_reading_ckpt import YourTransformerModel
from _4_ordering import load_stats_from_model, load_singular_values_from_model
from _3_similarity import calculate_similarity
from tools import save_to_txt


def group_heads_by_importance_and_similarity(model, importance_matrix, similarity_matrix):
    """
    按照重要性对头进行分组，优先选择最重要的头并与最相似的头进行配对。
    每次配对后失活已分组的头部，行和列同时失活，且避免与自己分组。
    """
    num_heads = model.num_heads
    grouped_heads = []
    
    # 标记哪些头部是活动的
    active_heads = np.ones(num_heads, dtype=bool)  # 每个头部是否活跃

    # 按重要性排序，重要性高的头排前面
    importance_matrix = process_importance_matrix(importance_matrix)
    sorted_heads_by_importance = torch.argsort(importance_matrix, descending=True).tolist()

    # 在加载相似性矩阵后，将对角线设置为负无穷，避免与自己分组
    similarity_matrix = similarity_matrix.clone()  # 保证不修改原矩阵
    for i in range(num_heads):
        similarity_matrix[i, i] = -float('inf')

    while len(sorted_heads_by_importance) > 0:
        # 选择最重要的未分组头
        head_idx = sorted_heads_by_importance.pop(0)
        
        # 如果该头已经失活，跳过它
        if not active_heads[head_idx]:
            continue
        
        # 找出与该头相似性最高的另一个头
        similarities = similarity_matrix[head_idx]
        
        # 只选择活动的头部，忽略失活头部
        valid_similarities = similarities[active_heads]
        
        # 如果活动的头部数量为1，跳过该轮分组
        if len(valid_similarities) == 1:
            break
        
        # 选择最大相似性的头
        max_similarity_idx = torch.argmax(valid_similarities).item()
        
        # 将其映射回原始的索引
        max_similarity_idx = np.where(active_heads)[0][max_similarity_idx]
        
        # 将这两个头部分组，并同时失活它们的行和列
        grouped_heads.append((head_idx, max_similarity_idx))
        active_heads[head_idx] = active_heads[max_similarity_idx] = False
        
        # 在相似性矩阵中同时失活行和列
        similarity_matrix[head_idx, :] = 0  # 清空该行
        similarity_matrix[:, head_idx] = 0  # 清空该列
        similarity_matrix[max_similarity_idx, :] = 0  # 清空该行
        similarity_matrix[:, max_similarity_idx] = 0  # 清空该列

    return grouped_heads


def process_importance_matrix(importance_matrix):
    """
    确保重要性矩阵是一个一维向量。
    """
    if importance_matrix.ndimension() > 1:
        importance_matrix = importance_matrix.view(-1)
    return importance_matrix


def group_heads_for_layer(model, similartiry_type, importance_type, layer_idx):
    """
    输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
    """
    # 获取相似性矩阵和重要性矩阵
    similarity_matrices = calculate_similarity(model)
    importance_matrix = load_stats_from_model(model, layer_idx).get(importance_type)
    print(importance_matrix)
    # importance_matrix = load_stats_from_model(model, layer_idx).get('K_var')

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)
    # similarity_matrix = layer_similarity.get('K_similarity_matrix')

    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 调用分组函数
    grouped_heads = group_heads_by_importance_and_similarity(model, importance_matrix, similarity_matrix)
    return grouped_heads

def group_heads_singular(model, similartiry_type, importance_type, layer_idx):
    """
    输入层索引，自动加载该层的相似性矩阵和重要性矩阵，并返回头部分组结果。
    """
    # 获取相似性矩阵和重要性矩阵
    similarity_matrices = calculate_similarity(model)
    importance_matrix = load_singular_values_from_model(model, layer_idx).get(importance_type)
    print(importance_matrix)

    # 获取指定层的相似性矩阵
    layer_similarity = similarity_matrices.get(f'Layer_{layer_idx}', {})
    similarity_matrix = layer_similarity.get(similartiry_type)
    # similarity_matrix = layer_similarity.get('K_similarity_matrix')

    if similarity_matrix is None or not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.zeros((model.num_heads, model.num_heads))

    # 调用分组函数
    grouped_heads = group_heads_by_importance_and_similarity(model, importance_matrix, similarity_matrix)
    return grouped_heads


# 示例调用
# 初始化模型
model = YourTransformerModel(num_heads=12, dim=768)
model.load_pretrained_qkv_weights()

# 定义所有的相似性矩阵和重要性矩阵的组合
similarity_keys = ['K_cosine', 'V_cosine']
importance_keys = ['K_singular', 'Q_singular', 'V_singular', 'KxQ_singular']
# importance_keys = ['K_mean', 'K_var', 'Q_mean', 'Q_var', 'V_mean', 'V_var', 'KxQ_mean', 'KxQ_var']

# 遍历相似性矩阵和重要性矩阵的组合
for similarity_key in similarity_keys:
    for importance_key in importance_keys:
        # 创建存储结果的字典
        group_schemes = {}
        
        # 遍历 12 层
        for layer_idx in range(12):
            # 假设 group_heads_for_layer 是根据相似性矩阵和重要性矩阵对头部分组的函数
            grouped_heads = group_heads_singular(model, similarity_key, importance_key, layer_idx)
            group_schemes[layer_idx] = grouped_heads

        # 格式化输出的字符串
        output_str = ""
        for layer_idx, grouped_heads in group_schemes.items():
            output_str += f"{grouped_heads},\n"

        # 创建输出文件夹路径（使用组合名称）
        output_dir = f"/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin/{similarity_key}_{importance_key}"
        os.makedirs(output_dir, exist_ok=True)

        # 定义输出文件路径
        output_path = os.path.join(output_dir, 'group.txt')

        # 保存到文件
        save_to_txt(output_path, output_str)

        print(f"Saved group scheme for {similarity_key} and {importance_key} to {output_path}")