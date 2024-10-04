# group_by_similarity_with_random_matrices_and_pooling.py

import torch

class GroupingModule:
    def __init__(self):
        pass

    def group_by_similarity(self, sim_matrix, group_size):
        """
        根据K之间的相似性矩阵，将K分组，每组group_size个头。
        Args:
            sim_matrix (torch.Tensor): 键之间的相似性矩阵，形状为 (num_kv_heads, num_kv_heads)
            group_size (int): 每组的键数量。
        Returns:
            List[List[int]]: 每个分组中的索引列表，每个列表中有 group_size 个索引。
        """
        num_heads = sim_matrix.shape[0]
        groups = []  # 用于存放每个组的结果
        ungrouped_heads = list(range(num_heads))  # 初始时所有head都未分组

        while len(ungrouped_heads) >= group_size:
            first_head = ungrouped_heads[0]
            sim_scores = sim_matrix[first_head]
            sorted_heads = sorted(ungrouped_heads, key=lambda x: sim_scores[x], reverse=True)
            group = sorted_heads[:group_size]
            groups.append(group)
            ungrouped_heads = [head for head in ungrouped_heads if head not in group]

        return groups


def calculate_similarity_matrix(matrices):
    """
    计算给定矩阵列表之间的点积相似度矩阵。
    Args:
        matrices (List[torch.Tensor]): 包含多个随机矩阵的列表，每个矩阵形状为 (dim, dim)
    Returns:
        torch.Tensor: 计算得到的相似度矩阵，形状为 (num_matrices, num_matrices)
    """
    num_matrices = len(matrices)
    sim_matrix = torch.zeros((num_matrices, num_matrices))

    # 计算矩阵间的点积相似度
    for i in range(num_matrices):
        xx = torch.sum(matrices[i] * matrices[i])
        for j in range(num_matrices):
            sim_matrix[i, j] = torch.sum(matrices[i] * matrices[j]) / xx  # 点积相似度

    return sim_matrix


def average_pooling(group, matrices):
    """
    对每组的矩阵进行对应位置的平均池化。
    Args:
        group (List[int]): 该组包含的矩阵的索引。
        matrices (List[torch.Tensor]): 矩阵列表，每个矩阵形状为 (dim, dim)
    Returns:
        torch.Tensor: 该组矩阵池化后的矩阵，形状为 (dim, dim)
    """
    # 取出该组的所有矩阵
    group_matrices = [matrices[idx] for idx in group]

    # 对应位置进行平均
    pooled_matrix = torch.mean(torch.stack(group_matrices), dim=0)
    
    return pooled_matrix


def main():
    num_matrices = 8
    dim = 4  # 每个随机矩阵的维度
    group_size = 2  # 每组包含的矩阵数量

    # 生成随机矩阵列表
    matrices = [torch.rand((dim, dim)) for _ in range(num_matrices)]

    # 计算相似度矩阵
    sim_matrix = calculate_similarity_matrix(matrices)

    # 打印相似度矩阵
    print("Similarity Matrix:\n", sim_matrix)

    # 创建实例并调用group_by_similarity进行分组
    grouping_module = GroupingModule()
    groups = grouping_module.group_by_similarity(sim_matrix, group_size)

    # 打印分组结果
    print("\nGenerated Groups:", groups)

    # 对每组进行平均池化
    for idx, group in enumerate(groups):
        pooled_matrix = average_pooling(group, matrices)
        print(f"\nPooled Matrix for Group {idx + 1} (Group Indices: {group}):\n", pooled_matrix)


if __name__ == "__main__":
    main()
