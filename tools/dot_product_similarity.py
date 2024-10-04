import numpy as np

def dot_product_similarity(matrices):
    """
    计算多个四维矩阵之间的点积相似度。
    
    Args:
        matrices (list of np.ndarray): 包含多个四维矩阵的列表，每个矩阵的形状为 (B, H, W, D)。
        
    Returns:
        similarity_matrix (np.ndarray): 点积相似度矩阵，形状为 (N, N)，其中 N 是矩阵的数量。
    """
    num_matrices = len(matrices)
    assert num_matrices > 1, "至少需要两个矩阵来计算相似度"
    
    # 初始化相似度矩阵
    similarity_matrix = np.zeros((num_matrices, num_matrices))
    
    # 遍历每对矩阵，计算它们的点积相似度
    for i in range(num_matrices):
        norm_factor = np.sum(matrices[i] * matrices[i])
        for j in range(i, num_matrices):
            # 计算两个四维矩阵的点积
            dot_product = np.sum(matrices[i] * matrices[j])/norm_factor
            similarity_matrix[i, j] = dot_product
            similarity_matrix[j, i] = dot_product  # 对称矩阵
    
    return similarity_matrix


# 示例：创建一组四维矩阵 (B, H, W, D)
B, H, W, D = 3, 4, 4, 4  # 批次大小、高度、宽度、深度
matrix1 = np.random.rand(B, H, W, D)
matrix2 = np.random.rand(B, H, W, D)
matrix3 = np.random.rand(B, H, W, D)

# 将矩阵放入列表
matrices = [matrix1, matrix2, matrix3]

# 计算点积相似度矩阵
similarity_matrix = dot_product_similarity(matrices)

# 打印相似度矩阵
print("点积相似度矩阵：")
print(similarity_matrix)
