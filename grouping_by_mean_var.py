import numpy as np
import pandas as pd

def load_mean_var(file_path=None):
    """
    返回15个固定的importance_order序列。
    """
    importance_orders = {}

    # 15个固定的importance_order序列
    fixed_orders = [
        [7, 5, 4, 3, 8, 1, 11, 2, 6, 10, 0, 9],
        [1, 5, 10, 8, 0, 4, 7, 9, 3, 6, 11, 2],
        [7, 5, 4, 3, 8, 1, 11, 2, 6, 10, 0, 9],
        [10, 1, 11, 3, 8, 2, 6, 0, 9, 4, 5, 7],
        [1, 5, 10, 0, 8, 4, 7, 9, 2, 6, 3, 11],
        [4, 5, 10, 1, 11, 3, 8, 2, 6, 0, 9, 7],
        [9, 10, 5, 0, 8, 3, 4, 6, 1, 2, 7, 11],
        [1, 5, 10, 7, 8, 0, 2, 9, 4, 6, 3, 11],
        [9, 10, 5, 0, 8, 3, 4, 6, 1, 2, 7, 11],
        [3, 2, 10, 0, 7, 9, 4, 6, 5, 11, 8, 1],
        [8, 5, 2, 3, 6, 0, 9, 4, 7, 10, 11, 1],
        [1, 8, 11, 4, 5, 6, 7, 9, 10, 0, 2, 3],
        [2, 3, 10, 8, 4, 1, 0, 11, 9, 7, 6, 5],
        [5, 8, 2, 0, 1, 10, 7, 3, 6, 9, 4, 11],
        [2, 3, 10, 8, 4, 1, 0, 11, 9, 7, 6, 5]
    ]

    # 为每个固定的order生成一个标签并存入字典
    labels = [
        "K_mean","K_var","K_mean-var", 
        "Q_mean","Q_var","Q_mean-var",
        "V_mean","V_var","V_mean-var",
        "KQ_mean","KQ_var","KQ_mean-var",
        "KQV_mean","KQV_var","KQV_mean-var",
    ]

    for i, order in enumerate(fixed_orders):
        importance_orders[labels[i]] = order

    return importance_orders



def importance_prioritized(similarity_matrix, importance_order):
    """
    第一种分组方法：给定重要性的顺序，从最重要的行开始分组，不能和自己分为一组。
    确保每个头只能被分到一个组。
    """
    n = similarity_matrix.shape[0]
    groups = []
    active_rows = set(importance_order)  # 按重要性排序的行的集合
    active_cols = set(range(n))  # 所有列都最开始是活跃的

    for row in importance_order:
        if row not in active_rows:
            continue
        
        # 如果没有更多活跃的列，则跳过
        available_cols = [c for c in active_cols if c != row]
        if not available_cols:
            continue

        # 找到该行中相似性最高的列，且该列不能是自己
        col = max(available_cols, key=lambda c: similarity_matrix[row, c])
        
        # 添加组：row 和相似性最高的列 col
        groups.append((row, col))

        # 失活：row 和 col 不再参与后续的分组
        active_rows.remove(row)
        active_rows.remove(col)
        active_cols.remove(row)
        active_cols.remove(col)
    
    return groups


def highest_similarity(similarity_matrix):
    """
    第二种分组方法：遍历矩阵，找出当前相似性最高的两个头分为一组，不能和自己分为一组。
    """
    n = similarity_matrix.shape[0]
    groups = []
    active = set(range(n))  # 活跃的行和列集合

    while len(active) > 1:
        # 找到相似性最大的两个头，且不能是自己
        max_sim = -1
        max_pair = (-1, -1)
        
        for i in active:
            for j in active:
                if i != j and similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    max_pair = (i, j)

        # 添加组：相似性最大的两个头 i 和 j
        i, j = max_pair
        groups.append((i, j))

        # 失活：i 和 j 的行和列不再参与分组
        active.remove(i)
        active.remove(j)

    return groups

# 从 Excel 文件中读取 12x12 相似性矩阵
def load_similarity_matrix_from_excel(file_path):
    """
    遍历 Excel 文件中的所有子表格，逐个读取相似性矩阵。
    """
    # 使用 pandas 的 ExcelFile 类来加载多个子表格
    xl = pd.ExcelFile(file_path)
    similarity_matrices = {}

    # 遍历所有子表格
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name, header=None)
        similarity_matrices[sheet_name] = df.values  # 将 DataFrame 转换为 numpy 数组

    return similarity_matrices

# 保存所有输出到txt文件
def save_to_txt(output_path, content):
    with open(output_path, 'a') as f:
        f.write(content + '\n')


# 文件路径
l2_norms_file_path = '/data/yjzhang/desktop/try/key-driven-gqa/output/arbitrary/share/l2_norms.txt'
excel_file_path = './output/arbitrary/share/similarity_matrices.xlsx'  # Excel 文件路径
output_txt_path = './output/arbitrary/share/grouping_mean_var.txt'  # 输出txt文件路径

# 加载重要性顺序
importance_orders = load_mean_var(l2_norms_file_path)

# 加载所有子表格中的相似性矩阵
similarity_matrices = load_similarity_matrix_from_excel(excel_file_path)

save_to_txt(output_txt_path, f"\nImportance_order:")
for label, importance_order in importance_orders.items():
    # 打印每个重要性顺序
    importance_order_str = f"Importance order for {label}: {importance_order}"
    save_to_txt(output_txt_path, importance_order_str)

# 遍历每个子表格的相似性矩阵并执行分组
for sheet_name, similarity_matrix in similarity_matrices.items():
    save_to_txt(output_txt_path, f"\nProcessing sheet: {sheet_name}")

    # 遍历每种 L2 范数类型并执行分组
    for label, importance_order in importance_orders.items():
    #     # 打印每个重要性顺序
    #     importance_order_str = f"Importance order for {label}: {importance_order}"
    #     save_to_txt(output_txt_path, importance_order_str)

        # 按给定重要性顺序分组
        groups_importance = importance_prioritized(similarity_matrix, importance_order)
        save_to_txt(output_txt_path, f"Importance-prioritized groups for {label}: {groups_importance}")

    # 按最高相似性分组
    groups_highest_similarity = highest_similarity(similarity_matrix)
    save_to_txt(output_txt_path, f"Highest similarity groups: {groups_highest_similarity}")

save_to_txt(output_txt_path, "\nAll processing complete.")
