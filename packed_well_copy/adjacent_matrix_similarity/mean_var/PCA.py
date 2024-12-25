import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成一个随机的高维数据集（假设有100个样本，每个样本有10个特征）
np.random.seed(42)
X = np.random.rand(100, 10)

# 1. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 初始化PCA，指定要降到2维
pca = PCA(n_components=2)

# 3. 拟合PCA模型并转换数据
X_pca = pca.fit_transform(X_scaled)

# 4. 查看每个主成分的方差比例
print(f'主成分方差解释比例: {pca.explained_variance_ratio_}')
print(X_pca)
