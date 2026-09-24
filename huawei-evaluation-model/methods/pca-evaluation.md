# PCA 主成分评价

## 适用场景

- 指标太多（>10），想降维后综合评分。
- 指标间有较强相关性。
- 获奖案例：C23102540029（PCA 降维+回归）。
- 注意：PCA 在 39 篇里出现 ≥15 次，但多作"降维/筛选指标"而非直接评价。

## 步骤

1. 数据标准化（Z-score）。
2. 求相关矩阵 R。
3. 求特征值 $\lambda_i$ 和特征向量。
4. 选累计方差贡献率 ≥ 85% 的前 k 个主成分。
5. 计算各主成分得分 $F_k$。
6. 综合得分 $F = \sum (\lambda_i / \sum\lambda) F_i$。

## 可运行代码

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 示例：10 个样本 × 8 个指标
np.random.seed(0)
X = np.random.randn(10, 8)
df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1,9)],
                 index=[f'方案{i}' for i in range(1,11)])

# 1. 标准化
Xs = StandardScaler().fit_transform(df)

# 2. PCA
pca = PCA()
pca.fit(Xs)

# 3. 选累计方差 ≥85% 的主成分
cumvar = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumvar >= 0.85) + 1
print(f"选前 {k} 个主成分，累计方差 {cumvar[k-1]:.2%}")

# 4. 主成分得分
scores = pca.transform(Xs)[:, :k]

# 5. 综合得分（按方差贡献率加权）
weights = pca.explained_variance_ratio_[:k] / pca.explained_variance_ratio_[:k].sum()
F = scores @ weights

result = pd.DataFrame({'综合得分': F}, index=df.index)
result['排名'] = result['综合得分'].rank(ascending=False).astype(int)
print(result.sort_values('排名'))
```

## 输出示例

```
选前 6 个主成分，累计方差 89.12%
         综合得分  排名
方案5    1.234    1
方案2    0.856    2
...
```

## 常见错误

1. **不标准化直接 PCA**：大量纲指标会主导主成分。
2. **主成分个数拍脑袋**：必须按累计方差贡献率（≥85%）或 Kaiser 准则（特征值>1）。
3. **主成分载荷解释不清**：论文里要写出每个主成分主要代表哪些指标。
4. **PCA 得分正负号无意义**：只看相对大小，不看绝对正负。

## 与其他方法对比

| 方法 | 指标数 | 主观性 | 适合场景 |
|---|---|---|---|
| AHP | ≤9 | 强 | 有专家经验 |
| 熵权/CRITIC | 不限 | 无 | 有客观数据 |
| PCA | >10 | 无 | 指标冗余、想降维 |
| TOPSIS | 不限 | 无（权重外给） | 方案排序 |