# TOPSIS 优劣解距离法

## 适用场景

- 权重已确定（主观/客观/组合），要给方案排序。
- 方案可量化、指标已归一化。
- 获奖案例：F23102520378（FAHP-熵权-TOPSIS）、F23107010086（熵权-TOPSIS）。

## 原理

构造正理想解（各指标最优值）和负理想解（各指标最劣值），计算每个方案到二者的距离，相对接近度越大越好。

## 步骤

1. 归一化矩阵 X（效益/成本方向统一）。
2. 向量规范化：$z_{ij} = x_{ij} / \sqrt{\sum x_{ij}^2}$。
3. 加权规范化：$v_{ij} = w_j \cdot z_{ij}$。
4. 正理想解 $V^+ = \max_j V$（效益型）/ $\min_j V$（成本型，已归一化后都是越大越好）。
5. 负理想解 $V^- = \min_j V$。
6. 距离：$D_i^+ = \sqrt{\sum (v_{ij} - V_j^+)^2}$，$D_i^- = \sqrt{\sum (v_{ij} - V_j^-)^2}$。
7. 相对接近度 $C_i = D_i^- / (D_i^+ + D_i^-)$，$C_i \in [0,1]$，越大越好。

## 可运行代码

```python
import numpy as np
import pandas as pd

def topsis(df, weights, benefit_cols, cost_cols):
    """
    df: 行=方案, 列=指标
    weights: 各指标权重（和为1）
    返回: 含 D+, D-, C, 排名 的 DataFrame
    """
    X = df.copy().astype(float)
    # 1. 极值归一化
    for c in benefit_cols:
        X[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min())
    for c in cost_cols:
        X[c] = (X[c].max() - X[c]) / (X[c].max() - X[c].min())
    # 2. 向量规范化
    Z = X / np.sqrt((X**2).sum(axis=0))
    # 3. 加权
    V = Z * weights
    # 4. 正负理想解（归一化后都是越大越好）
    v_pos = V.max(axis=0)
    v_neg = V.min(axis=0)
    # 5. 距离
    d_pos = np.sqrt(((V - v_pos)**2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg)**2).sum(axis=1))
    # 6. 相对接近度
    C = d_neg / (d_pos + d_neg)
    res = pd.DataFrame({'D+': d_pos, 'D-': d_neg, '相对接近度C': C}, index=df.index)
    res['排名'] = res['相对接近度C'].rank(ascending=False).astype(int)
    return res.sort_values('排名')

# 示例数据
data = pd.DataFrame({
    '经济': [85, 70, 90, 65, 75],
    '社会': [80, 85, 75, 90, 70],
    '环境': [60, 75, 80, 70, 85],
    '成本': [300, 280, 350, 260, 310],
    '能耗': [20, 18, 25, 16, 22],
    '风险': [0.3, 0.2, 0.5, 0.1, 0.4],
}, index=['A','B','C','D','E'])

# 假设熵权法得到的权重
w = np.array([0.199, 0.176, 0.147, 0.144, 0.158, 0.176])

result = topsis(
    data, w,
    benefit_cols=['经济','社会','环境'],
    cost_cols=['成本','能耗','风险']
)
print(result)
```

## 输出示例

```
           D+        D-  相对接近度C  排名
D    0.158524  0.238253     0.600472    1
B    0.136140  0.190888     0.583705    2
A    0.153354  0.169885     0.525572    3
C    0.219860  0.170204     0.436350    4
E    0.205052  0.133471     0.394274    5
```

## 变体

- **熵权-TOPSIS**：先用熵权法算权重，再跑 TOPSIS。
- **灰色关联-TOPSIS**：把欧氏距离换成灰色关联贴近度，适合样本少。
- **FAHP-TOPSIS**：用三角模糊数表示正负理想解，适合指标模糊。

## 常见错误

1. **归一化方向错**：成本型指标忘记反向。
2. **权重和不为 1**：先归一化权重。
3. **不向量规范化直接乘权重**：标准 TOPSIS 要先做 $\sqrt{\sum x^2}$ 规范化。
4. **D+ + D- = 0**：某方案刚好等于正负理想解，会出除零，加 1e-9。

## 论文写作建议

- 表：列出各方案 D+、D-、C、排名。
- 图：水平条形图按 C 值排序（用 huawei-chart-style 的 04-评价与对比类）。
- 结论："D 方案相对接近度 0.60，排名第一，主要优势在于成本与风险控制；C 方案经济最强但成本偏高，排名第四。"