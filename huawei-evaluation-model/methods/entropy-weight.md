# 熵权法（客观赋权）

## 适用场景

- 有客观数据，不想依赖专家主观判断。
- 指标变异越大，区分度越高，应给越大权重。
- 获奖案例：D23104860011、F23107010086、F23102520378。

## 原理

熵是信息论中不确定性的度量。某指标下各方案取值差异越大，信息熵越小，提供的信息量越多，权重应越大。

## 步骤

1. 数据归一化（效益型正向、成本型负向）。
2. 计算第 j 指标下第 i 方案的比重 $p_{ij}$。
3. 计算熵值 $e_j = -k \sum p_{ij} \ln p_{ij}$，其中 $k = 1/\ln(n)$。
4. 计算差异系数 $d_j = 1 - e_j$。
5. 权重 $w_j = d_j / \sum d_j$。

## 可运行代码

```python
import numpy as np
import pandas as pd

def entropy_weight(df, benefit_cols, cost_cols):
    """
    df: 行=方案, 列=指标
    benefit_cols: 效益型列名
    cost_cols: 成本型列名
    返回: 权重 Series, 归一化后的数据 DataFrame
    """
    X = df.copy().astype(float)
    # 1. 极值归一化
    for c in benefit_cols:
        X[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min())
    for c in cost_cols:
        X[c] = (X[c].max() - X[c]) / (X[c].max() - X[c].min())
    # 2. 整体平移，避免 ln(0)
    X = (X - X.min()) / (X.max() - X.min()) + 1e-6
    # 3. 计算比重
    P = X / X.sum(axis=0)
    # 4. 熵值
    n = len(X)
    k = 1.0 / np.log(n)
    e = -k * (P * np.log(P)).sum(axis=0)
    # 5. 差异系数 & 权重
    d = 1 - e
    w = d / d.sum()
    return w, X

# 示例数据
data = pd.DataFrame({
    '经济': [85, 70, 90, 65, 75],
    '社会': [80, 85, 75, 90, 70],
    '环境': [60, 75, 80, 70, 85],
    '成本': [300, 280, 350, 260, 310],
    '能耗': [20, 18, 25, 16, 22],
    '风险': [0.3, 0.2, 0.5, 0.1, 0.4],
}, index=['A','B','C','D','E'])

w, X_norm = entropy_weight(
    data,
    benefit_cols=['经济','社会','环境'],
    cost_cols=['成本','能耗','风险']
)
print("熵权法权重：")
print(w.round(4))
```

## 输出示例

```
熵权法权重：
经济    0.1990
社会    0.1759
环境    0.1472
成本    0.1442
能耗    0.1578
风险    0.1759
dtype: float64
```

## 常见错误

1. **不区分指标方向**：成本型指标直接正向归一化会完全反向。
2. **出现 0 没处理**：$0 \cdot \ln 0$ 会出 NaN，必须平移（+1e-6）。
3. **样本太少（n<5）**：熵权不稳定，建议结合 AHP。
4. **极端值敏感**：某指标一个 outliers 会把权重拉很高，先缩尾。

## 与其他方法联用

- 熵权 + TOPSIS → 最经典组合（见 topsis.md）。
- 熵权 + AHP → 组合赋权（见 combination-weight.md）。
- 熵权 + 灰色关联 → 灰色关联度加权求和。