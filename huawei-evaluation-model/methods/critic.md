# CRITIC 客观赋权法

## 适用场景

- 指标间有冲突性（某指标高了另一指标必然低）。
- 比熵权法多考虑了"指标间冲突性"。
- 获奖案例：E23103570015、E23103530067（与熵权对比使用）。

## 原理

权重由两部分决定：
1. **对比强度**：标准差 $\sigma_j$，越大区分度越高。
2. **冲突性**：$\sum_{k=1}^p (1 - r_{jk})$，与其他指标相关性越低，冲突越大，信息量越多。

$$C_j = \sigma_j \sum_{k=1}^p (1 - r_{jk})$$
$$w_j = C_j / \sum C_j$$

## 可运行代码

```python
import numpy as np
import pandas as pd

def critic_weight(df, benefit_cols, cost_cols):
    X = df.copy().astype(float)
    for c in benefit_cols:
        X[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min())
    for c in cost_cols:
        X[c] = (X[c].max() - X[c]) / (X[c].max() - X[c].min())
    # 标准差
    sigma = X.std(axis=0, ddof=1)
    # 相关矩阵
    corr = X.corr().values
    # 冲突性
    conflict = np.sum(1 - corr, axis=1)
    # 信息量
    C = sigma * conflict
    w = C / C.sum()
    return w

data = pd.DataFrame({
    '经济': [85, 70, 90, 65, 75],
    '社会': [80, 85, 75, 90, 70],
    '环境': [60, 75, 80, 70, 85],
    '成本': [300, 280, 350, 260, 310],
    '能耗': [20, 18, 25, 16, 22],
    '风险': [0.3, 0.2, 0.5, 0.1, 0.4],
}, index=['A','B','C','D','E'])

w = critic_weight(data,
    benefit_cols=['经济','社会','环境'],
    cost_cols=['成本','能耗','风险'])
print("CRITIC 权重：")
print(w.round(4))
```

## 输出示例

```
CRITIC 权重：
经济    0.2939
社会    0.1237
环境    0.2327
成本    0.1152
能耗    0.1168
风险    0.1177
dtype: float64
```

## 与熵权法对比

| 维度 | 熵权法 | CRITIC |
|---|---|---|
| 考虑变异 | ✓ | ✓ |
| 考虑冲突 | ✗ | ✓ |
| 适合指标独立 | ✓ | 一般 |
| 适合指标相关 | 一般 | ✓ |

论文里可以两个都算，对比权重差异，说明为什么选 CRITIC。