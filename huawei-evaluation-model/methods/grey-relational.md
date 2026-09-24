# 灰色关联分析

## 适用场景

- 样本少（n<10）、信息不完全、数据灰度大。
- 医疗题（E 组）分析因素与疗效的关联程度。
- 获奖案例：D23104860011、E23100650012、E23102550019、E23103530067。

## 原理

比较各方案曲线与"理想最优方案"曲线的几何相似度，越相似关联度越大。

## 步骤

1. 确定参考数列 $X_0$（各指标最优值）和比较数列 $X_i$（各方案）。
2. 无量纲化（初值化 / 均值化 / 归一化）。
3. 计算关联系数：
$$\xi_i(k) = \frac{\min_i \min_k |x_0(k)-x_i(k)| + \rho \max_i \max_k |x_0(k)-x_i(k)|}{|x_0(k)-x_i(k)| + \rho \max_i \max_k |x_0(k)-x_i(k)|}$$
4. 权重加权求关联度 $r_i = \sum_k w_k \xi_i(k)$。
5. $r_i$ 越大，方案越优。

$\rho$ 分辨系数，通常取 0.5。

## 可运行代码

```python
import numpy as np
import pandas as pd

def grey_relational(df, benefit_cols, cost_cols, weights=None, rho=0.5):
    X = df.copy().astype(float)
    # 归一化
    for c in benefit_cols:
        X[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min())
    for c in cost_cols:
        X[c] = (X[c].max() - X[c]) / (X[c].max() - X[c].min())
    # 参考数列 = 最优值
    ref = X.max(axis=0)
    # 差值矩阵
    diff = np.abs(X - ref)
    dmin = diff.values.min()
    dmax = diff.values.max()
    # 关联系数
    xi = (dmin + rho * dmax) / (diff + rho * dmax)
    # 关联度（等权或加权）
    if weights is None:
        r = xi.mean(axis=1)
    else:
        r = xi @ np.array(weights)
    res = pd.DataFrame({'灰色关联度': r}, index=df.index)
    res['排名'] = res['灰色关联度'].rank(ascending=False).astype(int)
    return res.sort_values('排名')

data = pd.DataFrame({
    '经济': [85, 70, 90, 65, 75],
    '社会': [80, 85, 75, 90, 70],
    '环境': [60, 75, 80, 70, 85],
    '成本': [300, 280, 350, 260, 310],
    '能耗': [20, 18, 25, 16, 22],
    '风险': [0.3, 0.2, 0.5, 0.1, 0.4],
}, index=['A','B','C','D','E'])

res = grey_relational(data,
    benefit_cols=['经济','社会','环境'],
    cost_cols=['成本','能耗','风险'])
print(res)
```

## 输出示例

```
    灰色关联度  排名
D    0.7980    1
B    0.6097    2
C    0.5190    3
A    0.5177    4
E    0.5150    5
```

## 常见错误

1. **参考数列选错**：必须是各指标最优值，不是平均值。
2. **不区分指标方向**：成本型指标要先反向。
3. **ρ 取 0.5 是惯例**，不要随便改；论文里要说明。
4. **样本大时灰色关联区分度低**：n>20 改用 TOPSIS。

## 与其他方法联用

- 灰色关联 + 熵权 → 加权关联度。
- 灰色关联 + TOPSIS → 用关联度代替欧氏距离。
- GM(1,1) 灰色预测 + 灰色关联分析 = D 题双碳路径标配。