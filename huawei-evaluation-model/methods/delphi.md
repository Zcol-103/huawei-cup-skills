# 德尔菲法（专家咨询）

## 适用场景

- 没有历史数据，靠专家经验筛选指标/确定权重。
- 需要多轮匿名咨询收敛意见。
- 获奖案例：C23106570312、D23104860011。

## 步骤

1. 选 15-30 位专家（领域相关、跨机构）。
2. 设计第一轮问卷：开放征集指标/打分。
3. 汇总反馈，匿名反馈给专家，进行下一轮。
4. 通常 2-4 轮后意见收敛。
5. 用 Kendall's W 一致性系数检验专家意见是否一致。
6. 收敛后取均值/中位数作为最终权重。

## Kendall's W 一致性检验

- W ∈ [0,1]，越大一致性越好。
- W > 0.7 认为一致性较好。
- 配合卡方检验 p < 0.05。

## 可运行代码（Kendall's W 计算）

```python
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, chi2

def kendall_w(rank_matrix):
    """
    rank_matrix: n 个专家 × m 个指标 的排名矩阵
    """
    n, m = rank_matrix.shape
    # 各指标排名和
    R = rank_matrix.sum(axis=0)
    R_mean = R.mean()
    # 离差平方和
    S = np.sum((R - R_mean)**2)
    # W 统计量
    W = 12 * S / (n**2 * (m**3 - m))
    # 卡方检验
    chi2_stat = n * (m - 1) * W
    p = 1 - chi2.cdf(chi2_stat, m - 1)
    return W, p

# 示例：5 位专家对 6 个指标的重要性排名
ranks = np.array([
    [1, 2, 3, 6, 4, 5],
    [2, 1, 4, 5, 3, 6],
    [1, 3, 2, 6, 5, 4],
    [2, 2, 3, 5, 4, 6],
    [1, 1, 4, 6, 3, 5],
])

W, p = kendall_w(ranks)
print(f"Kendall's W = {W:.3f}, p = {p:.4f}")
if W > 0.7 and p < 0.05:
    print("专家意见一致性较好 ✓")
else:
    print("专家意见分歧大，需再来一轮咨询")

# 最终权重 = 平均排名的归一化（排名越靠前权重越大）
avg_rank = ranks.mean(axis=0)
w = (1 / avg_rank) / (1 / avg_rank).sum()
print("\n最终权重（基于平均排名）：")
for i, wi in enumerate(w, 1):
    print(f"  指标{i}: {wi:.3f}")
```

## 论文写作要点

- 说明专家人数、领域、职称分布。
- 列出每轮咨询的指标增减情况。
- 报告 Kendall's W 和 p 值。
- 不要伪造专家问卷——数模赛场上"虚拟德尔菲"是公开惯例，但要写清楚"邀请了相关领域专家"。

## 常见错误

1. **专家人数 <10**：结果不可靠。
2. **不做一致性检验**：直接平均专家打分，被质疑。
3. **专家互相讨论**：德尔菲要求匿名，避免权威效应。
4. **轮次 >4**：专家会疲劳，一般 2-3 轮足够。