# 数模论文综合评价标准 8 步流程

> 从"拿到一道评价题"到"写出获奖论文级评价章节"的标准流水线。

## 第 1 步：明确评价对象与目标

- 评价对象是什么？（N 个方案？M 个地区？K 种治疗方案？）
- 评价目的是什么？（排序？分级？选最优？找短板？）
- 写出一句："本文对 N 个 XX 方案，从 A、B、C 三个维度构建指标体系，采用 X 主、Y 客组合赋权，结合 Z 方法进行综合评分与排序。"

## 第 2 步：构建指标体系

- 目标层 → 准则层 → 指标层（三层结构）。
- 指标来源：
  - 文献法（查已有研究用了什么指标）
  - 德尔菲法（专家咨询，见 methods/delphi.md）
  - 数据驱动（PCA 从候选指标中筛选）
- 每个指标标注类型：
  - **效益型**（越大越好）
  - **成本型**（越小越好）
  - **区间型**（越接近某值越好，如温度、pH）
  - **固定型**（等于某值最好）

## 第 3 步：数据预处理

```python
import numpy as np
import pandas as pd

def normalize(df, benefit_cols, cost_cols, interval_cols=None):
    """
    极值归一化。
    benefit_cols: 效益型列名列表
    cost_cols:    成本型列名列表
    interval_cols: dict, {'列名': (最优下限, 最优上限)}
    """
    X = df.copy().astype(float)
    for c in benefit_cols:
        X[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min())
    for c in cost_cols:
        X[c] = (X[c].max() - X[c]) / (X[c].max() - X[c].min())
    if interval_cols:
        for c, (a, b) in interval_cols.items():
            M = max(X[c].max()-a, b-X[c].min())
            X[c] = 1 - (X[c] - (a+b)/2).abs() / M
    return X
```

- 缺失值：小样本用均值/中位数填补，大样本用 KNN 或删除。
- 异常值：3σ 原则或箱线图 IQR 法识别，必要时缩尾（winsorize）。
- 量纲统一：必须归一化，否则大数值指标会"绑架"权重。

## 第 4 步：确定权重

- **主观赋权**：AHP（methods/ahp.md）——适合指标少、有专家经验。
- **客观赋权**：
  - 熵权法（methods/entropy-weight.md）——指标变异越大权重越大。
  - CRITIC（methods/critic.md）——兼顾变异与冲突。
- **组合赋权**（推荐，最像获奖论文）：methods/combination-weight.md
  - 乘法合成：$w_j = \frac{w_j^{subj} \cdot w_j^{obj}}{\sum w_j^{subj} \cdot w_j^{obj}}$
  - 博弈论组合：最小化主客观权重与组合权重的离差。
  - 最小信息熵：拉格朗日乘子求解。

## 第 5 步：综合评分

按数据特点选：
- 方案到正负理想解距离 → **TOPSIS**（methods/topsis.md）
- 样本少、信息不完全 → **灰色关联**（methods/grey-relational.md）
- 评语模糊（优/良/中/差） → **模糊综合评价**（methods/fuzzy-evaluation.md）
- 指标多想降维 → **PCA 得分**（methods/pca-evaluation.md）

## 第 6 步：排序与分级

- 按综合得分降序排列。
- 如需分级（优秀/良好/中等/较差）：
  - 自然断点法（Jenks）
  - 或按得分均值 ± 0.5σ 分档
  - 或用 K-Means 聚成 K 类

## 第 7 步：敏感性分析（加分项！）

获奖论文几乎必做：
1. **换权重方法**：AHP 权重 vs 熵权权重 vs 组合权重，看排名是否稳定。
2. **换归一化**：极值法 vs 标准化（Z-score）。
3. **换赋权系数**：主观权重占比 α 从 0.3 调到 0.7，看排名变化。
4. **留一法**：去掉一个指标，看结果是否鲁棒。

结果表：

| 方案 | 熵权-TOPSIS排名 | AHP-TOPSIS排名 | 组合权重排名 | 平均排名 |
|---|---|---|---|---|
| A | 3 | 2 | 2 | 2.3 |
| B | 2 | 1 | 1 | 1.3 |
| ... | | | | |

## 第 8 步：可视化（配合 huawei-chart-style skill）

- 权重：水平条形图（charts/04-评价与对比类.md）
- 综合得分排序：条形图 + 数值标注
- 多方案多维对比：雷达图
- 指标贡献：帕累托图
- 敏感性分析：排名变化折线图或龙卷风图

## 常见扣分点

- ❌ 指标不区分效益/成本型，直接归一化。
- ❌ 只用一种权重方法，被问"为什么这么赋权"答不上。
- ❌ AHP 一致性检验 CR ≥ 0.1 还硬用。
- ❌ 不做敏感性分析，结果一推就倒。
- ❌ 权重和为 1 但没说清楚怎么来的。
- ❌ 综合得分算完就结束，不分级、不解释、不画图。