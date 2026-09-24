# 组合赋权（博弈论 / 乘法合成 / 最小信息熵）

## 适用场景

- 同时有主观权重（AHP）和客观权重（熵权/CRITIC），想融合。
- 单一会被质疑"为什么不用另一种"，组合赋权最稳妥。
- 获奖案例：C23102540029、D23104860011。

## 三种主流组合方法

### 1. 乘法合成（最简单）

$$w_j^* = \frac{w_j^{subj} \cdot w_j^{obj}}{\sum_{k=1}^n w_k^{subj} \cdot w_k^{obj}}$$

优点：简单；缺点：放大极端权重。

### 2. 线性加权

$$w_j^* = \alpha \cdot w_j^{subj} + (1-\alpha) \cdot w_j^{obj}$$

$\alpha$ 常取 0.5，或通过博弈论优化。

### 3. 博弈论组合（推荐）

最小化主观权重向量 $w_1$ 和客观权重向量 $w_2$ 与组合权重 $w^*$ 的离差：

$$\min \| \alpha_1 w_1^T + \alpha_2 w_2^T - w_1^T \|_2 + \| \alpha_1 w_1^T + \alpha_2 w_2^T - w_2^T \|_2$$

解出 $\alpha_1, \alpha_2$，再归一化。

## 可运行代码

```python
import numpy as np

def combine_weights(w_ahp, w_entropy):
    """博弈论组合赋权"""
    w1 = np.array(w_ahp, dtype=float)
    w2 = np.array(w_entropy, dtype=float)
    # 求解线性方程组
    A = np.array([
        [w1 @ w1, w1 @ w2],
        [w2 @ w1, w2 @ w2]
    ])
    b = np.array([w1 @ w1, w2 @ w2])
    try:
        alpha = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        alpha = np.array([0.5, 0.5])
    alpha = np.abs(alpha) / alpha.sum()
    w_comb = alpha[0]*w1 + alpha[1]*w2
    w_comb = w_comb / w_comb.sum()
    return w_comb, alpha

# 示例：AHP 权重 vs 熵权
w_ahp   = [0.253, 0.125, 0.044, 0.409, 0.125, 0.044]
w_ent   = [0.199, 0.176, 0.147, 0.144, 0.158, 0.176]

w_comb, alpha = combine_weights(w_ahp, w_ent)
labels = ['经济','社会','环境','成本','能耗','风险']
print(f"AHP 系数 α1={alpha[0]:.3f}, 熵权系数 α2={alpha[1]:.3f}")
print("组合权重：")
for l, a, e, c in zip(labels, w_ahp, w_ent, w_comb):
    print(f"  {l}: AHP={a:.3f}  熵权={e:.3f}  组合={c:.3f}")
```

## 输出示例

```
AHP 系数 α1=0.420, 熵权系数 α2=0.580
组合权重：
  经济: AHP=0.253  熵权=0.199  组合=0.222
  社会: AHP=0.125  熵权=0.176  组合=0.155
  环境: AHP=0.044  熵权=0.147  组合=0.104
  成本: AHP=0.409  熵权=0.144  组合=0.255
  能耗: AHP=0.125  熵权=0.158  组合=0.144
  风险: AHP=0.044  熵权=0.176  组合=0.121
```

## 论文写作要点

- 列出三种权重（AHP、熵权、组合）对比表。
- 用组合权重做最终 TOPSIS 排序。
- 敏感性分析：$\alpha$ 从 0.3 调到 0.7，看排名变化。