# 2a: Fully Symmetric Setting 实验实现指南

> **实验编号**：2a_fully_symmetric
>
> **对应论文**：Section 4.2 第一段 + Appendix "Experiment under full symmetry"
>
> **对应图表**：Figure 4(a), Appendix Figure sup_acc_totalsymmetry, sup_loss_totalsymmetry

---

## 1. 实验目标

验证在**完全对称设定**下（即对称锚点对拥有相同的函数，且函数不通过基本函数组合得到），Mamba 是否能通过对称性泛化。

论文发现：标准 Mamba 可以在 "34" 的测试集上达到 100% 准确率，但无法通过对称性学到 "43" 的结果。

### 预期结果

- "34" 测试准确率接近 100%
- "43" 测试准确率接近随机猜测水平

---

## 2. 论文原文要点

> We design a task with fully symmetric setting as follows. The symmetric anchor pairs have the same function but they do not composite by elementary functions; only function "43" is unseen during the training. This leaves the symmetric solution as the only possible one.

> To validate Mamba's difficulty in solving the composite function task under fully symmetric settings, we set dimension of model to 128, while keeping all other settings consistent with those used in the phase diagram experiments.

---

## 3. 与标准 composite task 的关键区别

### 3.1 函数定义变化

标准 composite task 中，4 个基本函数可以组合。在 fully symmetric setting 中：

- 每个 anchor pair 的函数是**独立定义**的（不通过基本函数组合）
- 对称 pair 拥有**相同的函数**：pair (a1, a2) 和 pair (a2, a1) 的 label 计算方式相同
- 只有 pair "43" 在训练时不可见

具体来说，16 个 pair 的函数满足：
- f(a1,a2) = f(a2,a1) 对所有 pair 成立（对称性）
- pair (4,3) 从训练集中排除
- 如果模型学会了对称性，它应该能从 pair (3,4) 的函数推断出 pair (4,3) 的函数

### 3.2 数据集构造

根据论文附录 "Experiment under full symmetry"：

- **d_model = 128**（而非标准的 32）
- 层数扫描：2, 3, 4, 5, 6
- 初始化：标准初始化 γ = 0.5
- 其他设定与 phase diagram 一致
- 使用 3 个固定随机种子

### 3.3 函数设计

为了使对称 pair 有相同的函数，一种自然的设计是：

对于 anchor pair (a1, a2)，令排序后的 pair 为 (min(a1,a2), max(a1,a2))，然后：

| 排序后的 pair | 函数 |
|---|---|
| (1,1) | (key + 5) % 100 |
| (1,2) | (key + 6) % 100 |
| (1,3) | (key + 3) % 100 |
| (1,4) | (key - 3) % 100 |
| (2,2) | (key + 1) % 100 |
| (2,3) | (key - 1) % 100 |
| (2,4) | (key - 7) % 100 |
| (3,3) | (key - 2) % 100 |
| (3,4) / (4,3) | (key - 6) % 100 |
| (4,4) | (key - 8) % 100 |

注意：这些函数**不是**通过 f_a1 ∘ f_a2 组合得到的，而是独立赋值，且 (3,4) 和 (4,3) 共享同一个函数。

**重要**：具体函数值需根据论文实际代码确定。上面只是一个符合约束的示例。关键约束是：
1. f(a1,a2) = f(a2,a1)（对称性）
2. 函数值不能通过基本函数组合推导出来

---

## 4. 实现规格

### 4.1 模型配置

```python
d_model = 128  # 注意：比标准实验大
d_state = 128
d_conv = 4
expand = 2
n_layers = [2, 3, 4, 5, 6]  # 扫描
gamma = 0.5  # 标准初始化
seeds = [seed1, seed2, seed3]  # 3 个固定种子
```

### 4.2 训练配置

与标准 phase diagram 实验一致：
- Optimizer: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01)
- LR: warmup 10 epochs (1e-5 → 2.5e-4) + cosine decay
- Batch size: 2048
- Epochs: 200
- Gradient clipping: max_norm=1.0

### 4.3 数据集

需要新建一个 `FullySymmetricDataset`（或修改 `CompositeFunctionDataset` 增加 `symmetric_mode` 参数）：

- 训练集：15 个 pair（排除 (4,3)），300,000 样本
- 测试集：仅 pair (4,3)
- 序列长度：8
- 词汇量：100
- 函数满足对称性：f(a1,a2) = f(a2,a1)

### 4.4 评估

- 记录 "34" 对应函数的准确率
- 记录 "43"（测试 pair）的准确率
- 由于函数对称，"43" 的正确答案 = "34" 的正确答案

---

## 5. 输出规格

```
results/2a_fully_symmetric/
├── L{n}_S{seed}/
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   ├── model_final.pt
│   └── model_best.pt
└── summary.json  # 汇总所有配置的结果
```

### 5.1 可视化

复现 Appendix Figure sup_acc_totalsymmetry：
- x 轴：随机种子
- y 轴：层数
- 颜色：准确率

---

## 6. 需要新建/修改的文件

```
src/data/fully_symmetric_task.py   # 或修改 composite_task.py
src/2a_fully_symmetric.py          # 训练脚本
experiments/2a_fully_symmetric_run.sh
```

---

## 7. 验收标准

1. 完成 5 层数 × 3 种子 = 15 个训练实验
2. "34" 测试准确率接近 100%（至少在某些配置下）
3. "43" 测试准确率接近随机猜测
4. 生成汇总结果和可视化图
