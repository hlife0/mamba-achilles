# 2b: All-One Convolution 实验实现指南

> **实验编号**：2b_allone_conv
>
> **对应论文**：Section 4.2 "Nonlinear convolution introduces asymmetry" — Figure 4(b)
>
> **对应图表**：Figure 4(b)
>
> **前置依赖**：标准 composite function task 数据集和训练流程

---

## 1. 实验目标

验证 Conv1d 的**权重不对称性**是 Mamba 偏好组合解（而非对称解）的根本原因。

方法：将 Mamba 所有层的 Conv1d kernel 权重设为全 1（消除卷积的不对称性），观察模型在 composite task 上是否改为偏好对称解。

### 预期结果

- 标准 Mamba (γ=1)：学到 composite solution，不学 symmetric solution
- All-one Conv1d Mamba (γ=1)：偏好 symmetric solution

---

## 2. 论文原文要点

> To better isolate the effect of convolution's asymmetry, we remove this asymmetry by setting all convolutional kernel weights to 1. As shown in the Fig. 4(b), once the asymmetry of convolution is eliminated, Mamba biases learning the symmetric solution to fit the composite task.

Appendix 补充：
> We use a five-layer Mamba model with small initialization (γ = 1), while all other settings are consistent with those used in the phase diagram experiments.

---

## 3. 实现策略

### 3.1 模型修改

不需要修改模型架构。只需在训练过程中**冻结 Conv1d 权重为全 1**：

```python
def set_allone_conv1d(model):
    """Set all Conv1d weights to 1 and freeze them."""
    for layer in model.backbone.backbone.layers:
        conv1d = layer.mixer.conv1d
        with torch.no_grad():
            conv1d.weight.fill_(1.0)
            if conv1d.bias is not None:
                conv1d.bias.fill_(0.0)
        conv1d.weight.requires_grad = False
        if conv1d.bias is not None:
            conv1d.bias.requires_grad = False
```

### 3.2 关键点

- Conv1d 权重在初始化后**立即设为全 1**
- 训练过程中 Conv1d 权重**不更新**（requires_grad=False）
- 偏置也设为 0 并冻结
- 其他参数正常训练

---

## 4. 实验配置

### 4.1 模型

```python
n_layers = 5
gamma = 1.0  # small initialization
d_model = 32
d_state = 128
d_conv = 4
expand = 2
```

### 4.2 训练

与标准 phase diagram 实验完全一致：
- Optimizer: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01)
- LR: warmup 10 epochs (1e-5 → 2.5e-4) + cosine decay
- Batch size: 2048
- Epochs: 200
- Gradient clipping: max_norm=1.0

### 4.3 数据集

使用**标准** composite function task（与 Experiment 0 完全相同）。

### 4.4 对照实验

Figure 4(b) 同时展示了两个模型的结果：
1. **Standard structure**：标准 Mamba, n_layers=5, γ=1.0（作为对照）
2. **1Conv1d**：全 1 Conv1d Mamba, n_layers=5, γ=1.0

两者使用相同数据和训练配置，唯一区别是 Conv1d 权重。

---

## 5. 评估指标

对于测试 pair (4,3)：
- **Composite accuracy**：预测 = (key-10)%100 的比率
- **Symmetric accuracy**：预测 = (key-6)%100 的比率

Figure 4(b) 展示训练过程中两个指标的变化曲线。

---

## 6. 输出规格

```
results/2b_allone_conv/
├── allone/                    # All-one Conv1d 结果
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   └── model_final.pt
├── standard/                  # 标准 Mamba 对照
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   └── model_final.pt
└── comparison_plot.png        # Figure 4(b) 复现
```

### 6.1 可视化

Figure 4(b) 是训练曲线图：
- x 轴：epoch
- y 轴：accuracy
- 两条曲线：comp. 和 sym.
- 两组：1Conv1d 和 Standard structure

---

## 7. 需要新建/修改的文件

```
src/2b_allone_conv.py              # 训练脚本（基于 0_train.py，增加 --allone_conv 参数）
experiments/2b_allone_conv_run.sh
```

可以复用 `src/0_train.py` 的大部分代码，只增加 `--allone_conv` flag 来控制是否冻结 Conv1d。

---

## 8. 验收标准

1. All-one Conv1d 模型偏好 symmetric solution（sym acc 显著高于 standard）
2. Standard 模型偏好 composite solution（comp acc 显著高于 sym acc）
3. 生成对比训练曲线图
