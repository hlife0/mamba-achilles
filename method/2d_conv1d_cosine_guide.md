# 2d: Conv1d Cosine Similarity 实验实现指南

> **实验编号**：2d_conv1d_cosine
>
> **对应论文**：Section 4.2 最后一段 + Appendix "Cosine similarity among convolution weights"
>
> **对应图表**：Appendix Figure sup_conv1d_cos

---

## 1. 实验目标

分析 Mamba Conv1d 卷积核参数在训练过程中的**不对称性**。

具体做法：比较训练开始（epoch 0）和训练结束（epoch 209/200）时 Conv1d kernel 的各个参数向量之间的余弦相似度。

### 预期结果

各个 Conv1d kernel 的参数向量之间近乎正交（余弦相似度接近 0），说明卷积参数具有强不对称性。

---

## 2. 论文原文要点

> Mamba's asymmetry bias originates from its nonlinear convolution, specifically from the asymmetry of its convolutional kernel parameters. To investigate this asymmetry, we examined the cosine similarity between the convolutional kernels at the beginning and the end of training, and found that they are nearly orthogonal.

> Figure caption: Cosine similarity of convolutional kernel parameters between epoch 0 and epoch 209 for the two-layer Mamba model. The model configuration is consistent with that used in the phase diagram experiments. In this figure, Mamba is configured with two layers and initialized with γ = 1.

---

## 3. 具体含义

### 3.1 Conv1d kernel 结构

Mamba 的 Conv1d 权重形状为 `(d_inner, 1, kernel_size)`，即 `(d_inner, 1, 4)`。

- `d_inner = d_model * expand = 32 * 2 = 64`
- 每个通道有 4 个权重参数：`c = [c_0, c_1, c_2, c_3]`

### 3.2 余弦相似度的计算

论文计算的是 Conv1d kernel 中**不同位置参数之间**的余弦相似度。

具体来说，对于一层的 Conv1d 权重 `W ∈ R^(d_inner, 1, 4)`：
- 提取 4 个参数向量：`w_i = W[:, 0, i]`，每个 `w_i ∈ R^(d_inner)`
- 计算所有 `(w_i, w_j)` 对之间的余弦相似度
- 结果是一个 4×4 的相似度矩阵

对每一层、每个时间点（epoch 0 和 epoch 末）分别计算，得到热力图。

---

## 4. 实验配置

### 4.1 模型

```python
n_layers = 2
gamma = 1.0
d_model = 32
d_state = 128
d_conv = 4
expand = 2
```

### 4.2 数据需求

本实验需要**两个时间点的 Conv1d 权重**：
1. 初始化后（epoch 0）的 Conv1d 权重
2. 训练完成后（epoch 200）的 Conv1d 权重

**两种获取方式**：

**方案 A**：从已有的训练流程中保存 epoch 0 和 final 的 checkpoint，提取 Conv1d 权重。

**方案 B**：在标准训练脚本中增加保存初始权重的逻辑，事后分析。

推荐方案 B：修改训练脚本，在训练开始前额外保存一份 `model_epoch0.pt`。

如果使用已有 checkpoint，只需从 `model_final.pt` 提取 final 权重，并重新初始化一个模型（相同 seed）获取 epoch 0 权重。

---

## 5. 实现规格

### 5.1 核心分析代码

```python
def compute_conv1d_cosine_similarity(conv1d_weight):
    """
    Compute cosine similarity matrix between Conv1d kernel position vectors.

    Args:
        conv1d_weight: tensor of shape (d_inner, 1, kernel_size)

    Returns:
        sim_matrix: (kernel_size, kernel_size) cosine similarity matrix
    """
    kernel_size = conv1d_weight.shape[2]
    vectors = []
    for i in range(kernel_size):
        v = conv1d_weight[:, 0, i]  # (d_inner,)
        vectors.append(v)

    sim_matrix = torch.zeros(kernel_size, kernel_size)
    for i in range(kernel_size):
        for j in range(kernel_size):
            cos_sim = F.cosine_similarity(vectors[i].unsqueeze(0),
                                           vectors[j].unsqueeze(0))
            sim_matrix[i, j] = cos_sim.item()

    return sim_matrix
```

### 5.2 可视化

生成热力图，每层两个子图（epoch 0 和 epoch final）：

```python
fig, axes = plt.subplots(n_layers, 2, figsize=(8, 4*n_layers))
for layer_idx in range(n_layers):
    # Epoch 0
    sns.heatmap(sim_epoch0[layer_idx], ax=axes[layer_idx, 0],
                vmin=-1, vmax=1, cmap='RdBu_r', annot=True)
    axes[layer_idx, 0].set_title(f'Layer {layer_idx} - Epoch 0')

    # Epoch final
    sns.heatmap(sim_final[layer_idx], ax=axes[layer_idx, 1],
                vmin=-1, vmax=1, cmap='RdBu_r', annot=True)
    axes[layer_idx, 1].set_title(f'Layer {layer_idx} - Epoch {final_epoch}')
```

---

## 6. 输出规格

```
results/2d_conv1d_cosine/
├── config.json
├── cosine_similarity.json    # 数值结果
└── conv1d_cosine_heatmap.png # 热力图
```

---

## 7. 需要新建的文件

```
src/2d_conv1d_cosine.py              # 分析脚本
experiments/2d_conv1d_cosine_run.sh
```

---

## 8. 验收标准

1. 生成每层 Conv1d 在 epoch 0 和 epoch final 的余弦相似度矩阵
2. 对角线值为 1.0（自身余弦相似度）
3. 非对角线值接近 0（近乎正交）
4. 生成可读的热力图
