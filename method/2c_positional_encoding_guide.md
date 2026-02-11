# 2c: Positional Encoding 实验实现指南

> **实验编号**：2c_positional_encoding
>
> **对应论文**：Section 4.2 最后一段 — Figure 4(c)
>
> **对应图表**：Figure 4(c)
>
> **前置依赖**：标准 composite function task 数据集

---

## 1. 实验目标

验证**显式位置编码**是否能帮助 Mamba 学到对称解。

论文选择了一个 Mamba 在标准设定下**既无法学到组合解也无法学到对称解**的配置（2 层, γ=0.5），加入位置编码后观察是否能学到对称解。

### 预期结果

- 标准 Mamba (2 层, γ=0.5)：composite 和 symmetric 准确率均较低
- 加入位置编码的 Mamba (2 层, γ=0.5)：偏好 symmetric solution

---

## 2. 论文原文要点

> For a Mamba network with initialization rate γ=0.5, it cannot learn composite function task by either composite function or symmetric function. We found that if positional encoding is explicitly included, as shown in the Fig. 4(c), such Mamba network learns composite task by symmetric function. Therefore, positional encoding is also critical for learning symmetric solution.

Appendix 补充：
> We use a two-layer Mamba model with standard initialization (γ = 0.5). All other settings are kept consistent with those used in the phase diagram experiments.

---

## 3. 实现策略

### 3.1 位置编码的加入方式

在 embedding 层之后、Mamba 层之前加入标准正弦位置编码（sinusoidal positional encoding）：

```python
class MambaForCompositeWithPE(nn.Module):
    """Mamba with explicit positional encoding."""

    def __init__(self, ...):
        super().__init__()
        # 标准 Mamba backbone
        self.backbone = MambaLMHeadModel(config)

        # 位置编码
        self.pos_encoding = self._make_sinusoidal_pe(max_len=8, d_model=d_model)

    def _make_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)

    def forward(self, input_ids):
        # 需要 hook 进 backbone 的 embedding 层之后加入 PE
        ...
```

**实现细节**：由于 `MambaLMHeadModel` 内部封装了 embedding，需要通过以下方式之一加入 PE：

1. **Monkey-patch embedding 层**：在 embedding 输出后加上位置编码
2. **Hook**：注册 forward hook 在 embedding 层之后
3. **自定义子类**：继承 MambaForComposite 并覆盖 forward

推荐方案 1（最简洁）：

```python
def add_positional_encoding(model, d_model, max_len=8):
    """Add sinusoidal PE to Mamba's embedding output."""
    pe = _make_sinusoidal_pe(max_len, d_model).to(model.device)
    original_embed = model.backbone.backbone.embedding

    class EmbeddingWithPE(nn.Module):
        def __init__(self, embed, pe):
            super().__init__()
            self.embed = embed
            self.pe = pe

        def forward(self, x):
            emb = self.embed(x)
            return emb + self.pe[:, :x.shape[1], :]

    model.backbone.backbone.embedding = EmbeddingWithPE(original_embed, pe)
```

---

## 4. 实验配置

### 4.1 模型

```python
n_layers = 2
gamma = 0.5  # standard initialization
d_model = 32
d_state = 128
d_conv = 4
expand = 2
```

### 4.2 训练

与标准 phase diagram 实验一致。

### 4.3 对照实验

Figure 4(c) 同时展示两个模型：
1. **Standard structure**：标准 Mamba, 2 层, γ=0.5
2. **Pos.**：加位置编码的 Mamba, 2 层, γ=0.5

---

## 5. 评估指标

同 2b 实验：
- Composite accuracy
- Symmetric accuracy
- 训练曲线对比

---

## 6. 输出规格

```
results/2c_positional_encoding/
├── with_pe/
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   └── model_final.pt
├── standard/
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   └── model_final.pt
└── comparison_plot.png    # Figure 4(c) 复现
```

---

## 7. 需要新建/修改的文件

```
src/models/mamba_pe.py             # 位置编码包装（或直接在训练脚本中实现）
src/2c_positional_encoding.py      # 训练脚本
experiments/2c_positional_encoding_run.sh
```

---

## 8. 验收标准

1. 标准 Mamba (2层, γ=0.5)：comp 和 sym 准确率均较低
2. PE Mamba (2层, γ=0.5)：sym 准确率显著提升
3. 生成对比训练曲线图
