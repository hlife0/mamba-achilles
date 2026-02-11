# 2e: Transformer + Convolution Phase Diagram 实验实现指南

> **实验编号**：2e_transformer_conv
>
> **对应论文**：Section 4.3 "Transformer with convolution biases asymmetric solution"
>
> **对应图表**：Figure 5 (phase_Transformer.png)
>
> **注意**：此实验需要实现一个 Transformer 模型，复杂度较高

---

## 1. 实验目标

验证将 Conv1d 引入 Transformer 后，Transformer 也会表现出类似 Mamba 的**不对称偏好**（偏好组合解而非对称解）。

### 预期结果

- 标准 Transformer（已知在不同 γ 下可分别学到 composite 或 symmetric solution）
- 加入 Conv1d 的 Transformer：要么无法泛化（小 γ），要么偏好 composite solution（大 γ），不再学到 symmetric solution

---

## 2. 论文原文要点

> We insert a convolution after the input to the Transformer and before the attention module (applying it to Q, K, and V). This is analogous to how Mamba applies nonlinear convolution before the SSM module.

> With convolution component, Transformer either can not generalize for small γ or fit data by symmetric solution for relative large γ.

Appendix "Transformer configurations"：
> The dimension of model is set to 128. The dimensions of the query and key vectors are set to 128, while the value vectors have a dimension of 256. The Transformer's feedforward network (FFN) uses a hidden dimension of 128, and configured with only a single attention head.

**注意**：Figure 5 caption 说 "(a) and (b) show the composite and symmetric solution accuracy of the Transformer after adding nonlinear convolution"。

---

## 3. 架构设计

### 3.1 Transformer with Convolution

```
Input → Embedding → [Transformer Block × N layers] → LM Head → Last Token Logits
```

每个 Transformer Block：
```
x → LayerNorm → Conv1d+SiLU → Linear(Q,K,V) → Attention → Residual → LayerNorm → FFN → Residual
```

或者更精确地，论文说 "after the input to the Transformer and before the attention module (applying it to Q, K, V)"：

```
x → LayerNorm → Linear → Conv1d+SiLU → Split(Q,K,V) → Attention → Residual → LayerNorm → FFN → Residual
```

### 3.2 关键参数

```python
d_model = 128
d_qk = 128      # query, key dimension
d_v = 256        # value dimension (2× expansion, mirroring Mamba's expand=2)
d_ffn = 128      # FFN hidden dimension
n_heads = 1
activation = 'silu'  # SiLU, same as Mamba
conv_kernel_size = 4  # same as Mamba
```

### 3.3 Conv1d 的具体位置

在 attention 计算之前，对线性投影后的 Q、K、V 分别应用因果 Conv1d + SiLU：

```python
class TransformerBlockWithConv(nn.Module):
    def __init__(self, d_model, d_qk, d_v, d_ffn, conv_kernel_size=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # QKV projection
        self.qk_proj = nn.Linear(d_model, 2 * d_qk)
        self.v_proj = nn.Linear(d_model, d_v)

        # Conv1d applied to Q, K, V
        self.conv_q = nn.Conv1d(d_qk, d_qk, conv_kernel_size, padding=conv_kernel_size-1, groups=1)
        self.conv_k = nn.Conv1d(d_qk, d_qk, conv_kernel_size, padding=conv_kernel_size-1, groups=1)
        self.conv_v = nn.Conv1d(d_v, d_v, conv_kernel_size, padding=conv_kernel_size-1, groups=1)

        # Output projection
        self.out_proj = nn.Linear(d_v, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        # Pre-norm attention with conv
        h = self.norm1(x)
        qk = self.qk_proj(h)
        q, k = qk.chunk(2, dim=-1)
        v = self.v_proj(h)

        # Apply causal conv + SiLU
        q = F.silu(self.conv_q(q.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2))
        k = F.silu(self.conv_k(k.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2))
        v = F.silu(self.conv_v(v.transpose(1,2))[:, :, :x.shape[1]].transpose(1,2))

        # Scaled dot-product attention (causal)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_qk ** 0.5)
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        attn.masked_fill_(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        x = x + self.out_proj(out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 4. Phase Diagram 配置

### 4.1 扫描参数

与 Mamba phase diagram 类似：
- **层数**：2, 3, 4, 5, 6, 7（根据图表确定）
- **Gamma**：0.1, 0.2, ..., 1.0, 1.5, 2.0
- **种子**：3 个固定种子取平均

### 4.2 训练配置

与标准实验一致（参考 Training setup）。

### 4.3 数据集

使用标准 composite function task。

---

## 5. 输出规格

```
results/2e_transformer_conv/
├── L{n}_G{gamma}/
│   ├── config.json
│   ├── metrics.json
│   ├── training_log.csv
│   └── model_final.pt
├── phase_diagram_comp.png    # Figure 5(a)
└── phase_diagram_symm.png    # Figure 5(b)
```

### 5.1 可视化

Phase diagram 热力图（同 Experiment 0）：
- x 轴：γ (initialization rate)
- y 轴：layers
- 颜色：accuracy

生成两张图：composite accuracy 和 symmetric accuracy。

---

## 6. 需要新建的文件

```
src/models/transformer_conv.py          # Transformer + Conv1d 模型
src/2e_transformer_conv.py              # 训练脚本
src/2e_transformer_conv_visualize.py    # Phase diagram 可视化
experiments/2e_transformer_conv_run.sh
```

---

## 7. 注意事项

### 7.1 公平比较

论文强调公平比较：
- Transformer 和 Mamba 使用相同的训练配置
- 参数量需可比（通过调整 FFN 维度）
- 使用相同的激活函数（SiLU）
- 使用相同的 Conv1d kernel size（4）

### 7.2 因果卷积

Conv1d 必须是**因果**的（causal），即位置 i 的输出只依赖 [i-3, i-2, i-1, i]。实现时使用 left padding。

### 7.3 复杂度

此实验是 5 个实验中复杂度最高的，因为需要从零实现 Transformer + Conv1d 模型。建议最后实现。

---

## 8. 验收标准

1. 实现 Transformer + Conv1d 模型，参数量与 Mamba 可比
2. 完成 phase diagram 扫描
3. Conv1d Transformer 在大 γ 时偏好 composite solution（与标准 Transformer 行为不同）
4. 生成 phase diagram 热力图
