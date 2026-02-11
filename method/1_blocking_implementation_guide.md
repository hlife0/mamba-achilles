# Information Blocking 实验实现指南

> **实验编号**：1_blocking
>
> **对应论文**：Section 4.1 "SSM does not function in composite tasks" — Information blocking 段落
>
> **对应图表**：Figure 6(a)
>
> **前置依赖**：需要一个训练好的、学到了 composite solution 的 Mamba checkpoint（来自 0_train phase diagram 实验）

---

## 1. 实验目标

验证 Mamba 的 SSM 模块在 composite function task 中**不传递关键信息**。

具体来说：在 SSM 矩阵 S 中，手动将 key 和两个 anchor token 到所有下游 token 的注意力条目清零（"blocking"），观察模型输出是否发生变化。如果输出几乎不变，说明 SSM 没有承担 key/anchor 信息的传递，信息完全由 Conv1d 传递。

### 预期结果

对 16 个 anchor pair 分别统计：blocking 后的预测 == 原始预测 的比率应接近 **100%**。

---

## 2. 论文原文要点（arxiv20260121.tex 第333-335行）

> We applied a causal intervention approach, manually blocking all information flow from the key and both anchors to the downstream tokens.
> The (i,j) element of S represents how much token i attends to token j. If this value is set to 0, it implies that token i cannot receive information from token j through the SSM.
> Our blocking mechanism is implemented by zeroing out the specific entries in S corresponding to the connections we wish to block.
> For various anchor pairs, the output after cutting these connections remains nearly identical to the original output, indicating that the SSM plays little role in transmitting this information.

> Each anchor pair is evaluated using 480 randomly generated sequences. (第351行)

---

## 3. Mamba2 内部结构与 Hook 点

### 3.1 模型层级结构

```
MambaForComposite
  .backbone (MambaLMHeadModel)
    .backbone (MixerModel)
      .embedding
      .layers[i] (Block)
        .norm (RMSNorm)
        .mixer (Mamba2)             ← 我们需要 hook 这里
          .in_proj (Linear)
          .conv1d (Conv1d, depthwise, kernel=4)
          .act (SiLU)
          内部调用: mamba_chunk_scan_combined(x, dt, A, B, C, ...)
      .norm_f (RMSNorm)
    .lm_head (Linear)
```

### 3.2 SSM 矩阵的数学定义

论文定义（第194-197行）：

```
Mask = F(A₀, dt)           # 下三角指数衰减掩码
I = Repeat(C Bᵀ, Nₕ)       # 类 attention score
S = Mask ⊙ I                # SSM 矩阵, shape=(Nₕ, s, s)
Y = S X̃ + X                # SSM 输出，+X 是 skip connection (D·x 项)
```

其中 S[i,j] 代表 token i 对 token j 的注意力权重。

### 3.3 Mamba2 代码中的对应关系

在 `mamba_ssm/modules/mamba2.py` 的 forward 方法中（非 fused 路径，第207-274行）：

```python
# 1. 投影
zxbcdt = self.in_proj(u)                           # (B, L, d_in_proj)

# 2. 分割
z0, x0, z, xBC, dt = torch.split(zxbcdt, [...], dim=-1)

# 3. Conv1d + SiLU (非线性卷积)
xBC = causal_conv1d_fn(xBC.T, conv1d.weight, bias, activation).T

# 4. 分割出 x, B, C
x, B, C = torch.split(xBC, [d_ssm, ngroups*d_state, ngroups*d_state], dim=-1)

# 5. SSM 计算 (在 ssd_combined.py 中)
y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, ...)
```

在 `_mamba_chunk_scan_combined_fwd`（ssd_combined.py 第281-331行）中：

```python
dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, ...)      # 累积衰减
states = _chunk_state_fwd(B, x, dt, dA_cumsum, ...) # 块状态
states = _state_passing_fwd(states, ...)             # 跨块传递
CB = _bmm_chunk_fwd(C, B, chunk_size, ...)           # CB 矩阵（类 attention）
out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, ...)
```

**关键**：SSM 矩阵 S 在这里是通过 `CB`（C⊗Bᵀ）和 `dA_cumsum`（指数衰减掩码）隐式计算的。它在 `_chunk_scan_fwd` 内部被组合为 `L * CB`（对应 Mask ⊙ I = S），然后乘以 x 得到输出。

---

## 4. 实现策略

### 4.1 核心难点

由于 `mamba_chunk_scan_combined` 是一个融合的 Triton kernel，**无法直接 hook 中间的 SSM 矩阵 S**。

**解决方案**：不用 fused path，而是在 Mamba2.forward 中强制走非 fused 路径（`self.use_mem_eff_path = False`），然后用一个**自定义的 SSM 计算函数**替代 `mamba_chunk_scan_combined`，在其中显式计算 S 矩阵并执行 blocking。

### 4.2 推荐实现方案：自定义 SSM 前向

不修改 mamba_ssm 源码。而是：

1. 加载训练好的模型
2. 对每层 Mamba2 模块设置 `use_mem_eff_path = False`，确保走非 fused 路径
3. 使用 `mamba_chunk_scan` 的参考实现（`ssd_chunk_scan_combined_ref` 或 `mamba_chunk_scan`）替代 fused kernel
4. 在参考实现中，显式构造 S 矩阵，执行 blocking，再计算输出

参考实现位于 `ssd_combined.py` 第621行 `ssd_chunk_scan_combined_ref`，以及第584行 `mamba_chunk_scan`。

### 4.3 关键代码路径

基于 `mamba_chunk_scan`（ssd_combined.py 第584-618行），SSM 计算步骤为：

```python
def mamba_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size).float()
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)

    # 1. 块状态
    states = chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True)
    # 2. 跨块传递
    states = state_passing(states, dA_cumsum[:, :, :, -1])
    # 3. 块内扫描（这里隐含了 S 矩阵）
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out
```

**blocking 需要在步骤 3 中介入**：在 `chunk_scan` 计算 S = L ⊙ CB 后、乘以 x 之前，把 S 中从 key/anchor 位置到下游的条目清零。

---

## 5. 详细实现规格

### 5.1 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| checkpoint_path | str | 训练好的 model_final.pt 路径 |
| config_path | str | 对应的 config.json 路径 |
| num_samples_per_pair | int | 每个 anchor pair 的测试序列数，论文为 **480** |
| seed | int | 随机种子 |

### 5.2 推荐的 Checkpoint 配置

论文附录 E.1.2（第866行）使用的消融实验配置：

```
n_layers = 5
gamma = 1.0
d_model = 32
d_state = 128
d_conv = 4
expand = 2
```

该配置下 Mamba 能学到 composite solution（accuracy ~90%+），适合做 blocking 实验。

如果 phase diagram 尚未完成该配置的训练，需要先单独训练一个。

### 5.3 数据生成

对 16 个 anchor pair 各生成 480 个测试序列：

```python
ANCHOR_PAIRS = [(i, j) for i in range(1, 5) for j in range(1, 5)]  # 16 对

for pair in ANCHOR_PAIRS:
    sequences = []
    for _ in range(480):
        # 随机 key (20-99)
        key = random.randint(20, 99)
        # 随机 position (0-5)
        pos = random.randint(0, 5)
        # 构造长度为 8 的序列
        seq = [random.randint(0, 99) for _ in range(8)]
        seq[pos] = key
        seq[pos + 1] = pair[0]
        seq[pos + 2] = pair[1]
        sequences.append(seq)
```

**注意**：这里不需要模余分离约束，因为这不是评估泛化性能，而是评估 blocking 干预的效果。只需确保序列格式正确。

### 5.4 Blocking 逻辑

对于一个序列，key 在位置 p，anchor1 在 p+1，anchor2 在 p+2：

```python
# 在 SSM 矩阵 S 上执行 blocking
# S shape: (nheads, seqlen, seqlen)
# S[h, i, j] = token i 对 token j 的注意力

# 阻断 key 到所有下游 token 的信息流
S[:, p+1:, p] = 0      # key → 下游 (包括 anchor 到后面的所有 token)

# 阻断 anchor1 到所有下游 token 的信息流
S[:, p+2:, p+1] = 0    # anchor1 → 下游

# 阻断 anchor2 到所有下游 token 的信息流
S[:, p+3:, p+2] = 0    # anchor2 → 下游
```

注意：这需要在**每一层**的 SSM 中都执行。

### 5.5 评估指标

```python
# 对每个 anchor pair 统计
match_count = 0
for seq in sequences:
    original_pred = model_forward_normal(seq).argmax()
    blocked_pred = model_forward_blocked(seq, key_pos=p).argmax()
    if original_pred == blocked_pred:
        match_count += 1

match_rate = match_count / len(sequences)  # 预期接近 1.0
```

### 5.6 输出

生成柱状图（对应论文 Figure 6a）：

- **x 轴**：16 个 anchor pair（"11", "12", ..., "44"）
- **y 轴**：match rate（blocking 后预测 == 原始预测 的比率）
- **预期**：所有柱子接近 100%

### 5.7 输出文件

```
results/1_blocking/
├── config.json              # 实验配置
├── blocking_results.json    # 每个 pair 的详细数字结果
├── blocking_bar_chart.png   # Figure 6a 复现图
└── stdout.log               # 运行日志
```

---

## 6. 代码结构

### 6.1 需要新建的文件

```
src/1_blocking.py              # 主实验脚本
experiments/1_blocking_run.sh   # 运行脚本
```

### 6.2 可复用的共享模块

```
src/models/mamba_wrapper.py     # 加载 checkpoint
src/data/composite_task.py      # 序列格式参考（但本实验自行生成指定 pair 的序列）
```

### 6.3 需要新建的共享模块

```
src/models/mamba_hooks.py       # Mamba2 内部 hook 工具（blocking 和 substitution 共用）
```

---

## 7. `mamba_hooks.py` 规格

此文件为 blocking 和 substitution 实验的共享基础设施。

### 7.1 核心功能：手动 SSM 前向

提供一个函数，替代 `mamba_chunk_scan_combined`，在其中：
1. 显式计算 SSM 矩阵 S（或其等价分解形式 L ⊙ CB）
2. 允许通过回调函数修改 S
3. 计算修改后的输出

```python
def manual_ssm_forward(x, dt, A, B, C, chunk_size, D=None, z=None,
                       dt_bias=None, dt_softplus=False,
                       ssm_modifier_fn=None):
    """
    手动实现 SSM 前向，支持中间干预。

    Args:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        dt_softplus: bool
        ssm_modifier_fn: callable(S, batch_metadata) -> S_modified
            对 SSM 矩阵的修改回调函数

    Returns:
        out: (batch, seqlen, nheads, headdim)
    """
    pass
```

### 7.2 禁用 fused path

```python
def disable_fused_path(model):
    """
    将模型所有 Mamba2 层的 use_mem_eff_path 设为 False，
    强制走非 fused 路径以便 hook。

    Args:
        model: MambaForComposite 实例
    """
    for layer in model.backbone.backbone.layers:
        layer.mixer.use_mem_eff_path = False
```

### 7.3 Monkey-patch SSM 计算

```python
def patch_mamba2_with_modifier(mamba2_module, ssm_modifier_fn):
    """
    Monkey-patch 一个 Mamba2 模块，使其 SSM 计算过程中
    调用 ssm_modifier_fn 修改 SSM 矩阵。

    实现方式：替换 mamba2_module.forward 中对 mamba_chunk_scan_combined 的调用，
    改为调用 manual_ssm_forward，传入 ssm_modifier_fn。

    Args:
        mamba2_module: Mamba2 实例
        ssm_modifier_fn: callable
    """
    pass
```

### 7.4 SSM 矩阵的简化计算

对于序列长度为 8 的情况，chunk_size 通常 >= 8（默认 256），整个序列在一个 chunk 内。此时 SSM 矩阵可以简化为：

```python
# 在单个 chunk 内
# dA: (batch, nheads, seqlen)
dt_processed = F.softplus(dt + dt_bias)          # (B, L, H)
dA = dt_processed * A                             # (B, L, H), A 为负值

# 指数衰减掩码 L: (seqlen, seqlen), 下三角
# L[i,j] = exp(sum(dA[k] for k in range(j+1, i+1))) for i > j, else 0
dA_cumsum = torch.cumsum(dA, dim=1)               # 沿序列维度累积
# L[i,j] = exp(dA_cumsum[i] - dA_cumsum[j])，仅下三角

# CB 矩阵: (batch, nheads, seqlen, seqlen)
# CB[b,h,i,j] = C[b,i,:] @ B[b,j,:] （对每个 head/group）
CB = torch.einsum("blgn, bsgn -> bgls", C, B)     # (B, ngroups, L, L)

# SSM 矩阵 S = L ⊙ CB
S = L * CB                                         # (B, H, L, L)

# 输出 (不含 D·x skip connection)
y_ssm = torch.einsum("bhij, bjhp -> bihp", S, x_tilde)  # (B, L, H, P)

# 加上 skip connection
y = y_ssm + D * x                                  # D·x 项
```

**关键洞察**：由于 `Y = S X̃ + X`（或 `Y = S X̃ + D·X`），即使把 S 中的条目清零，X 仍然通过 skip connection 保留。这正是 blocking 实验能成功的原因 — Conv1d 的信息已经编码在 X 中。

---

## 8. `1_blocking.py` 主脚本规格

### 8.1 命令行接口

```python
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to model_final.pt')
parser.add_argument('--config', type=str, required=True,
                    help='Path to config.json from training')
parser.add_argument('--num_samples', type=int, default=480,
                    help='Number of samples per anchor pair')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='results/1_blocking')
```

### 8.2 主流程

```python
def main():
    # 1. 加载配置和模型
    config = load_config(args.config)
    model = MambaForComposite(
        n_layers=config['n_layers'],
        gamma=config['gamma'],
        ... # 其他参数
    )
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # 2. 禁用 fused path
    disable_fused_path(model)

    # 3. 对 16 个 anchor pair 分别实验
    results = {}
    for a1 in range(1, 5):
        for a2 in range(1, 5):
            pair = (a1, a2)

            # 3a. 生成 480 个序列
            sequences, key_positions = generate_sequences(pair, args.num_samples, args.seed)

            # 3b. 正常前向，获取原始预测
            with torch.no_grad():
                original_preds = model(sequences).argmax(dim=-1)

            # 3c. Blocking 前向
            # 对每层 Mamba2 注册 blocking modifier
            def blocking_modifier(S, key_pos):
                """将 key/anchor1/anchor2 到下游的 SSM 条目清零"""
                p = key_pos
                S[:, :, p+1:, p] = 0    # key → downstream
                S[:, :, p+2:, p+1] = 0  # anchor1 → downstream
                S[:, :, p+3:, p+2] = 0  # anchor2 → downstream
                return S

            with torch.no_grad():
                blocked_preds = model_forward_with_blocking(
                    model, sequences, key_positions, blocking_modifier
                )

            # 3d. 计算 match rate
            match_rate = (original_preds == blocked_preds).float().mean().item()
            results[f"{a1}{a2}"] = match_rate

    # 4. 保存结果 & 生成图表
    save_results(results, args.output_dir)
    plot_bar_chart(results, args.output_dir)
```

### 8.3 序列生成函数

```python
def generate_sequences(anchor_pair, num_samples, seed):
    """
    为指定 anchor pair 生成测试序列。

    Args:
        anchor_pair: (int, int), e.g. (1, 2)
        num_samples: int, 论文中为 480
        seed: int

    Returns:
        sequences: torch.LongTensor, shape=(num_samples, 8)
        key_positions: torch.LongTensor, shape=(num_samples,)
    """
    rng = random.Random(seed)
    sequences = []
    key_positions = []

    for _ in range(num_samples):
        pos = rng.randint(0, 5)  # key 位置 0-5
        key = rng.randint(20, 99)
        seq = [rng.randint(0, 99) for _ in range(8)]
        # 确保填充位置不含 anchor 值 1-4（避免意外 anchor pair）
        for i in range(8):
            if i not in (pos, pos+1, pos+2):
                while seq[i] in (1, 2, 3, 4):
                    seq[i] = rng.randint(5, 99)
        seq[pos] = key
        seq[pos + 1] = anchor_pair[0]
        seq[pos + 2] = anchor_pair[1]
        sequences.append(seq)
        key_positions.append(pos)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(key_positions, dtype=torch.long)
```

---

## 9. 可视化规格

### 9.1 柱状图（Figure 6a）

```python
def plot_bar_chart(results, output_dir):
    """
    绘制 blocking 实验柱状图。

    Args:
        results: dict, key="11","12",...,"44", value=match_rate (0-1)
        output_dir: str

    样式：
        - x 轴: anchor pair 名称
        - y 轴: match rate (0% - 100%)
        - 每个柱子标注数值
        - 标题: "Accuracy under Information Blocking"
        - 虚线标注 100% 参考线
    """
    pass
```

---

## 10. 注意事项

### 10.1 fused vs non-fused path

Mamba2 默认使用 fused Triton kernel（`use_mem_eff_path=True`），此路径下 Conv1d + SSM 全部融合在一个 kernel 中，无法 hook。**必须**设置 `use_mem_eff_path = False`。

注意：非 fused 路径的数值结果与 fused 路径可能有微小浮点差异，但对 argmax 预测不影响。

### 10.2 序列长度与 chunk_size

本实验序列长度为 8，远小于默认 chunk_size（256）。因此整个序列只有一个 chunk，SSM 矩阵 S 的形状简化为 `(batch, nheads, 8, 8)`。不需要处理跨 chunk 的状态传递。

### 10.3 batch 处理

由于不同序列的 key 位置可能不同，blocking 的 mask 对每个样本不同。有两种处理方式：

- **方案 A**（简单）：逐样本推理，每个样本独立 blocking。效率低但实现简单。
- **方案 B**（高效）：按 key_position 分组，同组内 batch 推理。最多 6 组（position 0-5）。

**推荐方案 B**，因为总共只有 16×480 = 7,680 个样本，按 position 分组后每组约 1,280 个样本，计算量很小。

### 10.4 关于 "每一层都 blocking"

论文没有明确说只 blocking 某一层还是所有层。从实验设计意图来看，应该是**所有层同时 blocking**，因为目的是证明 SSM 整体不传递信息。

---

## 11. 验收标准

1. 对 16 个 anchor pair 各 480 个序列完成 blocking 实验
2. 生成 blocking_results.json 包含所有数值
3. 生成 blocking_bar_chart.png 柱状图
4. 所有 anchor pair 的 match rate > 95%（论文结果为接近 100%）
5. 实验可通过命令行一键运行
