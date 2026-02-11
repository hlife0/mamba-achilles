# Information Substitution 实验实现指南

> **实验编号**：1_substitution
>
> **对应论文**：Section 4.1 "SSM does not function in composite tasks" — Information substitution 段落
>
> **对应图表**：Figure 6(b)
>
> **前置依赖**：
> - 需要一个训练好的、学到了 composite solution 的 Mamba checkpoint（同 1_blocking）
> - 共享模块 `src/models/mamba_hooks.py`（与 1_blocking 共用）

---

## 1. 实验目标

验证 Mamba 的 Conv1d **已经编码了足够的信息来完成 composite function task**。

具体来说：对于一个 anchor pair 非 "43" 的序列，将其 Conv1d+SiLU 之后的**下游 token 隐藏状态**替换为一个 anchor pair 为 "43" 的参考序列的对应状态（两个序列除 anchor pair 不同外，其余元素完全相同）。如果替换后模型输出坍缩到 "43" 对应的结果，说明 Conv1d 后的状态已经包含了所有关键信息。

### 预期结果

对 16 个 anchor pair 各 480 个序列：替换后的预测 == "43" 参考序列的预测，比率接近 **100%**。

---

## 2. 论文原文要点（arxiv20260121.tex 第338-345行）

> To further verify that Mamba relies on convolution for information transmission, we conducted an information substitution experiment: if Mamba encodes all necessary information through convolution, then transferring the resulting state to another sequence should enable it to produce the same output as the original.
>
> For all sequences with anchor pairs other than '43', as illustrated in Fig.5, we replaced the post-convolution hidden states of the downstream tokens with those from a "43" sequence, where all other elements are identical except for the anchor pair.
>
> We found that the outputs of nearly all anchor pairs collapse to the output corresponding to the "43" anchor pair.

> Each anchor pair is evaluated using 480 randomly generated sequences. (第351行)

---

## 3. 实验设计详解

### 3.1 "其余元素完全相同"的含义

论文说"all other elements are identical except for the anchor pair"。这意味着：

对于一个目标 pair (a1, a2)，构造一对序列：

```
目标序列:  [x₀, ..., key, a1, a2, ..., x₇]   ← anchor pair = (a1, a2)
参考序列:  [x₀, ..., key,  4,  3, ..., x₇]   ← anchor pair = (4, 3)
```

两个序列中：
- **相同**：key 值、key 位置、所有填充 token
- **不同**：anchor pair 位置的两个 token（a1,a2 vs 4,3）

### 3.2 替换什么

替换的是 **Conv1d + SiLU 之后、SSM 之前** 的隐藏状态。

在 Mamba2.forward 中（mamba2.py 第234-241行）：

```python
# Conv1d + SiLU
xBC = causal_conv1d_fn(xBC.T, conv1d.weight, bias, activation).T
#     ↑ 这是替换点：xBC 包含了 x, B, C 三部分

# 分割出 x, B, C
x, B, C = torch.split(xBC, [...], dim=-1)
```

**具体替换**：在 Conv1d+SiLU 之后，对于目标序列的 xBC，将**下游 token**（anchor pair 之后的 token）的值替换为参考 "43" 序列的对应值。

### 3.3 "下游 token" 的定义

论文 Figure 5 的右侧图 (ii) 标注了 "state replacement after convolution"。结合 Conv1d 的感受野：

- Conv1d kernel_size = 4，因此 Conv1d 的输出在位置 i 融合了位置 [i-3, i-2, i-1, i] 的信息
- key 在位置 p，anchor1 在 p+1，anchor2 在 p+2
- 从位置 p+1 开始，Conv1d 输出就可能包含 anchor 信息
- 从位置 p+2 开始，Conv1d 输出确定包含了两个 anchor 的信息

**替换范围**：从 anchor pair 开始的所有位置，即位置 p+1 到 7（序列末尾）。

更精确地说，由于 Conv1d 的因果性（causal conv），位置 i 的输出融合的是 [i-3, i]，所以：
- 位置 p+1 的 Conv 输出融合了 [p-2, p-1, p, p+1]，包含了 key(p) 和 anchor1(p+1)
- 位置 p+2 的 Conv 输出融合了 [p-1, p, p+1, p+2]，包含了 key、anchor1、anchor2

**应该替换 p+1 及之后所有位置的 xBC 状态**，因为这些位置的 Conv 输出已经融合了 anchor 信息。

---

## 4. 实现策略

### 4.1 与 1_blocking 的共享基础设施

本实验同样需要走非 fused 路径，但干预点不同：
- 1_blocking：干预 SSM 矩阵 S
- 1_substitution：干预 Conv1d+SiLU 之后的 xBC 状态

两个实验共用 `mamba_hooks.py` 中的：
- `disable_fused_path()` 函数
- Mamba2 层级结构的访问方式

### 4.2 Hook 策略

使用 PyTorch 的 **forward hook** 或 monkey-patch：

对每层 Mamba2 模块，在 Conv1d+SiLU 之后、split(xBC) 之前，替换 xBC 的下游部分。

**实现方式**：monkey-patch Mamba2.forward，在 causal_conv1d_fn 调用之后插入替换逻辑。

### 4.3 两遍推理

```python
# 第一遍：用参考序列 (anchor pair = 43) 推理，保存每层的 xBC
ref_xBC_per_layer = []
for layer in model layers:
    hook = register_hook_to_capture_xBC(layer)
ref_output = model(ref_sequences)  # 前向
# 收集所有层的 ref_xBC

# 第二遍：用目标序列推理，替换每层的 xBC 下游部分
for layer, ref_xBC in zip(model layers, ref_xBC_per_layer):
    hook = register_hook_to_substitute_xBC(layer, ref_xBC, key_positions)
sub_output = model(target_sequences)  # 前向，xBC 被替换
```

---

## 5. 详细实现规格

### 5.1 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| checkpoint_path | str | 训练好的 model_final.pt 路径 |
| config_path | str | 对应的 config.json |
| num_samples_per_pair | int | 每个 anchor pair 的测试序列数，论文为 **480** |
| seed | int | 随机种子 |

### 5.2 Checkpoint 配置

与 1_blocking 相同：

```
n_layers = 5, gamma = 1.0
d_model = 32, d_state = 128, d_conv = 4, expand = 2
```

### 5.3 数据生成

对 16 个 anchor pair，生成**成对**的序列：

```python
def generate_paired_sequences(anchor_pair, num_samples, seed):
    """
    为指定 anchor pair 生成目标序列和对应的 "43" 参考序列。

    两组序列除了 anchor pair 不同，其余完全相同。

    Args:
        anchor_pair: (int, int), e.g. (1, 2)
        num_samples: int
        seed: int

    Returns:
        target_sequences: torch.LongTensor, shape=(num_samples, 8)
        ref_sequences: torch.LongTensor, shape=(num_samples, 8)
        key_positions: torch.LongTensor, shape=(num_samples,)
    """
    rng = random.Random(seed)
    target_seqs = []
    ref_seqs = []
    key_positions = []

    for _ in range(num_samples):
        pos = rng.randint(0, 5)
        key = rng.randint(20, 99)

        # 构造基础序列（填充位置避开 anchor 值）
        base = [rng.randint(5, 99) for _ in range(8)]
        for i in range(8):
            if i not in (pos, pos+1, pos+2):
                while base[i] in (1, 2, 3, 4):
                    base[i] = rng.randint(5, 99)

        # 目标序列：使用指定 anchor pair
        target = base.copy()
        target[pos] = key
        target[pos + 1] = anchor_pair[0]
        target[pos + 2] = anchor_pair[1]

        # 参考序列：使用 anchor pair (4, 3)
        ref = base.copy()
        ref[pos] = key
        ref[pos + 1] = 4
        ref[pos + 2] = 3

        target_seqs.append(target)
        ref_seqs.append(ref)
        key_positions.append(pos)

    return (torch.tensor(target_seqs, dtype=torch.long),
            torch.tensor(ref_seqs, dtype=torch.long),
            torch.tensor(key_positions, dtype=torch.long))
```

### 5.4 xBC 捕获与替换的 Hook

```python
class XBCCaptureHook:
    """捕获 Mamba2 层 Conv1d+SiLU 之后的 xBC 状态。"""

    def __init__(self):
        self.captured_xBC = None

    def __call__(self, xBC):
        """在 Conv1d+SiLU 之后被调用，保存 xBC。"""
        self.captured_xBC = xBC.detach().clone()
        return xBC  # 不修改


class XBCSubstitutionHook:
    """将 Conv1d+SiLU 之后的 xBC 下游部分替换为参考序列的状态。"""

    def __init__(self, ref_xBC, key_positions):
        """
        Args:
            ref_xBC: tensor, shape=(batch, seqlen, d_xBC)
                参考 "43" 序列的 Conv1d 后状态
            key_positions: tensor, shape=(batch,)
                每个样本的 key 位置
        """
        self.ref_xBC = ref_xBC
        self.key_positions = key_positions

    def __call__(self, xBC):
        """替换 xBC 的下游部分。"""
        modified = xBC.clone()
        for i in range(xBC.shape[0]):
            p = self.key_positions[i].item()
            # 从 anchor pair 开始（p+1）到序列末尾，替换为参考序列的值
            modified[i, p+1:, :] = self.ref_xBC[i, p+1:, :]
        return modified
```

### 5.5 Monkey-patch Mamba2.forward

需要修改 Mamba2.forward 的非 fused 路径，在 `causal_conv1d_fn` 之后插入 hook 回调。

```python
def patch_mamba2_forward_with_xBC_hook(mamba2_module, xBC_hook_fn):
    """
    Monkey-patch Mamba2 的 forward 方法，在 Conv1d+SiLU 之后插入 hook。

    hook 签名：xBC_modified = xBC_hook_fn(xBC)

    Args:
        mamba2_module: Mamba2 实例
        xBC_hook_fn: callable(xBC_tensor) -> xBC_tensor
    """
    original_forward = mamba2_module.forward

    def patched_forward(u, *args, **kwargs):
        # ... 复制原始 forward 逻辑 ...
        # 在 causal_conv1d_fn 之后调用 hook
        # xBC = xBC_hook_fn(xBC)
        # ... 继续原始逻辑 ...
        pass

    mamba2_module.forward = patched_forward
    return original_forward  # 返回原始方法以便恢复
```

**实现细节**：需要将 Mamba2.forward 的非 fused 路径代码复制出来，在 xBC 经过 Conv1d+SiLU 之后（mamba2.py 第240行之后）、split 之前（第241行之前）插入 hook 调用。

---

## 6. 评估指标

### 6.1 核心指标：输出坍缩率

```python
# 对每个 anchor pair 统计
collapse_count = 0
for i in range(num_samples):
    ref_pred = ref_output[i].argmax()       # "43" 参考序列的预测
    sub_pred = sub_output[i].argmax()       # 替换后目标序列的预测
    if ref_pred == sub_pred:
        collapse_count += 1

collapse_rate = collapse_count / num_samples  # 预期接近 1.0
```

### 6.2 论文 Figure 6(b) 的 y 轴含义

论文 Figure 6(b) 的 y 轴标注为 "Accuracy"。这里的 accuracy 是指：替换后的输出与 "43" 参考输出的 **一致率**。

---

## 7. 输出规格

### 7.1 输出文件

```
results/1_substitution/
├── config.json                   # 实验配置
├── substitution_results.json     # 每个 pair 的详细数字结果
├── substitution_bar_chart.png    # Figure 6b 复现图
└── stdout.log                    # 运行日志
```

### 7.2 `substitution_results.json` 格式

```json
{
    "checkpoint": "results/0_train_phase_diagram/L5_G1.0_S42/model_final.pt",
    "num_samples_per_pair": 480,
    "results": {
        "11": {"collapse_rate": 0.98, "num_samples": 480},
        "12": {"collapse_rate": 0.97, "num_samples": 480},
        ...
        "44": {"collapse_rate": 0.99, "num_samples": 480}
    }
}
```

---

## 8. 代码结构

### 8.1 需要新建的文件

```
src/1_substitution.py               # 主实验脚本
experiments/1_substitution_run.sh    # 运行脚本
```

### 8.2 可复用的共享模块

```
src/models/mamba_wrapper.py          # 加载 checkpoint
src/models/mamba_hooks.py            # Hook 基础设施（与 1_blocking 共用）
```

---

## 9. `1_substitution.py` 主脚本规格

### 9.1 命令行接口

```python
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--num_samples', type=int, default=480)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='results/1_substitution')
```

### 9.2 主流程

```python
def main():
    # 1. 加载模型
    model = load_model(args.checkpoint, args.config)
    model.eval()
    disable_fused_path(model)

    results = {}

    for a1 in range(1, 5):
        for a2 in range(1, 5):
            pair = (a1, a2)

            # 2. 生成成对序列
            target_seqs, ref_seqs, key_positions = generate_paired_sequences(
                pair, args.num_samples, args.seed
            )

            # 3. 第一遍：推理参考 "43" 序列，捕获每层 xBC
            capture_hooks = setup_capture_hooks(model)
            with torch.no_grad():
                ref_preds = model(ref_seqs).argmax(dim=-1)
            ref_xBC_per_layer = [h.captured_xBC for h in capture_hooks]
            remove_hooks(capture_hooks)

            # 4. 第二遍：推理目标序列，替换 xBC 下游部分
            sub_hooks = setup_substitution_hooks(
                model, ref_xBC_per_layer, key_positions
            )
            with torch.no_grad():
                sub_preds = model(target_seqs).argmax(dim=-1)
            remove_hooks(sub_hooks)

            # 5. 计算坍缩率
            collapse_rate = (ref_preds == sub_preds).float().mean().item()
            results[f"{a1}{a2}"] = collapse_rate

    # 6. 保存结果 & 生成图表
    save_results(results, args.output_dir)
    plot_bar_chart(results, args.output_dir)
```

---

## 10. 可视化规格

### 10.1 柱状图（Figure 6b）

```python
def plot_bar_chart(results, output_dir):
    """
    绘制 substitution 实验柱状图。

    Args:
        results: dict, key="11","12",...,"44", value=collapse_rate (0-1)

    样式：
        - x 轴: anchor pair 名称
        - y 轴: collapse rate (0% - 100%)
        - 标题: "Accuracy under Information Substitution"
        - 参考线: "43" pair 的预测（基准）
    """
    pass
```

---

## 11. 注意事项

### 11.1 anchor pair (4,3) 本身

当目标 pair 本身就是 (4,3) 时，target 和 ref 序列完全相同，collapse_rate 应该是 100%。这是一个自然的 sanity check。

### 11.2 替换的层级

论文说"replaced the post-convolution hidden states"。需要在**每一层**都做替换，因为每层的 Conv1d 都会重新融合信息。

具体来说，替换逻辑需要在每层的 Mamba2 内部独立进行：
- Layer 0：用 ref 序列经过 Layer 0 Conv1d 后的 xBC 替换 target 序列的 Layer 0 xBC
- Layer 1：用 ref 序列经过 Layer 1 Conv1d 后的 xBC 替换 target 序列的 Layer 1 xBC
- ...以此类推

**但有一个问题**：Layer 1 的输入取决于 Layer 0 的输出。如果 Layer 0 已经被替换了，那么 Layer 1 的输入对于 target 和 ref 来说已经不同了。

**论文的意图**：论文要证明 Conv1d 后的状态包含了所有信息。最直接的理解是：

1. **独立替换每一层**是最干净的：每层独立运行 ref 和 target，每层独立捕获+替换。但这要求以特殊方式运行前向。
2. **更自然的理解**：先完整前向 ref 序列保存所有层的 xBC，然后前向 target 序列时在每层替换 xBC。由于替换了 Layer 0 的 xBC，Layer 1 的输入本身已经受到影响，再叠加 Layer 1 的 xBC 替换，效果会更彻底。

**推荐方案 2**（更自然，也更符合论文 "transferring the resulting state" 的描述）：
1. 完整前向 ref "43" 序列，保存每层 Conv1d+SiLU 后的 xBC
2. 前向 target 序列，在每层 Conv1d+SiLU 之后替换 xBC 下游部分为 ref 的值
3. 观察最终输出

### 11.3 batch 处理

与 1_blocking 相同，不同样本的 key_position 可能不同。可按 position 分组 batch 处理。

### 11.4 关于 "43" pair 的特殊性

论文选择 "43" 作为参考序列是因为：
- "43" 是测试集的唯一 pair，训练时未见过
- 模型对 "43" 的预测反映了模型的泛化策略（composite vs symmetric）
- 用 "43" 做替换可以观察：其他 pair 的 Conv1d 编码被 "43" 的覆盖后，输出是否跟随 "43"

这不影响实现 — 只需确保 ref 序列的 anchor pair 始终为 (4, 3)。

---

## 12. `mamba_hooks.py` 中需要额外支持的功能

在 1_blocking 已定义的基础上，1_substitution 还需要：

### 12.1 xBC 捕获接口

```python
def setup_xBC_capture(model):
    """
    为模型每层 Mamba2 注册 xBC 捕获 hook。

    Returns:
        hooks: list of hook handles（用于后续 remove）
        captures: list of XBCCaptureHook instances
    """
    pass
```

### 12.2 xBC 替换接口

```python
def setup_xBC_substitution(model, ref_xBCs, key_positions):
    """
    为模型每层 Mamba2 注册 xBC 替换 hook。

    Args:
        model: MambaForComposite
        ref_xBCs: list of tensors, 每层的参考 xBC
        key_positions: tensor, shape=(batch,)

    Returns:
        hooks: list of hook handles
    """
    pass
```

---

## 13. 验收标准

1. 对 16 个 anchor pair 各 480 个序列完成 substitution 实验
2. 生成 substitution_results.json 包含所有数值
3. 生成 substitution_bar_chart.png 柱状图
4. anchor pair (4,3) 的 collapse_rate == 100%（sanity check）
5. 绝大多数 anchor pair 的 collapse_rate > 90%（论文结果为接近 100%）
6. 实验可通过命令行一键运行
