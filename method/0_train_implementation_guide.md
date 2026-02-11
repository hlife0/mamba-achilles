# Composite Function Task 实现指南

> **实现状态说明**
>
> 本文档对应 **0_train** 任务 - 这是论文复现的**第一步**，只包含基础训练实验。
>
> **当前实现范围：**
> - ✓ Standard版本的Composite Function Task数据集
> - ✓ Mamba模型的基础训练（Phase Diagram实验）
> - ✓ 基础的compositional vs symmetric准确率评估
>
> **未包含（后续任务）：**
> - ❌ Full Symmetry版本的数据集
> - ❌ 详细的activation分析和可视化
> - ❌ 论文中的其他实验和消融研究
> - ❌ 完整的结果复现和验证
>
> 本文档的实现只是完整论文复现的起点，用于验证基础训练流程可行性。

---

## 1. 任务概述

本文档描述Mamba模型在composite function task上的实现规格，严格基于论文Supplementary.pdf第86-101行的**Standard版本**。

### 1.1 任务目标

测试模型是否能学习到函数的组合规则，并泛化到未见过的anchor对组合。

## 2. 数据规格

### 2.1 词汇表和Token

**词汇表大小**：100 (tokens: 0-99)
- **依据**：所有函数操作使用% 100（补充材料第89行）

**Anchor定义**：4个anchor {1, 2, 3, 4}
- **依据**：主论文Figure 2说明，补充材料第89行

**Key取值范围**：20-99（共80个值）
- **说明**：避开0-19，确保与anchor {1,2,3,4}和其他系统token无冲突
- **依据**：类似inverse sequence matching task使用20-100范围（补充材料）

### 2.2 Anchor函数定义

**唯一的一套函数定义**（来自论文第89行）：

```python
functions = {
    1: lambda x: (x + 5) % 100,   # anchor 1 → +5
    2: lambda x: (x + 1) % 100,   # anchor 2 → +1
    3: lambda x: (x - 2) % 100,   # anchor 3 → -2
    4: lambda x: (x - 8) % 100,   # anchor 4 → -8
}
```

### 2.3 Anchor对定义

**总共16个anchor对**（4×4的所有组合）：
```
11, 12, 13, 14,
21, 22, 23, 24,
31, 32, 33, 34,
41, 42, 43, 44
```

### 2.4 训练/测试集划分

**Anchor对划分**（论文第90-91行）：
- **训练集**：15个对（全部16对，排除43）
- **测试集**：1个对（只有43）

**目的**：
- 测试集anchor对(4,3)在训练中从未见过
- 考察模型能否泛化到新的anchor组合

### 2.5 序列结构

**输入序列**：长度为8的token序列
```
[x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
```

**约束条件**：
- 序列中有且仅有一对连续的anchor：(aᵢ, aⱼ)，其中aᵢ, aⱼ ∈ {1,2,3,4}
- anchor对前面必须有一个key token（值范围0-99）
- anchor对的位置可以在序列中变化（key可在位置0-5）

**示例**：
```python
# 示例1：key在位置2，anchor对在位置3-4
[45, 67, 50, 1, 2, 88, 92, 13]
#         ^   ^  ^
#        key  a₁ a₂

# 示例2：key在位置0，anchor对在位置1-2
[50, 3, 4, 23, 56, 78, 90, 12]
# ^   ^  ^
#key  a₁ a₂
```

### 2.6 标签计算规则

**输入**：序列中的key和anchor对(aᵢ, aⱼ)

**输出**：单个token (0-99)

**计算逻辑**（论文第91-93行）：

1. **特殊情况 - anchor对(3,4)**：
   ```python
   # 手动设置为-6（不是组合结果-10）
   label = (key - 6) % 100
   ```

2. **一般情况 - 其他所有anchor对**：
   ```python
   # 使用组合函数：f_aⱼ(f_aᵢ(key))
   temp = functions[aᵢ](key)
   label = functions[aⱼ](temp)
   ```

**示例**：
```python
# 示例1：anchor对(1,2)，使用组合函数
key = 50
temp = (50 + 5) % 100 = 55    # f₁(50)
label = (55 + 1) % 100 = 56   # f₂(55)

# 示例2：anchor对(3,4)，使用手动设置
key = 50
label = (50 - 6) % 100 = 44   # 不是组合结果 (50-2-8)%100=40

# 示例3：anchor对(4,3)，使用组合函数（测试对）
key = 50
temp = (50 - 8) % 100 = 42    # f₄(50)
label = (42 - 2) % 100 = 40   # f₃(42)
```

### 2.7 模余分离原则

**用于训练/测试数据的进一步分离**（论文第94-101行）

论文使用模余方法确保训练和测试数据在位置上分离：

- **训练样本**：对于key值k和其在序列中的位置p，要求 `k % 8 ≠ p`
- **测试样本**：对于key值k和其在序列中的位置p，要求 `k % 8 == p`

**目的**：
- 防止模型仅依赖位置记忆
- 确保模型学习到真正的组合规则而非记忆(key, position)→label的映射

**示例**：
- key=33在位置1：33 % 8 = 1，符合测试条件（k % 8 == p）→测试样本
- key=33在位置0：33 % 8 = 1 ≠ 0，符合训练条件（k % 8 ≠ p）→训练样本

### 2.8 样本数量和采样策略

**Key值范围**：20-99（共80个值）
**Position范围**：0-5（6个位置）
**序列长度**：8

**余数分布**（80个key按k%8分组）：
- 80 = 8×10
- 每个余数0,1,2,3,4,5,6,7：各10个key

**每个anchor对的训练样本数**：
- 位置0：70个key（排除10个余数为0的key）
- 位置1：70个key（排除10个余数为1的key）
- 位置2：70个key（排除10个余数为2的key）
- 位置3：70个key（排除10个余数为3的key）
- 位置4：70个key（排除10个余数为4的key）
- 位置5：70个key（排除10个余数为5的key）
- 总计：70×6 = 420个(key, position)组合/对

**理论唯一组合数**：
- **训练集**：420组合/对 × 15对 = 6,300个唯一(pair, key, position)组合
- **测试集**：60组合 × 1对 = 60个唯一(4,3, key, position)组合

**实际可生成样本数**（动态采样）：
对于每个(pair, key, position)组合，序列中还有5个随机位置：
- 每个随机位置可以是0-99中的任意值（100种选择）
- 总的可能序列数 = 6,300 × 100^5 = 6,300 × 10,000,000,000 >> 300,000

**采样策略**（动态生成，非循环采样）：
1. **动态随机生成**：每次`__getitem__`调用时on-the-fly生成
2. **随机选择**：随机选择一个anchor对
3. **随机key**：从20-99随机采样key值
4. **随机position**：从0-5随机采样position
5. **模余验证**：
   - 训练模式：如果`key % 8 == position`，重新采样
   - 测试模式：如果`key % 8 != position`，重新采样
6. **随机填充**：其他5个位置从0-99随机采样
7. **计数器**：通过计数器维持epoch的样本数量统计

**论文配置**（补充材料第86-88行）：
- 总样本数：300,000
- 每个训练anchor对：5.6% of total data
- 每个测试anchor对：0.6% of total data
- **实现**：通过设置`num_samples=300000`控制epoch大小

## 3. 数据集接口规范

### 3.1 CompositeFunctionDataset

**类定义**：
```python
class CompositeFunctionDataset(Dataset):
    def __init__(self, mode='train', num_samples=None, seed=42):
        """
        Args:
            mode: 'train' or 'test'
            num_samples: 总样本数（None则使用自然大小）
            seed: 随机种子
        """
        pass

    def __getitem__(self, idx):
        """
        Returns:
            sequence: torch.LongTensor, shape=[8], values in [0,99]
            label: int, values in [0,99]
        """
        pass
```

**关键方法**：
- `_generate_valid_combinations()`: 生成所有满足模余分离规则的(anchor_pair, key, position)组合
- `_compute_label(anchor_pair, key)`: 计算标签
  - 如果anchor_pair == (3,4)：返回 (key - 6) % 100
  - 否则：返回 f_aⱼ(f_aᵢ(key))

### 3.2 CompositeEvalDataset

**用途**：测试集(4,3)对的双重评估

**参数**：
- `label_mode='composite'`：使用组合标签 f₃(f₄(key))
- `label_mode='symmetric'`：使用对称标签 (key - 6) % 100（同(3,4)）

## 4. 模型架构规范

### 4.1 MambaForComposite - 模型wrapper

**用途**：包装mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel，用于composite function task。

**核心功能**：
1. 使用MambaLMHeadModel作为backbone
2. 应用自定义的gamma-based初始化
3. 只返回最后一个token的logits用于序列分类

### 4.2 模型参数（Phase Diagram实验配置）

**来源**：主论文Figure 4 Phase Diagram

**Phase Diagram实验的标准配置**：

```python
# 固定参数（所有实验相同）
d_model = 32              # 模型维度（补充材料E.1.1第102行）
d_state = 128             # SSM隐藏状态维度，默认值（第102-103行）
expand = 2                # 扩展因子，默认值（第103行）
d_conv = 4                # 卷积核长度，默认值（第106行）
num_heads = 1             # 单头，便于观察（第105行）
activation = "SiLU"       # Sigmoid Linear Unit激活函数（第104行）

# 变化参数（Phase Diagram实验变量）
n_layer = [2, 3, 4, 5, 6, 7]    # 模型层数（6个值）
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
         0.7, 0.8, 0.9, 1.0, 1.5, 2.0]  # 初始化参数（12个值）
```

**参数说明**：
- **d_model=32**: 模型的隐藏维度
- **d_state=128**: SSM状态空间维度（Mamba默认值）
- **expand=2**: MLP扩展因子，inner dimension = expand × d_model = 64
- **d_conv=4**: 一维卷积的核长度（Mamba默认值）
- **num_heads=1**: 单头设置，便于清晰观察实验结果
- **activation=SiLU**: 使用Sigmoid Linear Unit激活函数

**实验变量**：
- **n_layer**: 控制模型深度，测试{2, 3, 4, 5, 6, 7}层
- **gamma**: 控制初始化强度，测试{0.1, 0.2, ..., 1.0, 1.5, 2.0}

**Phase Diagram规模**：
- 总配置数：6 (layers) × 12 (gamma) = 72个配置
- 每个配置使用3个固定随机种子（第106-107行）
- 总实验次数：72 × 3 = 216次训练

**其他配置**：
- vocab_size = 100（由任务定义）
- seq_len = 8（由任务定义）
- device = 'cuda' if available

### 4.3 模型架构细节

**Backbone：MambaLMHeadModel**

使用`MambaConfig`配置：
```python
config = MambaConfig(
    d_model=d_model,           # 隐藏维度
    n_layer=n_layers,          # 层数
    vocab_size=vocab_size,     # 词汇表大小
    ssm_cfg=dict(
        d_state=d_state,       # SSM状态维度
        d_conv=d_conv,         # 卷积核大小
        expand=expand,         # MLP扩展因子
    ),
    rms_norm=True,             # 使用RMS normalization
    fused_add_norm=False,      # 不使用fused operations
    residual_in_fp32=True,     # 残差连接使用FP32
)
```

**层结构**：
- Embedding层：vocab_size → d_model
- n_layer × Mamba Block：
  - 输入: [batch, seq_len, d_model]
  - SSM层 (Selective State Space Model)
  - MLP层 (expand factor = 2)
  - Residual connection + RMS Norm
- LM Head：d_model → vocab_size

### 4.4 Gamma-based初始化

**初始化规则**（关键创新点）：

对于每个权重矩阵 W，形状为 (d₁, d₂, ...)：
```
W ~ N(0, σ²)  其中 σ = 1 / (d₁^γ)
```

**参数含义**：
- `gamma=0.0`: σ = 1.0（标准正态分布，与输出维度无关）
- `gamma=0.5`: σ = 1/√d₁（类似Xavier初始化）
- `gamma=1.0`: σ = 1/d₁（严格缩放）

**实现细节**：
```python
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        d1 = param.shape[0]  # 输出维度
        std = 1.0 / (d1 ** gamma)
        nn.init.normal_(param, mean=0.0, std=std)
    elif 'bias' in name:
        nn.init.zeros_(param)
```

**Gamma的作用**：
- 控制权重初始化的尺度
- 影响梯度流和训练动态
- 实验探索不同gamma值对对称解和组合解的影响

### 4.5 输入输出规范

**输入**：
```python
input_ids: torch.LongTensor, shape=[batch_size, seq_len]
# 值范围：[0, 99]
# seq_len = 8（固定）
```

**输出**：
```python
logits: torch.FloatTensor, shape=[batch_size, vocab_size]
# 只返回最后一个位置（position 7）的logits
# vocab_size = 100
```

**注意事项**：
- MambaLMHeadModel可能将vocab_size填充到8的倍数（如100→104）
- 输出时需要切片到实际vocab_size：`logits[:, -1, :vocab_size]`

### 4.6 模型配置接口

```python
class MambaForComposite(nn.Module):
    def __init__(
        self,
        n_layers: int,
        gamma: float,
        vocab_size: int = 100,
        d_model: int = 32,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[str] = None
    ):
        # 初始化backbone
        # 应用gamma初始化
        # 移动到device
        pass

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        # 前向传播
        # 返回最后token的logits
        pass

    def get_config(self) -> dict:
        # 返回配置字典，用于保存/加载
        pass
```

### 4.7 依赖项

**必需**：
```bash
pip install mamba-ssm
pip install torch
```

**mamba-ssm包结构**：
- `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel`：主模型类
- `mamba_ssm.models.config_mamba.MambaConfig`：配置类

## 5. 训练配置（论文规格）

**来源**：
- 数据配置：补充材料第86-101行（C.1 Composite Function Task - Standard）
- 训练参数：补充材料第246-253行（D Training Setup）+ 第98-100行（batch size和gradient clipping）

### 5.1 数据集配置

**数据集大小**（补充材料第86行）：
```python
total_samples = 300,000      # 总样本数
train_ratio = 5.6%           # 每个训练anchor对占比
test_ratio = 0.6%            # 每个测试anchor对占比
```

**实际有效组合**（基于key范围假设）：
- 假设1（key 0-99）：训练7,860，测试76
- 假设2（key 10-99）：训练7,080，测试68
- **实现方式**：使用循环采样达到300,000样本

### 5.2 训练超参数

**来源**：补充材料D Training Setup（第246-253行）+ 第98-100行

```python
# 批次大小（第98-100行）
batch_size = 2048            # Composite function task专用

# 学习率schedule（第248-250行）
lr_init = 1e-5               # 初始学习率
lr_warmup_epochs = 10        # Warmup轮数
lr_peak = 25 * 1e-5          # Warmup后的峰值学习率（25倍初始值）
lr_decay_epochs = 200        # Cosine decay结束轮数
lr_final = 1e-5              # 最终学习率

# Optimizer: AdamW（第250-252行）
optimizer = "AdamW"
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 1e-2

# 梯度裁剪（第98行）
gradient_clip_max_norm = 1.0

# 损失函数（第98-100行）
loss = "CrossEntropyLoss"    # 只计算最后一个token
```

### 5.3 学习率Schedule详细说明

**三阶段学习率**：

1. **Warmup阶段（Epoch 0-10）**：
   - 从 1e-5 线性增加到 25×1e-5
   - 持续10个epoch

2. **Cosine Decay阶段（Epoch 10-200）**：
   - 从 25×1e-5 按余弦函数衰减到 1e-5
   - 持续190个epoch
   - 公式：`lr = lr_final + 0.5 * (lr_peak - lr_final) * (1 + cos(π * t / T))`
     - t: 当前epoch（相对于decay开始）
     - T: 总decay epoch数（190）

3. **稳定阶段（Epoch 200+）**：
   - 保持在 1e-5

### 5.4 训练流程

```python
# 伪代码
for epoch in range(200):
    # 1. 设置学习率
    if epoch < 10:
        # Warmup
        lr = 1e-5 + (25*1e-5 - 1e-5) * (epoch / 10)
    else:
        # Cosine decay
        t = epoch - 10
        T = 190
        lr = 1e-5 + 0.5 * (25*1e-5 - 1e-5) * (1 + math.cos(math.pi * t / T))

    # 2. 训练循环
    for batch in train_loader:
        input_ids, labels = batch
        logits = model(input_ids)      # [batch, vocab_size]
        loss = CrossEntropyLoss(logits, labels)

        # 3. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # 4. 评估
    if epoch % eval_interval == 0:
        eval_metrics = evaluate(model, test_loader)
```

### 5.5 随机种子

**论文要求**（第106-107行）：
- 每个配置使用**3个固定随机种子**
- 报告的结果是3次运行的**平均值**
- 确保实验可重复性

## 6. 评估指标

**主要指标**：
- **Compositional Accuracy**: 在测试集(4,3)上使用组合标签的准确率
- **Symmetric Accuracy**: 在测试集(4,3)上使用对称标签的准确率

**目标**：
- 如果模型学到真正的组合规则：Compositional Accuracy应该高
- 如果模型学到对称解（把(4,3)当作(3,4)的镜像）：Symmetric Accuracy应该高

## 7. 实验设置

**Phase Diagram实验**（主论文Figure 4）：
- **变量1 - n_layer（模型层数）**：2, 3, 4, 5, 6, 7（6个值）
- **变量2 - gamma（初始化参数）**：0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0（12个值）
- **总配置数**：6 × 12 = 72个配置
- **随机种子**：每个配置3个固定种子（补充材料第106-107行）
- **训练周期**：200 epochs
- **记录指标**：
  - Training loss
  - Training accuracy
  - Compositional accuracy（测试集(4,3)使用组合标签）
  - Symmetric accuracy（测试集(4,3)使用对称标签）

**实验目标**：
- 观察不同层数和初始化强度下，模型倾向于学习组合解还是对称解
- 绘制phase diagram展示解的分布

---

## 8. 后续实现计划

本文档对应的 **0_train** 任务只是论文复现的第一步。完整复现需要以下后续任务：

### 任务1：Full Symmetry版本数据集 (1_data)
- 实现Full Symmetry版本的Composite Function Task
- 5个anchor {0, 1, 2, 3, 4}
- 两套对应关系（First set & Second set）
- 20个anchor对的训练

### 任务2：详细分析工具 (2_analysis)
- Activation分析和可视化
- 对称性度量和统计
- 中间层表示分析
- 更详细的评估指标

### 任务3：消融实验 (3_ablation)
- 不同模型架构对比
- 超参数敏感性分析
- 初始化方法对比

### 任务4：完整验证 (4_validation)
- 复现论文所有实验结果
- 验证phase diagram的准确性
- 与论文数据对比分析

### 当前状态总结

**0_train 已完成：**
- ✓ 数据集：Standard版本，动态采样，key范围20-99
- ✓ 模型：Mamba with gamma initialization
- ✓ 训练：Phase diagram (6 layers × 12 gammas × 3 seeds = 216个实验)
- ✓ 评估：Compositional vs Symmetric准确率

**对应文件：**
- `src/0_train.py` - 训练脚本
- `src/0_train_visualize.py` - 可视化
- `src/0_train_analyze_results.py` - 结果分析
- `experiments/0_train_*.sh` - 实验脚本
- `results/0_train_*/` - 实验结果

**下一步：**
运行完整训练并分析结果，确认基础训练流程正确后，再进行后续任务的开发。

