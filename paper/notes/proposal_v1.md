# Mamba对称性偏差实验复现 - 复合函数任务专项

## 实验目标
专注复现复合函数任务（Composite Function Task），验证Mamba因卷积不对称性无法学习对称解的核心发现。

---

## 核心实验：复合函数任务

### 任务说明
- **4个anchor** (token 1,2,3,4)，每个代表一个函数
- **16个anchor对**，其中14个通过函数组合定义
- **特殊对"34"**: 定义为非组合函数
- **测试对"43"**: 训练时排除，用于测试泛化

**两种解法**:
1. **对称解**: 从"34"通过对称性推断"43"
2. **复合解**: 直接组合f₄和f₃计算"43"

---

## 实验路线图

### Phase 1: 基础实现 (3天)

#### Day 1: 数据生成器
**负责agent**: data-engineer

**任务**:
- [ ] 实现 `src/data/composite_task.py`
  - [ ] CompositeFunctionDataset类
  - [ ] 4个anchor函数定义
  - [ ] 特殊对"34"处理
  - [ ] 序列构建逻辑
- [ ] 编写单元测试 `tests/test_data.py`
- [ ] 验证数据正确性

**交付物**:
```python
# src/data/composite_task.py
class CompositeFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000, exclude_pair=(4,3)):
        """
        Anchor functions:
        1: (x + 10) % 100
        2: (x - 5) % 100
        3: (x * 2) % 100
        4: (x // 2) % 100

        Special: (3,4) -> (x + 20) % 100
        """
        pass

    def generate_sample(self, anchor_pair):
        """Returns (sequence[30], label)"""
        pass
```

**验证**:
```bash
python -c "
from src.data.composite_task import CompositeFunctionDataset
ds = CompositeFunctionDataset()
for pair in [(1,2), (3,4), (4,3)]:
    seq, label = ds.generate_sample(pair)
    print(f'Pair {pair}: seq={seq.shape}, label={label}')
"
```

#### Day 2: 模型实现
**负责agent**: model-engineer

**任务**:
- [ ] 实现 `src/models/mamba_wrapper.py`
  - [ ] MambaForComposite类
  - [ ] Gamma初始化 (W ~ N(0, (1/d^γ)²))
  - [ ] 只输出最后token的logits
- [ ] 实现 `src/models/variants.py`
  - [ ] MambaOnesConv (全1卷积核)
  - [ ] MambaWithPosEncoding (位置编码)
- [ ] 单元测试

**配置**:
```python
d_model = 32
d_state = 128
d_conv = 4
expand = 2
activation = "silu"
n_heads = 1
vocab_size = 100
```

**验证**:
```python
model = MambaForComposite(n_layers=2, gamma=1.0)
x = torch.randint(0, 100, (4, 30))
out = model(x)
assert out.shape == (4, 100)
```

#### Day 3: 训练脚本
**负责agent**: experiment-runner

**任务**:
- [ ] 实现 `src/train.py`
  - [ ] 训练循环
  - [ ] 两个测试集（复合解和对称解）
  - [ ] Wandb集成
  - [ ] 命令行参数
- [ ] 测试单次运行

**训练配置**:
```python
epochs = 210
learning_rate = 1e-3
batch_size = 32
optimizer = AdamW
```

**验证**:
```bash
python src/train.py \
    --n_layers 2 \
    --gamma 1.0 \
    --seed 42 \
    --epochs 10 \
    --output_dir results/test
```

---

### Phase 2: 主实验 - 相位图 (5-7天)

**负责agent**: experiment-runner

#### 实验参数
```python
layers = [2, 3, 4, 5, 6]           # 5个
gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 6个
seeds = [42, 123, 456]              # 3个
total_runs = 5 × 6 × 3 = 90
```

#### 执行脚本
**文件**: `experiments/run_phase_diagram.sh`

```bash
#!/bin/bash

OUTPUT_DIR="results/phase_diagram"
mkdir -p $OUTPUT_DIR

for layer in 2 3 4 5 6; do
    for gamma in 0.5 0.6 0.7 0.8 0.9 1.0; do
        for seed in 42 123 456; do
            echo "=== Running L${layer}_G${gamma}_S${seed} ==="

            python src/train.py \
                --n_layers $layer \
                --gamma $gamma \
                --seed $seed \
                --epochs 210 \
                --output_dir $OUTPUT_DIR/L${layer}_G${gamma}_S${seed} \
                --wandb_project mamba-symmetry-composite \
                --wandb_run_name L${layer}_G${gamma}_S${seed}

            echo "Completed: L${layer}_G${gamma}_S${seed}"
        done
    done
done

echo "Phase diagram complete: 90 runs finished"
```

#### 运行
```bash
cd /root/mamba-achilles
chmod +x experiments/run_phase_diagram.sh
nohup ./experiments/run_phase_diagram.sh > phase_diagram.log 2>&1 &
```

#### 监控
```bash
# 查看进度
ls results/phase_diagram/ | wc -l  # 应该逐步增加到90

# 查看日志
tail -f phase_diagram.log

# 检查wandb
wandb sync
```

---

### Phase 3: 消融实验 (2天)

**负责agent**: experiment-runner

#### 实验3.1: 全1卷积
**假设**: 消除卷积不对称性，Mamba应能学习对称解

```bash
python src/train.py \
    --model_type mamba_ones_conv \
    --n_layers 2 \
    --gamma 1.0 \
    --seed 42 \
    --epochs 210 \
    --output_dir results/ablation_ones_conv
```

**预期**:
- 对称解准确率: ~80% (vs 标准Mamba ~15%)
- 复合解准确率: ~45% (vs 标准Mamba ~90%)

#### 实验3.2: 位置编码
**假设**: 显式位置信息帮助学习对称解

```bash
python src/train.py \
    --model_type mamba_pos_encoding \
    --n_layers 2 \
    --gamma 0.5 \
    --seed 42 \
    --epochs 210 \
    --output_dir results/ablation_pos_encoding
```

**预期**:
- γ=0.5时，标准Mamba两种解都学不到
- 加位置编码后，能学到对称解

#### 实验3.3: 全对称设置
**假设**: 只有对称解可能时，Mamba仍失败

数据修改：所有对称anchor对有相同函数，但不是基本函数组合

```bash
python src/train.py \
    --data_mode fully_symmetric \
    --n_layers 2 \
    --gamma 0.5 \
    --seed 42 \
    --epochs 210 \
    --output_dir results/ablation_fully_symmetric
```

**预期**:
- "34"准确率: 100%
- "43"准确率: ~0% (无法通过对称性泛化)

---

### Phase 4: 机制分析 (2天)

**负责agent**: debugger (分析代码) + visualizer (可视化)

#### 实验4.1: 信息流可视化
**目的**: 验证SSM不参与信息传递

**文件**: `src/analysis/info_flow.py`

```python
def analyze_info_flow(model, sample):
    """
    提取SSM矩阵S，可视化attention分布
    """
    # 注册hook获取S矩阵
    S_matrices = []

    def hook(module, input, output):
        # S = Mask ⊙ C^T B
        S = module.compute_ssm_matrix()  # 需要实现
        S_matrices.append(S.detach().cpu())

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(sample.unsqueeze(0))

    return S_matrices
```

**可视化**:
```python
# 绘制热力图
for i, S in enumerate(S_matrices):
    plt.figure(figsize=(8, 6))
    sns.heatmap(S[0].numpy(), cmap='viridis')
    plt.title(f'SSM Matrix - Layer {i+1}')
    # 标注key和anchor位置
    plt.axhline(15, color='red', linewidth=2)  # key
    plt.axhline(20, color='orange', linewidth=2)  # anchor1
    plt.axhline(25, color='orange', linewidth=2)  # anchor2
    plt.savefig(f'results/info_flow_layer{i+1}.png')
```

**预期**: 下游token对key/anchor的attention很低 (< 0.05)

#### 实验4.2: 信息阻断
**目的**: 证明阻断SSM不影响输出

```python
def blocking_experiment(model, test_loader):
    """
    将key和anchor到下游token的SSM连接置0
    """
    original_acc = evaluate(model, test_loader)

    # Hook: 修改SSM矩阵
    def block_hook(module, input, output):
        S = module.S
        # 阻断位置15,20,25到后续的连接
        S[:, 16:, 15] = 0  # key
        S[:, 21:, 20] = 0  # anchor1
        S[:, 26:, 25] = 0  # anchor2
        module.S = S

    # 注册并测试
    register_hooks(model, block_hook)
    blocked_acc = evaluate(model, test_loader)

    print(f"Original: {original_acc:.3f}")
    print(f"Blocked: {blocked_acc:.3f}")
    print(f"Difference: {abs(original_acc - blocked_acc):.3f}")
```

**预期**: 差异 < 2%，证明SSM未被使用

#### 实验4.3: 信息替换
**目的**: 证明卷积后状态包含所有信息

```python
def substitution_experiment(model, samples):
    """
    用"43"的卷积后状态替换其他anchor对
    """
    # 获取"43"的卷积后状态
    reference_state = get_conv_output(model, sample_43)

    results = {}
    for pair in all_pairs:
        if pair == (4, 3):
            continue

        sample = generate_sample(pair)

        # 替换卷积后状态
        with replaced_conv_state(model, reference_state):
            output = model(sample)
            pred = output.argmax()

        results[pair] = pred

    # 检查是否都坍缩到"43"的输出
    reference_output = model(sample_43).argmax()
    collapse_rate = sum(pred == reference_output for pred in results.values()) / len(results)

    return collapse_rate
```

**预期**: > 90% 的输出坍缩到"43"的结果

#### 实验4.4: 卷积核相似度
**目的**: 证明卷积核参数高度不对称

```python
def analyze_conv_weights(model):
    """
    计算训练前后卷积核参数的余弦相似度
    """
    # 提取卷积核权重
    conv_weights = []
    for layer in model.mamba.layers:
        w = layer.mixer.conv1d.weight.data  # [out_ch, in_ch, kernel]
        conv_weights.append(w)

    # 计算两两余弦相似度
    similarity_matrix = compute_cosine_similarity(conv_weights)

    # 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Convolution Kernel Cosine Similarity')
    plt.savefig('results/conv_similarity.png')

    return similarity_matrix
```

**预期**: 近乎正交 (相似度 ≈ 0)

---

### Phase 5: 可视化与分析 (2天)

#### Day 1: 生成图表
**负责agent**: visualizer

**任务**:
- [ ] 相位图（6×6热力图 × 2）
- [ ] 消融实验对比图
- [ ] 信息流可视化
- [ ] 卷积核相似度热力图
- [ ] 训练曲线

**脚本**: `src/visualize.py`

```bash
python src/visualize.py \
    --results_dir results/phase_diagram \
    --output_dir results/figures
```

**输出**:
- `phase_diagram_composite.png`
- `phase_diagram_symmetric.png`
- `ablation_comparison.png`
- `info_flow_*.png`
- `conv_similarity.png`

#### Day 2: 撰写报告
**负责agent**: analyst

**任务**:
- [ ] 聚合所有实验结果
- [ ] 计算统计量（mean ± std）
- [ ] 与论文对比
- [ ] 统计显著性检验
- [ ] 撰写markdown报告

**报告**: `results/REPORT.md`

包含:
1. 实验概述
2. 相位图结果表格
3. 消融实验表格
4. 机制分析发现
5. 与论文对比
6. 结论

---

## 时间线总结

| 周 | 任务 | 产出 |
|----|------|------|
| 1 | Phase 1: 基础实现 | 数据、模型、训练脚本可运行 |
| 2-3 | Phase 2: 相位图实验 | 90次实验完成 |
| 3 | Phase 3: 消融实验 | 3组消融对比 |
| 4 | Phase 4: 机制分析 | 信息流/阻断/替换实验 |
| 4 | Phase 5: 可视化分析 | 所有图表和报告 |

**总计**: 约4周

---

## 成功标准

### 最小可行版本 (MVP)
- [ ] 相位图实验完成（至少3×3网格，1个seed）
- [ ] 复现Mamba偏向复合解的趋势
- [ ] 至少1个消融实验（全1卷积）

### 完整版本
- [ ] 完整6×6相位图（3个seeds）
- [ ] 所有3个消融实验
- [ ] 4个机制分析实验
- [ ] 所有可视化和报告
- [ ] 定量结果与论文误差 < ±5%

### 扩展版本 (Optional)
- [ ] 测试更大模型 (d_model=128)
- [ ] 更多层数 (7-10层)
- [ ] Mamba2-2.7b在该任务上的表现

---

## 检查点

**Week 1结束**:
```bash
# 应该能运行
python src/train.py --n_layers 2 --gamma 1.0 --epochs 10

# 输出示例
Epoch 10: train_loss=0.234, composite_acc=0.89, symmetric_acc=0.12
```

**Week 2结束**:
```bash
# 应该有
ls results/phase_diagram/ | wc -l  # >= 30 (部分完成)
```

**Week 3结束**:
```bash
# 应该有
ls results/phase_diagram/ | wc -l  # = 90 (全部完成)
ls results/ablation*/  # 3个目录
```

**Week 4结束**:
```bash
# 应该有
ls results/figures/*.png  # >= 5张图
cat results/REPORT.md  # 完整报告
```

---

## 快速启动命令

```bash
# 1. 创建目录结构
mkdir -p src/{data,models,analysis}
mkdir -p tests
mkdir -p experiments
mkdir -p results/{phase_diagram,ablations,figures}

# 2. 开始实现
# 让data-engineer实现数据生成器
# 让model-engineer实现模型
# 让experiment-runner准备训练脚本

# 3. 测试单次运行
python src/train.py --n_layers 2 --gamma 1.0 --epochs 10 --output_dir results/test

# 4. 启动相位图实验
./experiments/run_phase_diagram.sh

# 5. 可视化结果
python src/visualize.py --results_dir results/phase_diagram

# 6. 生成报告
# 让analyst分析结果并撰写报告
```

---

**重点**: 专注于复合函数任务，深入理解Mamba的对称性偏差问题。逆序列任务暂时搁置。
