# 流程08：损伤识别验证

## 📋 功能说明

基于逐维度阈值，对三类验证数据（裂纹、腐蚀、无损伤）进行损伤识别验证。

**核心功能**：
- ✅ 基于验证集健康样本计算每个维度的阈值
- ✅ 对三类验证数据（crack/corrosion/health）的所有样本进行测试
- ✅ 可视化每个维度的残差是否超过阈值
- ✅ 生成检测结果统计汇总

**预期行为**：
- **crack 和 corrosion**：期望检测到（超阈值），未检测到视为失败
- **health**：期望不检测到（不超阈值），检测到视为误报

---

## 📁 输入文件

| 文件路径 | 说明 | 来源流程 |
|---------|------|---------|
| `03_preprocess_training_data_output/preprocessed_data_raw.npz` | 训练集健康样本数据 | 流程03 |
| `04_train_model_output/validation_indices.csv` | 验证集索引 | 流程04 |
| `04_train_model_output/autoencoder.pth` | 训练好的模型 | 流程04 |
| `07_preprocess_validation_data_output/health/preprocessed_data_raw.npz` | 无损伤样本 | 流程07 |
| `07_preprocess_validation_data_output/crack/preprocessed_data_raw.npz` | 裂纹损伤样本 | 流程07 |
| `07_preprocess_validation_data_output/corrosion/preprocessed_data_raw.npz` | 腐蚀损伤样本 | 流程07 |

---

## 📤 输出文件

所有输出文件位于 `08_damage_validation_output/` 目录（扁平化结构）：

### 1. 阈值数据

| 文件名 | 说明 | 数据列 |
|--------|------|-------|
| `dimension_thresholds.csv` | 每个维度的阈值 | dimension, mean, tau, upper_threshold, lower_threshold |

### 2. 裂纹损伤验证

| 文件名 | 说明 |
|--------|------|
| `combined_residuals_crack.png` | 裂纹样本残差组合图（随机6个样本，3x2布局） |
| `combined_residuals_crack.csv` | 对应的数据文件 |
| `combined_residuals_crack_failed.png` | 裂纹未检出样本残差组合图（最多6个） |
| `combined_residuals_crack_failed.csv` | 对应的数据文件 |
| `crack_detection_summary.png` | 裂纹检测结果统计汇总（2x2布局） |
| `crack_detection_summary.csv` | 对应的数据文件 |

### 3. 腐蚀损伤验证

| 文件名 | 说明 |
|--------|------|
| `combined_residuals_corrosion.png` | 腐蚀样本残差组合图（随机6个样本，3x2布局） |
| `combined_residuals_corrosion.csv` | 对应的数据文件 |
| `combined_residuals_corrosion_failed.png` | 腐蚀未检出样本残差组合图（最多6个） |
| `combined_residuals_corrosion_failed.csv` | 对应的数据文件 |
| `corrosion_detection_summary.png` | 腐蚀检测结果统计汇总（2x2布局） |
| `corrosion_detection_summary.csv` | 对应的数据文件 |

### 4. 健康样本验证

| 文件名 | 说明 |
|--------|------|
| `combined_residuals_health.png` | 健康样本残差组合图（随机6个样本，3x2布局） |
| `combined_residuals_health.csv` | 对应的数据文件 |
| `combined_residuals_health_failed.png` | 健康样本误报残差组合图（最多6个） |
| `combined_residuals_health_failed.csv` | 对应的数据文件 |
| `health_detection_summary.png` | 健康样本检测结果统计汇总（2x2布局） |
| `health_detection_summary.csv` | 对应的数据文件 |

---

## ⚙️ 参数配置

所有参数集中在脚本顶部的"参数配置区"，按自然逻辑顺序编写。

### 核心参数

```python
# --- 模型结构配置（需与流程04一致）---
# ⚠️ 关键：这些参数必须与04_train_model_main.py中完全一致！
ENCODER_DIMS = [768, 384, 192]      # 编码器各层维度
LATENT_DIM = 192                    # 潜在空间维度
DECODER_DIMS = [192, 384, 768]      # 解码器各层维度
DROPOUT = 0.0                       # Dropout比例
ACTIVATION = "relu"                 # 激活函数（relu/gelu/tanh/sigmoid）
```

**⚠️ 重要说明**：
- 这些参数值来自于流程04实际训练时使用的配置
- 如果流程04修改了模型结构，必须同步修改此处参数
- 模型结构不匹配会导致加载模型时出错

### 阈值计算方法

支持三种阈值计算方法，通过 `TAU_METHOD` 参数选择：

#### 方法1：quantile_abs（分位数法）

```python
TAU_METHOD = "quantile_abs"
VAL_QUANTILE_BASE = 0.995          # 99.5%分位数
```

基于验证集残差绝对值的分位数计算阈值。
- **阈值**: τ = quantile(|residuals|, q)
- **判定**: |residual| > τ 视为超阈值
- **特点**: 简单直观，不考虑残差符号

#### 方法2：kstd_abs（绝对值K倍标准差法）⭐ 默认

```python
TAU_METHOD = "kstd_abs"
TAU_KSTD_K = 3.0                   # k值（通常取2-3）
```

基于残差绝对值的均值和标准差计算阈值。
- **阈值**: τ = |μ| + k·σ
- **判定**: |residual| > τ 视为超阈值
- **特点**: 考虑残差分布特性，更稳健

#### 方法3：mean_kstd（均值K倍标准差法）

```python
TAU_METHOD = "mean_kstd"
TAU_MEAN_KSTD_K = 3.0              # k值（通常取2-3）
```

基于残差均值的K倍标准差计算阈值（保留符号）。
- **阈值**: τ = μ + k·σ
- **判定**: residual > τ 或 residual < -τ 视为超阈值
- **特点**: 保留残差方向信息，适用于有偏差的残差分布

### 可视化参数

```python
RANDOM_SEED = 42                   # 随机种子，用于选择可视化样本
NUM_VIS_SAMPLES = 6                # 每类数据随机可视化样本数（3x2布局）
NUM_FAILED_SAMPLES = 6             # 失败样本可视化数量（3x2布局）
PLOT_DPI = 300                     # 图片分辨率
```

### 计算参数

```python
BATCH_SIZE = 512                   # 批处理大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### 数据类型配置

```python
DATA_TYPES = ['crack', 'corrosion', 'health']  # 要处理的验证数据类型
```

可根据需要调整，例如只处理部分类型：
```python
DATA_TYPES = ['crack']             # 仅处理裂纹
DATA_TYPES = ['crack', 'health']   # 处理裂纹和健康
```

---

## 🚀 使用方法

### 1. 基本用法

```bash
# 确保已完成流程03、04、07
python 08_damage_validation_main.py
```

### 2. 执行流程

```
流程08执行步骤：
├── Step 1/5: 加载训练集数据和验证集索引
├── Step 2/5: 加载训练好的模型
├── Step 3/5: 计算每个维度的阈值
├── Step 4/5: 对三类验证数据进行全样本测试
│   ├── 处理 CRACK 数据
│   ├── 处理 CORROSION 数据
│   └── 处理 HEALTH 数据
└── Step 5/5: 打印最终统计总结
```

### 3. 预期输出示例

```
======================================================================
流程08：损伤识别验证
======================================================================

[Output] Directory: c:\data\AE-main\script\08_damage_validation_output

[Device] Using: cuda

[Step 1/5] Loading training data and validation indices...
  - Training data shape: (2000, 150)
  - Validation samples: 200

[Step 2/5] Loading trained model...
  - Loaded model from: c:\data\AE-main\script\04_train_model_output\autoencoder.pth

[Step 3/5] Computing per-dimension thresholds...
  - Computing thresholds using 200 validation samples...
  - Residual matrix shape: (200, 150)
  - Using method: kstd_abs (k=3.0)
  - Threshold statistics:
    Min: 0.001234, Max: 0.098765
    Mean: 0.023456, Median: 0.019876

[Step 4/5] Testing validation data for all damage types...

======================================================================
  Processing CRACK data...
======================================================================
  - Loaded crack data: (100, 150)
  - Total 100 samples will be tested
  - Selected 6 samples for visualization: [5, 23, 34, 56, 78, 91]
  - Completed: Processed 100 samples

  [Visualization] Creating combined plot for 6 random samples...
    - Saved figure: ...\combined_residuals_crack.png
    - Saved data: ...\combined_residuals_crack.csv

  [Visualization] Creating plot for 2 failed samples...
    Failed sample indices: [12, 67]
    - Saved figure: ...\combined_residuals_crack_failed.png
    - Saved data: ...\combined_residuals_crack_failed.csv

  [Visualization] Generating CRACK detection summary...
    - Saved figure: ...\crack_detection_summary.png
    - Saved data: ...\crack_detection_summary.csv

  [CRACK Statistics]
  ------------------------------------------------------------------
    Total Samples:        100
    Detected Samples:     98 (98.0%)
    Undetected Samples:   2 (2.0%)
  ------------------------------------------------------------------
    Exceeded Dimensions:
      - Mean:             45.23
      - Median:           42.0
      - Min:              0
      - Max:              89
      - Std:              18.45
  ------------------------------------------------------------------
    Exceed Ratio:
      - Mean:             30.15%
      - Median:           28.00%
      - Max:              59.33%
  ------------------------------------------------------------------

[类似输出 for CORROSION and HEALTH...]

======================================================================
[Step 5/5] Overall Summary
======================================================================

CRACK:
----------------------------------------------------------------------
  Detection Rate:       98.0%
  Mean Exceeded Dims:   45.23
  Median Exceeded Dims: 42.0
  True Positives:       98 / 100 (98.0%)

CORROSION:
----------------------------------------------------------------------
  Detection Rate:       96.0%
  Mean Exceeded Dims:   38.67
  Median Exceeded Dims: 35.0
  True Positives:       96 / 100 (96.0%)

HEALTH:
----------------------------------------------------------------------
  Detection Rate:       3.0%
  Mean Exceeded Dims:   0.15
  Median Exceeded Dims: 0.0
  False Positives:      3 / 100 (3.0%)

======================================================================
流程08完成！
======================================================================

[Output Files]:
  - Threshold data: ...\dimension_thresholds.csv
  - All visualizations: ...\08_damage_validation_output\*.png
```

---

## 📊 输出说明

### 1. 残差组合图（3x2布局）

每个子图显示一个样本的：
- **蓝色柱状图**: 未超阈值的维度残差
- **红色柱状图**: 超阈值的维度残差
- **绿色虚线**: 阈值边界（上下）
- **标题**: 样本索引和超阈值维度数/比例

**解读**：
- 红色柱子越多 → 损伤越明显
- 对于 crack/corrosion：期望看到大量红色柱子
- 对于 health：期望看到全部蓝色柱子

### 2. 检测结果统计汇总（2x2布局）

**左上**：每个样本的超阈值维度数柱状图
- 横轴：样本索引
- 纵轴：超阈值维度数
- 颜色：检测成功/失败（根据数据类型）

**右上**：超阈值维度数分布直方图
- 显示所有样本的超阈值维度分布
- 标注均值和中位数

**左下**：检出率饼图
- 显示检测成功/失败的样本比例
- crack/corrosion：检测到是成功
- health：未检测到是成功

**右下**：超阈值比例累积分布
- 显示超阈值比例的累积分布
- 帮助理解整体检测性能

### 3. 失败样本组合图

**crack/corrosion 失败样本**：
- 显示未被检测到的损伤样本（n_exceed == 0）
- 用于分析漏检原因

**health 失败样本**：
- 显示被误报为损伤的健康样本（n_exceed > 0）
- 用于分析误报原因

---

## 🎯 评估指标

### 损伤样本（crack/corrosion）

| 指标 | 说明 | 期望值 |
|------|------|--------|
| 检测率 | 被检测到的样本比例 | ≥ 95% |
| 平均超阈值维度数 | 每个样本超阈值的维度数 | ≥ 30 |
| 漏检率 | 未被检测到的样本比例 | ≤ 5% |

### 健康样本（health）

| 指标 | 说明 | 期望值 |
|------|------|--------|
| 误报率 | 被误报为损伤的样本比例 | ≤ 5% |
| 平均超阈值维度数 | 每个样本超阈值的维度数 | ≈ 0 |
| 特异性 | 正确识别为健康的样本比例 | ≥ 95% |

---

## 🔧 调试技巧

### 1. 检测率不理想

**问题**：损伤样本检测率低于预期

**可能原因**：
- 阈值过于严格（TAU_KSTD_K 过大）
- 阈值方法不适合当前数据分布

**解决方案**：
```python
# 降低 k 值，使阈值更宽松
TAU_KSTD_K = 2.0  # 从 3.0 降低到 2.0

# 或尝试其他阈值方法
TAU_METHOD = "quantile_abs"
VAL_QUANTILE_BASE = 0.99  # 从 0.995 降低到 0.99
```

### 2. 误报率过高

**问题**：健康样本误报率过高

**可能原因**：
- 阈值过于宽松（TAU_KSTD_K 过小）
- 验证集选择不当

**解决方案**：
```python
# 提高 k 值，使阈值更严格
TAU_KSTD_K = 4.0  # 从 3.0 提高到 4.0

# 或调整分位数
VAL_QUANTILE_BASE = 0.999  # 从 0.995 提高到 0.999
```

### 3. 查看详细残差

**需求**：想看更多样本的残差分布

**解决方案**：
```python
# 增加可视化样本数
NUM_VIS_SAMPLES = 12  # 需调整为 3 的倍数

# 查看更多失败样本
NUM_FAILED_SAMPLES = 12
```

### 4. 只测试特定数据类型

**需求**：只想测试裂纹损伤

**解决方案**：
```python
DATA_TYPES = ['crack']  # 只处理裂纹
```

---

## ⚠️ 注意事项

### 1. 前置依赖

**必须先完成**：
- ✅ 流程03：训练数据预处理
- ✅ 流程04：模型训练
- ✅ 流程07：验证数据预处理

### 2. 模型结构一致性 ⚠️

**关键**：本流程中的模型结构参数必须与流程04完全一致！

```python
# 08_damage_validation_main.py 中的参数
ENCODER_DIMS = [768, 384, 192]
LATENT_DIM = 192
DECODER_DIMS = [192, 384, 768]
DROPOUT = 0.0
ACTIVATION = "relu"

# 必须与 04_train_model_main.py 中的参数一致：
# AE_ENCODER_DIMS = [768, 384, 192]
# AE_LATENT_DIM = 192
# AE_DECODER_DIMS = [192, 384, 768]
# AE_DROPOUT = 0.0
# AE_ACTIVATION = 'relu'
```

**如何检查**：
1. 打开 `04_train_model_main.py`
2. 查找参数配置区的 `AE_ENCODER_DIMS` 等参数
3. 确保与流程08中的参数完全一致

**常见错误**：
```
RuntimeError: Error(s) in loading state_dict for Autoencoder:
    size mismatch for encoder.0.weight: copying a param with shape torch.Size([768, 252])
    from checkpoint, the shape in current model is torch.Size([512, 252]).
```

**解决方法**：
- 检查并同步模型结构参数
- 确保激活函数、Dropout等配置一致

### 3. 阈值方法选择

**建议**：
- 初次使用：`kstd_abs` (k=3.0)
- 残差分布偏斜：`quantile_abs` (q=0.995)
- 需要方向信息：`mean_kstd` (k=3.0)

### 4. 内存占用

**说明**：
- 一次性加载所有验证样本
- 批量处理减少内存占用
- GPU 内存不足时降低 `BATCH_SIZE`

### 5. 随机性控制

**说明**：
- 使用固定的 `RANDOM_SEED = 42`
- 保证每次运行选择相同的可视化样本
- 便于结果复现和对比

---

## 📈 典型结果

### 良好的结果

```
CRACK:
  Detection Rate:       98.0%
  Mean Exceeded Dims:   45.23
  
CORROSION:
  Detection Rate:       96.0%
  Mean Exceeded Dims:   38.67
  
HEALTH:
  False Positives:      3 / 100 (3.0%)
```

**特征**：
- 损伤检测率 > 95%
- 健康样本误报率 < 5%
- 平均超阈值维度数显著 > 0

### 需要调整的结果

```
CRACK:
  Detection Rate:       75.0%  ← 过低
  Mean Exceeded Dims:   12.5   ← 过低
  
HEALTH:
  False Positives:      15 / 100 (15.0%)  ← 过高
```

**建议**：
- 调整阈值计算方法
- 检查模型训练质量
- 验证数据质量

---

## 🔗 相关流程

| 流程 | 关系 | 说明 |
|------|------|------|
| 流程03 | 前置依赖 | 提供训练集数据 |
| 流程04 | 前置依赖 | 提供模型和验证集索引 |
| 流程05 | 参考 | 阈值方法研发（可选） |
| 流程07 | 前置依赖 | 提供验证集数据 |
| 流程09 | 后续 | 时历动画渲染 |

---

## 📚 参考文献

阈值计算方法参考：
1. **分位数法**: 常用于异常检测，简单直观
2. **K倍标准差法**: 基于统计学的3σ原则，更稳健
3. **均值K倍标准差法**: 保留残差方向信息，适用于有偏分布

---

**重构日期**: 2025年11月2日  
**重构版本**: v1.0  
**重构状态**: ✅ 完成
