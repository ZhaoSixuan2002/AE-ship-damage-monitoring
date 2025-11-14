# 流程09：时历动画渲染

## 功能说明

基于验证数据（crack/corrosion/health）生成测点损伤可疑度随时间变化的热力图动画。

**核心功能**：
- 读取验证数据，对每个样本依次预测残差
- 计算超阈值倍数，更新252个测点的透明度
- 使用滑动窗口平均方式平滑透明度变化
- 生成4×63热力图动画（红色渐变，透明度随时间变化）
- 支持批量处理三种验证数据类型（crack/corrosion/health）

---

## 输入文件

### 必需输入

1. **训练好的模型**
   - 路径：`04_train_model_output/autoencoder.pth`
   - 来源：流程04（模型训练）

2. **训练数据**（用于计算阈值）
   - 路径：`03_preprocess_training_data_output/preprocessed_data_raw.npz`
   - 来源：流程03（训练数据预处理）

3. **验证数据**（三种类型）
   - 路径：`07_preprocess_validation_data_output/{crack|corrosion|health}/preprocessed_data_raw.npz`
   - 来源：流程07（验证数据预处理）

---

## 输出文件

### 输出目录结构

```
09_render_time_history_output/
├── dimension_thresholds.csv              # 每个维度的阈值
├── crack/                                # crack类型输出
│   ├── opacity_animation.mp4             # 透明度时历动画（MP4格式，需要FFmpeg）
│   │   或 opacity_animation.gif          # 透明度时历动画（GIF格式，通用）
│   ├── opacity_data.csv                  # 每个时间步每个测点的透明度数据 [T, D]
│   ├── exceed_history.csv                # 每个时间步每个测点的超阈值倍数历史 [T, D]
│   ├── first_9_frames_grid.png           # 前9帧的3x3组合大图，带ID标注（300 dpi）
│   ├── opacity_heatmap_3x3_samples.png   # 抽样3x3子图热力图（300 dpi）
│   └── opacity_heatmap_3x3_samples.csv   # 3x3子图对应的数据
├── corrosion/                            # corrosion类型输出（结构同上）
│   └── ...
└── health/                               # health类型输出（结构同上）
    └── ...
```

### 输出文件说明

1. **dimension_thresholds.csv**
   - 每个维度的阈值，用于判断超阈值倍数
   - 列：`dimension`, `tau`

2. **opacity_animation.mp4/.gif**
   - 透明度时历动画
   - 4×63热力图，颜色为红色渐变
   - 透明度100% = 完全透明（白色），透明度0% = 完全不透明（深红色）
   - 帧率：10 FPS（可配置）

3. **opacity_data.csv**
   - 每个时间步每个测点的透明度数据
   - 列：`time_step`, `dim_0`, `dim_1`, ..., `dim_251`
   - 行数 = 处理的样本数

4. **exceed_history.csv**
   - 每个时间步每个测点的超阈值倍数历史
   - 列：`time_step`, `dim_0`, `dim_1`, ..., `dim_251`
   - 行数 = 处理的样本数

5. **first_9_frames_grid.png**
   - 前9帧的3x3组合大图
   - 每个格子带ID标注（0~251）
   - 高分辨率（300 dpi），适合论文发表

6. **opacity_heatmap_3x3_samples.png**
   - 抽样9个时间步的3x3子图热力图
   - 默认抽样样本：[0, 10, 20, 30, 40, 50, 60, 70, 80]
   - 高分辨率（300 dpi），适合论文发表

---

## 参数配置

所有参数在脚本顶部的"参数配置区"集中管理。

### 路径配置

```python
OUTPUT_DIR_04 = "04_train_model_output"           # 模型输入目录
OUTPUT_DIR_07 = "07_preprocess_validation_data_output"  # 验证数据输入目录
OUTPUT_DIR = "09_render_time_history_output"      # 输出目录
```

### 模型架构参数（必须与流程04一致）

```python
AE_ENCODER_DIMS = [768, 384, 192]  # 编码器各隐藏层维度
AE_LATENT_DIM = 192                # 潜在空间维度
AE_DECODER_DIMS = [192, 384, 768]  # 解码器各隐藏层维度
AE_DROPOUT = 0.0                   # Dropout概率
AE_ACTIVATION = "relu"             # 激活函数：'relu', 'elu', 'leaky_relu'
```

### 阈值计算参数（必须与流程08一致）

```python
THRESHOLD_METHOD = "quantile_abs"  # 阈值方法：'quantile_abs', 'kstd_abs', 'mean_kstd'
THRESHOLD_QUANTILE = 0.95          # quantile_abs方法的分位数（0.90 ~ 0.99）
THRESHOLD_K_SIGMA = 3.0            # kstd_abs/mean_kstd方法的k倍标准差（1.0 ~ 5.0）
```

**说明**：
- `quantile_abs`：基于验证集残差绝对值的分位数
- `kstd_abs`：基于验证集残差绝对值的k倍标准差
- `mean_kstd`：基于验证集残差绝对值的均值+k倍标准差

### 透明度计算参数

```python
OPACITY_DECAY_FACTOR = 10.0        # 透明度衰减因子：平均超阈值1倍 = 减少10%透明度
OPACITY_WINDOW_SIZE = 10           # 滑动窗口大小（平滑超阈值倍数波动）
```

**说明**：
- 透明度计算公式：`opacity = 100% - avg_exceed_ratio × DECAY_FACTOR`
- 滑动窗口用于平滑超阈值倍数的波动，避免动画过于跳跃
- 推荐范围：
  - `OPACITY_DECAY_FACTOR`: 5.0 ~ 20.0
  - `OPACITY_WINDOW_SIZE`: 5 ~ 20

### 数据处理参数

```python
MAX_SAMPLES = -1                   # 最大处理样本数：-1表示全部，正整数表示处理前N个样本
PROCESS_DATA_TYPES = ["crack", "corrosion", "health"]  # 要处理的数据类型列表
```

### 热力图参数

```python
GRID_ROWS = 4                      # 热力图行数（252个测点 = 4行 × 63列）
GRID_COLS = 63                     # 热力图列数
COLORMAP = "Reds"                  # 颜色映射：'Reds', 'YlOrRd', 'OrRd'等
```

### 动画参数

```python
ANIMATION_FPS = 10                 # 帧率（每秒帧数）
SAVE_FORMAT = "mp4"                # 保存格式：'mp4'（需要FFmpeg）或'gif'（通用）
ANIMATION_DPI = 100                # 动画分辨率
```

**说明**：
- MP4格式需要安装FFmpeg，否则自动降级为GIF
- 推荐帧率：5 ~ 20 FPS
- 推荐DPI：80 ~ 150（过高会导致文件过大）

### 3x3子图热力图参数

```python
HEATMAP_3X3_SAMPLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # 要展示的9个样本索引
HEATMAP_DPI = 300                  # 3x3子图分辨率
```

---

## 执行方法

### 基本用法

```bash
# 处理所有数据类型（crack, corrosion, health）
python 09_render_time_history_main.py

# 或显式指定所有类型
python 09_render_time_history_main.py --type all
```

### 指定数据类型

```bash
# 只处理crack类型
python 09_render_time_history_main.py --type crack

# 只处理crack和corrosion类型
python 09_render_time_history_main.py --type crack corrosion

# 只处理health类型
python 09_render_time_history_main.py --type health
```

### 命令行参数

- `--type` 或 `-t`：指定要处理的数据类型
  - 不指定：处理所有类型（crack, corrosion, health）
  - `all`：处理所有类型
  - 多个类型：空格分隔，例如 `crack corrosion`

---

## 工作流程

### 执行流程

```
1. 加载训练数据（用于计算阈值）
   ↓
2. 加载模型
   ↓
3. 计算阈值（使用指定方法）
   ↓
4. 对每个数据类型：
   4.1 加载验证数据
   4.2 计算透明度时间序列
       - 对每个样本预测残差
       - 计算超阈值倍数
       - 使用滑动窗口平均
       - 更新透明度
   4.3 保存透明度和超阈值倍数数据
   4.4 生成前9帧组合图
   4.5 生成3x3子图热力图
   4.6 生成动画
```

### 透明度计算逻辑

```
初始状态：
  - 所有测点透明度 = 100%（完全透明）

对每个样本：
  1. 预测残差 = 模型输出 - 真实值
  2. 计算超阈值倍数 = |残差| / 阈值（仅当|残差| > 阈值）
  3. 滑动窗口平均超阈值倍数（平滑波动）
  4. 更新透明度 = 100% - 平均超阈值倍数 × 衰减因子
  5. 限制透明度范围在 [0%, 100%]

可视化映射：
  - 透明度100% → 白色（无损伤可疑）
  - 透明度0% → 深红色（高损伤可疑）
```

---

## 输出示例

### 透明度数据示例（opacity_data.csv）

```csv
time_step,dim_0,dim_1,dim_2,...,dim_251
0,100.0,100.0,100.0,...,100.0
1,98.5,99.2,100.0,...,97.8
2,95.3,97.1,99.5,...,93.2
...
```

### 超阈值倍数历史示例（exceed_history.csv）

```csv
time_step,dim_0,dim_1,dim_2,...,dim_251
0,0.0,0.0,0.0,...,0.0
1,0.15,0.08,0.0,...,0.22
2,0.47,0.29,0.05,...,0.68
...
```

### 3x3子图热力图数据示例（opacity_heatmap_3x3_samples.csv）

```csv
sample_index,mean_opacity,min_opacity,max_opacity
0,100.0,100.0,100.0
10,92.5,75.3,100.0
20,85.2,45.7,99.8
...
```

---

## 注意事项

### 1. 前置依赖

**必须先完成以下流程**：
- 流程03：训练数据预处理（生成preprocessed_data_raw.npz）
- 流程04：模型训练（生成autoencoder.pth）
- 流程07：验证数据预处理（生成各类型的preprocessed_data_raw.npz）

### 2. 参数一致性

**关键参数必须与前置流程一致**：
- 模型架构参数必须与流程04完全一致
- 阈值计算方法建议与流程08一致（但可以独立调整）

### 3. 内存占用

- 动画生成可能占用较大内存（取决于样本数和帧率）
- 如果内存不足，可以：
  - 减少 `MAX_SAMPLES`
  - 降低 `ANIMATION_DPI`
  - 使用GIF格式代替MP4

### 4. FFmpeg依赖

- MP4格式需要安装FFmpeg
- 如果FFmpeg不可用，脚本会自动降级为GIF格式
- 推荐安装FFmpeg以获得更好的压缩效果

### 5. 计算时间

- 透明度计算时间取决于样本数和维度数
- 动画生成时间取决于帧数和分辨率
- 典型运行时间（100个样本）：约1~3分钟

### 6. 输出文件大小

- 动画文件可能较大（取决于帧数和分辨率）
- MP4格式通常比GIF格式小
- 高分辨率PNG图片（300 dpi）较大，适合论文发表

---

## 参数调优建议

### 快速测试

```python
MAX_SAMPLES = 20                   # 只处理前20个样本
ANIMATION_FPS = 5                  # 降低帧率
ANIMATION_DPI = 80                 # 降低分辨率
SAVE_FORMAT = "gif"                # 使用GIF格式
```

### 论文发表

```python
MAX_SAMPLES = -1                   # 处理全部样本
ANIMATION_FPS = 10                 # 标准帧率
ANIMATION_DPI = 100                # 标准分辨率
SAVE_FORMAT = "mp4"                # MP4格式（文件更小）
HEATMAP_DPI = 300                  # 高分辨率静态图
```

### 演示展示

```python
MAX_SAMPLES = 50                   # 适中的样本数
ANIMATION_FPS = 15                 # 较高帧率（更流畅）
ANIMATION_DPI = 120                # 较高分辨率
SAVE_FORMAT = "mp4"                # MP4格式
```

---

## 故障排除

### 问题1：找不到模型文件

**错误信息**：
```
[Error] Model file not found: 04_train_model_output/autoencoder.pth
```

**解决方法**：
- 先运行 `04_train_model_main.py` 训练模型

### 问题2：找不到验证数据

**错误信息**：
```
Validation data not found for crack: 07_preprocess_validation_data_output/crack/preprocessed_data_raw.npz
```

**解决方法**：
- 先运行 `06_generate_validation_data_main.py` 生成验证数据
- 再运行 `07_preprocess_validation_data_main.py` 预处理验证数据

### 问题3：FFmpeg不可用

**警告信息**：
```
[Warning] FFmpeg not available
[Info] Falling back to GIF format...
```

**解决方法**：
- 安装FFmpeg：`conda install ffmpeg` 或 从官网下载
- 或者接受GIF格式（通用但文件较大）

### 问题4：内存不足

**错误信息**：
```
MemoryError: Unable to allocate array
```

**解决方法**：
- 减少 `MAX_SAMPLES`（如设为50或100）
- 降低 `ANIMATION_DPI`（如设为80）
- 使用GIF格式代替MP4

### 问题5：动画生成过慢

**解决方法**：
- 减少样本数 `MAX_SAMPLES`
- 降低帧率 `ANIMATION_FPS`
- 降低分辨率 `ANIMATION_DPI`

---

## 输出解读

### 透明度的含义

- **透明度100%**：测点完全透明（白色），表示无损伤可疑
- **透明度50%**：测点半透明（浅红色），表示中等损伤可疑
- **透明度0%**：测点完全不透明（深红色），表示高损伤可疑

### 动画的含义

- **时间轴**：每一帧对应一个验证样本（按样本顺序）
- **空间分布**：4×63热力图对应252个测点的空间位置
- **颜色变化**：红色加深表示损伤可疑度增加

### 预期结果

- **健康样本（health）**：
  - 透明度应保持较高（接近100%）
  - 颜色应保持较浅（接近白色）
  
- **损伤样本（crack/corrosion）**：
  - 损伤区域透明度应降低
  - 损伤区域颜色应加深（红色）
  - 非损伤区域应保持高透明度

---

## 相关文件

- **前置流程**：
  - `03_preprocess_training_data_main.py` - 训练数据预处理
  - `04_train_model_main.py` - 模型训练
  - `06_generate_validation_data_main.py` - 生成验证数据
  - `07_preprocess_validation_data_main.py` - 验证数据预处理

- **后续流程**：
  - `10_render_vtu_animation_main.py` - VTU 3D渲染动画

- **参考流程**：
  - `08_damage_validation_main.py` - 损伤识别验证（阈值计算参考）

---

## 版本信息

- **脚本版本**：1.0
- **创建日期**：2025-01-02
- **最后更新**：2025-01-02
- **作者**：GitHub Copilot
- **依赖库**：numpy, pandas, matplotlib, torch

---

## 更新日志

### v1.0 (2025-01-02)
- ✅ 初始版本
- ✅ 基于原 `time_history_animation.py` 重构
- ✅ 参数集中在脚本顶部
- ✅ 所有工具函数内嵌
- ✅ 独立输出目录 `09_render_time_history_output/`
- ✅ 支持三种验证数据类型（crack/corrosion/health）
- ✅ 支持命令行参数指定数据类型
- ✅ 新增3x3子图热力图输出
- ✅ 详细的进度反馈和错误处理
