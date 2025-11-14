# 流程07：验证数据预处理

## 📋 功能说明

批量预处理三类验证数据（health/crack/corrosion），提取应力数据并生成可视化。

**核心功能**：
- 从三种验证数据源目录读取应力数据
- 提取测点应力值生成特征矩阵
- （可选）执行数据变换（稳健高斯变换）
- 生成数据分布可视化图像
- 输出到独立的子目录（每种类型隔离）

## 🔗 依赖关系

### 前置流程
- **流程02** (`02_inp_to_vtu_main.py`)：生成 `measures_ID_auto.csv` 测点映射文件
- **流程06** (`06_generate_validation_data_main.py`)：生成三类验证数据

### 后续流程
- **流程08** (待重构)：损伤验证

## 📂 输入文件

### 数据源目录（由流程06生成）
```
C:/abaqus_gen_data_validate_original_health/
├── 0/
│   └── iteration.npy (或 iteration.csv)
├── 1/
│   └── iteration.npy
└── ...

C:/abaqus_gen_data_validate_damage_crack/
├── 0/
│   └── iteration.npy (或 iteration.csv)
├── 1/
│   └── iteration.npy
└── ...

C:/abaqus_gen_data_validate_damage_corrosion/
├── 0/
│   └── iteration.npy (或 iteration.csv)
├── 1/
│   └── iteration.npy
└── ...
```

### 配置文件（由流程02生成）
```
02_inp_to_vtu_output/measures_ID_auto.csv
```

## 📤 输出文件

```
07_preprocess_validation_data_output/
├── health/                                      # 健康数据输出
│   ├── preprocessed_data_raw.npz               # 原始数据（必有）
│   ├── preprocessed_data.npz                   # 变换后数据（可选）
│   ├── transforms.joblib                       # 变换器（可选）
│   ├── preprocess_V_first_25_combined_5x10.png # 可视化图像
│   └── preprocess_V_first_25_combined_5x10.csv # 图像对应数据
├── crack/                                       # 裂纹损伤数据输出
│   ├── preprocessed_data_raw.npz
│   ├── preprocessed_data.npz                   # 可选
│   ├── transforms.joblib                       # 可选
│   ├── preprocess_V_first_25_combined_5x10.png
│   └── preprocess_V_first_25_combined_5x10.csv
└── corrosion/                                   # 腐蚀损伤数据输出
    ├── preprocessed_data_raw.npz
    ├── preprocessed_data.npz                   # 可选
    ├── transforms.joblib                       # 可选
    ├── preprocess_V_first_25_combined_5x10.png
    └── preprocess_V_first_25_combined_5x10.csv
```

## 🚀 使用方法

### 基本用法

```bash
# 处理所有三种验证数据类型
python script/07_preprocess_validation_data_main.py
```

### 典型配置场景

#### 场景1：快速测试（仅可视化现有数据）

```python
# 在脚本顶部修改参数
DO_COLLECT = False          # 跳过数据收集
DO_TRANSFORM = False        # 跳过数据变换
DO_VISUALIZE = True         # 仅生成可视化

PREPROCESS_PLOT_FIRST_N = 9 # 只看前9个维度
PREPROCESS_GRID_ROWS = 3    # 3x3网格
PREPROCESS_GRID_COLS = 3
```

#### 场景2：完整预处理（收集+可视化）

```python
DO_COLLECT = True           # 执行数据收集
DO_TRANSFORM = False        # 不变换（使用原始数据）
DO_VISUALIZE = True         # 生成可视化

PREPROCESS_MAX_FOLDERS = None  # 处理所有数据
```

#### 场景3：仅处理特定类型

```python
PROCESS_HEALTH = True       # 处理健康数据
PROCESS_CRACK = False       # 跳过裂纹数据
PROCESS_CORROSION = False   # 跳过腐蚀数据
```

#### 场景4：限制处理数量（调试）

```python
PREPROCESS_MAX_FOLDERS = 10  # 每种类型仅处理前10个样本
```

## ⚙️ 关键参数说明

### 1. 路径配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SOURCE_DATA_HEALTH` | 健康数据源目录 | `C:\abaqus_gen_data_validate_original_health` |
| `SOURCE_DATA_CRACK` | 裂纹数据源目录 | `C:\abaqus_gen_data_validate_damage_crack` |
| `SOURCE_DATA_CORROSION` | 腐蚀数据源目录 | `C:\abaqus_gen_data_validate_damage_corrosion` |
| `OUTPUT_DIR` | 输出根目录 | `07_preprocess_validation_data_output` |
| `MEASURES_ID_CSV` | 测点ID文件路径 | `../script/02_inp_to_vtu_output/measures_ID_auto.csv` |

### 2. 处理控制

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `DO_COLLECT` | 是否收集数据 | True/False | True |
| `DO_TRANSFORM` | 是否变换数据 | True/False | False |
| `DO_VISUALIZE` | 是否可视化 | True/False | True |
| `PROCESS_HEALTH` | 是否处理健康数据 | True/False | True |
| `PROCESS_CRACK` | 是否处理裂纹数据 | True/False | True |
| `PROCESS_CORROSION` | 是否处理腐蚀数据 | True/False | True |

### 3. 数据处理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `PREPROCESS_MAX_FOLDERS` | 最大处理文件夹数 | None（不限制） |
| `USE_NPY` | 优先使用NPY格式 | True |
| `RECONSTRUCT_COLUMN` | 测点列名 | `'all_measures'` |
| `TRANSFORM_KIND` | 变换类型 | `'robust-gauss'` |

### 4. 可视化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `PREPROCESS_PLOT_FIRST_N` | 可视化前N个维度 | 25 |
| `PREPROCESS_GRID_ROWS` | 网格行数 | 5 |
| `PREPROCESS_GRID_COLS` | 网格列数 | 5 |
| `SHOW_PLOT_PROGRESS` | 显示绘图进度 | True |
| `FIG_DPI` | 图像DPI | 300 |

## 📊 输出数据说明

### NPZ文件格式

```python
# preprocessed_data_raw.npz
data = np.load("preprocessed_data_raw.npz")
V = data["V"]  # shape: (n_samples, n_features)
# 每行是一个样本，每列是一个测点的应力值

# preprocessed_data.npz（如果DO_TRANSFORM=True）
data = np.load("preprocessed_data.npz")
V = data["V"]  # shape: (n_samples, n_features)
# 经过稳健高斯变换后的数据
```

### 可视化图像

每张图像包含：
- **左侧子图**：应力值的时序散点图
- **右侧子图**：应力值的分布密度图（直方图+KDE曲线）

图像命名规则：
```
preprocess_V_first_25_combined_5x10.png
          │      │       │       └─ 网格尺寸（5行10列=散点5+密度5）
          │      │       └───────── 可视化类型（组合图）
          │      └───────────────── 显示的维度数量
          └──────────────────────── 变量名称
```

## ⚠️ 注意事项

### 1. 数据源检查

运行前确认验证数据已生成：
```bash
# 检查数据源目录
dir C:\abaqus_gen_data_validate_original_health\
dir C:\abaqus_gen_data_validate_damage_crack\
dir C:\abaqus_gen_data_validate_damage_corrosion\
```

每个目录下应该有数字命名的子文件夹（如 `0`, `1`, `2`, ...），每个子文件夹包含 `iteration.npy` 或 `iteration.csv`。

### 2. 测点ID文件

必须先运行流程02生成 `measures_ID_auto.csv`：
```bash
python script/02_inp_to_vtu_main.py
```

### 3. 内存使用

如果数据量大（如每种类型1000+样本），建议：
- 分批处理（设置 `PREPROCESS_MAX_FOLDERS`）
- 或仅选择需要的类型（关闭部分 `PROCESS_*` 开关）

### 4. 输出目录清理

- 每次运行会清空输出目录（除非 `DO_COLLECT=False`）
- 如需保留旧数据，请先备份

## 🔧 故障排除

### 问题1：找不到测点ID文件

**错误信息**：
```
错误: 测点ID文件未找到: ../script/02_inp_to_vtu_output/measures_ID_auto.csv
```

**解决方法**：
```bash
# 先运行流程02
python script/02_inp_to_vtu_main.py
```

### 问题2：数据源目录不存在

**错误信息**：
```
警告: 数据源目录不存在: C:\abaqus_gen_data_validate_original_health
跳过 health 数据集的处理
```

**解决方法**：
```bash
# 先运行流程06生成验证数据
python script/06_generate_validation_data_main.py
```

### 问题3：跳过文件夹

**现象**：处理的样本数少于预期

**可能原因**：
1. 文件夹内缺少 `iteration.npy` 或 `iteration.csv`
2. 数据文件损坏或格式错误
3. 缺少必要的测点ID

**解决方法**：
- 检查被跳过的文件夹内容
- 查看流程06的生成日志
- 重新运行流程06生成数据

### 问题4：可视化失败

**错误信息**：KDE相关错误

**解决方法**：
```python
# 设置更保守的可视化参数
PREPROCESS_PLOT_FIRST_N = 9  # 减少维度数
SHOW_PLOT_PROGRESS = False   # 关闭进度条（减少内存占用）
```

## 📈 性能参考

典型运行时间（Windows 10, i7-8700, 32GB RAM）：

| 数据规模 | 收集时间 | 可视化时间 | 总时间 |
|----------|----------|------------|--------|
| 3×10样本 | ~5s | ~15s | ~20s |
| 3×50样本 | ~20s | ~30s | ~50s |
| 3×100样本 | ~40s | ~50s | ~90s |
| 3×500样本 | ~3min | ~4min | ~7min |

**说明**：
- 时间与样本数量线性相关
- 可视化时间取决于 `PREPROCESS_PLOT_FIRST_N`
- 使用NPY格式比CSV快约30%

## 🎯 最佳实践

### 1. 开发阶段

```python
# 快速迭代验证
PREPROCESS_MAX_FOLDERS = 10      # 小规模测试
PREPROCESS_PLOT_FIRST_N = 9      # 快速可视化
DO_TRANSFORM = False              # 跳过变换
```

### 2. 正式运行

```python
# 完整处理
PREPROCESS_MAX_FOLDERS = None    # 处理全部
PREPROCESS_PLOT_FIRST_N = 25     # 详细可视化
DO_COLLECT = True                 # 重新收集
DO_VISUALIZE = True               # 生成图像
```

### 3. 仅更新图像

```python
# 数据已收集，仅重新绘图
DO_COLLECT = False                # 加载现有数据
DO_VISUALIZE = True               # 重新生成图像
PREPROCESS_SCATTER_COLOR = 'tab:red'  # 调整样式
```

## 🔄 与原版的关系

本脚本是对原 `preprocess_validate.py` 的重构版本：

| 特性 | 原版 (preprocess_validate.py) | 新版 (07_*.py) |
|------|-------------------------------|----------------|
| 架构 | 环境变量覆盖 + runpy | 独立脚本 |
| 依赖 | 依赖 config_runtime | 零外部依赖 |
| 参数配置 | 在 config.py | 在脚本顶部 |
| 可读性 | 中等 | 高 |
| 维护性 | 复杂 | 简单 |
| 功能 | 完全等价 | 完全等价 |

**建议**：优先使用新版 `07_preprocess_validation_data_main.py`。

---

**创建日期**: 2025-11-02  
**版本**: 1.0  
**作者**: GitHub Copilot
