# 流程10：VTU 3D损伤动画渲染

## 功能概述

流程10将损伤识别结果映射到船舱3D模型上，生成直观的3D损伤可疑度动画。支持三种验证数据类型（crack/corrosion/health），使用PyVista渲染高质量动画。

## 主要特性

✅ **批量处理三种数据类型**: crack（裂纹）、corrosion（腐蚀）、health（健康）  
✅ **自动阈值计算**: 支持三种方法（quantile_abs/kstd_abs/mean_kstd）  
✅ **滑动窗口平滑**: 使用滑动窗口平均超阈值倍数，减少波动  
✅ **直接单元映射**: VTU_Index = Abaqus_ID - 1，高效准确  
✅ **灵活的相机视角**: 通过交互式查看器调整并保存相机位置  
✅ **多种输出格式**: 支持GIF和MP4（需要FFmpeg）  
✅ **完整的数据记录**: 每个动画都有对应的CSV数据文件  
✅ **自包含设计**: 无外部依赖，所有代码内嵌  

## 依赖关系

### 前置流程
1. **流程02**: INP转VTU（生成VTU模型和测点映射）
2. **流程04**: 模型训练（提供自编码器模型）
3. **流程07**: 验证数据预处理（提供三种类型验证数据）

### 输入文件
```
02_inp_to_vtu_output/
├── whole_from_inp.vtu          # VTU模型文件
└── measures_ID_auto.csv        # 测点ID映射

04_train_model_output/
└── autoencoder.pth             # 训练好的模型

07_preprocess_validation_data_output/
├── crack/preprocessed_data_raw.npz
├── corrosion/preprocessed_data_raw.npz
└── health/preprocessed_data_raw.npz

script/
└── camera_position.json        # 相机位置配置（由helper工具生成）
```

### 输出文件
```
10_render_vtu_animation_output/
├── dimension_thresholds.csv    # 每个维度的阈值
├── crack/
│   ├── damage_suspicion_animation.gif  # 3D动画
│   └── damage_suspicion_timeline.csv   # 可疑度时间序列数据
├── corrosion/
│   ├── damage_suspicion_animation.gif
│   └── damage_suspicion_timeline.csv
└── health/
    ├── damage_suspicion_animation.gif
    └── damage_suspicion_timeline.csv
```

## 使用方法

### 方法1：处理所有类型（默认）

```bash
python 10_render_vtu_animation_main.py
```

处理 crack、corrosion、health 三种类型。

### 方法2：处理指定类型

```bash
# 只处理crack类型
python 10_render_vtu_animation_main.py --type crack

# 处理多个类型
python 10_render_vtu_animation_main.py --type crack corrosion

# 明确指定处理所有类型
python 10_render_vtu_animation_main.py --type all
```

### 方法3：使用交互式查看器调整相机（首次运行推荐）

```bash
# 第一步：启动交互式查看器
python 10_interactive_vtu_viewer_helper.py --data crack

# 在交互窗口中：
# - 使用鼠标旋转、缩放、平移模型
# - 调整到满意的视角
# - 关闭窗口（相机位置自动保存到 camera_position.json）

# 第二步：使用保存的相机位置渲染动画
python 10_render_vtu_animation_main.py
```

## 参数配置

所有参数都在脚本顶部的"参数配置区"，可根据需要修改：

### 模型架构参数
```python
AE_ENCODER_DIMS = [768, 384, 192]  # 编码器各隐藏层维度
AE_LATENT_DIM = 192                # 潜在空间维度
AE_DECODER_DIMS = [192, 384, 768]  # 解码器各隐藏层维度
AE_DROPOUT = 0.0                   # Dropout概率
AE_ACTIVATION = "relu"             # 激活函数
```

⚠️ **注意**: 必须与流程04的参数完全一致！

### 阈值计算参数
```python
THRESHOLD_METHOD = "quantile_abs"  # 阈值方法
THRESHOLD_QUANTILE = 0.95          # 分位数（0.90 ~ 0.99）
THRESHOLD_K_SIGMA = 3.0            # k倍标准差（1.0 ~ 5.0）
```

📌 **建议**: 与流程08保持一致以确保结果可比性。

### 损伤可疑度计算参数
```python
SUSPICION_WINDOW_SIZE = 10         # 滑动窗口大小（样本数）
DAMAGE_SCALE_FACTOR = 10.0         # 损伤缩放因子
```

💡 **说明**: 
- `SUSPICION_WINDOW_SIZE`: 越大越平滑，但响应越慢
- `DAMAGE_SCALE_FACTOR`: 控制颜色映射的灵敏度

### 数据处理参数
```python
MAX_SAMPLES = -1                   # 最大样本数（-1=全部）
PROCESS_DATA_TYPES = ["crack", "corrosion", "health"]
```

### 3D渲染参数
```python
RENDER_FPS = 10                    # 帧率（每秒帧数）
RENDER_FORMAT = "gif"              # 保存格式：'mp4'或'gif'
RENDER_DPI = 100                   # 渲染分辨率
RENDER_WINDOW_WIDTH = 1920         # 窗口宽度
RENDER_WINDOW_HEIGHT = 1080        # 窗口高度
RENDER_BACKGROUND = "white"        # 背景颜色
RENDER_CMAP = "coolwarm"           # 颜色映射
RENDER_CLIM_MIN = 0                # 颜色范围最小值
RENDER_CLIM_MAX = 100              # 颜色范围最大值
RENDER_OPACITY = 1.0               # 模型不透明度
RENDER_SHOW_EDGES = False          # 是否显示网格边缘
RENDER_SAVE_VTU_FRAMES = False     # 是否保存每帧VTU文件
```

🎨 **颜色映射说明**:
- `coolwarm`: 蓝色（健康）→ 白色 → 红色（损伤）
- `RdYlGn_r`: 绿色（健康）→ 黄色 → 红色（损伤）
- `Reds`: 白色 → 红色（强调损伤区域）

## 交互式查看器控制

### 鼠标操作
- **左键拖拽**: 旋转模型
- **中键拖拽**: 平移视图
- **滚轮**: 缩放
- **右键拖拽**: 缩放（备选）

### 键盘快捷键
- **Space**: 播放/暂停
- **右箭头**: 下一帧
- **左箭头**: 上一帧
- **+**: 加速播放
- **-**: 减速播放
- **R**: 重置相机视角
- **C**: 打印当前相机位置（控制台）
- **Q**: 退出查看器

## 技术细节

### 损伤可疑度计算流程

1. **阈值计算**: 使用训练数据计算每个维度的阈值
   ```
   残差 = |预测值 - 真实值|
   阈值 = quantile(残差, 0.95)  # 或其他方法
   ```

2. **超阈值倍数**: 对每个验证样本计算超阈值倍数
   ```
   超阈值倍数 = 残差 / 阈值  (仅当残差 > 阈值时)
   ```

3. **滑动窗口平滑**: 使用窗口平均减少波动
   ```
   平均超阈值倍数 = mean(最近N个样本的超阈值倍数)
   ```

4. **损伤可疑度映射**: 映射到0-100范围
   ```
   损伤可疑度 = clip(平均超阈值倍数 × 缩放因子, 0, 100)
   ```

### 单元映射机制

使用**直接映射**方法：
```python
VTU_Cell_Index = Abaqus_Element_ID - 1
```

优点：
- ✅ 简单高效（O(1)向量化操作）
- ✅ 不需要额外的映射文件
- ✅ 适用于从INP直接转换的VTU文件

### 3D渲染流程

1. **准备阶段**: 加载VTU模型和损伤可疑度数据
2. **映射阶段**: 将252个测点映射到17万+单元
3. **渲染阶段**: 逐帧渲染并写入视频/GIF
4. **保存阶段**: 输出动画文件和CSV数据

## 性能优化建议

### 减少渲染时间
```python
MAX_SAMPLES = 50              # 限制帧数（如50帧）
RENDER_WINDOW_WIDTH = 1280    # 降低分辨率
RENDER_WINDOW_HEIGHT = 720
RENDER_FPS = 5                # 降低帧率
```

### 提高动画质量
```python
MAX_SAMPLES = -1              # 使用全部样本
RENDER_WINDOW_WIDTH = 2560    # 提高分辨率
RENDER_WINDOW_HEIGHT = 1440
RENDER_DPI = 150              # 提高DPI
RENDER_FORMAT = "mp4"         # 使用MP4（需要FFmpeg）
```

### 内存优化
- 如果内存不足，设置 `MAX_SAMPLES` 为较小值
- 不要启用 `RENDER_SAVE_VTU_FRAMES`（会占用大量磁盘空间）

## 典型运行时间

| 样本数 | 分辨率 | 格式 | 运行时间 |
|--------|--------|------|----------|
| 20     | 1920x1080 | GIF  | ~1分钟  |
| 50     | 1920x1080 | GIF  | ~2分钟  |
| 100    | 1920x1080 | GIF  | ~4分钟  |
| 100    | 1920x1080 | MP4  | ~3分钟  |

⚠️ **注意**: 时间取决于硬件配置（CPU/GPU、内存）

## 输出文件说明

### dimension_thresholds.csv
记录每个维度的阈值，格式：
```csv
dimension,tau
0,0.123456
1,0.234567
...
```

### damage_suspicion_timeline.csv
记录每个时间步每个测点的损伤可疑度，格式：
```csv
Frame,Sample_Index,Dim_000,Dim_001,...,Dim_251
0,0,0.000000,0.000000,...,5.234567
1,1,0.123456,0.234567,...,6.345678
...
```

### damage_suspicion_animation.gif / .mp4
3D动画文件，展示损伤可疑度随时间变化：
- 蓝色区域：健康（损伤可疑度接近0）
- 红色区域：损伤（损伤可疑度接近100）
- 白色区域：中等可疑度

## 常见问题

### Q1: "PyVista not available"
**解决**: 安装PyVista
```bash
pip install pyvista
```

### Q2: "Camera position file not found"
**解决**: 先运行交互式查看器生成相机位置
```bash
python 10_interactive_vtu_viewer_helper.py --data crack
# 调整视角后关闭窗口
```

### Q3: FFmpeg not available (MP4格式失败)
**解决**: 
- 方法1: 安装FFmpeg并添加到系统PATH
- 方法2: 使用GIF格式（设置 `RENDER_FORMAT = "gif"`）

### Q4: 动画渲染速度慢
**解决**: 
- 减少样本数: `MAX_SAMPLES = 20`
- 降低分辨率: `RENDER_WINDOW_WIDTH = 1280`
- 降低帧率: `RENDER_FPS = 5`

### Q5: 内存不足
**解决**: 
- 限制样本数: `MAX_SAMPLES = 50`
- 关闭VTU帧保存: `RENDER_SAVE_VTU_FRAMES = False`

### Q6: 颜色映射不直观
**解决**: 尝试不同的颜色映射
```python
RENDER_CMAP = "RdYlGn_r"  # 绿→黄→红（交通灯）
RENDER_CMAP = "Reds"      # 白→红（强调损伤）
```

### Q7: 相机视角不理想
**解决**: 重新运行交互式查看器
```bash
python 10_interactive_vtu_viewer_helper.py --data crack
# 使用鼠标调整到理想视角
# 关闭窗口保存新的相机位置
```

## 最佳实践

### 1. 首次运行流程
```bash
# 步骤1: 使用交互式查看器调整相机
python 10_interactive_vtu_viewer_helper.py --data crack

# 步骤2: 测试渲染（小样本）
# 修改脚本参数: MAX_SAMPLES = 10
python 10_render_vtu_animation_main.py --type crack

# 步骤3: 完整渲染（全部样本和类型）
# 修改脚本参数: MAX_SAMPLES = -1
python 10_render_vtu_animation_main.py
```

### 2. 参数调优顺序
1. 先用小样本（10-20帧）快速迭代
2. 调整颜色映射和缩放因子
3. 确认效果后使用全部样本
4. 最后调整分辨率和格式

### 3. 批量处理策略
```bash
# 分别处理以便更好控制
python 10_render_vtu_animation_main.py --type crack
python 10_render_vtu_animation_main.py --type corrosion
python 10_render_vtu_animation_main.py --type health
```

## 扩展功能

### 保存VTU帧文件序列（用于ParaView）
```python
RENDER_SAVE_VTU_FRAMES = True
```

生成 `frame_XXXX.vtu` 文件序列，可在ParaView中：
1. 打开任一帧文件
2. 使用时间轴控制播放
3. 应用各种滤波器和高级可视化

### 自定义颜色映射
```python
# 方案1: 使用预定义颜色映射
RENDER_CMAP = "viridis"     # 蓝→绿→黄
RENDER_CMAP = "plasma"      # 紫→粉→黄
RENDER_CMAP = "inferno"     # 黑→红→黄

# 方案2: 自定义颜色范围
RENDER_CLIM_MIN = 10        # 忽略低于10的值
RENDER_CLIM_MAX = 80        # 饱和高于80的值
```

### 多视角渲染
通过修改 `camera_position.json` 生成不同视角的动画：
```bash
# 视角1: 正面视图
python 10_interactive_vtu_viewer_helper.py --data crack
# 调整到正面视图，关闭窗口保存
cp camera_position.json camera_position_front.json

# 视角2: 侧面视图
# 调整到侧面视图，关闭窗口保存
cp camera_position.json camera_position_side.json

# 分别渲染
cp camera_position_front.json camera_position.json
python 10_render_vtu_animation_main.py --type crack

cp camera_position_side.json camera_position.json
python 10_render_vtu_animation_main.py --type crack
```

## 与流程09的区别

| 特性 | 流程09（时历动画） | 流程10（VTU动画） |
|------|-------------------|-------------------|
| 可视化类型 | 2D热力图 (4×63网格) | 3D模型渲染 |
| 输出格式 | PNG静态图 + MP4动画 | GIF/MP4 3D动画 |
| 数据映射 | 直接显示252个测点 | 映射到17万+单元 |
| 相机视角 | 固定俯视图 | 可自定义3D视角 |
| 交互性 | 无 | 支持交互式查看器 |
| 渲染时间 | 快（~2分钟） | 较慢（~4分钟） |
| 适用场景 | 快速损伤趋势分析 | 详细3D损伤可视化 |

💡 **建议**: 两个流程互补使用，流程09用于快速分析，流程10用于展示和报告。

## 依赖库

- **numpy**: 数组计算
- **pandas**: 数据处理和CSV读写
- **torch**: 模型推理
- **pyvista**: VTU文件读取和3D渲染
- **json**: 相机位置配置读写

所有依赖已在项目环境中安装完成。

## 输出示例

### 动画展示
![VTU Animation Example](../output/figures/vtu_render/damage_suspicion_animation.gif)

### 数据记录
```
10_render_vtu_animation_output/
├── dimension_thresholds.csv       # 252行×2列
├── crack/
│   ├── damage_suspicion_animation.gif  # 约10-50MB
│   └── damage_suspicion_timeline.csv   # T行×254列
├── corrosion/
│   ├── damage_suspicion_animation.gif
│   └── damage_suspicion_timeline.csv
└── health/
    ├── damage_suspicion_animation.gif
    └── damage_suspicion_timeline.csv
```

## 更新日志

### 2025-11-02
- ✅ 完成流程10重构
- ✅ 实现三种数据类型批量处理
- ✅ 创建交互式查看器工具
- ✅ 废弃config依赖，参数集中在脚本顶部
- ✅ 使用直接单元映射（VTU_Index = Abaqus_ID - 1）
- ✅ 支持相机位置保存和加载
- ✅ 完善的错误处理和进度反馈

## 下一步

流程10完成后，可以：
1. ✅ 生成三种类型的3D损伤动画
2. ✅ 使用动画进行损伤可视化展示
3. ✅ 结合流程09的2D热力图进行综合分析
4. 📝 准备论文的可视化素材
5. 📊 制作项目演示视频

---

**重构负责人**: GitHub Copilot  
**完成日期**: 2025-11-02  
**版本**: 1.0  
**状态**: ✅ 已完成并测试
