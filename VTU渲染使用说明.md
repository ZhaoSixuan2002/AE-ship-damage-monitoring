# VTU 3D渲染动画 - 使用说明

## 功能概述

已成功实现基于PyVista的3D损伤可疑度动画渲染功能，可将损伤识别结果映射到船舱VTU模型上，生成直观的3D动画展示。

## 实现的功能

### 1. 核心脚本
- **script/render_vtu_animation.py**: 主渲染脚本，实现以下功能：
  - 加载训练好的自编码器模型和阈值
  - 读取验证数据（crack/corrosion/health）
  - 计算每个样本252个测点的损伤可疑度
  - 将可疑度映射到VTU模型的17万+单元
  - 生成3D动画（GIF或MP4格式）

### 2. 配置参数（config.py中）

```python
# 总控开关
ENABLE_VTU_RENDER = 1  # 是否执行VTU 3D渲染动画环节

# 数据参数
VTU_RENDER_DATA_TYPE = 'crack'              # 渲染数据类型: 'crack'|'corrosion'|'health'
VTU_RENDER_MAX_SAMPLES = 20                 # 最大样本数（帧数）
VTU_RENDER_SUSPICION_MODE = 'exceedance_ratio'  # 可疑度计算模式

# 动画参数
VTU_RENDER_FPS = 10                         # 帧率
VTU_RENDER_FORMAT = 'gif'                   # 格式: 'gif'|'mp4'
VTU_RENDER_SAVE_VTU_FRAMES = 0              # 是否保存每帧VTU文件

# 可视化参数
VTU_RENDER_CMAP = 'coolwarm'                # 颜色映射: 蓝(健康)→红(损伤)
VTU_RENDER_CLIM_MIN = 0.0                   # 颜色范围最小值
VTU_RENDER_CLIM_MAX = 3.0                   # 颜色范围最大值（超阈值倍数）
VTU_RENDER_CAMERA = 'iso'                   # 相机视角
VTU_RENDER_WINDOW_WIDTH = 1920              # 窗口宽度
VTU_RENDER_WINDOW_HEIGHT = 1080             # 窗口高度
VTU_RENDER_OPACITY = 0.9                    # 模型不透明度
VTU_RENDER_SHOW_EDGES = 0                   # 是否显示网格边缘
VTU_RENDER_BACKGROUND = 'white'             # 背景颜色
```

### 3. 输出文件

执行后会在 `output/figures/vtu_render/` 目录下生成：

- **damage_suspicion_animation.gif**: 3D动画文件
- **damage_suspicion_timeline.csv**: 每个时间步252个测点的损伤可疑度数据

## 使用方法

### 方法1: 通过主控脚本运行（推荐）

```powershell
# 在 script/config.py 中设置 ENABLE_VTU_RENDER = 1
python main.py
```

### 方法2: 单独运行渲染脚本

```powershell
cd script
python render_vtu_animation.py
```

## 技术细节

### 数据流程

1. **加载阈值**: 从 `output/logs/dimension_thresholds.csv` 读取每个维度的阈值
2. **加载验证数据**: 从 `output/preprocess_validate/{crack|corrosion|health}/preprocessed_data_raw.npz` 读取
3. **计算可疑度**: 
   - 对每个样本预测残差
   - 计算超阈值倍数作为损伤可疑度
   - 支持三种模式：超阈值倍数/二值化/归一化
4. **映射到VTU**: 通过 `script/cell_matches.csv` 将252个测点映射到VTU模型的对应单元
5. **渲染动画**: 使用PyVista逐帧渲染并保存为GIF或MP4

### 可疑度计算模式

- **exceedance_ratio** (默认): 超阈值倍数，范围[0, 10]，直观显示超出程度
- **binary**: 二值化，超阈值为1，否则为0
- **normalized**: 归一化到[0, 1]，使用sigmoid函数平滑过渡

### 颜色映射推荐

- **coolwarm**: 蓝(健康)→白→红(损伤)，适合双向对比
- **RdYlGn_r**: 绿(健康)→黄→红(损伤)，符合交通灯直觉
- **Reds**: 白→红，强调损伤区域
- **jet**: 彩虹色，最大动态范围

## 性能优化建议

由于VTU模型包含176151个单元，渲染速度较慢：

1. **减少帧数**: 设置 `VTU_RENDER_MAX_SAMPLES` 为较小值（如10-30）
2. **降低分辨率**: 减小 `VTU_RENDER_WINDOW_WIDTH/HEIGHT`
3. **使用MP4**: `VTU_RENDER_FORMAT = 'mp4'` 比GIF更高效
4. **关闭边缘**: `VTU_RENDER_SHOW_EDGES = 0` 加快渲染

## 查看动画

生成的GIF文件可在以下位置查看：
```
output/figures/vtu_render/damage_suspicion_animation.gif
```

可使用：
- 任何图像查看器（Windows照片查看器、IrfanView等）
- 浏览器（直接拖拽GIF文件到浏览器）
- ParaView（如果保存了VTU帧文件）

## 扩展功能

### 保存VTU帧文件序列

如需在ParaView中进行交互式查看：

1. 设置 `VTU_RENDER_SAVE_VTU_FRAMES = 1`
2. 运行脚本后会在 `output/figures/vtu_render/` 生成 `frame_XXXX.vtu` 文件序列
3. 在ParaView中打开任一帧文件，使用时间轴控制播放

### 切换数据类型

- 裂纹损伤: `VTU_RENDER_DATA_TYPE = 'crack'`
- 腐蚀损伤: `VTU_RENDER_DATA_TYPE = 'corrosion'`
- 健康结构: `VTU_RENDER_DATA_TYPE = 'health'`

## 依赖库

- **pyvista**: VTU文件读取和3D渲染
- **imageio**: GIF动画生成
- **torch**: 模型推理
- **numpy, pandas**: 数据处理

所有依赖已在项目环境中安装完成。

## 故障排查

### 问题1: "VTU file not found"
- 确保 `script/whole/Step-1_1.vtu` 文件存在
- 检查路径设置是否正确

### 问题2: "Threshold file not found"
- 需先运行损伤验证 (`ENABLE_DAMAGE_VALIDATION = 1`)
- 确保 `output/logs/dimension_thresholds.csv` 已生成

### 问题3: 渲染速度慢
- 减少 `VTU_RENDER_MAX_SAMPLES` 到10-20
- 降低窗口分辨率
- 使用MP4格式

### 问题4: "Validation data not found"
- 需先运行验证数据预处理 (`ENABLE_VALIDATE_PREPROCESSING = 1`)
- 确保对应类型的NPZ文件存在

## 完成状态

✅ 脚本实现完成  
✅ 配置参数集成  
✅ 主控流程集成  
✅ 测试运行成功  
✅ 动画生成验证  

当前配置下（20帧，17万单元），完整渲染时间约2-3分钟。
