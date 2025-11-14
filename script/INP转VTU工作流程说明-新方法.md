# INP 到 VTU 直接转换工具使用说明

## 概述

本工具提供了一种全新的、更简洁的方式将 Abaqus INP 文件直接转换为 VTU 格式，**无需通过 ODB 中转**，并且 **Abaqus 单元 ID 自动与 VTU 行索引一一对应**，不再需要复杂的映射文件。

## 优势对比

### 旧方法（使用 ODB 中转）
```
INP → Abaqus → ODB → VTK → VTU
                ↓
           需要提取单元中心坐标
                ↓
           需要构建映射关系（cell_centers.txt + cell_matches.csv）
```

### 新方法（直接转换）
```
INP → VTU（直接）
    ↓
单元 ID 自动对应 VTU 索引
```

**优势：**
- ✅ 不需要 ODB 文件
- ✅ 不需要 cell_centers.txt
- ✅ 不需要 cell_matches.csv
- ✅ 单元 ID 映射简单：`VTU_Index = Abaqus_ID - 1`
- ✅ 转换速度快
- ✅ 流程简化

## 文件说明

### 1. `inp_to_vtu_direct.py`
**功能：** 将 Abaqus INP 文件直接转换为 VTU 格式

**输入：**
- `element_groups.inp` - Abaqus INP 文件

**输出：**
- `whole_from_inp.vtu` - VTU 格式的网格文件（176,151 个单元）
- `inp_vtu_cell_info.csv` - 单元信息统计（包含单元 ID、中心坐标等）

**使用方法：**
```bash
cd c:\data\AE-main\script
python inp_to_vtu_direct.py
```

**运行时间：** 约 10-30 秒（取决于 INP 文件大小）

### 2. `verify_inp_vtu_mapping.py`
**功能：** 验证 INP 到 VTU 的映射关系是否正确

**验证内容：**
- ✓ 所有 252 个测点 ID 是否都存在于 VTU 文件中
- ✓ 映射公式是否正确
- ✓ VTU 文件是否能被 PyVista 正确读取

**使用方法：**
```bash
cd c:\data\AE-main\script
python verify_inp_vtu_mapping.py
```

## 核心映射关系

### 公式
```python
VTU_Cell_Index = Abaqus_Element_ID - 1
```

### 示例
| Abaqus Element ID | VTU Cell Index |
|-------------------|----------------|
| 225               | 224            |
| 418               | 417            |
| 839               | 838            |
| 173475            | 173474         |

## 与现有脚本的集成

### 方法 1：直接修改现有脚本

在 `view_damage_animation_interactive.py` 和 `time_history_animation.py` 中：

**旧代码（需要映射文件）：**
```python
# 加载 VTU 模型
vtu_path = os.path.join(project_root, "script", "whole", "Step-1_1.vtu")
base_mesh = load_vtu_model(vtu_path)

# 加载映射关系
mapping_path = os.path.join(project_root, "script", "cell_matches.csv")
cell_mapping = load_cell_mapping(mapping_path)

# 映射可疑度到 VTU
vtu_suspicion = map_suspicion_to_vtu(
    suspicion_timeline,
    measure_ids,
    cell_mapping,  # 需要复杂的映射字典
    base_mesh.n_cells
)
```

**新代码（直接映射）：**
```python
# 加载 VTU 模型（使用新的 VTU 文件）
vtu_path = os.path.join(project_root, "script", "whole_from_inp.vtu")
base_mesh = pv.read(vtu_path)

# 加载测点 ID
measures_path = os.path.join(project_root, "script", "measures_ID.csv")
measures_df = pd.read_csv(measures_path)
measure_ids = measures_df['all_measures'].dropna().astype(int).values  # [225, 418, ...]

# 直接映射（无需映射文件）
def map_suspicion_to_vtu_direct(suspicion_timeline, measure_ids, total_cells):
    """
    直接将测点可疑度映射到 VTU 单元
    
    Args:
        suspicion_timeline: [T, D] 时间序列可疑度
        measure_ids: [D] Abaqus 单元 ID 数组
        total_cells: VTU 总单元数
    
    Returns:
        vtu_suspicion_timeline: [T, total_cells] VTU 格式的可疑度
    """
    T, D = suspicion_timeline.shape
    vtu_suspicion_timeline = np.zeros((T, total_cells), dtype=np.float32)
    
    # 转换：VTU 索引 = Abaqus ID - 1
    vtu_indices = measure_ids - 1
    
    # 直接赋值（无需循环查找映射）
    vtu_suspicion_timeline[:, vtu_indices] = suspicion_timeline
    
    return vtu_suspicion_timeline

# 使用
vtu_suspicion = map_suspicion_to_vtu_direct(
    suspicion_timeline,
    measure_ids,
    base_mesh.n_cells
)
```

### 方法 2：创建新的简化脚本

你也可以基于新的 VTU 文件创建全新的、更简洁的可视化脚本。

## 完整示例代码

```python
"""
使用新 VTU 文件的简化版损伤可视化脚本
"""
import os
import numpy as np
import pandas as pd
import pyvista as pv

# 文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VTU_FILE = os.path.join(SCRIPT_DIR, "whole_from_inp.vtu")
MEASURES_CSV = os.path.join(SCRIPT_DIR, "measures_ID.csv")

# 1. 加载 VTU 模型
mesh = pv.read(VTU_FILE)
print(f"VTU loaded: {mesh.n_cells} cells")

# 2. 加载测点 ID
df = pd.read_csv(MEASURES_CSV)
measure_ids = df['all_measures'].dropna().astype(int).values
print(f"Measure points: {len(measure_ids)}")

# 3. 假设你已经有了损伤可疑度数据（来自模型预测）
# suspicion_values: [252] 数组，每个测点的可疑度
suspicion_values = np.random.rand(len(measure_ids)) * 100  # 示例数据

# 4. 映射到 VTU（超级简单！）
vtu_suspicion = np.zeros(mesh.n_cells)
vtu_indices = measure_ids - 1  # 关键映射：VTU索引 = Abaqus ID - 1
vtu_suspicion[vtu_indices] = suspicion_values

# 5. 添加到网格并可视化
mesh.cell_data['damage_suspicion'] = vtu_suspicion

# 6. 可视化
plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    scalars='damage_suspicion',
    cmap='coolwarm',
    clim=[0, 100],
    show_edges=False
)
plotter.show()
```

## 性能对比

| 方法 | 转换时间 | 映射复杂度 | 需要的文件 |
|------|----------|------------|------------|
| **旧方法** | ~5-10 分钟 | O(n×m) KDTree 搜索 | INP + ODB + VTU + cell_centers.txt + cell_matches.csv |
| **新方法** | ~10-30 秒 | O(1) 直接索引 | INP + VTU |

## 常见问题

### Q1: 为什么 VTU 索引是 Abaqus ID - 1？
**A:** 因为：
- Abaqus 单元 ID 从 1 开始编号
- VTU/数组索引从 0 开始编号
- meshio 在转换时保持了单元的定义顺序

### Q2: 所有测点都能正确映射吗？
**A:** 是的！验证脚本已确认所有 252 个测点都存在于 VTU 文件中。

### Q3: 这个方法适用于其他 INP 文件吗？
**A:** 适用于大多数标准 Abaqus INP 文件，但需要注意：
- INP 文件必须包含完整的节点和单元定义
- 支持常见单元类型（C3D8, S4R, S3 等）
- 如果 INP 文件格式特殊，可能需要调整解析器

### Q4: 原来的 VTU 文件还能用吗？
**A:** 可以，但新方法更简单。建议逐步迁移到新方法。

## 下一步

1. ✅ **已完成**：INP 转 VTU 脚本 (`inp_to_vtu_direct.py`)
2. ✅ **已完成**：映射验证脚本 (`verify_inp_vtu_mapping.py`)
3. 🔄 **可选**：修改现有可视化脚本以使用新 VTU 文件
4. 🔄 **可选**：创建简化版的可视化脚本

## 技术细节

### 支持的单元类型
- **实体单元**: C3D8, C3D8R, C3D6, C3D4, C3D10, C3D20
- **壳单元**: S4, S4R, S3, S8R
- **平面单元**: CPS4, CPS3, CPE4, CPE3

### 文件格式
- **输入**: Abaqus INP (ASCII)
- **输出**: VTU (VTK Unstructured Grid, XML format)

### 依赖库
```bash
pip install meshio numpy pandas pyvista
```

## 联系与反馈

如有问题或建议，请在项目中提出 issue。

---

**创建日期**: 2025-11-01  
**作者**: AI Assistant  
**版本**: 1.0
