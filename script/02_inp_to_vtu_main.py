"""
流程02：INP转VTU并提取元素集主脚本
功能：
    1. 将 Abaqus INP 文件转换为 VTU 格式
    2. 自动提取 INP 文件中的元素集定义 (*Elset)
    3. 生成测点列表文件 measures_ID_auto.csv（供后续流程使用）
    4. 生成元素集汇总报告

重要：此步骤必须在流程03（预处理）之前执行，因为流程03依赖 measures_ID_auto.csv

输入：element_groups.inp（Abaqus INP 文件，包含元素集定义）
输出：
    - 02_inp_to_vtu_output/whole_from_inp.vtu（VTU网格文件）
    - 02_inp_to_vtu_output/measures_ID_auto.csv（测点ID列表，后续流程依赖）
    - 02_inp_to_vtu_output/element_sets_summary.txt（元素集汇总报告）
"""

import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import meshio
except ImportError:
    print("错误: 未找到 meshio 库。请安装：")
    print("  pip install meshio")
    sys.exit(1)

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 1. 路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '02_inp_to_vtu_output')

# --- 2. 输入文件 ---
INP_FILE_NAME = "element_groups.inp"                        # INP文件名（位于script目录）
INP_FILE_PATH = os.path.join(os.path.dirname(__file__), INP_FILE_NAME)

# --- 3. 输出文件 ---
OUTPUT_VTU_NAME = "whole_from_inp.vtu"                      # VTU网格文件
OUTPUT_MEASURES_CSV = "measures_ID_auto.csv"                # 测点ID列表（重要：后续流程依赖）
OUTPUT_ELSETS_SUMMARY = "element_sets_summary.txt"          # 元素集汇总报告

# --- 4. 元素集配置 ---
TARGET_ELSET_NAME = 'all_measures'                          # 要提取的元素集名称
VERBOSE = True                                              # 是否显示详细信息

# ========================================
# 元素集解析函数
# ========================================

def parse_elsets_from_inp(inp_file_path):
    """
    从 INP 文件中提取所有元素集定义
    
    Args:
        inp_file_path: INP 文件路径
    
    Returns:
        dict: {elset_name: [element_ids...]}
    """
    print(f"\n[元素集解析] 从INP文件提取元素集...")
    
    element_sets = {}
    current_elset_name = None
    current_elset_data = []
    is_generate = False
    
    with open(inp_file_path, 'r', encoding='latin1') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('**'):
                continue
            
            # 检测 *Elset 关键字
            if line.upper().startswith('*ELSET'):
                # 保存之前的元素集
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if VERBOSE:
                        print(f"  ✓ {current_elset_name}: {len(current_elset_data)} 个元素")
                
                # 重置
                current_elset_data = []
                is_generate = False
                
                # 解析元素集名称
                # 格式: *Elset, elset=all_measures, instance=Part-1-1
                match = re.search(r'elset\s*=\s*([^,\s]+)', line, re.IGNORECASE)
                if match:
                    current_elset_name = match.group(1).strip()
                    
                    # 检查是否为 generate 类型
                    if 'generate' in line.lower():
                        is_generate = True
                else:
                    current_elset_name = None
                
                continue
            
            # 遇到其他关键字则结束当前元素集
            if line.startswith('*') and not line.startswith('**'):
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if VERBOSE:
                        print(f"  ✓ {current_elset_name}: {len(current_elset_data)} 个元素")
                
                current_elset_name = None
                current_elset_data = []
                is_generate = False
                continue
            
            # 解析元素集数据
            if current_elset_name:
                if is_generate:
                    # Generate 格式: start, end, increment
                    # 例如: 128754, 128757, 1
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        try:
                            start = int(parts[0])
                            end = int(parts[1])
                            increment = int(parts[2]) if len(parts) >= 3 else 1
                            
                            generated_ids = list(range(start, end + 1, increment))
                            current_elset_data.extend(generated_ids)
                        except ValueError:
                            pass
                else:
                    # 逗号分隔的单元ID列表
                    try:
                        ids = [int(x.strip()) for x in line.split(',') if x.strip()]
                        current_elset_data.extend(ids)
                    except ValueError:
                        pass
        
        # 保存最后一个元素集
        if current_elset_name and current_elset_data:
            element_sets[current_elset_name] = current_elset_data
            if VERBOSE:
                print(f"  ✓ {current_elset_name}: {len(current_elset_data)} 个元素")
    
    print(f"\n[元素集解析] 共找到 {len(element_sets)} 个元素集")
    
    return element_sets


def save_elsets_summary(element_sets, output_path):
    """
    保存元素集汇总报告
    
    Args:
        element_sets: 元素集字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("元素集汇总报告\n")
        f.write("=" * 70 + "\n\n")
        
        for name, ids in sorted(element_sets.items()):
            f.write(f"[{name}]\n")
            f.write(f"  - 元素数量: {len(ids)}\n")
            f.write(f"  - ID范围: {min(ids)} 到 {max(ids)}\n")
            f.write(f"  - 前10个ID: {ids[:10]}\n")
            f.write("\n")
    
    print(f"[汇总报告] 已保存到: {output_path}")


def generate_measures_csv(element_sets, output_csv_path, elset_name='all_measures'):
    """
    从元素集生成测点CSV文件
    
    Args:
        element_sets: 元素集字典
        output_csv_path: 输出CSV路径
        elset_name: 元素集名称 (默认 'all_measures')
    
    Returns:
        list: 测点ID列表
    """
    print(f"\n[测点生成] 从元素集 '{elset_name}' 生成测点CSV...")
    
    if elset_name not in element_sets:
        print(f"[警告] 未找到元素集 '{elset_name}'！")
        print(f"[信息] 可用的元素集: {list(element_sets.keys())}")
        return None
    
    measure_ids = sorted(element_sets[elset_name])
    
    # 创建 DataFrame (与原 measures_ID.csv 格式一致)
    df = pd.DataFrame({
        'all_measures': measure_ids
    })
    
    # 保存到CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"[成功] 测点CSV已保存: {output_csv_path}")
    print(f"  - 测点总数: {len(measure_ids)}")
    print(f"  - ID范围: {min(measure_ids)} 到 {max(measure_ids)}")
    print(f"  - 前10个ID: {measure_ids[:10]}")
    
    return measure_ids


# ========================================
# 网格解析函数
# ========================================

def parse_inp_manually(inp_file_path):
    """
    手动解析 Abaqus INP 文件的几何部分
    
    Args:
        inp_file_path: INP 文件路径
    
    Returns:
        mesh: meshio.Mesh 对象
    """
    print(f"\n[网格解析] 开始解析INP文件...")
    
    nodes = {}
    elements = {}
    element_types = {}
    
    current_section = None
    current_element_type = None
    in_part = False
    in_assembly = False
    
    with open(inp_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('**'):
                continue
            
            if line.startswith('*'):
                line_upper = line.upper()
                
                if line_upper.startswith('*PART'):
                    in_part = True
                    in_assembly = False
                    continue
                
                elif line_upper.startswith('*END PART'):
                    in_part = False
                    continue
                
                elif line_upper.startswith('*ASSEMBLY'):
                    in_assembly = True
                    in_part = False
                    if VERBOSE:
                        print(f"  - 进入Assembly部分（行 {line_num}），停止几何解析")
                    break
                
                if not in_part:
                    continue
                
                if line_upper.startswith('*NODE'):
                    current_section = 'NODE'
                    current_element_type = None
                
                elif line_upper.startswith('*ELEMENT'):
                    current_section = 'ELEMENT'
                    parts = line.split(',')
                    for part in parts:
                        if 'TYPE=' in part.upper():
                            current_element_type = part.split('=')[1].strip()
                            break
                
                else:
                    if current_section in ['NODE', 'ELEMENT']:
                        current_section = None
                        current_element_type = None
                
                continue
            
            if current_section == 'NODE':
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0].strip())
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[node_id] = [x, y, z]
                    except ValueError:
                        pass
            
            elif current_section == 'ELEMENT' and current_element_type:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        element_id = int(parts[0].strip())
                        node_ids = [int(p.strip()) for p in parts[1:]]
                        elements[element_id] = node_ids
                        element_types[element_id] = current_element_type
                    except ValueError:
                        pass
    
    print(f"[网格解析] 解析完成")
    print(f"  - 节点数: {len(nodes)}, 元素数: {len(elements)}")
    
    if len(nodes) == 0 or len(elements) == 0:
        raise ValueError("未找到节点或元素")
    
    # 转换为 meshio 格式
    sorted_node_ids = sorted(nodes.keys())
    node_id_to_index = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)
    
    elements_by_type = {}
    for elem_id, node_ids in elements.items():
        elem_type = element_types[elem_id]
        if elem_type not in elements_by_type:
            elements_by_type[elem_type] = []
        
        node_indices = [node_id_to_index[nid] for nid in node_ids]
        elements_by_type[elem_type].append((elem_id, node_indices))
    
    for elem_type in elements_by_type:
        elements_by_type[elem_type].sort(key=lambda x: x[0])
    
    # Abaqus到VTK类型映射
    abaqus_to_vtk = {
        'C3D8': 'hexahedron', 'C3D8R': 'hexahedron',
        'C3D6': 'wedge', 'C3D4': 'tetra',
        'S4': 'quad', 'S4R': 'quad', 'S3': 'triangle',
    }
    
    cells = []
    for elem_type, elem_list in elements_by_type.items():
        vtk_type = abaqus_to_vtk.get(elem_type, elem_type.lower())
        connectivity = np.array([elem[1] for elem in elem_list], dtype=int)
        cells.append((vtk_type, connectivity))
        print(f"  - {elem_type} -> {vtk_type}: {len(elem_list)} 个元素")
    
    mesh = meshio.Mesh(points=points, cells=cells)
    return mesh


# ========================================
# 主程序
# ========================================

print("\n" + "=" * 70)
print("流程02：INP转VTU并自动提取元素集")
print("=" * 70)

try:
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置输出文件路径
    vtu_file = os.path.join(OUTPUT_DIR, OUTPUT_VTU_NAME)
    measures_csv = os.path.join(OUTPUT_DIR, OUTPUT_MEASURES_CSV)
    elsets_summary = os.path.join(OUTPUT_DIR, OUTPUT_ELSETS_SUMMARY)
    
    # 检查输入文件
    if not os.path.exists(INP_FILE_PATH):
        raise FileNotFoundError(f"INP文件不存在: {INP_FILE_PATH}")
    
    print(f"\n输入文件: {INP_FILE_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 步骤 1: 解析元素集
    print("\n" + "=" * 70)
    print("步骤 1/5: 解析元素集")
    print("=" * 70)
    element_sets = parse_elsets_from_inp(INP_FILE_PATH)
    
    # 步骤 2: 保存元素集汇总
    print("\n" + "=" * 70)
    print("步骤 2/5: 保存元素集汇总")
    print("=" * 70)
    save_elsets_summary(element_sets, elsets_summary)
    
    # 步骤 3: 生成测点CSV
    print("\n" + "=" * 70)
    print("步骤 3/5: 生成测点CSV")
    print("=" * 70)
    measure_ids = generate_measures_csv(element_sets, measures_csv, TARGET_ELSET_NAME)
    
    # 步骤 4: 解析网格
    print("\n" + "=" * 70)
    print("步骤 4/5: 解析网格")
    print("=" * 70)
    mesh = parse_inp_manually(INP_FILE_PATH)
    
    # 步骤 5: 保存VTU
    print("\n" + "=" * 70)
    print("步骤 5/5: 保存VTU文件")
    print("=" * 70)
    print(f"[转换] 保存VTU文件: {vtu_file}")
    mesh.write(vtu_file, file_format="vtu")
    print(f"[成功] VTU文件已保存")
    
    # 总结
    print("\n" + "=" * 70)
    print("流程02完成！")
    print("=" * 70)
    print(f"\n输出文件：")
    print(f"  1. VTU文件:        {os.path.abspath(vtu_file)}")
    print(f"  2. 测点CSV:        {os.path.abspath(measures_csv)}")
    print(f"     ↳ 重要：此文件将被后续流程使用")
    print(f"  3. 元素集汇总:      {os.path.abspath(elsets_summary)}")
    
    if measure_ids:
        print(f"\n自动提取的测点信息：")
        print(f"  - 测点总数: {len(measure_ids)}")
        print(f"  - 前10个ID: {measure_ids[:10]}")
        print(f"  - 后10个ID: {measure_ids[-10:]}")
    
    print(f"\n关键优势：")
    print(f"  ✓ 无需手动准备 measures_ID.csv！")
    print(f"  ✓ 所有数据直接从INP文件提取")
    print(f"  ✓ 后续流程请使用 '{OUTPUT_MEASURES_CSV}'")
    print()

except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
