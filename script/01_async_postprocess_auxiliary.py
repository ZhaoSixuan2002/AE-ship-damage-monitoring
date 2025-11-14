"""
流程01辅助脚本：异步后处理
功能：使用 Abaqus Python 执行，打开指定 ODB，导出 CSV/NPY，可选删除 ODB
设计用于与下一次求解并行执行

用法（由主脚本调用）：
    abaqus python 01_async_postprocess_auxiliary.py -- <odb_path> <output_folder> <iteration_num>

输出：
    <output_folder>/iteration.csv
    <output_folder>/iteration.npy
    <output_folder>/timings.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

# 仅使用 odbAccess，避免引入 CAE GUI 依赖
try:
    from odbAccess import openOdb
except Exception:
    try:
        from OdbAccess import openOdb  # type: ignore
    except Exception:
        raise

# ========================================
# 参数配置区（直接编写，无需外部配置文件）
# ========================================

# --- 后处理控制参数 ---
POSTPROCESS_DELETE_ODB = True                           # 后处理完成后是否删除ODB文件
ENABLE_TIMING_LOG = True                                # 是否记录阶段耗时

# --- 应力提取参数 ---
DISPLAY_ELEMENT_SET = 'MIDDLEWHOLE'                     # 目标元素集名称
STRESS_POSITION = 'INTEGRATION_POINT'                   # 应力提取位置：'INTEGRATION_POINT' 或 'CENTROID' 或 'ELEMENT_NODAL'
SECTION_FILTER = 'SNEG'                                 # 截面点过滤：'ALL' 或 'SNEG' 或 'SPOS' 或 'SMID'
AGGREGATION = 'mean'                                    # 单元聚合方法：'mean' 或 'max' 或 'median' 或 'p90'/'p95'/'p99'
EXPORT_SCOPE = 'ELEMENT'                                # 导出域：'ELEMENT' 或 'NODE'
NODE_AGGREGATION = 'mean'                               # 节点聚合方法：'none' 或 'mean' 或 'max' 或 'median'

# --- 输出格式参数 ---
EXPORT_CSV = True                                       # 是否导出CSV文件
EXPORT_BIN_FORMATS = ['npy']                            # 二进制导出格式列表：['npy', 'npz'] 等
CSV_SORT_LABELS = False                                 # CSV是否按标签排序
CSV_BUFFER_MB = 16                                      # CSV缓冲区大小(MB)
EXPORT_ELEMENT_CENTERS_ONCE = True                      # 首次迭代是否导出元素中心坐标

# ========================================
# 解析命令行参数
# ========================================
if len(sys.argv) < 4:
    print("用法: 01_async_postprocess_auxiliary.py -- <odb_path> <output_folder> <iteration_num>")
    sys.exit(2)

# 支持可能存在的 "--"
args = [a for a in sys.argv[1:] if a != "--"]
odb_path = os.path.abspath(args[-3])
output_folder = os.path.abspath(args[-2])
iteration_num = int(args[-1])

# ========================================
# 辅助函数
# ========================================

def _accept_section(v):
    """根据SECTION_FILTER判断是否接受该截面点"""
    if SECTION_FILTER.upper() == "ALL":
        return True
    try:
        sp = getattr(v, "sectionPoint", None)
        desc = ""
        if sp is not None:
            desc = getattr(sp, "description", "") or str(sp)
        desc = str(desc).upper()
        target = SECTION_FILTER.upper()
        if target == "SNEG":
            return ("SNEG" in desc) or ("-1.0" in desc)
        if target == "SPOS":
            return ("SPOS" in desc) or ("1.0" in desc)
        if target == "SMID":
            return ("SMID" in desc) or ("SAVG" in desc) or ("MID" in desc)
        return True
    except Exception:
        return True


def _aggregate(vals, method):
    """聚合函数"""
    if not vals:
        return None
    m = method.lower()
    if m == "max":
        return max(vals)
    if m == "median":
        try:
            import statistics
            return statistics.median(vals)
        except Exception:
            vs = sorted(vals)
            n = len(vs)
            return vs[n // 2] if n % 2 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])
    if m.startswith("p") and len(m) >= 2:
        try:
            p = float(m[1:])
            p = max(0.0, min(100.0, p))
        except Exception:
            p = 95.0
        vs = sorted(vals)
        if not vs:
            return None
        k = int(round((p / 100.0) * (len(vs) - 1)))
        return vs[k]
    # 默认 mean
    return sum(vals) / float(len(vals))


# ========================================
# 主处理流程
# ========================================

# 计时
T = {}
_t0_total = time.time()

# 安全检查
if not os.path.exists(odb_path):
    print(f"[post] ODB文件不存在: {odb_path}")
    sys.exit(1)
if not os.path.isdir(output_folder):
    try:
        os.makedirs(output_folder)
    except Exception:
        pass

# 打开 ODB（只读）
_t = time.time()
odb = openOdb(odb_path, readOnly=True)
T["open_odb"] = time.time() - _t

# 选择最后一帧
step_names = list(odb.steps.keys())
step = odb.steps[step_names[0]]
frame = step.frames[-1]

# 目标元素集
elset = None
try:
    elset = odb.rootAssembly.elementSets[DISPLAY_ELEMENT_SET]
except Exception:
    try:
        for inst in odb.rootAssembly.instances.values():
            if DISPLAY_ELEMENT_SET in inst.elementSets:
                elset = inst.elementSets[DISPLAY_ELEMENT_SET]
                break
    except Exception:
        elset = None

# 应力场（优先 Mises 标量）
stress = frame.fieldOutputs["S"]

# 确定提取位置
_scope = EXPORT_SCOPE.upper()
if _scope == "NODE":
    try:
        from abaqusConstants import ELEMENT_NODAL
        _pos = ELEMENT_NODAL
    except Exception:
        _pos = None
else:
    try:
        from abaqusConstants import CENTROID, INTEGRATION_POINT
        _map = {
            "INTEGRATION_POINT": INTEGRATION_POINT,
            "CENTROID": CENTROID,
            "ELEMENT_NODAL": None,
        }
        _pos = _map.get(STRESS_POSITION.upper(), None)
    except Exception:
        _pos = None

# 提取/标量变换
_t = time.time()
try:
    sub = (
        stress.getSubset(region=elset, position=_pos)
        if (_pos is not None and elset is not None)
        else (
            stress.getSubset(position=_pos)
            if _pos is not None
            else (stress.getSubset(region=elset) if elset is not None else stress)
        )
    )
except Exception:
    sub = stress

try:
    from abaqusConstants import MISES
    mises = sub.getScalarField(invariant=MISES)
except Exception:
    try:
        mises = sub.getScalarField(invariant="Mises")
    except Exception:
        mises = None
T["extract_field"] = time.time() - _t

# 收集 values
_t = time.time()
try:
    values_iter = list(mises.values) if mises is not None else []
except Exception:
    values_iter = []
if not values_iter:
    try:
        values_iter = list(sub.values)
    except Exception:
        values_iter = []
T["collect_values"] = time.time() - _t

csv_final_path = os.path.join(output_folder, "iteration.csv")

# 写出（增大缓冲、可选排序）
buf_size = max(0, CSV_BUFFER_MB) * 1024 * 1024
rows_out = None
colnames = None
_t_write = time.time()

if _scope == "NODE":
    # 节点域
    node_agg = NODE_AGGREGATION.lower()
    add_coords = (iteration_num == 1) and EXPORT_ELEMENT_CENTERS_ONCE

    if node_agg == "none":
        rows_out = []
        colnames = ["Node Label", "S-Mises"] + (["X", "Y", "Z"] if add_coords else [])
        if add_coords:
            node_coords = {}
            _t = time.time()
            try:
                for inst in odb.rootAssembly.instances.values():
                    for n in inst.nodes:
                        node_coords[int(n.label)] = tuple(n.coordinates)
            except Exception:
                node_coords = {}
            T["build_node_coords"] = time.time() - _t
        for v in values_iter:
            try:
                if not _accept_section(v):
                    continue
                sval = None
                try:
                    sval = float(v.data)
                except Exception:
                    try:
                        sval = float(v.mises)
                    except Exception:
                        sval = float(v.data.mises)
                if math.isnan(sval) or math.isinf(sval):
                    continue
                nlab = int(v.nodeLabel)
                if add_coords:
                    c = node_coords.get(nlab)
                    if c is not None:
                        rows_out.append([nlab, sval, c[0], c[1], c[2]])
                    else:
                        rows_out.append([nlab, sval, "", "", ""])
                else:
                    rows_out.append([nlab, sval])
            except Exception:
                continue
    else:
        node_to_vals = {}
        for v in values_iter:
            try:
                if not _accept_section(v):
                    continue
                sval = None
                try:
                    sval = float(v.data)
                except Exception:
                    try:
                        sval = float(v.mises)
                    except Exception:
                        sval = float(v.data.mises)
                if math.isnan(sval) or math.isinf(sval):
                    continue
                nlab = int(v.nodeLabel)
                node_to_vals.setdefault(nlab, []).append(sval)
            except Exception:
                continue
        order = sorted(node_to_vals.keys()) if CSV_SORT_LABELS else node_to_vals.keys()
        rows_out = []
        if add_coords:
            node_coords = {}
            _t = time.time()
            try:
                for inst in odb.rootAssembly.instances.values():
                    for n in inst.nodes:
                        node_coords[int(n.label)] = tuple(n.coordinates)
            except Exception:
                node_coords = {}
            T["build_node_coords"] = time.time() - _t
            colnames = ["Node Label", "S-Mises", "X", "Y", "Z"]
            for nlab in order:
                vals = node_to_vals[nlab]
                val = _aggregate(vals, node_agg)
                if val is None:
                    continue
                c = node_coords.get(nlab)
                if c is not None:
                    rows_out.append([nlab, val, c[0], c[1], c[2]])
                else:
                    rows_out.append([nlab, val, "", "", ""])
        else:
            colnames = ["Node Label", "S-Mises"]
            for nlab in order:
                vals = node_to_vals[nlab]
                val = _aggregate(vals, node_agg)
                if val is None:
                    continue
                rows_out.append([nlab, val])
else:
    # 单元域
    elem_to_vals = {}
    for v in values_iter:
        try:
            if not _accept_section(v):
                continue
            lab = int(v.elementLabel)
            sval = None
            try:
                sval = float(v.data)
            except Exception:
                try:
                    sval = float(v.mises)
                except Exception:
                    sval = float(v.data.mises)
            if math.isnan(sval) or math.isinf(sval):
                continue
            elem_to_vals.setdefault(lab, []).append(sval)
        except Exception:
            continue
    
    agg_method = AGGREGATION.lower()
    elem_to_mises = {}
    for lab, vals in elem_to_vals.items():
        val = _aggregate(vals, agg_method)
        if val is not None:
            elem_to_mises[lab] = val
    
    add_coords = (iteration_num == 1) and EXPORT_ELEMENT_CENTERS_ONCE
    order = list(sorted(elem_to_mises.keys()) if CSV_SORT_LABELS else elem_to_mises.keys())
    rows = []
    colnames = ["Element Label", "S-Mises"] + (["X", "Y", "Z"] if add_coords else [])
    
    if add_coords:
        # 构建元素-节点 与 节点-坐标
        elem_nodes, node_coords = {}, {}
        _t = time.time()
        allowed = None
        try:
            allowed = set(int(e.label) for e in (elset.elements if elset else []))
        except Exception:
            allowed = None
        try:
            for inst in odb.rootAssembly.instances.values():
                try:
                    for n in inst.nodes:
                        node_coords[int(n.label)] = tuple(n.coordinates)
                except Exception:
                    pass
                try:
                    for e in inst.elements:
                        elab = int(e.label)
                        if (allowed is not None) and (elab not in allowed):
                            continue
                        elem_nodes[elab] = list(e.connectivity)
                except Exception:
                    pass
        finally:
            T["build_elem_nodes"] = time.time() - _t
        
        for lab in order:
            val = elem_to_mises[lab]
            coords = None
            nlist = elem_nodes.get(lab)
            if nlist:
                xs = []
                ys = []
                zs = []
                for nlab in nlist:
                    c = node_coords.get(int(nlab))
                    if c is not None:
                        x, y, z = c
                        xs.append(float(x))
                        ys.append(float(y))
                        zs.append(float(z))
                if xs:
                    coords = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
            if coords is not None:
                rows.append([lab, val, coords[0], coords[1], coords[2]])
            else:
                rows.append([lab, val, "", "", ""])
    else:
        for lab in order:
            rows.append([lab, elem_to_mises[lab]])

    rows_out = rows

# 写 CSV（可选）
if EXPORT_CSV:
    with open(csv_final_path, "w", newline="", buffering=buf_size) as f:
        w = csv.writer(f)
        w.writerow(colnames)
        w.writerows(rows_out)
    T["write_csv"] = time.time() - _t_write
else:
    T["write_csv"] = 0.0

# 二进制导出（可选）
try:
    _bin_formats = [s.strip().lower() for s in EXPORT_BIN_FORMATS]
    if _bin_formats:
        _t_b = time.time()
        # 确保有 rows_out/colnames
        if (rows_out is None or colnames is None) and os.path.exists(csv_final_path):
            try:
                rows_out = []
                with open(csv_final_path, newline="") as f:
                    r = csv.reader(f)
                    header = next(r, None)
                    colnames = header
                    for row in r:
                        rows_out.append(row)
            except Exception:
                rows_out = None
        if rows_out is not None and colnames is not None:
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            import numpy as _np
            arr = _np.array([[_to_float(v) for v in row] for row in rows_out], dtype=float)
            if "npy" in _bin_formats:
                npy_path = os.path.splitext(csv_final_path)[0] + ".npy"
                _np.save(npy_path, arr)
            if "npz" in _bin_formats:
                npz_path = os.path.splitext(csv_final_path)[0] + ".npz"
                _np.savez(npz_path, data=arr, columns=colnames)
        T["binary_export"] = time.time() - _t_b
except Exception:
    pass

# 清理与关闭
try:
    odb.close()
except Exception:
    pass

# 写入计时日志
if ENABLE_TIMING_LOG:
    try:
        _timing_csv = os.path.join(output_folder, "timings.csv")
        prev_rows = []
        hdr = None
        epoch_from_single = None
        
        # 读取已有 timings.csv（由single_run写入）
        if os.path.exists(_timing_csv):
            try:
                with open(_timing_csv) as f:
                    rdr = csv.reader(f)
                    hdr = next(rdr, None)
                    for row in rdr:
                        if not row:
                            continue
                        prev_rows.append(row)
                if hdr and ("epoch" in hdr) and prev_rows:
                    i_epoch = hdr.index("epoch")
                    try:
                        epoch_from_single = float(prev_rows[0][i_epoch])
                    except Exception:
                        epoch_from_single = None
            except Exception:
                prev_rows = []
                hdr = None

        # 组装本次步骤
        post_steps_order = [
            "open_odb", "extract_field", "collect_values",
            "build_node_coords", "build_elem_nodes",
            "write_csv", "binary_export",
        ]
        seq = [(k, T[k]) for k in post_steps_order if k in T]
        start0 = 0.0
        if epoch_from_single is not None:
            start0 = _t0_total - epoch_from_single
        
        header = None
        if hdr and ("start_rel" in hdr) and ("end_rel" in hdr) and ("kind" in hdr):
            header = hdr if ("epoch" in hdr) else (hdr + ["epoch"])
        else:
            header = ["step", "seconds", "start_rel", "end_rel", "kind", "epoch"]
        
        # 生成本次 post 行
        post_rows = []
        cursor = start0
        for k, dur in seq:
            s = cursor
            e = cursor + float(dur)
            post_rows.append([k, f"{dur:.3f}", f"{s:.3f}", f"{e:.3f}", "post"])
            cursor = e
        
        # 合并写回
        with open(_timing_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            # 处理旧行
            if prev_rows and len(prev_rows[0]) <= 2:
                off = 0.0
                for r in prev_rows:
                    step = r[0]
                    secs = float(r[1]) if len(r) > 1 else 0.0
                    s = off
                    e = off + secs
                    w.writerow([
                        step, f"{secs:.3f}", f"{s:.3f}", f"{e:.3f}", "single",
                        epoch_from_single if epoch_from_single is not None else ""
                    ])
                    off = e
            else:
                if header and ("epoch" in header) and hdr and ("epoch" not in hdr):
                    for r in prev_rows:
                        w.writerow(r + [epoch_from_single if epoch_from_single is not None else ""])
                else:
                    for r in prev_rows:
                        w.writerow(r)
            # 写入post行
            for r in post_rows:
                w.writerow(r)
    except Exception:
        pass

# 删除 ODB（可选）
if POSTPROCESS_DELETE_ODB:
    try:
        os.remove(odb_path)
    except Exception:
        try:
            time.sleep(1.0)
            os.remove(odb_path)
        except Exception:
            pass

print(f"[post] 迭代 {iteration_num} 完成 -> {csv_final_path}")
