"""
流程06辅助脚本：单次Abaqus计算（验证数据生成）
在Abaqus noGUI模式下执行，完成一次完整的加载、计算、后处理流程
与训练数据生成的区别：从命令行接收CAE文件路径，支持不同损伤类型的CAE文件
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *

try:
    import odbAccess as _odbAccess
except Exception:
    _odbAccess = None
import csv
import math
import os
import shutil
import sys
import time

import displayGroupOdbToolset as dgo

# ========================================
# 固定参数配置（验证数据生成专用）
# ========================================
WORK_DIR = r'C:\ABAQUS\temp'
DISPLAY_ELEMENT_SET = 'MIDDLEWHOLE'

# 渲染和输出控制
ENABLE_LOAD_IMAGE = False
ENABLE_STRESS_IMAGE = False
ENABLE_TIMING_LOG = True
FAST_EXIT_AFTER_CSV = True
AUTO_CLOSE_CAE_AFTER_RUN = True
CAE_EXIT_SAVE_MODE = "temp"
CAE_FORCE_EXIT_IF_NEEDED = True

# 后处理控制
ASYNC_POSTPROCESS = True
STRESS_POSITION = "INTEGRATION_POINT"
SECTION_FILTER = "ALL"
AGGREGATION = "mean"
EXPORT_SCOPE = "ELEMENT"
NODE_AGGREGATION = "mean"
EXPORT_ELEMENT_CENTERS_ONCE = True

# ODB优化
REDUCE_ODB_FIELD_OUTPUT = True
FIELD_OUTPUT_VARS = ["S"]
FIELD_OUTPUT_FREQUENCY = "LAST_INCREMENT"
FIELD_OUTPUT_EVERY_N = 0
REDUCE_HISTORY_OUTPUT = True
DISABLE_TEXT_PRINT = True
JOB_PRECISION = "SINGLE"

# 步参数（保持默认，不强制修改）
STEP_NLGEOM = None
STEP_INITIAL_INC = None
STEP_MAX_INC = None
STEP_MIN_INC = None
STEP_MAX_NUM_INC = None

# 显示控制
PLOT_LEGEND_MODE = "compact"
LEGEND_FONT_SIZE = 50
ANNOTATION_FONT_SIZE = 48
HIDE_MESH_EDGES_LOAD = True
HIDE_MESH_EDGES_STRESS = True
ANNO_SHOW_TITLE = False
ANNO_SHOW_STATE = False
ANNO_SHOW_ANNOTATIONS = False
ANNO_SHOW_TRIAD = False
ANNO_SHOW_COMPASS = False

# ========================================
# 从命令行参数获取配置
# ========================================
# 参数顺序：cae_file, output_folder, h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2, iteration_num
if len(sys.argv) < 14:
    print("Error: Insufficient arguments")
    print("Usage: abaqus python script.py -- <cae_file> <output_folder> <h1> <h2> <h3> <draft> <mx1> <my1> <mz1> <mx2> <my2> <mz2> <iteration_num>")
    sys.exit(1)

cae_file_path = sys.argv[-13]
output_folder = sys.argv[-12]
h1 = float(sys.argv[-11])
h2 = float(sys.argv[-10])
h3 = float(sys.argv[-9])
draft = float(sys.argv[-8])
mx1 = float(sys.argv[-7])
my1 = float(sys.argv[-6])
mz1 = float(sys.argv[-5])
mx2 = float(sys.argv[-4])
my2 = float(sys.argv[-3])
mz2 = float(sys.argv[-2])
iteration_num = int(sys.argv[-1])

abaqus_work_dir = WORK_DIR
display_element_set = DISPLAY_ELEMENT_SET

print("=" * 60)
print(f"验证数据单次运行 - 迭代 {iteration_num}")
print("=" * 60)
print(f"CAE file: {cae_file_path}")
print(f"Output folder: {output_folder}")
print(f"Display element set: {display_element_set}")
print("=" * 60)

try:
    t0_total = time.time()
    timings = {}
    
    # --- 1. 打开CAE文件 ---
    print("Opening CAE file...")
    t0 = time.time()
    openMdb(pathName=cae_file_path)
    print("CAE file opened successfully")
    timings["open_cae"] = time.time() - t0
    
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # --- 2. 施加载荷 ---
    print("Applying loads...")
    t0 = time.time()
    
    # 修改表达式场 - 左侧货舱
    exp_left = (
        "3e-12*9810*(  ( " + str(h1) + " - Z - 0.000004*(pow(Y, 2) + pow(X -0, 2)) ) + fabs(" +
        str(h1) + " - Z - 0.000004*(pow(Y, 2) + pow(X -0, 2)) )  )/2"
    )
    mdb.models["Model-1"].analyticalFields["left"].setValues(expression=exp_left)
    mdb.models["Model-1"].loads["left1"].setValues(field="left")
    mdb.models["Model-1"].loads["left2"].setValues(field="left")
    
    # 中间货舱
    exp_mid = (
        "3e-12*9810*(  ( " + str(h2) + " - Z - 0.000004*(pow(Y, 2) + pow(X -66000, 2)) ) + fabs(" +
        str(h2) + " - Z - 0.000004*(pow(Y, 2) + pow(X -66000, 2)) )  )/2"
    )
    mdb.models["Model-1"].analyticalFields["mid"].setValues(expression=exp_mid)
    mdb.models["Model-1"].loads["mid1"].setValues(field="mid")
    mdb.models["Model-1"].loads["mid2"].setValues(field="mid")
    mdb.models["Model-1"].loads["mid3"].setValues(field="mid")
    
    # 右侧货舱
    exp_right = (
        "3e-12*9810*(  ( " + str(h3) + " - Z - 0.000004*(pow(Y, 2) + pow(X -132000, 2)) ) + fabs(" +
        str(h3) + " - Z - 0.000004*(pow(Y, 2) + pow(X -132000, 2)) )  )/2"
    )
    mdb.models["Model-1"].analyticalFields["right"].setValues(expression=exp_right)
    mdb.models["Model-1"].loads["right1"].setValues(field="right")
    mdb.models["Model-1"].loads["right2"].setValues(field="right")
    
    # 水压力
    exp_water = "1.025e-12*9810*( (" + str(draft) + "-Z) + fabs((" + str(draft) + "-Z)))/2"
    mdb.models["Model-1"].analyticalFields["water"].setValues(expression=exp_water)
    mdb.models["Model-1"].loads["water"].setValues(field="water")
    
    # 修改集中力矩
    mdb.models["Model-1"].loads["mom1"].setValues(
        cm1=mx1, cm2=my1, cm3=mz1, distributionType=UNIFORM, field=""
    )
    mdb.models["Model-1"].loads["mom2"].setValues(
        cm1=mx2, cm2=my2, cm3=mz2, distributionType=UNIFORM, field=""
    )
    
    print("Loads applied successfully")
    timings["apply_loads"] = time.time() - t0
    
    # --- 3. 模型优化设置 ---
    job_name = "Job-1"
    
    # 预清理锁文件
    try:
        _lck_path = os.path.join(abaqus_work_dir, job_name + ".lck")
        if os.path.exists(_lck_path):
            try:
                os.remove(_lck_path)
                print("[pre-clean] Removed stale lock:", _lck_path)
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        mdl = mdb.models["Model-1"]
        
        # 步参数调整
        try:
            for _sname, _step in mdl.steps.items():
                _kwargs = {}
                if STEP_NLGEOM is not None:
                    _kwargs["nlgeom"] = bool(STEP_NLGEOM)
                if STEP_INITIAL_INC is not None:
                    _kwargs["initialInc"] = float(STEP_INITIAL_INC)
                if STEP_MAX_INC is not None:
                    _kwargs["maxInc"] = float(STEP_MAX_INC)
                if STEP_MIN_INC is not None:
                    _kwargs["minInc"] = float(STEP_MIN_INC)
                if STEP_MAX_NUM_INC is not None:
                    _kwargs["maxNumInc"] = int(STEP_MAX_NUM_INC)
                if _kwargs:
                    try:
                        _step.setValues(**_kwargs)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 降低Field Output
        try:
            if REDUCE_ODB_FIELD_OUTPUT:
                _vars_tuple = tuple(FIELD_OUTPUT_VARS) if FIELD_OUTPUT_VARS else None
                for _rname, _req in mdl.fieldOutputRequests.items():
                    try:
                        if str(FIELD_OUTPUT_FREQUENCY).upper() == "LAST_INCREMENT":
                            if _vars_tuple:
                                _req.setValues(variables=_vars_tuple, frequency=LAST_INCREMENT)
                            else:
                                _req.setValues(frequency=LAST_INCREMENT)
                        elif str(FIELD_OUTPUT_FREQUENCY).upper() == "EVERY_N" and (FIELD_OUTPUT_EVERY_N and FIELD_OUTPUT_EVERY_N > 0):
                            if _vars_tuple:
                                _req.setValues(variables=_vars_tuple, numIntervals=int(FIELD_OUTPUT_EVERY_N))
                            else:
                                _req.setValues(numIntervals=int(FIELD_OUTPUT_EVERY_N))
                        else:
                            if _vars_tuple:
                                _req.setValues(variables=_vars_tuple)
                    except Exception:
                        try:
                            if _vars_tuple:
                                _req.setValues(variables=_vars_tuple)
                        except Exception:
                            pass
        except Exception:
            pass
        
        # 降低History Output
        try:
            if REDUCE_HISTORY_OUTPUT:
                for _rname, _req in mdl.historyOutputRequests.items():
                    _done = False
                    for _sname in list(mdl.steps.keys()):
                        try:
                            _req.deactivate(_sname)
                            _done = True
                        except Exception:
                            pass
                    if not _done:
                        try:
                            _req.setValues(frequency=LAST_INCREMENT)
                        except Exception:
                            try:
                                _req.setValues(numIntervals=1)
                            except Exception:
                                pass
        except Exception:
            pass
        
        # 关闭文本打印
        try:
            if DISABLE_TEXT_PRINT:
                mdb.jobs[job_name].setValues(
                    historyPrint=OFF, modelPrint=OFF, contactPrint=OFF, echoPrint=OFF
                )
        except Exception:
            pass
        
        # 设定精度
        try:
            if str(JOB_PRECISION).upper() == "SINGLE":
                mdb.jobs[job_name].setValues(precision=SINGLE)
            elif str(JOB_PRECISION).upper() == "DOUBLE":
                mdb.jobs[job_name].setValues(precision=DOUBLE)
        except Exception:
            pass
    except Exception:
        pass
    
    # --- 4. 提交计算 ---
    print("Submitting job...")
    t0 = time.time()
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    print("Job submitted, waiting for completion...")
    mdb.jobs[job_name].waitForCompletion()
    print("Job completed successfully")
    timings["solve_job"] = time.time() - t0
    
    # --- 5. 异步后处理：仅移动ODB ---
    if ASYNC_POSTPROCESS:
        try:
            odb_path = os.path.join(abaqus_work_dir, job_name + ".odb")
            if not os.path.exists(odb_path):
                raise Exception(f"ODB file not found: {odb_path}")
            t_mv = time.time()
            dest_odb = os.path.join(output_folder, "iteration.odb")
            if os.path.exists(dest_odb):
                try:
                    os.remove(dest_odb)
                except Exception:
                    pass
            shutil.move(odb_path, dest_odb)
            timings["move_odb"] = time.time() - t_mv
        except Exception as e:
            print("[warn] Move ODB failed:", str(e))
        
        # 写入阶段计时
        try:
            total_time = time.time() - t0_total
            timings["total"] = total_time
            if ENABLE_TIMING_LOG:
                _timing_csv = os.path.join(output_folder, "timings.csv")
                offset = 0.0
                ordered = list(timings.items())
                with open(_timing_csv, "w") as tf:
                    tf.write("step,seconds,start_rel,end_rel,kind\n")
                    for k, v in ordered:
                        s = offset
                        e = offset + float(v)
                        tf.write(f"{k},,{round(v, 3):.3f},{s:.3f},{e:.3f}\n")
                        offset = e
                print("Timings saved:", _timing_csv)
        except Exception:
            pass
        
        # 快速退出
        os._exit(0)
    
    # --- 6. 同步后处理（如未启用异步）---
    print("Post-processing results...")
    t0 = time.time()
    odb_path = os.path.join(abaqus_work_dir, job_name + ".odb")
    
    if not os.path.exists(odb_path):
        raise Exception(f"ODB file not found: {odb_path}")
    
    _t_open_odb = time.time()
    if _odbAccess is not None:
        try:
            o3 = _odbAccess.openOdb(path=odb_path, readOnly=True)
        except Exception:
            o3 = session.openOdb(name=odb_path)
    else:
        o3 = session.openOdb(name=odb_path)
    timings["open_odb"] = time.time() - _t_open_odb
    
    # 获取数据帧
    try:
        step_names = list(o3.steps.keys())
        step = o3.steps[step_names[0]]
        frame = step.frames[-1]
        try:
            _fi = os.path.join(output_folder, "frames_info.txt")
            with open(_fi, "w") as _f:
                _f.write(f"step_name: {step_names[0]}\n")
                _f.write(f"num_frames: {len(step.frames)}\n")
                try:
                    _f.write("last_frame_value: {}\n".format(getattr(frame, "frameValue", "")))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        frame = list(o3.steps.values())[0].frames[-1]
    
    # 获取目标单元集
    elset = None
    try:
        elset = o3.rootAssembly.elementSets[display_element_set]
    except Exception:
        try:
            for inst in o3.rootAssembly.instances.values():
                if display_element_set in inst.elementSets:
                    elset = inst.elementSets[display_element_set]
                    break
        except Exception:
            elset = None
    
    # 应力场
    stress = frame.fieldOutputs["S"]
    
    _pos_map = {
        "INTEGRATION_POINT": INTEGRATION_POINT,
        "CENTROID": CENTROID,
        "ELEMENT_NODAL": ELEMENT_NODAL,
    }
    
    _export_scope = str(EXPORT_SCOPE).upper()
    if _export_scope == "NODE":
        _pos = _pos_map.get("ELEMENT_NODAL", None)
    else:
        _pos = _pos_map.get(str(STRESS_POSITION).upper(), None)
    
    _t_extract = time.time()
    try:
        if _pos is not None:
            sub = stress.getSubset(region=elset, position=_pos) if elset else stress.getSubset(position=_pos)
        else:
            sub = stress if elset is None else stress.getSubset(region=elset)
    except Exception:
        sub = stress
    
    try:
        mises = sub.getScalarField(invariant=MISES)
    except Exception:
        try:
            mises = sub.getScalarField(invariant="Mises")
        except Exception:
            mises = None
    timings["extract_field"] = time.time() - _t_extract
    
    def _accept_section(v):
        if str(SECTION_FILTER).upper() == "ALL":
            return True
        try:
            sp = getattr(v, "sectionPoint", None)
            desc = ""
            if sp is not None:
                desc = getattr(sp, "description", "") or str(sp)
            desc = str(desc).upper()
            target = str(SECTION_FILTER).upper()
            if target == "SNEG":
                return ("SNEG" in desc) or ("-1.0" in desc)
            if target == "SPOS":
                return ("SPOS" in desc) or ("1.0" in desc)
            if target == "SMID":
                return ("SMID" in desc) or ("SAVG" in desc) or ("MID" in desc)
            return True
        except Exception:
            return True
    
    _t_collect = time.time()
    try:
        values_iter = list(mises.values) if mises is not None else []
    except Exception:
        values_iter = []
    finally:
        timings["collect_values"] = time.time() - _t_collect
    
    if not values_iter:
        try:
            values_iter = list(sub.values)
        except Exception:
            values_iter = []
    
    def _aggregate(vals, method):
        if not vals:
            return None
        if method == "max":
            return max(vals)
        if method == "median":
            try:
                import statistics
                return statistics.median(vals)
            except Exception:
                vals_sorted = sorted(vals)
                n = len(vals_sorted)
                return vals_sorted[n // 2] if n % 2 == 1 else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
        if method.startswith("p") and len(method) >= 2:
            try:
                p = float(method[1:])
                p = max(0.0, min(100.0, p))
            except Exception:
                p = 95.0
            vals_sorted = sorted(vals)
            if not vals_sorted:
                return None
            k = int(round((p / 100.0) * (len(vals_sorted) - 1)))
            return vals_sorted[k]
        return sum(vals) / float(len(vals))
    
    csv_final_path = os.path.join(output_folder, "iteration.csv")
    
    # 元素导出
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
                    try:
                        sval = float(v.data.mises)
                    except Exception:
                        continue
            if math.isnan(sval) or math.isinf(sval):
                continue
            elem_to_vals.setdefault(lab, []).append(sval)
        except Exception:
            continue
    
    agg_method = str(AGGREGATION).lower()
    elem_to_mises = {}
    for lab, vals in elem_to_vals.items():
        val = _aggregate(vals, agg_method)
        if val is not None:
            elem_to_mises[lab] = val
    
    _add_coords = (iteration_num == 1) and bool(EXPORT_ELEMENT_CENTERS_ONCE)
    
    with open(csv_final_path, "w", newline="") as f:
        w = csv.writer(f)
        if _add_coords:
            elem_nodes = {}
            node_coords = {}
            try:
                allowed = None
                try:
                    allowed = set(int(e.label) for e in (elset.elements if elset else []))
                except Exception:
                    allowed = None
                _t_build_maps = time.time()
                for inst in o3.rootAssembly.instances.values():
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
                timings["build_elem_nodes"] = time.time() - _t_build_maps
            except Exception:
                elem_nodes = {}
            
            w.writerow(["Element Label", "S-Mises", "X", "Y", "Z"])
            for lab in sorted(elem_to_mises.keys()):
                val = elem_to_mises[lab]
                coords = None
                try:
                    nlist = elem_nodes.get(lab)
                    if nlist:
                        xs, ys, zs = [], [], []
                        for nlab in nlist:
                            c = node_coords.get(int(nlab))
                            if c is not None:
                                x, y, z = c
                                xs.append(float(x))
                                ys.append(float(y))
                                zs.append(float(z))
                        if xs:
                            coords = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
                except Exception:
                    coords = None
                if coords is not None:
                    w.writerow([lab, val, coords[0], coords[1], coords[2]])
                else:
                    w.writerow([lab, val, "", "", ""])
        else:
            w.writerow(["Element Label", "S-Mises"])
            for lab in sorted(elem_to_mises.keys()):
                w.writerow([lab, elem_to_mises[lab]])
    
    print("CSV report saved")
    timings["write_csv"] = time.time() - t0
    
    # --- 7. 清理 ---
    if FAST_EXIT_AFTER_CSV:
        total_time = time.time() - t0_total
        timings["total"] = total_time
        if ENABLE_TIMING_LOG:
            _timing_csv = os.path.join(output_folder, "timings.csv")
            offset = 0.0
            ordered = list(timings.items())
            with open(_timing_csv, "w") as tf:
                tf.write("step,seconds,start_rel,end_rel,kind\n")
                for k, v in ordered:
                    s = offset
                    e = offset + float(v)
                    tf.write(f"{k},,{round(v, 3):.3f},{s:.3f},{e:.3f}\n")
                    offset = e
            print("Timings saved:", _timing_csv)
        print("[fast-exit] Exiting...")
        os._exit(0)
    
    # 常规关闭
    _t_close = time.time()
    try:
        if "o3" in globals() and hasattr(o3, "close"):
            o3.close()
        else:
            session.odbs[odb_path].close()
    except Exception:
        try:
            session.odbs[odb_path].close()
        except Exception:
            pass
    timings["close_odb"] = time.time() - _t_close
    
    print("=" * 60)
    total_time = time.time() - t0_total
    timings["total"] = total_time
    print(f"Iteration {iteration_num} completed successfully")
    print("=" * 60)
    
    # 写入计时
    try:
        if ENABLE_TIMING_LOG:
            _timing_csv = os.path.join(output_folder, "timings.csv")
            offset = 0.0
            ordered = list(timings.items())
            with open(_timing_csv, "w") as tf:
                tf.write("step,seconds,start_rel,end_rel,kind\n")
                for k, v in ordered:
                    s = offset
                    e = offset + float(v)
                    tf.write(f"{k},,{round(v, 3):.3f},{s:.3f},{e:.3f}\n")
                    offset = e
            print("Timings saved:", _timing_csv)
    except Exception:
        pass
    
    # 自动退出
    try:
        if AUTO_CLOSE_CAE_AFTER_RUN:
            mode = str(CAE_EXIT_SAVE_MODE).lower()
            if mode == "temp":
                try:
                    _tmp_cae = os.path.join(abaqus_work_dir, "_auto_saved_model.cae")
                    mdb.saveAs(pathName=_tmp_cae)
                except Exception:
                    pass
            elif mode == "save":
                try:
                    mdb.save()
                except Exception:
                    pass
            try:
                session.exit()
            except Exception:
                try:
                    session.Exit()
                except Exception:
                    if CAE_FORCE_EXIT_IF_NEEDED:
                        os._exit(0)
    except Exception:
        if CAE_FORCE_EXIT_IF_NEEDED:
            os._exit(0)

except Exception as e:
    print("=" * 60)
    print(f"ERROR in iteration {iteration_num}:")
    print(str(e))
    print("=" * 60)
    
    error_file = os.path.join(output_folder, "abaqus_error.log")
    with open(error_file, "w") as f:
        f.write(f"Error in iteration {iteration_num}\n")
        f.write(str(e))
        f.write("\n")
    
    sys.exit(1)
