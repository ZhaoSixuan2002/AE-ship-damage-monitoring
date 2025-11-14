"""
流程01：生成训练数据主脚本
功能：批量调用Abaqus进行计算，每次迭代启动和关闭Abaqus以避免内存累积
使用方法：python 01_generate_training_data_main.py
"""

import csv
import os
import random
import shutil
import subprocess
import sys
import time

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 1. 基本路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  # 工作区根目录
GEN_DATA_DIR = r'C:\abaqus_gen_data'                    # 生成数据根目录（外部存储）
WORK_DIR = r'C:\ABAQUS\temp'                            # Abaqus工作目录
CAE_FILE_PATH = os.path.join(WORKSPACE_DIR, 'script', 'FEM_model', '0924-sunshangmoxing.cae')  # CAE文件路径
DISPLAY_ELEMENT_SET = 'MIDDLEWHOLE'                     # 显示元素集名称

# --- 2. 批处理控制参数 ---
NUM_ITERATIONS = 2                                   # 迭代次数
LOAD_MODE = 'random'                                    # 载荷模式：'random' 或 'predefined'
ITER_SUFFIX_CONTINUE = True                             # 是否续接已有编号
ITER_SUFFIX_START = 0                                   # 起始编号（>0时优先使用，否则根据CONTINUE自动判断）

# 预定义载荷案例（仅在LOAD_MODE='predefined'时使用）
PREDEFINED_LOAD_CASES = [
    # (h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2)
    (10474.62, 9695.98, 24053.05, 23494.74, -1488676.11, -91230634.55, 5227965.98, 6120170.63, 53864029.05, -54246137.73),
]
PREDEFINED_CASE_INDEX = 0                               # 使用的预定义案例索引

# --- 3. Abaqus配置 ---
ABAQUS_EXE = 'abaqus'                                   # Abaqus可执行路径或命令名
ABAQUS_SHOW_GUI = False                                 # 是否显示GUI（False=noGUI模式）

# --- 4. 异步后处理配置 ---
ASYNC_POSTPROCESS = True                                # 是否启用异步后处理
MAX_POST_WORKERS = 1                                    # 并发后处理进程数（1-8）
POSTPROCESS_USE_ABAQUS_PYTHON = True                    # 使用abaqus python执行后处理

# --- 5. 输出配置 ---
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '01_generate_training_data_output')  # 本脚本输出目录

# ========================================
# 脚本路径
# ========================================
single_run_script = os.path.join(os.path.dirname(__file__), "single_run.py")
postprocess_script = os.path.join(os.path.dirname(__file__), "01_async_postprocess_auxiliary.py")

# ========================================
# 后台后处理进程管理
# ========================================
post_workers = []  # [{process, it, odb, out_dir}]
post_tasks = []    # [(odb_path, output_folder, iteration_num)]


def _prune_workers(wait_any=False):
    """清理已完成的后台进程；若 wait_any=True 且无空位，则阻塞等待一个完成。"""
    global post_workers
    alive = []
    for w in post_workers:
        p = w["process"]
        rc = p.poll()
        if rc is None:
            alive.append(w)
    post_workers = alive
    if wait_any and len(post_workers) >= MAX_POST_WORKERS:
        w0 = post_workers[0]
        w0["process"].wait()
        try:
            post_workers.remove(w0)
        except Exception:
            pass


def _launch_post(odb_path, out_dir, itnum, abaqus_cmd_resolved, launch_via_shell):
    """尝试启动一个后台后处理任务；若达到并发上限则入队等待。"""
    if not ASYNC_POSTPROCESS:
        return
    _prune_workers(wait_any=False)
    if len(post_workers) >= MAX_POST_WORKERS:
        post_tasks.append((odb_path, out_dir, itnum))
        return
    
    # 组装命令
    if POSTPROCESS_USE_ABAQUS_PYTHON:
        if launch_via_shell:
            cmd = f'"{abaqus_cmd_resolved}" python "{postprocess_script}" -- "{odb_path}" "{out_dir}" {itnum}'
        else:
            cmd = [abaqus_cmd_resolved, "python", postprocess_script, "--", odb_path, out_dir, str(itnum)]
    else:
        if launch_via_shell:
            cmd = f'"{abaqus_cmd_resolved}" cae noGUI="{postprocess_script}" -- "{odb_path}" "{out_dir}" {itnum}'
        else:
            cmd = [abaqus_cmd_resolved, "cae", "noGUI=" + postprocess_script, "--", odb_path, out_dir, str(itnum)]

    # 环境
    _env = os.environ.copy()
    _script_dir = os.path.dirname(__file__)
    _env["PYTHONPATH"] = (_script_dir + os.pathsep + _env.get("PYTHONPATH", "")) if _env.get("PYTHONPATH") else _script_dir
    
    p = subprocess.Popen(cmd, stdout=None, stderr=None, cwd=WORK_DIR, shell=launch_via_shell, env=_env)
    post_workers.append({"process": p, "it": itnum, "odb": odb_path, "out_dir": out_dir})


def _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell):
    """在容量允许时启动排队中的后处理任务。"""
    _prune_workers(wait_any=False)
    while post_tasks and len(post_workers) < MAX_POST_WORKERS:
        odb_path, out_dir, itnum = post_tasks.pop(0)
        _launch_post(odb_path, out_dir, itnum, abaqus_cmd_resolved, launch_via_shell)


def progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=50, fill="#"):
    """进度条函数"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()


def _find_max_suffix(path):
    """查找目录中最大的数字编号"""
    try:
        max_id = 0
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            if name.isdigit():
                try:
                    n = int(name)
                    if n > max_id:
                        max_id = n
                except Exception:
                    pass
        return max_id
    except Exception:
        return 0


# ========================================
# 主程序
# ========================================
print("=" * 60)
print("流程01：生成训练数据")
print("=" * 60)
print(f"总迭代次数: {NUM_ITERATIONS}")
print(f"载荷模式: {LOAD_MODE}")
print(f"输出目录: {GEN_DATA_DIR}")
print(f"Abaqus GUI: {'ON' if ABAQUS_SHOW_GUI else 'OFF'}")
print("=" * 60)

# 预创建输出/工作目录
if not os.path.exists(GEN_DATA_DIR):
    os.makedirs(GEN_DATA_DIR)
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 基础检查：CAE 文件
if not os.path.isfile(CAE_FILE_PATH):
    print(f"错误: CAE文件不存在 {CAE_FILE_PATH}")
    sys.exit(1)
else:
    print(f"使用CAE文件: {CAE_FILE_PATH}")

# 校验 Abaqus 命令
abaqus_cmd_resolved = None
if os.path.isabs(ABAQUS_EXE):
    if os.path.exists(ABAQUS_EXE):
        abaqus_cmd_resolved = ABAQUS_EXE
    else:
        print(f"错误: 指定的 ABAQUS_EXE 不存在: {ABAQUS_EXE}")
        sys.exit(1)
else:
    which_path = shutil.which(ABAQUS_EXE)
    if which_path:
        abaqus_cmd_resolved = which_path
    else:
        common_path = r"C:\SIMULIA\Commands\abaqus.bat"
        print(f"错误: 找不到 Abaqus 命令 '{ABAQUS_EXE}' (未在 PATH 中)。")
        if os.path.exists(common_path):
            print(f"检测到常见安装路径存在: {common_path}")
            print(f"建议修改脚本中的 ABAQUS_EXE = r'{common_path}'")
        sys.exit(1)

print(f"使用Abaqus命令: {abaqus_cmd_resolved}")

# Windows 下如果是 .bat/.cmd，需要通过 shell 启动
is_windows = os.name == "nt"
launch_via_shell = False
if is_windows and abaqus_cmd_resolved.lower().endswith((".bat", ".cmd")):
    launch_via_shell = True
    print("[info] 检测到Windows批处理启动器；将通过 shell=True 调用")

# 统计变量
total_start_time = time.time()
successful_runs = 0
failed_runs = 0

# 计算起始后缀编号
if isinstance(ITER_SUFFIX_START, int) and ITER_SUFFIX_START > 0:
    start_suffix = ITER_SUFFIX_START
elif ITER_SUFFIX_CONTINUE:
    start_suffix = _find_max_suffix(GEN_DATA_DIR) + 1
else:
    start_suffix = 1

print(f"起始编号: {start_suffix} (模式: {'manual' if (isinstance(ITER_SUFFIX_START, int) and ITER_SUFFIX_START > 0) else ('continue' if ITER_SUFFIX_CONTINUE else 'reset')})")

# 主循环
for i in range(NUM_ITERATIONS):
    iteration_start_time = time.time()
    iteration_num = start_suffix + i

    # 创建输出文件夹
    output_folder = os.path.join(GEN_DATA_DIR, str(iteration_num))
    os.makedirs(output_folder)

    # 生成或选择载荷参数
    if LOAD_MODE == "random":
        h1 = random.uniform(4000, 30000)
        h2 = random.uniform(4000, 15000)
        h3 = random.uniform(4000, 30000)
        draft = random.uniform(10000, 30000)
        mx1 = random.uniform(1e6, 1e7) * random.choice([-1, 1])
        my1 = random.uniform(1e7, 1e8) * random.choice([-1, 1])
        mz1 = random.uniform(1e6, 1e7) * random.choice([-1, 1])
        mx2 = random.uniform(1e6, 1e7) * random.choice([-1, 1])
        my2 = random.uniform(1e7, 1e8) * random.choice([-1, 1])
        mz2 = random.uniform(1e6, 1e8) * random.choice([-1, 1])
    elif LOAD_MODE == "predefined":
        if PREDEFINED_CASE_INDEX < len(PREDEFINED_LOAD_CASES):
            h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2 = PREDEFINED_LOAD_CASES[PREDEFINED_CASE_INDEX]
        else:
            print("\n错误: PREDEFINED_CASE_INDEX 超出范围。退出。")
            sys.exit(1)
    else:
        print("\n错误: 无效的 LOAD_MODE。退出。")
        sys.exit(1)

    # 保存载荷参数到CSV
    load_params_file = os.path.join(output_folder, "loading_conditions.csv")
    with open(load_params_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h1", "h2", "h3", "draft", "mx1", "my1", "mz1", "mx2", "my2", "mz2"])
        writer.writerow([h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2])

    # 构建Abaqus命令
    use_gui = ABAQUS_SHOW_GUI
    if launch_via_shell:
        gui_token = "script" if use_gui else "noGUI"
        cmd_str = (
            f'"{abaqus_cmd_resolved}" cae {gui_token}="{single_run_script}" -- "{output_folder}" {h1} {h2} {h3} {draft} '
            f"{mx1} {my1} {mz1} {mx2} {my2} {mz2} {iteration_num}"
        )
        cmd = cmd_str
    else:
        token = "script=" if use_gui else "noGUI="
        cmd = [
            abaqus_cmd_resolved, "cae", token + single_run_script, "--",
            output_folder, str(h1), str(h2), str(h3), str(draft),
            str(mx1), str(my1), str(mz1), str(mx2), str(my2), str(mz2), str(iteration_num),
        ]

    # 执行Abaqus计算
    try:
        _env = os.environ.copy()
        _script_dir = os.path.dirname(__file__)
        _existing_pythonpath = _env.get("PYTHONPATH", "")
        if _existing_pythonpath:
            _env["PYTHONPATH"] = _script_dir + os.pathsep + _existing_pythonpath
        else:
            _env["PYTHONPATH"] = _script_dir
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=WORK_DIR, shell=launch_via_shell, env=_env
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            successful_runs += 1
        else:
            failed_runs += 1
            print(f"\n[迭代 {iteration_num}] Abaqus 进程返回非零退出码: {process.returncode}")

    except Exception as e:
        failed_runs += 1
        print(f"\n[迭代 {iteration_num}] 调用 Abaqus 失败: {e}")

    # 异步后处理：启动后台CSV导出任务
    try:
        if ASYNC_POSTPROCESS:
            expected_odb = os.path.join(output_folder, "iteration.odb")
            odb_path = None
            if os.path.exists(expected_odb):
                odb_path = expected_odb
            else:
                try:
                    for fn in os.listdir(output_folder):
                        if fn.lower().endswith(".odb"):
                            odb_path = os.path.join(output_folder, fn)
                            break
                except Exception:
                    odb_path = None
            if odb_path and os.path.exists(odb_path):
                _launch_post(odb_path, output_folder, iteration_num, abaqus_cmd_resolved, launch_via_shell)
                _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell)
    except Exception:
        pass

    # 计算本轮耗时
    iteration_end_time = time.time()
    iteration_time = iteration_end_time - iteration_start_time

    # 更新进度
    progress_bar(
        i + 1, NUM_ITERATIONS,
        prefix="整体进度:", suffix=f"完成 (时间: {iteration_time:.1f}s)", length=50
    )

    # 短暂延迟，确保资源完全释放
    time.sleep(2)

# 等待所有后台后处理完成
try:
    if ASYNC_POSTPROCESS:
        _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell)
        while post_workers:
            _prune_workers(wait_any=True)
            _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell)
except Exception:
    pass

total_end_time = time.time()
total_time = total_end_time - total_start_time

print("\n" + "=" * 60)
print("所有迭代完成！")
print("=" * 60)
print(f"总迭代次数: {NUM_ITERATIONS}")
print(f"成功: {successful_runs}")
print(f"失败: {failed_runs}")
print(f"总用时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
print(f"平均每次迭代: {total_time / NUM_ITERATIONS:.2f} 秒")
print("=" * 60)

# 写入日志文件
log_file = os.path.join(OUTPUT_DIR, "data_generation_log.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("流程01：生成训练数据 - 运行日志\n")
    f.write("=" * 60 + "\n")
    f.write(f"总迭代次数: {NUM_ITERATIONS}\n")
    f.write(f"成功: {successful_runs}\n")
    f.write(f"失败: {failed_runs}\n")
    f.write(f"总用时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)\n")
    f.write(f"平均每次迭代: {total_time / NUM_ITERATIONS:.2f} 秒\n")
    f.write(f"数据存储路径（外部）: {GEN_DATA_DIR}\n")
    f.write("=" * 60 + "\n")

print(f"\n日志已保存到: {log_file}")
print(f"注意：实际数据存储在外部路径: {GEN_DATA_DIR}")
