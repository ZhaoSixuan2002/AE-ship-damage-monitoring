"""
流程06：生成验证数据主脚本
功能：生成四种验证数据集（health/crack/corrosion/multi），用于模型验证
     每种类型使用不同的CAE文件，生成到独立的外部存储目录
使用方法：
    python 06_generate_validation_data_main.py                    # 生成所有数据集
    python 06_generate_validation_data_main.py --datasets multi   # 只生成multi数据集
    python 06_generate_validation_data_main.py --datasets health crack  # 生成health和crack
"""

import argparse
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
WORK_DIR = r'C:\ABAQUS\temp'                            # Abaqus工作目录
DISPLAY_ELEMENT_SET = 'MIDDLEWHOLE'                     # 显示元素集名称

# 四种验证数据类型的CAE文件配置
VALIDATE_HEALTH_CAE_FILE = os.path.join(WORKSPACE_DIR, 'script', 'FEM_model', '0924-sunshangmoxing.cae')
VALIDATE_CRACK_CAE_FILE = os.path.join(WORKSPACE_DIR, 'script', 'FEM_model', '0924-sunshangmoxing-liefeng-B2-5.cae')
VALIDATE_CORROSION_CAE_FILE = os.path.join(WORKSPACE_DIR, 'script', 'FEM_model', '0924-sunshangmoxing-liefeng-AB3-7.cae')
VALIDATE_MULTI_CAE_FILE = os.path.join(WORKSPACE_DIR, 'script', 'FEM_model', '0924-sunshangmoxing-zonghe.cae')

# 四种验证数据类型的输出目录配置（外部存储）
VALIDATE_HEALTH_OUTPUT_DIR = r'C:\abaqus_gen_data_validate_original_health'
VALIDATE_CRACK_OUTPUT_DIR = r'C:\abaqus_gen_data_validate_damage_crack'
VALIDATE_CORROSION_OUTPUT_DIR = r'C:\abaqus_gen_data_validate_damage_corrosion'
VALIDATE_MULTI_OUTPUT_DIR = r'C:\abaqus_gen_data_validate_damage_multi'

# --- 2. 批处理控制参数 ---
NUM_ITERATIONS = 100                                    # 每种类型的迭代次数
LOAD_MODE = 'random'                                    # 载荷模式：'random' 或 'predefined'
ITER_SUFFIX_CONTINUE = True                             # 是否续接已有编号
ITER_SUFFIX_START = 0                                   # 起始编号（>0时优先使用，否则根据CONTINUE自动判断）

# 预定义载荷案例（仅在LOAD_MODE='predefined'时使用）
PREDEFINED_LOAD_CASES = [
    # (h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2)
    (10474.62, 9695.98, 24053.05, 23494.74, -1488676.11, -91230634.55, 5227965.98, 6120170.63, 53864029.05, -54246137.73),
]
PREDEFINED_CASE_INDEX = 0                               # 使用的预定义案例索引

# --- 3. 验证数据生成控制 ---
# 选择要生成的数据类型（可选择一个或多个）
# 注意：这些默认值会被命令行参数覆盖
GENERATE_HEALTH = True                                  # 生成健康结构验证数据
GENERATE_CRACK = True                                   # 生成裂纹损伤验证数据
GENERATE_CORROSION = True                               # 生成腐蚀损伤验证数据
GENERATE_MULTI = True                                   # 生成多位置综合损伤验证数据

# --- 4. Abaqus配置 ---
ABAQUS_EXE = 'abaqus'                                   # Abaqus可执行路径或命令名
ABAQUS_SHOW_GUI = False                                 # 是否显示GUI（False=noGUI模式）

# --- 5. 异步后处理配置 ---
ASYNC_POSTPROCESS = True                                # 是否启用异步后处理
MAX_POST_WORKERS = 1                                    # 并发后处理进程数（1-8）
POSTPROCESS_USE_ABAQUS_PYTHON = True                    # 使用abaqus python执行后处理

# --- 6. 输出配置 ---
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '06_generate_validation_data_output')  # 本脚本输出目录

# ========================================
# 脚本路径
# ========================================
single_run_script = os.path.join(os.path.dirname(__file__), "06_single_run_auxiliary.py")
postprocess_script = os.path.join(os.path.dirname(__file__), "06_async_postprocess_auxiliary.py")

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


def generate_validation_dataset(data_type, cae_file, output_base_dir, num_iterations, 
                                 abaqus_cmd_resolved, launch_via_shell):
    """
    生成一个验证数据集
    
    参数：
        data_type: 数据类型名称 ('health', 'crack', 'corrosion')
        cae_file: CAE文件路径
        output_base_dir: 输出基础目录
        num_iterations: 迭代次数
        abaqus_cmd_resolved: Abaqus命令路径
        launch_via_shell: 是否通过shell启动
    
    返回：
        (successful_runs, failed_runs): 成功和失败的迭代次数
    """
    print("\n" + "=" * 60)
    print(f"开始生成 {data_type.upper()} 验证数据")
    print("=" * 60)
    print(f"CAE文件: {cae_file}")
    print(f"输出目录: {output_base_dir}")
    print(f"迭代次数: {num_iterations}")
    print("=" * 60)
    
    # 检查CAE文件
    if not os.path.isfile(cae_file):
        print(f"错误: CAE文件不存在 {cae_file}")
        return (0, num_iterations)
    
    # 创建输出目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 计算起始编号
    if isinstance(ITER_SUFFIX_START, int) and ITER_SUFFIX_START > 0:
        start_suffix = ITER_SUFFIX_START
    elif ITER_SUFFIX_CONTINUE:
        start_suffix = _find_max_suffix(output_base_dir) + 1
    else:
        start_suffix = 1
    
    print(f"起始编号: {start_suffix}")
    
    successful_runs = 0
    failed_runs = 0
    dataset_start_time = time.time()
    
    # 主循环
    for i in range(num_iterations):
        iteration_start_time = time.time()
        iteration_num = start_suffix + i
        
        # 创建输出文件夹
        output_folder = os.path.join(output_base_dir, str(iteration_num))
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
                print(f"\n错误: PREDEFINED_CASE_INDEX 超出范围。跳过 {data_type}。")
                return (0, num_iterations)
        else:
            print(f"\n错误: 无效的 LOAD_MODE。跳过 {data_type}。")
            return (0, num_iterations)
        
        # 保存载荷参数到CSV
        load_params_file = os.path.join(output_folder, "loading_conditions.csv")
        with open(load_params_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["h1", "h2", "h3", "draft", "mx1", "my1", "mz1", "mx2", "my2", "mz2"])
            writer.writerow([h1, h2, h3, draft, mx1, my1, mz1, mx2, my2, mz2])
        
        # 构建Abaqus命令（传递CAE文件路径作为额外参数）
        use_gui = ABAQUS_SHOW_GUI
        if launch_via_shell:
            gui_token = "script" if use_gui else "noGUI"
            cmd_str = (
                f'"{abaqus_cmd_resolved}" cae {gui_token}="{single_run_script}" -- "{cae_file}" "{output_folder}" '
                f'{h1} {h2} {h3} {draft} {mx1} {my1} {mz1} {mx2} {my2} {mz2} {iteration_num}'
            )
            cmd = cmd_str
        else:
            token = "script=" if use_gui else "noGUI="
            cmd = [
                abaqus_cmd_resolved, "cae", token + single_run_script, "--",
                cae_file, output_folder, str(h1), str(h2), str(h3), str(draft),
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
                print(f"\n[{data_type} - 迭代 {iteration_num}] Abaqus 进程返回非零退出码: {process.returncode}")
        
        except Exception as e:
            failed_runs += 1
            print(f"\n[{data_type} - 迭代 {iteration_num}] 调用 Abaqus 失败: {e}")
        
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
            i + 1, num_iterations,
            prefix=f"{data_type.upper()} 进度:", 
            suffix=f"完成 (时间: {iteration_time:.1f}s)", 
            length=50
        )
        
        # 短暂延迟，确保资源完全释放
        time.sleep(2)
    
    dataset_end_time = time.time()
    dataset_time = dataset_end_time - dataset_start_time
    
    print(f"\n{data_type.upper()} 数据集生成完成！")
    print(f"成功: {successful_runs}, 失败: {failed_runs}")
    print(f"用时: {dataset_time:.2f} 秒 ({dataset_time / 60:.2f} 分钟)")
    
    return (successful_runs, failed_runs)


# ========================================
# 主程序
# ========================================

# 解析命令行参数
parser = argparse.ArgumentParser(
    description='生成验证数据集（health/crack/corrosion/multi）',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
  python 06_generate_validation_data_main.py                      # 生成所有数据集
  python 06_generate_validation_data_main.py --datasets multi     # 只生成multi数据集
  python 06_generate_validation_data_main.py --datasets health crack  # 生成health和crack
    """
)
parser.add_argument(
    '--datasets',
    nargs='+',
    choices=['health', 'crack', 'corrosion', 'multi'],
    help='指定要生成的数据集类型（可指定多个）。不指定则生成所有数据集。'
)
args = parser.parse_args()

# 根据命令行参数设置生成标志
if args.datasets:
    # 如果指定了数据集，则只生成指定的
    GENERATE_HEALTH = 'health' in args.datasets
    GENERATE_CRACK = 'crack' in args.datasets
    GENERATE_CORROSION = 'corrosion' in args.datasets
    GENERATE_MULTI = 'multi' in args.datasets
else:
    # 如果没有指定，则生成所有数据集
    GENERATE_HEALTH = True
    GENERATE_CRACK = True
    GENERATE_CORROSION = True
    GENERATE_MULTI = True

print("=" * 60)
print("流程06：生成验证数据")
print("=" * 60)
print(f"每种类型迭代次数: {NUM_ITERATIONS}")
print(f"载荷模式: {LOAD_MODE}")
print(f"Abaqus GUI: {'ON' if ABAQUS_SHOW_GUI else 'OFF'}")
print("=" * 60)
print("将生成以下验证数据集:")
if GENERATE_HEALTH:
    print(f"  - HEALTH (健康结构): {VALIDATE_HEALTH_OUTPUT_DIR}")
if GENERATE_CRACK:
    print(f"  - CRACK (裂纹损伤): {VALIDATE_CRACK_OUTPUT_DIR}")
if GENERATE_CORROSION:
    print(f"  - CORROSION (腐蚀损伤): {VALIDATE_CORROSION_OUTPUT_DIR}")
if GENERATE_MULTI:
    print(f"  - MULTI (多位置综合损伤): {VALIDATE_MULTI_OUTPUT_DIR}")
print("=" * 60)

# 预创建工作目录和输出目录
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

# 总体统计
total_start_time = time.time()
all_results = {}

# 生成各类验证数据
if GENERATE_HEALTH:
    health_success, health_failed = generate_validation_dataset(
        'health', VALIDATE_HEALTH_CAE_FILE, VALIDATE_HEALTH_OUTPUT_DIR, 
        NUM_ITERATIONS, abaqus_cmd_resolved, launch_via_shell
    )
    all_results['health'] = (health_success, health_failed)

if GENERATE_CRACK:
    crack_success, crack_failed = generate_validation_dataset(
        'crack', VALIDATE_CRACK_CAE_FILE, VALIDATE_CRACK_OUTPUT_DIR, 
        NUM_ITERATIONS, abaqus_cmd_resolved, launch_via_shell
    )
    all_results['crack'] = (crack_success, crack_failed)

if GENERATE_CORROSION:
    corrosion_success, corrosion_failed = generate_validation_dataset(
        'corrosion', VALIDATE_CORROSION_CAE_FILE, VALIDATE_CORROSION_OUTPUT_DIR, 
        NUM_ITERATIONS, abaqus_cmd_resolved, launch_via_shell
    )
    all_results['corrosion'] = (corrosion_success, corrosion_failed)

if GENERATE_MULTI:
    multi_success, multi_failed = generate_validation_dataset(
        'multi', VALIDATE_MULTI_CAE_FILE, VALIDATE_MULTI_OUTPUT_DIR, 
        NUM_ITERATIONS, abaqus_cmd_resolved, launch_via_shell
    )
    all_results['multi'] = (multi_success, multi_failed)

# 等待所有后台后处理完成
try:
    if ASYNC_POSTPROCESS:
        print("\n等待所有后台后处理任务完成...")
        _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell)
        while post_workers:
            _prune_workers(wait_any=True)
            _drain_post_tasks(abaqus_cmd_resolved, launch_via_shell)
        print("所有后台后处理任务已完成")
except Exception:
    pass

total_end_time = time.time()
total_time = total_end_time - total_start_time

# 打印总结
print("\n" + "=" * 60)
print("所有验证数据集生成完成！")
print("=" * 60)
for data_type, (success, failed) in all_results.items():
    print(f"{data_type.upper()}: 成功 {success}, 失败 {failed}")
print(f"总用时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
print("=" * 60)

# 写入日志文件
log_file = os.path.join(OUTPUT_DIR, "data_generation_log.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("流程06：生成验证数据 - 运行日志\n")
    f.write("=" * 60 + "\n")
    f.write(f"每种类型迭代次数: {NUM_ITERATIONS}\n")
    f.write(f"载荷模式: {LOAD_MODE}\n")
    f.write("\n生成结果:\n")
    for data_type, (success, failed) in all_results.items():
        f.write(f"  {data_type.upper()}: 成功 {success}, 失败 {failed}\n")
    f.write(f"\n总用时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)\n")
    f.write("\n数据存储路径（外部）:\n")
    if 'health' in all_results:
        f.write(f"  HEALTH: {VALIDATE_HEALTH_OUTPUT_DIR}\n")
    if 'crack' in all_results:
        f.write(f"  CRACK: {VALIDATE_CRACK_OUTPUT_DIR}\n")
    if 'corrosion' in all_results:
        f.write(f"  CORROSION: {VALIDATE_CORROSION_OUTPUT_DIR}\n")
    if 'multi' in all_results:
        f.write(f"  MULTI: {VALIDATE_MULTI_OUTPUT_DIR}\n")
    f.write("=" * 60 + "\n")

print(f"\n日志已保存到: {log_file}")
print("注意：实际数据存储在外部路径（见上述输出）")
