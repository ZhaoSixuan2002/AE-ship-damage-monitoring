"""
流程07：验证数据预处理主脚本
功能：批量预处理四类验证数据（health/crack/corrosion/multi），并分别输出到独立文件夹
     处理逻辑与流程03相同，但针对四种验证数据类型分别执行

依赖：流程02和流程06必须先执行

输入：
    - C:/abaqus_gen_data_validate_original_health/ 目录下的健康数据
    - C:/abaqus_gen_data_validate_damage_crack/ 目录下的裂纹损伤数据
    - C:/abaqus_gen_data_validate_damage_corrosion/ 目录下的腐蚀损伤数据
    - C:/abaqus_gen_data_validate_damage_multi/ 目录下的多位置综合损伤数据
    - 02_inp_to_vtu_output/measures_ID_auto.csv（由流程02生成）

输出：
    - 07_preprocess_validation_data_output/health/preprocessed_data_raw.npz
    - 07_preprocess_validation_data_output/crack/preprocessed_data_raw.npz
    - 07_preprocess_validation_data_output/corrosion/preprocessed_data_raw.npz
    - 07_preprocess_validation_data_output/multi/preprocessed_data_raw.npz
    - 07_preprocess_validation_data_output/health/*.png (可视化图像)
    - 07_preprocess_validation_data_output/crack/*.png (可视化图像)
    - 07_preprocess_validation_data_output/corrosion/*.png (可视化图像)
    - 07_preprocess_validation_data_output/multi/*.png (可视化图像)

使用方法：
    python 07_preprocess_validation_data_main.py                     # 处理所有数据集
    python 07_preprocess_validation_data_main.py --datasets multi    # 只处理multi数据集
    python 07_preprocess_validation_data_main.py --datasets health crack  # 处理health和crack
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import stat
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from tqdm import tqdm

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 1. 路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 四种验证数据源目录（由流程06生成）
SOURCE_DATA_HEALTH = r'C:\abaqus_gen_data_validate_original_health'
SOURCE_DATA_CRACK = r'C:\abaqus_gen_data_validate_damage_crack'
SOURCE_DATA_CORROSION = r'C:\abaqus_gen_data_validate_damage_corrosion'
SOURCE_DATA_MULTI = r'C:\abaqus_gen_data_validate_damage_multi'

# 输出目录
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '07_preprocess_validation_data_output')

# --- 2. 数据处理参数 ---
MEASURES_ID_CSV = '../script/02_inp_to_vtu_output/measures_ID_auto.csv'  # 测点ID文件（由流程02生成）
RECONSTRUCT_COLUMN = 'all_measures'                         # 重构列名（自监督学习）
PREPROCESS_MAX_FOLDERS = None                               # 最大处理文件夹数（None=不限制）
USE_NPY = True                                              # 优先使用NPY格式（False则使用CSV）
SEED = 42                                                   # 随机种子

# --- 3. 处理流程开关（对三种数据类型的总开关）---
DO_COLLECT = True                                           # 执行数据收集
DO_TRANSFORM = False                                        # 执行数据变换（稳健高斯变换）
DO_VISUALIZE = True                                         # 执行可视化

# --- 4. 数据类型选择（选择要处理的验证数据类型）---
# 注意：这些默认值会被命令行参数覆盖
PROCESS_HEALTH = True                                       # 处理健康数据
PROCESS_CRACK = True                                        # 处理裂纹损伤数据
PROCESS_CORROSION = True                                    # 处理腐蚀损伤数据
PROCESS_MULTI = True                                        # 处理多位置综合损伤数据

# --- 5. 变换参数（仅当DO_TRANSFORM=True时生效）---
TRANSFORM_KIND = 'robust-gauss'                             # 变换类型：'none', 'minmax', 'robust-gauss'

# --- 6. 可视化参数 ---
PREPROCESS_PLOT_FIRST_N = 25                                # 可视化前N个维度
SHOW_PLOT_PROGRESS = True                                   # 显示绘图进度条
PREPROCESS_GRID_ROWS = 5                                    # 网格行数
PREPROCESS_GRID_COLS = 5                                    # 网格列数（每行样本数）

# 可视化样式参数
PREPROCESS_SCATTER_COLOR = 'tab:blue'                       # 散点图颜色
PREPROCESS_DENSITY_COLOR = 'tab:orange'                     # 密度图颜色
PREPROCESS_SCATTER_SIZE = 1.5                               # 散点大小
PREPROCESS_DENSITY_XLIM = (-3.5, 3.5)                       # 密度图X轴范围（变换后使用，原始数据自动）
PREPROCESS_SHOW_STATS = False                               # 是否显示统计信息
PREPROCESS_SHOW_GRID = False                                # 是否显示网格线
PREPROCESS_KDE_LINEWIDTH = 0.5                              # KDE曲线线宽
PREPROCESS_HIST_BINS = 100                                  # 柱状图桶数
PREPROCESS_HIST_ALPHA = 0.20                                # 柱状图透明度

# 图像输出参数
FIG_DPI = 300                                               # 图片DPI

# 绘图风格（学术风）
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
}

# ========================================
# 内嵌工具函数：数据变换
# ========================================

TransformKind = Literal["none", "minmax", "robust-gauss"]

@dataclass
class FittedTransforms:
    """变换器封装类"""
    kind: TransformKind
    scaler_V: object | None = None
    gauss_V: QuantileTransformer | None = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> FittedTransforms:
        return joblib.load(path)


def fit_transforms(
    V: np.ndarray, kind: TransformKind = "robust-gauss", random_state: int = 42
) -> FittedTransforms:
    """拟合数据变换器"""
    if kind == "none":
        return FittedTransforms(kind="none")

    if kind == "minmax":
        sv = MinMaxScaler()
        sv.fit(V)
        return FittedTransforms(kind="minmax", scaler_V=sv)

    if kind == "robust-gauss":
        # 第一步：稳健缩放（抗异常值）
        sv = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        V1 = sv.fit_transform(V)

        # 第二步：边际高斯化（分位到标准正态）
        gv = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, V.shape[0]),
            subsample=int(1e9),
            random_state=random_state,
        )
        gv.fit(V1)

        return FittedTransforms(kind="robust-gauss", scaler_V=sv, gauss_V=gv)

    raise ValueError(f"Unknown transform kind: {kind}")


def apply_transforms(V: np.ndarray, tf: FittedTransforms) -> np.ndarray:
    """应用变换器"""
    if tf.kind == "none":
        return V
    if tf.kind == "minmax":
        return tf.scaler_V.transform(V)
    if tf.kind == "robust-gauss":
        V1 = tf.scaler_V.transform(V)
        return tf.gauss_V.transform(V1)
    raise ValueError(f"Unknown transform kind: {tf.kind}")


# ========================================
# 内嵌工具函数：绘图
# ========================================

def apply_plot_style():
    """应用学术风绘图样式"""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(
    fig: plt.Figure,
    output_dir: str,
    name: str,
    data: dict[str, np.ndarray] | None = None,
):
    """保存图像和同名数据源"""
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(img_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    if data:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)


def safe_kde(data: np.ndarray, grid: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """安全的核密度估计"""
    clean = data[np.isfinite(data)]
    if clean.size == 0:
        return None, None
    if clean.size == 1 or np.allclose(clean, clean[0]):
        center = clean[0]
        span = 1.0 if not np.isfinite(center) else max(abs(center) * 0.05, 1e-3)
        xs = np.linspace(center - span, center + span, 512)
        ys = np.exp(-0.5 * ((xs - center) / (span if span > 0 else 1e-3)) ** 2)
        ys /= np.trapezoid(ys, xs)
        return xs, ys

    if grid is None:
        low, high = np.percentile(clean, [0.5, 99.5])
        if np.isclose(low, high):
            delta = max(abs(low) * 0.05, 1e-3)
            low -= delta
            high += delta
        xs = np.linspace(low, high, 512)
    else:
        xs = grid

    std = clean.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        std = max(abs(clean.mean()) * 0.05, 1e-3)
    bandwidth = 1.06 * std * clean.size ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1e-2

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(clean[:, None])
    log_density = kde.score_samples(xs[:, None])
    density = np.exp(log_density)
    density /= np.trapezoid(density, xs)
    return xs, density


def plot_grid_scatter_density(
    data: np.ndarray,
    title_list: list[str],
    output_dir: str,
    name: str,
    samples_per_row: int = 5,
    max_rows: int = 5,
    scatter_color: str = "tab:blue",
    density_color: str = "tab:green",
    scatter_size: float = 1.0,
    density_xlim: Optional[tuple[float, float]] = None,
    show_grid: bool = True,
    kde_linewidth: float = 1.0,
    progress_callback: Optional[Callable[[int, str], range]] = None,
) -> None:
    """标准化的网格子图绘制函数"""
    n_features = min(data.shape[1], len(title_list))
    cols = samples_per_row * 2  # 每个样本占2列（左散点，右密度）
    rows = max(1, min(max_rows, math.ceil(n_features / samples_per_row)))
    max_samples = rows * samples_per_row
    n = min(n_features, max_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.35), sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, cols)
    
    data_to_csv: dict[str, np.ndarray] = {"sample_idx": np.arange(data.shape[0])}
    global_scatter_ymin, global_scatter_ymax = np.inf, -np.inf
    global_density_ymax = 0.0
    scatter_axes: list[plt.Axes] = []
    density_axes: list[plt.Axes] = []
    
    # 进度迭代器
    if progress_callback:
        iter_range = progress_callback(rows * samples_per_row, "绘图进度")
    else:
        iter_range = range(rows * samples_per_row)
    
    # 绘制所有子图
    for idx in iter_range:
        sample_index = idx
        row = idx // samples_per_row
        col_sample = idx % samples_per_row
        scatter_ax = axes[row, col_sample * 2]
        density_ax = axes[row, col_sample * 2 + 1]
        
        if sample_index >= n:
            scatter_ax.axis("off")
            density_ax.axis("off")
            continue
        
        series = data[:, sample_index]
        sample_idx_arr = np.arange(series.shape[0])
        
        # 左边：绘制散点图
        scatter_ax.scatter(sample_idx_arr, series, s=scatter_size, color=scatter_color, alpha=0.7, linewidths=0)
        if show_grid:
            scatter_ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        scatter_ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
        scatter_ax.set_title(title_list[sample_index], fontsize=9, pad=2)
        
        # 收集散点图的Y轴范围
        clean = series[np.isfinite(series)]
        if clean.size:
            global_scatter_ymin = min(global_scatter_ymin, float(clean.min()))
            global_scatter_ymax = max(global_scatter_ymax, float(clean.max()))
        
        # 右边：绘制密度图（统计图）
        if clean.size == 0:
            density_ax.axis("off")
            continue
        
        grid_low, grid_high = np.percentile(clean, [0.5, 99.5])
        if np.isclose(grid_low, grid_high):
            spread = max(abs(grid_low) * 0.05, 1e-3)
            grid_low -= spread
            grid_high += spread
        grid = np.linspace(grid_low, grid_high, 512)
        
        xs, ys = safe_kde(series, grid)
        if xs is None:
            density_ax.axis("off")
            continue
        
        hist_vals, _, _ = density_ax.hist(
            series,
            bins=PREPROCESS_HIST_BINS,
            density=True,
            color=density_color,
            alpha=PREPROCESS_HIST_ALPHA,
            edgecolor="black",
            linewidth=0.5,
        )
        density_ax.fill_between(xs, ys, color=density_color, alpha=PREPROCESS_HIST_ALPHA)
        density_ax.plot(xs, ys, color=density_color, lw=kde_linewidth)
        density_ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        if show_grid:
            density_ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        density_ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
        
        ymax_local = max(
            (hist_vals.max() if hist_vals.size else 0.0), (ys.max() if ys is not None else 0.0)
        )
        global_density_ymax = max(global_density_ymax, ymax_local)
        
        scatter_axes.append(scatter_ax)
        density_axes.append(density_ax)
        data_to_csv[f"feature_{sample_index}"] = series
    
    # 统一设置所有散点图的Y轴范围
    if np.isfinite(global_scatter_ymin) and np.isfinite(global_scatter_ymax) and global_scatter_ymin < global_scatter_ymax:
        pad = 0.02 * (global_scatter_ymax - global_scatter_ymin)
        for ax in scatter_axes:
            ax.set_ylim(global_scatter_ymin - pad, global_scatter_ymax + pad)
            ax.set_xlim(0, max(0, data.shape[0] - 1))
    
    # 统一设置所有密度图的Y轴范围
    if global_density_ymax > 0:
        for ax in density_axes:
            ax.set_ylim(0, global_density_ymax * 1.05)
    
    # 如果指定了密度图的X轴范围
    if density_xlim is not None:
        for ax in density_axes:
            ax.set_xlim(density_xlim)
    
    # 隐藏除了第一行第一列外的所有刻度标签
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if not (r == 0 and c in (0, 1)):
                ax.set_xticklabels([])
                ax.set_yticklabels([])
    
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    save_figure(fig, output_dir=output_dir, name=name, data=data_to_csv)


# ========================================
# 辅助函数
# ========================================

def _handle_remove_error(func, path, exc_info):
    """处理删除文件时的权限错误"""
    err = exc_info[1]
    if isinstance(err, PermissionError):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            func(path)
        except Exception:
            raise err
    else:
        raise err


def _remove_path(path: str) -> None:
    """安全删除文件或目录"""
    if not os.path.exists(path):
        return
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, onerror=_handle_remove_error)
        else:
            os.remove(path)
    except PermissionError:
        print(f"[清理] 警告: 无法删除 {path}（权限不足）。请关闭相关程序后重试。")


def _prepare_output_workspace(output_dir: str, remove_raw: bool) -> None:
    """清理输出目录"""
    if os.path.exists(output_dir):
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                # 如果需要删除原始数据，或者不是原始数据文件，就删除
                if remove_raw or not item.endswith('_raw.npz'):
                    _remove_path(item_path)
        except Exception as e:
            print(f"[清理] 警告: 清理 {output_dir} 时出错: {e}")
    
    os.makedirs(output_dir, exist_ok=True)


def process_dataset(
    data_type: str,
    source_dir: str,
    output_subdir: str,
    v_ids: list[int],
    num_points: int,
) -> None:
    """处理单个数据集"""
    
    print("\n" + "=" * 60)
    print(f"处理验证数据集: {data_type.upper()}")
    print("=" * 60)
    print(f"数据源目录: {source_dir}")
    print(f"输出目录: {output_subdir}")
    print(f"最大处理文件夹数: {PREPROCESS_MAX_FOLDERS if PREPROCESS_MAX_FOLDERS else '不限制'}")
    print(f"数据收集: {'是' if DO_COLLECT else '否'}")
    print(f"数据变换: {'是' if DO_TRANSFORM else '否'} ({TRANSFORM_KIND if DO_TRANSFORM else 'N/A'})")
    print(f"可视化: {'是' if DO_VISUALIZE else '否'}")
    print("=" * 60)
    
    # 输出路径
    processed_data_path_npz = os.path.join(output_subdir, "preprocessed_data.npz")
    raw_data_path_npz = os.path.join(output_subdir, "preprocessed_data_raw.npz")
    transforms_path = os.path.join(output_subdir, "transforms.joblib")
    
    # ========================================
    # 步骤 1：数据收集
    # ========================================
    V_data = None
    if DO_COLLECT:
        print(f"\n[步骤1] 数据收集 - {data_type}")
        print("-" * 60)
        
        if not os.path.isdir(source_dir):
            print(f"警告: 数据源目录不存在: {source_dir}")
            print(f"跳过 {data_type} 数据集的处理")
            return

        # 清理输出目录
        _prepare_output_workspace(output_subdir, remove_raw=DO_TRANSFORM)

        # 获取文件夹列表
        folders = []
        for f in os.listdir(source_dir):
            fp = os.path.join(source_dir, f)
            if not os.path.isdir(fp):
                continue
            if f.isdigit():
                folders.append(f)
                continue
            # 兼容非数字命名
            try:
                has_npy = any(x.lower().endswith('.npy') for x in os.listdir(fp))
                has_csv = any(x.lower().endswith('.csv') for x in os.listdir(fp))
                if (USE_NPY and has_npy) or ((not USE_NPY) and has_csv):
                    folders.append(f)
            except Exception:
                continue
        
        folders = sorted(folders, key=lambda x: int(x) if x.isdigit() else 0)
        total_folders = len(folders)
        if PREPROCESS_MAX_FOLDERS is not None:
            folders = folders[:PREPROCESS_MAX_FOLDERS]
        
        print(f"找到 {total_folders} 个文件夹, 处理前 {len(folders)} 个")

        total_samples = len(folders)
        V_data = np.zeros((total_samples, num_points))

        sample_idx = 0
        skipped_folders = []
        required_ids_set = set(v_ids)

        for folder in tqdm(folders, desc=f"收集进度({data_type})"):
            try:
                folder_path = os.path.join(source_dir, folder)
                expected_file = "iteration.npy" if USE_NPY else "iteration.csv"
                file_path = os.path.join(folder_path, expected_file)
                
                if not os.path.exists(file_path):
                    ext = ".npy" if USE_NPY else ".csv"
                    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
                    if len(files) == 1:
                        file_path = os.path.join(folder_path, files[0])
                    else:
                        skipped_folders.append(folder)
                        continue

                if USE_NPY:
                    arr = np.load(file_path)
                    cols = (
                        ["Element Label", "S-Mises"]
                        if arr.shape[1] == 2
                        else ["Element Label", "S-Mises", "X", "Y", "Z"][: arr.shape[1]]
                    )
                    df = pd.DataFrame(arr, columns=cols).set_index("Element Label")
                else:
                    df = pd.read_csv(file_path, skipinitialspace=True).set_index("Element Label")

                actual_labels_set = set(df.index)
                missing_required_ids = required_ids_set - actual_labels_set
                if missing_required_ids:
                    raise ValueError(f"缺少必要的ID: {sorted(list(missing_required_ids))}")

                V_data[sample_idx] = df.loc[v_ids]["S-Mises"].values
                sample_idx += 1

            except Exception:
                skipped_folders.append(folder)
                continue

        if sample_idx < total_samples:
            print(f"\n成功处理 {sample_idx} / {len(folders)} 个文件夹")
            V_data = V_data[:sample_idx]

        if skipped_folders:
            print(f"跳过 {len(skipped_folders)} 个文件夹")

        # 保存原始数据
        np.savez(raw_data_path_npz, V=V_data)
        print(f"已保存原始数据到: {raw_data_path_npz}")
        print(f"数据形状: {V_data.shape}")

    else:
        print(f"\n[步骤1] 跳过数据收集，加载已有数据 - {data_type}")
        print("-" * 60)
        
        if os.path.exists(raw_data_path_npz):
            V_data = np.load(raw_data_path_npz)["V"]
            print(f"已从 {raw_data_path_npz} 加载数据")
            print(f"数据形状: {V_data.shape}")
        else:
            print(f"警告: 未找到原始数据文件: {raw_data_path_npz}")
            print(f"跳过 {data_type} 数据集的处理")
            return

    # ========================================
    # 步骤 2：数据变换
    # ========================================
    Vt: Optional[np.ndarray] = None
    tf: Optional[FittedTransforms] = None

    if DO_TRANSFORM:
        print(f"\n[步骤2] 数据变换 - {data_type}")
        print("-" * 60)
        
        tf = fit_transforms(V_data, kind=TRANSFORM_KIND, random_state=SEED)
        tf.save(transforms_path)
        print(f"已保存变换器到: {transforms_path}")
        
        Vt = apply_transforms(V_data, tf)
        np.savez(processed_data_path_npz, V=Vt)
        print(f"已保存变换后数据到: {processed_data_path_npz}")
        print(f"变换后数据形状: {Vt.shape}")
        
        if tf.kind == "robust-gauss":
            print(f"变换器维度: {tf.scaler_V.center_.shape}")
    else:
        print(f"\n[步骤2] 跳过数据变换 - {data_type}")
        print("-" * 60)
        Vt = V_data
        print("将直接使用原始数据")

    # ========================================
    # 步骤 3：可视化
    # ========================================
    if not DO_VISUALIZE:
        print(f"\n跳过可视化 - {data_type}")
    else:
        print(f"\n[步骤3] 数据可视化 - {data_type}")
        print("-" * 60)
        
        apply_plot_style()
        
        target = "V"
        compare_mode = DO_TRANSFORM and Vt is not None and tf is not None
        id_list = v_ids
        
        def _progress_iter(count: int, desc: str):
            return tqdm(range(count), desc=desc) if SHOW_PLOT_PROGRESS else range(count)
        
        # 准备标题列表
        title_list = [f"{target}[{i}]  ID {id_list[i]}" for i in range(len(id_list))]
        
        if compare_mode:
            # 变换模式：绘制变换后的数据
            print(f"绘制变换后数据（前{min(PREPROCESS_PLOT_FIRST_N, Vt.shape[1])}个维度）")
            plot_grid_scatter_density(
                data=Vt,
                title_list=title_list,
                output_dir=output_subdir,
                name=f"preprocess_{target}_first_{min(PREPROCESS_PLOT_FIRST_N, Vt.shape[1])}_combined_{PREPROCESS_GRID_ROWS}x{PREPROCESS_GRID_COLS}",
                samples_per_row=PREPROCESS_GRID_COLS,
                max_rows=PREPROCESS_GRID_ROWS,
                scatter_color=PREPROCESS_SCATTER_COLOR,
                density_color=PREPROCESS_DENSITY_COLOR,
                scatter_size=PREPROCESS_SCATTER_SIZE,
                density_xlim=PREPROCESS_DENSITY_XLIM,
                show_grid=PREPROCESS_SHOW_GRID,
                kde_linewidth=PREPROCESS_KDE_LINEWIDTH,
                progress_callback=_progress_iter if SHOW_PLOT_PROGRESS else None,
            )
        else:
            # 非变换模式：绘制原始数据
            print(f"绘制原始数据（前{min(PREPROCESS_PLOT_FIRST_N, V_data.shape[1])}个维度）")
            plot_grid_scatter_density(
                data=V_data,
                title_list=title_list,
                output_dir=output_subdir,
                name=f"preprocess_{target}_first_{min(PREPROCESS_PLOT_FIRST_N, V_data.shape[1])}_combined_{PREPROCESS_GRID_ROWS}x{PREPROCESS_GRID_COLS}",
                samples_per_row=PREPROCESS_GRID_COLS,
                max_rows=PREPROCESS_GRID_ROWS,
                scatter_color=PREPROCESS_SCATTER_COLOR,
                density_color=PREPROCESS_DENSITY_COLOR,
                scatter_size=PREPROCESS_SCATTER_SIZE,
                density_xlim=None,  # 原始数据自动范围
                show_grid=PREPROCESS_SHOW_GRID,
                kde_linewidth=PREPROCESS_KDE_LINEWIDTH,
                progress_callback=_progress_iter if SHOW_PLOT_PROGRESS else None,
            )
        
        print(f"可视化图像已保存到: {output_subdir}")

    # ========================================
    # 总结输出
    # ========================================
    print("\n" + "=" * 60)
    print(f"数据集 {data_type.upper()} 处理完成！")
    print("=" * 60)

    only_visualize = DO_VISUALIZE and not DO_COLLECT and not DO_TRANSFORM
    if only_visualize:
        print("仅执行了可视化：")
        print(f"- 图像与CSV: {output_subdir}")
    else:
        print("输出文件：")
        if os.path.exists(raw_data_path_npz):
            if DO_TRANSFORM:
                print(f"- 原始数据: {raw_data_path_npz}")
            else:
                print(f"- 数据文件(原始): {raw_data_path_npz}")
                print(f"  ↳ 注意: DO_TRANSFORM=False，后续流程将使用此文件")
        
        if DO_TRANSFORM:
            if os.path.exists(transforms_path):
                print(f"- 变换器: {transforms_path}")
            if os.path.exists(processed_data_path_npz):
                print(f"- 变换后数据: {processed_data_path_npz}")
                print(f"  ↳ 后续流程将使用此文件")
        
        if DO_VISUALIZE:
            print(f"- 可视化图像: {output_subdir}/*.png")

    print("=" * 60)


# ========================================
# 主程序
# ========================================

# 解析命令行参数
parser = argparse.ArgumentParser(
    description='预处理验证数据集（health/crack/corrosion/multi）',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
  python 07_preprocess_validation_data_main.py                       # 处理所有数据集
  python 07_preprocess_validation_data_main.py --datasets multi      # 只处理multi数据集
  python 07_preprocess_validation_data_main.py --datasets health crack  # 处理health和crack
    """
)
parser.add_argument(
    '--datasets',
    nargs='+',
    choices=['health', 'crack', 'corrosion', 'multi'],
    help='指定要处理的数据集类型（可指定多个）。不指定则处理所有数据集。'
)
args = parser.parse_args()

# 根据命令行参数设置处理标志
if args.datasets:
    # 如果指定了数据集，则只处理指定的
    PROCESS_HEALTH = 'health' in args.datasets
    PROCESS_CRACK = 'crack' in args.datasets
    PROCESS_CORROSION = 'corrosion' in args.datasets
    PROCESS_MULTI = 'multi' in args.datasets
else:
    # 如果没有指定，则处理所有数据集
    PROCESS_HEALTH = True
    PROCESS_CRACK = True
    PROCESS_CORROSION = True
    PROCESS_MULTI = True

print("=" * 60)
print("流程07：验证数据预处理")
print("=" * 60)
print("说明：批量预处理四类验证数据（health/crack/corrosion/multi）")
print("=" * 60)

# 初始化随机种子
random.seed(SEED)
np.random.seed(SEED)

# 读取测点ID配置文件（由流程02生成）
measures_csv_path = os.path.join(os.path.dirname(__file__), MEASURES_ID_CSV)
if not os.path.exists(measures_csv_path):
    print(f"\n错误: 测点ID文件未找到: {measures_csv_path}")
    print("请先执行流程02 (02_inp_to_vtu_main.py) 以生成此文件。")
    raise FileNotFoundError(f"测点ID文件不存在: {measures_csv_path}")

measures_df = pd.read_csv(measures_csv_path)
v_col = RECONSTRUCT_COLUMN if RECONSTRUCT_COLUMN in measures_df.columns else "all_measures"
v_ids = measures_df[v_col].dropna().astype(int).tolist()
num_points = len(v_ids)
print(f"\n重构维度: {num_points} (列 '{v_col}')")

# 数据集配置
datasets = []
if PROCESS_HEALTH:
    datasets.append({
        'type': 'health',
        'source': SOURCE_DATA_HEALTH,
        'output': os.path.join(OUTPUT_DIR, 'health'),
    })
if PROCESS_CRACK:
    datasets.append({
        'type': 'crack',
        'source': SOURCE_DATA_CRACK,
        'output': os.path.join(OUTPUT_DIR, 'crack'),
    })
if PROCESS_CORROSION:
    datasets.append({
        'type': 'corrosion',
        'source': SOURCE_DATA_CORROSION,
        'output': os.path.join(OUTPUT_DIR, 'corrosion'),
    })
if PROCESS_MULTI:
    datasets.append({
        'type': 'multi',
        'source': SOURCE_DATA_MULTI,
        'output': os.path.join(OUTPUT_DIR, 'multi'),
    })

if not datasets:
    print("\n警告: 未选择任何数据集进行处理")
    print("请在配置区将 PROCESS_HEALTH、PROCESS_CRACK、PROCESS_CORROSION 或 PROCESS_MULTI 设置为 True")
else:
    print(f"\n将处理以下数据集: {', '.join([d['type'] for d in datasets])}")

# 处理每个数据集
for dataset in datasets:
    process_dataset(
        data_type=dataset['type'],
        source_dir=dataset['source'],
        output_subdir=dataset['output'],
        v_ids=v_ids,
        num_points=num_points,
    )

# ========================================
# 总体完成总结
# ========================================
print("\n" + "=" * 80)
print("流程07全部完成！")
print("=" * 80)
print("处理的数据集：")
for dataset in datasets:
    print(f"  - {dataset['type']}: {dataset['output']}")
print("=" * 80)
