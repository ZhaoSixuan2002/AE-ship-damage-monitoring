"""
流程05：确定阈值（阈值方法研发）
功能：基于训练好的自编码器模型和验证集，确定逐维残差阈值
依赖：
    - 03_preprocess_training_data_output/preprocessed_data_raw.npz（或preprocessed_data.npz）
    - 04_train_model_output/autoencoder.pth
    - 04_train_model_output/validation_indices.csv
输出：
    - 05_determine_thresholds_output/val_dim_stats.csv（每个维度的统计信息和阈值）
    - 05_determine_thresholds_output/val_recon_errors.csv（所有验证样本的逐维残差）
    - 05_determine_thresholds_output/val_tau_by_dimension.png（逐维阈值汇总图）
    - 05_determine_thresholds_output/val_tau_by_dimension.csv（对应数据）
    - 05_determine_thresholds_output/val_residuals_by_dim_5x10.png（随机25维的残差分布图）
    - 05_determine_thresholds_output/val_residuals_by_dim_5x10.csv（对应数据）
    - 05_determine_thresholds_output/val_samples_combined_5x10.png（随机25个样本的残差分布图）
    - 05_determine_thresholds_output/val_samples_combined_5x10.csv（对应数据）
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from typing import Tuple, Optional
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# ========================================
# 参数配置区（按自然逻辑顺序）
# ========================================

# --- 1. 目录配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "05_determine_thresholds_output")

# 输入依赖
PREPROCESS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "03_preprocess_training_data_output")
TRAIN_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "04_train_model_output")

# --- 2. 阈值计算方法配置 ---
# 支持三种方法：
# - "quantile_abs": 基于|残差|的分位数（默认）
# - "kstd_abs": 基于标准差的k-sigma原则（tau = k * std）
# - "mean_kstd": 基于均值和标准差（tau = |mean| + k * std）
TAU_METHOD = "kstd_abs"  # 可选: "quantile_abs", "kstd_abs", "mean_kstd"
VAL_QUANTILE_BASE = 0.995  # quantile_abs方法的分位数（0~1）
TAU_KSTD_K = 3.0  # kstd_abs和mean_kstd方法的k值（通常取3~5）

# --- 3. 可视化配置 ---
RANDOM_SEED = 42  # 随机选择的种子
SHOW_DIM_PLOT = True  # 是否生成随机25维度的详细图
SHOW_SAMPLE_PLOT = True  # 是否生成随机25样本的详细图
PREPROCESS_HIST_BINS = 100  # 直方图bins数量
PREPROCESS_SCATTER_SIZE = 1.5  # 散点大小
PREPROCESS_DENSITY_COLOR = "tab:orange"  # 密度图颜色
PREPROCESS_SCATTER_COLOR = "tab:blue"  # 散点图颜色

# --- 4. 数据变换配置 ---
# 是否使用预处理时的变换（若为True，需要transforms.joblib文件）
DO_TRANSFORM = False

# --- 5. 设备配置 ---
# 自动检测CUDA或使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 6. 模型架构配置（必须与训练时一致） ---
AE_ENCODER_DIMS = [768, 384, 192]  # 编码器隐藏层维度
AE_LATENT_DIM = 192  # 潜空间维度
AE_DECODER_DIMS = [192, 384, 768]  # 解码器隐藏层维度
AE_DROPOUT = 0.0  # Dropout比例
AE_ACTIVATION = 'relu'  # 激活函数

# --- 7. 批处理配置 ---
# 批量预测时的batch size
AE_BATCH_SIZE = 512

# --- 8. 绘图风格配置（学术风格） ---
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
# 内嵌工具函数
# ========================================

def apply_style():
    """应用学术论文风格"""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(fig, name: str, data: Optional[dict] = None):
    """保存图片和对应数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存图片
    fig_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] 图片: {fig_path}")
    
    # 保存对应数据
    if data is not None:
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        print(f"[保存] 数据: {csv_path}")


def safe_kde(data, grid):
    """安全的KDE计算，处理异常情况"""
    try:
        if len(data) < 3:
            return None, None
        clean = data[np.isfinite(data)]
        if len(clean) < 3:
            return None, None
        
        # 使用Scott规则估计带宽
        kde = stats.gaussian_kde(clean, bw_method='scott')
        ys = kde(grid)
        return grid, ys
    except Exception:
        return None, None


class FittedTransforms:
    """数据变换器类（内嵌实现）"""
    
    def __init__(self, method: str = "none"):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.q01 = None
        self.q99 = None
    
    @staticmethod
    def load(path: str):
        """加载变换器"""
        return joblib.load(path)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆变换"""
        if self.method == "none":
            return X
        elif self.method == "standardize":
            return X * self.std + self.mean
        elif self.method == "minmax":
            return X * (self.max - self.min) + self.min
        elif self.method == "robust":
            return X * (self.q99 - self.q01) + self.q01
        else:
            return X


def inverse_transforms(X: np.ndarray, tf: FittedTransforms) -> np.ndarray:
    """便捷函数：逆变换"""
    return tf.inverse_transform(X)


# ========================================
# 自编码器模型定义
# ========================================

class Autoencoder(nn.Module):
    """自编码器模型（与训练时相同的结构）"""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list,
        latent_dim: int,
        decoder_dims: list,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }.get(activation, nn.ReLU)

        # 编码器
        enc = []
        prev = input_dim
        for h in encoder_dims:
            enc += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                enc += [nn.Dropout(dropout)]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        # 解码器
        dec = []
        prev = latent_dim
        for h in decoder_dims:
            dec += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                dec += [nn.Dropout(dropout)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

        # Xavier 初始化线性层
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ========================================
# 数据加载函数
# ========================================

def load_vector_data() -> np.ndarray:
    """加载预处理后的向量数据
    
    优先加载preprocessed_data_raw.npz（未归一化版本）
    若不存在则加载preprocessed_data.npz（归一化版本）
    """
    raw_path = os.path.join(PREPROCESS_OUTPUT_DIR, "preprocessed_data_raw.npz")
    processed_path = os.path.join(PREPROCESS_OUTPUT_DIR, "preprocessed_data.npz")
    
    if os.path.exists(raw_path):
        data = np.load(raw_path, allow_pickle=True)
        print(f"[数据] 加载原始数据: {raw_path}")
    elif os.path.exists(processed_path):
        data = np.load(processed_path, allow_pickle=True)
        print(f"[数据] 加载处理后数据: {processed_path}")
    else:
        raise FileNotFoundError(f"未找到预处理数据文件:\n{raw_path}\n{processed_path}")
    
    V = data["V"]
    print(f"[数据] 加载完成: V.shape={V.shape}")
    return V


def load_model(input_dim: int, device: torch.device) -> Autoencoder:
    """加载训练好的模型"""
    model_path = os.path.join(TRAIN_OUTPUT_DIR, "autoencoder.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    # 创建模型（使用配置的架构参数）
    model = Autoencoder(
        input_dim=input_dim,
        encoder_dims=AE_ENCODER_DIMS,
        latent_dim=AE_LATENT_DIM,
        decoder_dims=AE_DECODER_DIMS,
        dropout=AE_DROPOUT,
        activation=AE_ACTIVATION
    )
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"[模型] 加载完成: {model_path}")
    print(f"[模型] 配置: input_dim={input_dim}, encoder_dims={AE_ENCODER_DIMS}, latent_dim={AE_LATENT_DIM}")
    
    return model


def load_validation_indices(total_samples: int) -> list:
    """加载验证集索引
    
    优先从validation_indices.csv加载
    若不存在则使用默认划分（后20%）
    """
    val_csv = os.path.join(TRAIN_OUTPUT_DIR, "validation_indices.csv")
    
    if os.path.exists(val_csv):
        val_idx = pd.read_csv(val_csv)["Validation Index"].astype(int).tolist()
        print(f"[验证集] 从文件加载: {val_csv}")
    else:
        # 回退：使用默认划分（后20%）
        val_ratio = 0.2
        val_size = max(1, int(total_samples * val_ratio))
        val_idx = list(range(total_samples - val_size, total_samples))
        print(f"[验证集] 使用默认划分: 后{val_ratio*100:.0f}%")
    
    print(f"[验证集] 样本数: {len(val_idx)}")
    return val_idx


# ========================================
# 核心计算函数
# ========================================

def compute_val_residuals(
    V: np.ndarray, 
    model: Autoencoder, 
    device: torch.device,
    val_indices: list,
    tf: Optional[FittedTransforms] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算验证集的残差
    
    返回:
        residuals: [Nv, D] 残差矩阵 (pred - true)
        true_vals: [Nv, D] 真实值
        pred_vals: [Nv, D] 预测值
    """
    print("\n" + "=" * 60)
    print("计算验证集残差")
    print("=" * 60)
    
    Xv = torch.from_numpy(V[val_indices].astype(np.float32)).to(device)
    Nv, D = Xv.shape
    
    # 批量预测
    print(f"[预测] 开始批量预测: {Nv}个样本, batch_size={AE_BATCH_SIZE}")
    preds = []
    with torch.no_grad():
        for i in range(0, Nv, AE_BATCH_SIZE):
            if i % (AE_BATCH_SIZE * 10) == 0:
                print(f"[预测] 进度: {i}/{Nv}")
            batch = Xv[i : i + AE_BATCH_SIZE]
            recon = model(batch)
            preds.append(recon.cpu().numpy())
    
    pred_arr = np.concatenate(preds, axis=0)
    true_arr = Xv.cpu().numpy()
    
    print(f"[预测] 完成: pred.shape={pred_arr.shape}, true.shape={true_arr.shape}")
    
    # 可选：逆变换到原始尺度
    if tf is not None and DO_TRANSFORM:
        print("[变换] 应用逆变换到原始尺度")
        true_plot = inverse_transforms(true_arr, tf)
        pred_plot = inverse_transforms(pred_arr, tf)
    else:
        true_plot = true_arr
        pred_plot = pred_arr
    
    # 计算残差
    residuals = pred_plot - true_plot
    print(f"[残差] 计算完成: residuals.shape={residuals.shape}")
    print(f"[残差] 统计: mean={np.mean(residuals):.6f}, std={np.std(residuals):.6f}")
    
    return residuals, true_plot, pred_plot


def compute_dimension_thresholds(
    residuals: np.ndarray,
    method: str = "kstd_abs",
    quantile: float = 0.995,
    k: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算每个维度的阈值
    
    参数:
        residuals: [Nv, D] 残差矩阵
        method: 阈值计算方法
            - "quantile_abs": 基于|残差|的分位数
            - "kstd_abs": 基于标准差的k-sigma原则（tau = k * std）
            - "mean_kstd": 基于均值和标准差（tau = |mean| + k * std）
        quantile: quantile_abs方法的分位数
        k: kstd_abs和mean_kstd方法的k值
    
    返回:
        mean: [D] 每个维度的均值
        std: [D] 每个维度的标准差
        tau: [D] 每个维度的阈值
    """
    print("\n" + "=" * 60)
    print(f"计算逐维阈值: 方法={method}")
    print("=" * 60)
    
    D = residuals.shape[1]
    
    # 计算基本统计量
    mean = np.nanmean(residuals, axis=0)
    std = np.nanstd(residuals, axis=0, ddof=1)
    
    print(f"[统计] mean范围: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"[统计] std范围: [{std.min():.6f}, {std.max():.6f}]")
    
    # 根据方法计算阈值
    if method == "quantile_abs":
        # 方法1：基于|残差|的分位数
        abs_res = np.abs(residuals)
        tau = np.nanquantile(abs_res, quantile, axis=0)
        print(f"[阈值] 使用分位数法: q={quantile}")
    elif method == "kstd_abs":
        # 方法2：基于标准差的k-sigma原则
        tau = k * std
        print(f"[阈值] 使用k-sigma法: k={k}")
    elif method == "mean_kstd":
        # 方法3：基于均值和标准差
        abs_mean = np.abs(mean)
        tau = abs_mean + k * std
        print(f"[阈值] 使用均值+k-sigma法: k={k}")
    else:
        raise ValueError(f"不支持的阈值方法: {method}")
    
    print(f"[阈值] tau范围: [{tau.min():.6f}, {tau.max():.6f}]")
    print(f"[阈值] tau统计: mean={tau.mean():.6f}, std={tau.std():.6f}")
    
    return mean, std, tau


# ========================================
# 可视化函数
# ========================================

def plot_random25_dims(
    residuals: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tau: np.ndarray,
    seed: int = 42
):
    """绘制随机25个维度的残差分布图（5x10网格）
    
    左列：样本索引 vs 残差散点图
    右列：残差密度图（直方图+KDE）+ 统计信息
    """
    print("\n" + "=" * 60)
    print("绘制随机25维度详细图")
    print("=" * 60)
    
    apply_style()
    
    Nv, D = residuals.shape
    rng = random.Random(seed)
    dims = sorted(rng.sample(range(D), min(25, D)))
    
    print(f"[可视化] 随机选择的25个维度: {dims}")
    
    rows = 5
    cols_per_dim = 2  # 每个维度2列：散点+密度
    total_cols = 5 * cols_per_dim  # 每行5个维度
    
    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 1.35, rows * 1.35), sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, total_cols)
    
    # 全局y轴范围
    global_scatter_ymin, global_scatter_ymax = np.inf, -np.inf
    global_density_ymax = 0.0
    scatter_axes = []
    density_axes = []
    
    # 用于保存数据的字典
    scatter_data = {}
    density_data = {}
    
    for i, dim in enumerate(dims):
        row_idx = i // 5
        col_offset = (i % 5) * 2
        ax_scatter = axes[row_idx, col_offset]
        ax_density = axes[row_idx, col_offset + 1]
        
        r = residuals[:, dim]
        sample_indices = np.arange(Nv)
        
        # 保存散点数据
        scatter_data[f"dim_{dim}_sample_idx"] = sample_indices
        scatter_data[f"dim_{dim}_residual"] = r
        
        # 左侧散点图：sample_index vs residual
        ax_scatter.scatter(sample_indices, r, s=1.5, color="tab:blue", alpha=0.7, linewidths=0)
        ax_scatter.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax_scatter.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax_scatter.tick_params(axis="both", which="both", direction="in", labelsize=8)
        ax_scatter.set_title(f"Dim {dim}", fontsize=9, pad=2)
        
        clean = r[np.isfinite(r)]
        if clean.size:
            global_scatter_ymin = min(global_scatter_ymin, float(clean.min()))
            global_scatter_ymax = max(global_scatter_ymax, float(clean.max()))
        
        # 右侧密度图：KDE + 统计信息
        if clean.size == 0:
            ax_density.axis("off")
        else:
            low, high = np.percentile(clean, [0.5, 99.5])
            if np.isclose(low, high):
                spread = max(abs(low) * 0.05, 1e-3)
                low -= spread
                high += spread
            grid = np.linspace(low, high, 512)
            xs, ys = safe_kde(clean, grid)
            
            if xs is None:
                ax_density.axis("off")
            else:
                # 直方图
                hist_vals, _, _ = ax_density.hist(
                    clean,
                    bins=PREPROCESS_HIST_BINS,
                    density=True,
                    color="tab:orange",
                    alpha=0.2,
                    edgecolor="black",
                    linewidth=0.5,
                )
                
                # KDE曲线
                ax_density.fill_between(xs, ys, color="tab:orange", alpha=0.2)
                ax_density.plot(xs, ys, color="tab:orange", lw=0.6)
                ax_density.axhline(0, color="black", lw=0.5, alpha=0.4)
                ax_density.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                ax_density.tick_params(axis="both", which="both", direction="in", labelsize=8)
                
                # 保存密度数据
                density_data[f"dim_{dim}_kde_x"] = xs
                density_data[f"dim_{dim}_kde_y"] = ys
                
                # 计算统计信息
                mu = float(mean[dim])
                sig = float(std[dim])
                var = sig ** 2
                tau_val = float(tau[dim])
                
                # 上下阈值边界
                upper_val = mu + tau_val
                lower_val = mu - tau_val
                
                # 绘制均值和阈值线
                ax_density.axvline(mu, color="tab:blue", linestyle="-", linewidth=0.4, alpha=0.8)
                ax_density.axvline(upper_val, color="red", linestyle="--", linewidth=0.4, alpha=0.9)
                ax_density.axvline(lower_val, color="red", linestyle=":", linewidth=0.4, alpha=0.9)
                
                # 文本标注
                txt = f"μ={mu:.3g}\nσ²={var:.3g}\nτ={tau_val:.3g}\nUpper={upper_val:.3g}\nLower={lower_val:.3g}"
                ax_density.text(0.98, 0.98, txt, transform=ax_density.transAxes, 
                              ha="right", va="top", fontsize=4,
                              bbox=dict(boxstyle="round,pad=0.15", fc="white", 
                                      ec="gray", alpha=0.8, linewidth=0.4))
                
                ymax_local = max((hist_vals.max() if hist_vals.size else 0.0), 
                               (ys.max() if ys is not None else 0.0))
                global_density_ymax = max(global_density_ymax, ymax_local)
                density_axes.append(ax_density)
        
        scatter_axes.append(ax_scatter)
    
    # 统一y轴范围
    if np.isfinite(global_scatter_ymin) and np.isfinite(global_scatter_ymax) and global_scatter_ymin < global_scatter_ymax:
        pad = 0.02 * (global_scatter_ymax - global_scatter_ymin)
        for ax in scatter_axes:
            ax.set_ylim(global_scatter_ymin - pad, global_scatter_ymax + pad)
            ax.set_xlim(0, max(0, Nv - 1))
    
    if global_density_ymax > 0:
        for ax in density_axes:
            ax.set_ylim(0, global_density_ymax * 1.05)
    
    # 隐藏除第一行第一列外的刻度标签
    rows_total, cols_total = axes.shape
    for r in range(rows_total):
        for c in range(cols_total):
            if not (r == 0 and c in (0, 1)):
                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
    
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    
    # 保存图片和散点数据（不包含KDE数据，因为长度不一致）
    save_figure(fig, name="val_residuals_by_dim_5x10", data=scatter_data)


def plot_random25_samples(residuals: np.ndarray, seed: int = 42):
    """绘制随机25个样本的残差分布图（5x10网格）
    
    格式与preprocess一致：左散点（维度索引 vs 残差）+ 右统计（直方图+KDE）
    residuals: [Nv, D] - 验证样本的残差矩阵
    """
    print("\n" + "=" * 60)
    print("绘制随机25样本详细图")
    print("=" * 60)
    
    apply_style()
    
    Nv, D = residuals.shape
    rng = random.Random(seed)
    sample_indices = sorted(rng.sample(range(Nv), min(25, Nv)))
    
    print(f"[可视化] 随机选择的25个样本: {sample_indices}")
    
    rows = 5
    cols_per_sample = 2  # 每个样本2列：散点+密度
    total_cols = 5 * cols_per_sample  # 每行5个样本
    
    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 1.35, rows * 1.35), sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, total_cols)
    
    # 全局y轴范围
    global_scatter_ymin, global_scatter_ymax = np.inf, -np.inf
    global_density_ymax = 0.0
    scatter_axes = []
    density_axes = []
    
    # 用于保存数据的字典
    data_to_save = {"dim_idx": np.arange(D)}
    
    for i, sample_idx in enumerate(sample_indices):
        row_idx = i // 5
        col_offset = (i % 5) * 2
        ax_scatter = axes[row_idx, col_offset]
        ax_density = axes[row_idx, col_offset + 1]
        
        r = residuals[sample_idx, :]  # 该样本在所有维度的残差
        dim_indices = np.arange(D)
        
        # 保存该样本的数据
        data_to_save[f"sample_{sample_idx}"] = r
        
        # 左侧散点图：dimension_index vs residual
        ax_scatter.scatter(dim_indices, r, s=PREPROCESS_SCATTER_SIZE, 
                          color=PREPROCESS_SCATTER_COLOR, alpha=0.7, linewidths=0)
        ax_scatter.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax_scatter.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax_scatter.tick_params(axis="both", which="both", direction="in", labelsize=8)
        ax_scatter.set_title(f"Sample {sample_idx}", fontsize=9, pad=2)
        
        clean = r[np.isfinite(r)]
        if clean.size:
            global_scatter_ymin = min(global_scatter_ymin, float(clean.min()))
            global_scatter_ymax = max(global_scatter_ymax, float(clean.max()))
        
        # 右侧密度图：KDE + 直方图
        if clean.size == 0:
            ax_density.axis("off")
        else:
            low, high = np.percentile(clean, [0.5, 99.5])
            if np.isclose(low, high):
                spread = max(abs(low) * 0.05, 1e-3)
                low -= spread
                high += spread
            grid = np.linspace(low, high, 512)
            xs, ys = safe_kde(clean, grid)
            
            if xs is None:
                ax_density.axis("off")
            else:
                # 直方图
                hist_vals, _, _ = ax_density.hist(
                    clean,
                    bins=PREPROCESS_HIST_BINS,
                    density=True,
                    color=PREPROCESS_DENSITY_COLOR,
                    alpha=0.2,
                    edgecolor="black",
                    linewidth=0.5,
                )
                
                # KDE曲线
                ax_density.fill_between(xs, ys, color=PREPROCESS_DENSITY_COLOR, alpha=0.2)
                ax_density.plot(xs, ys, color=PREPROCESS_DENSITY_COLOR, lw=0.6)
                ax_density.axhline(0, color="black", lw=0.5, alpha=0.4)
                ax_density.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                ax_density.tick_params(axis="both", which="both", direction="in", labelsize=8)
                
                ymax_local = max((hist_vals.max() if hist_vals.size else 0.0), 
                               (ys.max() if ys is not None else 0.0))
                global_density_ymax = max(global_density_ymax, ymax_local)
                density_axes.append(ax_density)
        
        scatter_axes.append(ax_scatter)
    
    # 统一y轴范围
    if np.isfinite(global_scatter_ymin) and np.isfinite(global_scatter_ymax) and global_scatter_ymin < global_scatter_ymax:
        pad = 0.02 * (global_scatter_ymax - global_scatter_ymin)
        for ax in scatter_axes:
            ax.set_ylim(global_scatter_ymin - pad, global_scatter_ymax + pad)
            ax.set_xlim(0, max(0, D - 1))
    
    if global_density_ymax > 0:
        for ax in density_axes:
            ax.set_ylim(0, global_density_ymax * 1.05)
    
    # 隐藏除第一行第一列外的刻度标签
    rows_total, cols_total = axes.shape
    for r in range(rows_total):
        for c in range(cols_total):
            if not (r == 0 and c in (0, 1)):
                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
    
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    
    save_figure(fig, name="val_samples_combined_5x10", data=data_to_save)


def plot_tau_summary(mean: np.ndarray, tau: np.ndarray):
    """绘制阈值汇总图"""
    print("\n" + "=" * 60)
    print("绘制逐维阈值汇总图")
    print("=" * 60)
    
    apply_style()
    
    D = tau.shape[0]
    upper_threshold = mean + tau
    lower_threshold = mean - tau
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(np.arange(D), upper_threshold, color="darkgreen", linewidth=1.5, 
           label="Upper Threshold (μ+τ)")
    ax.plot(np.arange(D), lower_threshold, color="green", linewidth=1.5, 
           label="Lower Threshold (μ-τ)")
    ax.fill_between(np.arange(D), lower_threshold, upper_threshold, 
                    alpha=0.15, color='tab:blue', label='Threshold Range')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    
    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Threshold Value")
    ax.set_title("Per-Dimension Threshold Boundaries")
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    
    fig.tight_layout()
    
    save_figure(fig, name="val_tau_by_dimension", 
               data={"dim": np.arange(D), 
                     "upper_threshold": upper_threshold, 
                     "lower_threshold": lower_threshold, 
                     "tau": tau})


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主程序"""
    print("\n" + "=" * 60)
    print("流程05：确定阈值（阈值方法研发）")
    print("=" * 60)
    print(f"[配置] 阈值方法: {TAU_METHOD}")
    if TAU_METHOD == "quantile_abs":
        print(f"[配置] 分位数: {VAL_QUANTILE_BASE}")
    else:
        print(f"[配置] k值: {TAU_KSTD_K}")
    print(f"[配置] 设备: {DEVICE}")
    print(f"[配置] 输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    V = load_vector_data()
    N, D = V.shape
    
    # 2. 加载模型
    model = load_model(input_dim=D, device=DEVICE)
    
    # 3. 加载验证集索引
    val_indices = load_validation_indices(N)
    
    # 4. 加载可选的变换器
    tf_path = os.path.join(PREPROCESS_OUTPUT_DIR, "transforms.joblib")
    tf = None
    if DO_TRANSFORM and os.path.exists(tf_path):
        tf = FittedTransforms.load(tf_path)
        print(f"[变换] 加载变换器: {tf_path}")
    elif DO_TRANSFORM:
        print(f"[警告] 未找到变换器文件: {tf_path}，将使用原始数据")
    
    # 5. 计算验证集残差
    residuals, true_vals, pred_vals = compute_val_residuals(V, model, DEVICE, val_indices, tf)
    
    # 6. 保存逐维残差
    print("\n" + "=" * 60)
    print("保存逐维残差")
    print("=" * 60)
    residuals_df = pd.DataFrame(residuals, columns=[f"dim_{i}" for i in range(D)])
    residuals_csv = os.path.join(OUTPUT_DIR, "val_recon_errors.csv")
    residuals_df.to_csv(residuals_csv, index=False)
    print(f"[保存] 逐维残差: {residuals_csv}")
    
    # 7. 计算逐维阈值
    mean, std, tau = compute_dimension_thresholds(
        residuals, 
        method=TAU_METHOD, 
        quantile=VAL_QUANTILE_BASE, 
        k=TAU_KSTD_K
    )
    
    # 8. 保存统计信息和阈值
    print("\n" + "=" * 60)
    print("保存统计信息和阈值")
    print("=" * 60)
    
    upper_threshold = mean + tau
    lower_threshold = mean - tau
    
    # 保存详细统计信息（包含所有需要的信息，不再需要单独的dimension_thresholds.csv）
    stats_df = pd.DataFrame({
        "dim": np.arange(D),
        "mean": mean,
        "std": std,
        "var": std ** 2,
        "tau": tau,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
    })
    stats_csv = os.path.join(OUTPUT_DIR, "val_dim_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[保存] 统计信息和阈值: {stats_csv}")
    
    # 9. 可视化：随机25维度详细图
    if SHOW_DIM_PLOT:
        plot_random25_dims(residuals, mean, std, tau, seed=RANDOM_SEED)
    
    # 10. 可视化：随机25样本详细图（格式与preprocess一致）
    if SHOW_SAMPLE_PLOT:
        plot_random25_samples(residuals, seed=RANDOM_SEED)
    
    # 11. 可视化：逐维阈值汇总图
    plot_tau_summary(mean, tau)
    
    # 12. 输出摘要
    print("\n" + "=" * 60)
    print("阈值确定完成")
    print("=" * 60)
    print(f"[摘要] 总维度数: {D}")
    print(f"[摘要] 验证样本数: {len(val_indices)}")
    print(f"[摘要] 阈值方法: {TAU_METHOD}")
    print(f"[摘要] 阈值范围: [{tau.min():.6f}, {tau.max():.6f}]")
    print(f"[摘要] 阈值均值: {tau.mean():.6f}")
    print(f"[摘要] 阈值标准差: {tau.std():.6f}")
    print(f"\n[输出] 所有结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
