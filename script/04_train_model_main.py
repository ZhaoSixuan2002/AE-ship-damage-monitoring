"""
流程04：模型训练主脚本
功能：训练自编码器模型，评估重构性能，计算损伤识别阈值

依赖：流程03必须先执行，以生成预处理后的数据文件

输入：
    - 03_preprocess_training_data_output/preprocessed_data_raw.npz (原始数据，DO_TRANSFORM=False时)
    - 03_preprocess_training_data_output/preprocessed_data.npz (变换后数据，DO_TRANSFORM=True时)
    - 03_preprocess_training_data_output/transforms.joblib (变换器，可选)
输出：
    - 04_train_model_output/autoencoder.pth (最佳模型权重)
    - 04_train_model_output/validation_indices.csv (验证集样本索引)
    - 04_train_model_output/training_losses.csv (训练/验证损失记录)
    - 04_train_model_output/training_validation_loss_combined.png (损失曲线-左线性右对数)
    - 04_train_model_output/reconstruction_multi_samples_*.png (重构对比网格图)
    - 04_train_model_output/val_residuals_random25_5x10.png (随机25维残差可视化)
    - 04_train_model_output/val_tau_by_dimension.png (逐维阈值汇总图)
    - 04_train_model_output/val_dim_stats.csv (逐维统计与阈值)
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 1. 路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PREPROCESS_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '03_preprocess_training_data_output')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', '04_train_model_output')

# --- 2. 数据源配置 ---
DO_TRANSFORM = False                                        # 使用变换后数据(True)或原始数据(False)
PROCESSED_NPZ_PATH = os.path.join(PREPROCESS_OUTPUT_DIR, 'preprocessed_data.npz')     # 变换后数据
RAW_NPZ_PATH = os.path.join(PREPROCESS_OUTPUT_DIR, 'preprocessed_data_raw.npz')       # 原始数据
TRANSFORMS_PATH = os.path.join(PREPROCESS_OUTPUT_DIR, 'transforms.joblib')            # 变换器

# --- 3. 模型架构参数 ---
AE_ENCODER_DIMS = [768, 384, 192]                           # 编码器隐藏层维度
AE_LATENT_DIM = 192                                         # 潜空间维度
AE_DECODER_DIMS = [192, 384, 768]                           # 解码器隐藏层维度
AE_DROPOUT = 0.0                                            # Dropout比例（0表示禁用）
AE_ACTIVATION = 'relu'                                      # 激活函数：'relu', 'gelu', 'tanh', 'sigmoid'

# --- 4. 训练超参数 ---
AE_BATCH_SIZE = 256                                         # 批大小
AE_EPOCHS = 2000                                            # 最大训练轮数
AE_LR = 0.0003                                              # 学习率
AE_WEIGHT_DECAY = 0.0                                       # 权重衰减（L2正则化）
AE_GRAD_CLIP = 1.0                                          # 梯度裁剪阈值（0表示禁用）
AE_TRAIN_SPLIT_RATIO = 0.9                                  # 训练集比例

# --- 5. 训练控制 ---
TRAIN_SHUFFLE = False                                       # 是否打乱训练数据
SEED = 42                                                   # 随机种子
RUN_TRAIN = True                                            # 执行训练
RUN_EVAL = False                                            # 执行评估（独立运行）
AUTO_TEST_AFTER_TRAIN = False                               # 训练后自动测试

# --- 6. 评估参数 ---
TEST_SAMPLE_INDICES = 'last'                                # 测试样本索引：'auto'(前10+后10), 'last'(最后N个), 'all', 或逗号分隔的索引如'0,5,10'
TEST_GRID_ROWS = 5                                          # 测试结果网格行数
TEST_GRID_COLS = 5                                          # 测试结果网格列数
VAL_QUANTILE_BASE = 0.995                                   # 验证分位数基准（仅在TAU_METHOD='quantile_abs'时使用）

# --- 7. 阈值计算方法配置 ---
TAU_METHOD = 'kstd_abs'                                     # 阈值计算方法：'quantile_abs'(分位数), 'kstd_abs'(均值+k倍标准差)
TAU_KSTD_K = 4.0                                            # kstd_abs方法的k值（使用4-sigma阈值）

# --- 8. 可视化参数 ---
FIG_DPI = 300                                               # 图片DPI
PREPROCESS_SCATTER_COLOR = 'tab:blue'                       # 散点图颜色
PREPROCESS_DENSITY_COLOR = 'tab:orange'                     # 密度图颜色
PREPROCESS_SCATTER_SIZE = 1.5                               # 散点大小
PREPROCESS_DENSITY_XLIM = None                              # 密度图X轴范围（None表示自动）
PREPROCESS_KDE_LINEWIDTH = 0.5                              # KDE曲线线宽
PREPROCESS_HIST_BINS = 100                                  # 柱状图桶数
PREPROCESS_HIST_ALPHA = 0.20                                # 柱状图透明度

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
# 内嵌工具函数：绘图
# ========================================

def apply_style():
    """应用学术风绘图样式"""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(
    fig: plt.Figure,
    name: str,
    subdir: str | None = None,
    data: dict | None = None,
):
    """保存图像和同名数据源
    - fig: Matplotlib Figure
    - name: 文件名（不含扩展名）
    - subdir: 可选子目录名（相对于OUTPUT_DIR）
    - data: 可选字典，将以CSV形式保存
    """
    out_dir = OUTPUT_DIR if subdir is None else os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(img_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    if data:
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)


def safe_kde(data: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """安全的核密度估计
    
    Args:
        data: 输入数据数组
        grid: 可选的网格点，如果未提供则自动生成
        
    Returns:
        (xs, density): 网格点和对应的密度值，如果无法计算则返回 (None, None)
    """
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


def inverse_transforms(Vt: np.ndarray | None, tf: FittedTransforms) -> np.ndarray | None:
    """逆变换：从变换后空间恢复到原始空间"""
    if Vt is None:
        return None
    if tf.kind == "none":
        return Vt
    if tf.kind == "minmax":
        return tf.scaler_V.inverse_transform(Vt)
    if tf.kind == "robust-gauss":
        Z1 = tf.gauss_V.inverse_transform(Vt)
        return tf.scaler_V.inverse_transform(Z1)
    raise ValueError(tf.kind)


# ========================================
# 模型定义
# ========================================

class Autoencoder(nn.Module):
    """最基础的多层感知机自编码器"""

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        latent_dim: int,
        decoder_dims: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }.get(activation, nn.ReLU)

        # 编码器
        enc: list[nn.Module] = []
        prev = input_dim
        for h in encoder_dims:
            enc += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                enc += [nn.Dropout(dropout)]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        # 解码器
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in decoder_dims:
            dec += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                dec += [nn.Dropout(dropout)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

        # Xavier 初始化线性层
        def _init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ========================================
# 数据加载
# ========================================

def load_vector_data() -> np.ndarray:
    """根据 DO_TRANSFORM 配置，加载合适的预处理矩阵 V（N×D）。
    
    - DO_TRANSFORM=True: 加载 preprocessed_data.npz（变换后的数据）
    - DO_TRANSFORM=False: 加载 preprocessed_data_raw.npz（原始尺度的数据）
    """
    if DO_TRANSFORM:
        path = PROCESSED_NPZ_PATH
        data_type = "preprocessed_data.npz"
    else:
        path = RAW_NPZ_PATH
        data_type = "preprocessed_data_raw.npz"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到预处理文件: {path}\n请先运行流程03生成 {data_type}")
    
    data = np.load(path)
    V = data["V"]
    print(f"[加载] 已从 {data_type} 加载数据，形状: {V.shape}")
    return V


# ========================================
# 训练流程
# ========================================

def train() -> None:
    """最基础训练流程：加载 V，划分训练/验证，MSE 训练，保存最佳权重与日志。"""
    print("\n" + "=" * 60)
    print("流程04：模型训练")
    print("=" * 60)
    
    # 清理旧输出
    def clean_old():
        print("正在清理旧输出...")
        if os.path.exists(OUTPUT_DIR):
            try:
                shutil.rmtree(OUTPUT_DIR)
            except Exception as e:
                print(f"[警告] 清理输出目录失败: {e}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("清理完成。\n")

    clean_old()
    apply_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 数据
    V = load_vector_data().astype(np.float32, copy=False)
    N, D = V.shape
    print(f"数据维度: N={N}, D={D}")
    if V.ndim != 2 or not np.isfinite(V).all():
        raise ValueError("V 必须是二维且不含 NaN/Inf")

    # 划分训练/验证（末尾作为验证集，保持时间顺序）
    val_ratio = 1.0 - AE_TRAIN_SPLIT_RATIO
    train_size = int(N * AE_TRAIN_SPLIT_RATIO)  # 精确的90%
    val_size = N - train_size  # 剩余的作为验证集
    X_train = torch.from_numpy(V[:train_size]).to(device)
    X_val = torch.from_numpy(V[train_size:]).to(device)
    val_indices = list(range(train_size, N))
    
    # 保存验证集索引
    pd.DataFrame({"Validation Index": val_indices}).to_csv(
        os.path.join(OUTPUT_DIR, "validation_indices.csv"), index=False
    )
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    # 模型/优化器/损失
    model = Autoencoder(D, AE_ENCODER_DIMS, AE_LATENT_DIM, AE_DECODER_DIMS, AE_DROPOUT, AE_ACTIVATION).to(device)
    optimizer = Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train), batch_size=AE_BATCH_SIZE, shuffle=TRAIN_SHUFFLE)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=AE_BATCH_SIZE, shuffle=False)

    print(f"\n开始训练 (共 {AE_EPOCHS} 轮)...")
    print("-" * 60)
    
    best_val = float("inf")
    train_losses, val_losses = [], []
    
    for epoch in range(AE_EPOCHS):
        model.train()
        running = 0.0
        count = 0
        for (xb,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            if AE_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), AE_GRAD_CLIP)
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        train_epoch = running / max(1, count)

        model.eval()
        vrunning = 0.0
        vcount = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                recon = model(xb)
                vloss = criterion(recon, xb)
                vrunning += vloss.item() * xb.size(0)
                vcount += xb.size(0)
        val_epoch = vrunning / max(1, vcount)

        train_losses.append(train_epoch)
        val_losses.append(val_epoch)
        
        # 每轮都打印日志
        print(f"Epoch {epoch+1:4d}/{AE_EPOCHS} | Train Loss: {train_epoch:.6f} | Val Loss: {val_epoch:.6f}")

        if val_epoch + 1e-9 < best_val:
            best_val = val_epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "autoencoder.pth"))

    print("-" * 60)
    print(f"训练完成！最佳验证损失: {best_val:.6f}")

    # 保存损失日志
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
        os.path.join(OUTPUT_DIR, "training_losses.csv"), index=False
    )
    
    xs = list(range(1, len(train_losses) + 1))
    
    # 合并图：左边线性损失，右边对数损失
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：线性损失
    ax1.plot(xs, train_losses, label="Train", color="tab:blue", linewidth=2)
    ax1.plot(xs, val_losses, label="Validation", color="tab:orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc='best')
    ax1.tick_params(axis="both", direction="in")
    
    # 右图：对数损失
    ax2.plot(xs, train_losses, label="Train", color="tab:blue", linewidth=2)
    ax2.plot(xs, val_losses, label="Validation", color="tab:orange", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_yscale("log")
    ax2.set_title("Training and Validation Loss (Log Scale)")
    ax2.legend(loc='best')
    ax2.tick_params(axis="both", direction="in")
    
    fig.tight_layout()
    save_figure(
        fig,
        name="training_validation_loss_combined",
        data={"epoch": xs, "train_loss": train_losses, "val_loss": val_losses},
    )

    print(f"\n训练曲线已保存到: {OUTPUT_DIR}")


# ========================================
# 评估流程
# ========================================

def _load_best_model(D: int, device: torch.device) -> Autoencoder:
    """加载最佳模型权重"""
    model = Autoencoder(D, AE_ENCODER_DIMS, AE_LATENT_DIM, AE_DECODER_DIMS, AE_DROPOUT, AE_ACTIVATION).to(device)
    w = os.path.join(OUTPUT_DIR, "autoencoder.pth")
    if not os.path.exists(w):
        raise FileNotFoundError(f"未找到模型权重: {w}")
    model.load_state_dict(torch.load(w, map_location=device))
    model.eval()
    return model


def _compute_val_threshold(
    V: np.ndarray, model: Autoencoder, device: torch.device
) -> Tuple[float, np.ndarray]:
    """在验证集上计算每样本重构MSE，并返回 base 分位阈值与误差数组。"""
    val_csv = os.path.join(OUTPUT_DIR, "validation_indices.csv")
    if os.path.exists(val_csv):
        val_idx = pd.read_csv(val_csv)["Validation Index"].astype(int).tolist()
    else:
        N = V.shape[0]
        val_ratio = 1.0 - AE_TRAIN_SPLIT_RATIO
        val_size = max(1, int(N * val_ratio))
        val_idx = list(range(N - val_size, N))

    Xv = torch.from_numpy(V[val_idx].astype(np.float32)).to(device)
    errs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xv.shape[0], AE_BATCH_SIZE):
            b = Xv[i : i + AE_BATCH_SIZE]
            r = model(b)
            e = torch.mean((b - r) ** 2, dim=1).cpu().numpy()
            errs.append(e)
    recon_errors = np.concatenate(errs, axis=0)
    tau_base = float(np.quantile(recon_errors, VAL_QUANTILE_BASE))
    return tau_base, recon_errors


def _score_single_sample(v: np.ndarray, model: Autoencoder, device: torch.device) -> Tuple[np.ndarray, float]:
    """返回：逐维误差 s_i（MSE）与样本总体 MSE=S。"""
    x = torch.from_numpy(v[None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        recon = model(x)
        sqerr = (x - recon) ** 2
        s_i = sqerr.squeeze(0).cpu().numpy()
        S = float(np.mean(s_i))
    return s_i, S


def _get_test_sample_indices(total_samples: int) -> list[int]:
    """根据配置确定测试样本索引列表"""
    indices_str = TEST_SAMPLE_INDICES.lower()
    max_samples = TEST_GRID_ROWS * TEST_GRID_COLS
    
    if indices_str == 'auto':
        # 默认：前10个 + 后10个
        first_10 = list(range(min(10, total_samples)))
        last_10 = list(range(max(0, total_samples - 10), total_samples))
        # 合并并去重
        indices = sorted(set(first_10 + last_10))
    elif indices_str == 'last':
        # 最后N个样本（N = 网格容量）
        start_idx = max(0, total_samples - max_samples)
        indices = list(range(start_idx, total_samples))
    elif indices_str == 'all':
        # 全部样本
        indices = list(range(total_samples))
    else:
        # 逗号分隔的索引
        try:
            indices = [int(x.strip()) for x in TEST_SAMPLE_INDICES.split(',') if x.strip()]
            indices = [i for i in indices if 0 <= i < total_samples]
        except:
            print(f"[警告] 无法解析 TEST_SAMPLE_INDICES='{TEST_SAMPLE_INDICES}'，使用默认 last 模式")
            start_idx = max(0, total_samples - max_samples)
            indices = list(range(start_idx, total_samples))
    
    # 限制最大数量为网格大小
    if len(indices) > max_samples:
        print(f"[警告] 测试样本数({len(indices)})超过网格容量({max_samples})，截取后{max_samples}个")
        indices = indices[-max_samples:]  # 取最后N个
    
    return indices


def _plot_multi_sample_grid(
    test_indices: list[int],
    all_true_vals: list[np.ndarray],
    all_pred_vals: list[np.ndarray],
    all_S_values: list[float],
    tau_base: float,
    is_original_scale: bool
) -> None:
    """绘制多样本网格图：每个样本两列（左：true vs pred 散点；右：残差密度直方图）

    说明：
    - 左侧子图延续原有"预测-真值"对比散点样式（细点、细网格、y=x参考线）。
    - 右侧子图参照预处理组合图样式，绘制残差 r = pred - true 的密度分布直方图，叠加KDE曲线与填充。
    - 所有密度子图的Y轴范围统一，X轴范围可由 PREPROCESS_DENSITY_XLIM 控制。
    """
    n_samples = len(test_indices)
    rows = TEST_GRID_ROWS
    cols = TEST_GRID_COLS
    
    # 创建子图网格（每个样本2列：左scatter，右density）
    total_cols = cols * 2
    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 1.35, rows * 1.35), sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, total_cols)
    
    # 为每个样本绘制（左散点 + 右密度）
    scatter_csv = {}
    density_axes: list[plt.Axes] = []
    global_density_ymax = 0.0
    for plot_idx in range(rows * cols):
        row = plot_idx // cols
        col_sample = plot_idx % cols
        ax_scatter = axes[row, col_sample * 2]
        ax_density = axes[row, col_sample * 2 + 1]
        
        if plot_idx >= n_samples:
            ax_scatter.axis("off")
            ax_density.axis("off")
            continue
        
        sample_idx = test_indices[plot_idx]
        true_vals = all_true_vals[plot_idx]
        pred_vals = all_pred_vals[plot_idx]
        S = all_S_values[plot_idx]
        is_anom = S > tau_base
        
        # 计算当前样本的数据范围（用于y=x参考线）
        combined = np.concatenate([true_vals, pred_vals])
        clean = combined[np.isfinite(combined)]
        if clean.size:
            local_min = float(clean.min())
            local_max = float(clean.max())
        else:
            local_min, local_max = 0, 1
        
        # 绘制 true vs pred 散点（点更小更细）
        ax_scatter.scatter(true_vals, pred_vals, s=PREPROCESS_SCATTER_SIZE, color=PREPROCESS_SCATTER_COLOR, alpha=0.6, linewidths=0, edgecolors='none')
        
        # 绘制 y=x 参考线（更细）
        ax_scatter.plot([local_min, local_max], [local_min, local_max], 'r--', lw=0.8, alpha=0.5)
        
        # 网格线更细
        ax_scatter.grid(True, alpha=0.15, linestyle="--", linewidth=0.3)
        ax_scatter.tick_params(axis="both", which="both", direction="in", labelsize=7, width=0.5, length=2)
        
        # 标题显示样本索引和MSE（字体兼容性）
        anom_mark = " [!]" if is_anom else ""
        ax_scatter.set_title(f"#{sample_idx}{anom_mark} MSE={S:.3g}", fontsize=8, pad=2)
        
        # 设置当前子图的独立范围（根据自己的数据）
        if clean.size:
            pad = 0.05 * (local_max - local_min) if local_max > local_min else 0.1
            ax_scatter.set_xlim(local_min - pad, local_max + pad)
            ax_scatter.set_ylim(local_min - pad, local_max + pad)
        
        # 保存到CSV
        scatter_csv[f"sample_{sample_idx}_true"] = true_vals
        scatter_csv[f"sample_{sample_idx}_pred"] = pred_vals

        # 右侧：残差密度直方图 + KDE（参考预处理绘图样式）
        residuals = pred_vals - true_vals  # 假定残差定义为 pred - true
        residuals_clean = residuals[np.isfinite(residuals)]
        if residuals_clean.size == 0:
            ax_density.axis("off")
        else:
            # 构造KDE网格并绘制
            low_p, high_p = np.percentile(residuals_clean, [0.5, 99.5])
            if np.isclose(low_p, high_p):
                spread = max(abs(low_p) * 0.05, 1e-3)
                low_p -= spread
                high_p += spread
            grid = np.linspace(low_p, high_p, 512)

            xs, ys = safe_kde(residuals_clean, grid)
            if xs is None:
                ax_density.axis("off")
            else:
                hist_vals, _, _ = ax_density.hist(
                    residuals_clean,
                    bins=PREPROCESS_HIST_BINS,
                    density=True,
                    color=PREPROCESS_DENSITY_COLOR,
                    alpha=PREPROCESS_HIST_ALPHA,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax_density.fill_between(xs, ys, color=PREPROCESS_DENSITY_COLOR, alpha=PREPROCESS_HIST_ALPHA)
                ax_density.plot(xs, ys, color=PREPROCESS_DENSITY_COLOR, lw=PREPROCESS_KDE_LINEWIDTH)
                ax_density.axhline(0, color="black", lw=0.5, alpha=0.4)
                ax_density.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                ax_density.tick_params(axis="both", which="both", direction="in", labelsize=8)

                ymax_local = max((hist_vals.max() if hist_vals.size else 0.0), (ys.max() if ys is not None else 0.0))
                global_density_ymax = max(global_density_ymax, ymax_local)
                density_axes.append(ax_density)

        # 保存残差到CSV
        scatter_csv[f"sample_{sample_idx}_residual"] = residuals
    
    # 统一密度图的Y轴与可选X轴范围；控制刻度标签显示（减小拥挤）
    if global_density_ymax > 0:
        for ax_d in density_axes:
            ax_d.set_ylim(0, global_density_ymax * 1.05)
    if PREPROCESS_DENSITY_XLIM is not None and PREPROCESS_DENSITY_XLIM:
        for ax_d in density_axes:
            ax_d.set_xlim(PREPROCESS_DENSITY_XLIM)

    scale_label = "Original" if is_original_scale else "Transformed"
    
    # 添加整体的轴标签
    fig.text(0.5, 0.02, f'True Values ({scale_label})', ha='center', fontsize=10)
    fig.text(0.02, 0.5, f'Predicted Values ({scale_label})', va='center', rotation='vertical', fontsize=10)
    
    fig.tight_layout(pad=0.6, rect=[0.03, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    
    save_figure(
        fig,
        name=f"reconstruction_multi_samples_{rows}x{cols}",
        data=scatter_csv,
    )
    print(f"[评估] 已保存多样本重构对比图: reconstruction_multi_samples_{rows}x{cols}.png")


def evaluate() -> None:
    """基础评估：
    - 计算验证集重构误差分布并给出分位阈值 tau_base；
    - 对多个样本绘制 True vs Pred 网格散点图（参考 preprocess 样式）；
    - 新增：逐维残差统计与阈值可视化（随机25维组合图 + 全部维度τ汇总图）。
    """
    print("\n" + "=" * 60)
    print("流程04：模型评估")
    print("=" * 60)
    
    apply_style()
    V = load_vector_data().astype(np.float32, copy=False)
    N, D = V.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_best_model(D, device)

    # 载入可选变换器（用于反变换到原始尺度绘图）
    tf_path = TRANSFORMS_PATH
    tf = FittedTransforms.load(tf_path) if os.path.exists(tf_path) else None

    # 阈值基础统计（样本级MSE分布），保留CSV用于调试
    tau_base, recon_errors = _compute_val_threshold(V, model, device)
    pd.DataFrame({"recon_error": recon_errors}).to_csv(
        os.path.join(OUTPUT_DIR, "val_recon_errors.csv"), index=False
    )

    # 确定测试样本列表
    test_indices = _get_test_sample_indices(N)
    print(f"[评估] 测试样本索引: {test_indices[:5]}...{test_indices[-5:] if len(test_indices) > 5 else ''} (共{len(test_indices)}个)")
    
    # 计算所有测试样本的重构结果
    all_true_vals = []
    all_pred_vals = []
    all_S_values = []
    all_is_anom = []
    
    for idx in test_indices:
        s_i, S = _score_single_sample(V[idx], model, device)
        is_anom = S > tau_base
        all_S_values.append(S)
        all_is_anom.append(is_anom)
        
        # 获取真实值和预测值
        x = torch.from_numpy(V[idx][None, :].astype(np.float32)).to(device)
        with torch.no_grad():
            recon = model(x)
            true_vals = x.squeeze(0).cpu().numpy()
            pred_vals = recon.squeeze(0).cpu().numpy()
        
        # 如果需要反变换
        if tf is not None and DO_TRANSFORM:
            true_vals_plot = inverse_transforms(true_vals.reshape(1, -1), tf).flatten()
            pred_vals_plot = inverse_transforms(pred_vals.reshape(1, -1), tf).flatten()
        else:
            true_vals_plot = true_vals
            pred_vals_plot = pred_vals
        
        all_true_vals.append(true_vals_plot)
        all_pred_vals.append(pred_vals_plot)
    
    # 绘制多样本网格散点图（参考 preprocess 样式）
    _plot_multi_sample_grid(
        test_indices, all_true_vals, all_pred_vals, all_S_values, tau_base, 
        tf is not None and DO_TRANSFORM
    )
    
    # 新增：逐维残差统计与阈值(tau)可视化 ----------------------
    try:
        # 计算验证集索引
        val_csv = os.path.join(OUTPUT_DIR, "validation_indices.csv")
        if os.path.exists(val_csv):
            val_idx = pd.read_csv(val_csv)["Validation Index"].astype(int).tolist()
        else:
            val_ratio = 1.0 - AE_TRAIN_SPLIT_RATIO
            val_size = max(1, int(N * val_ratio))
            val_idx = list(range(N - val_size, N))

        # 批量预测（验证集）
        Xv = torch.from_numpy(V[val_idx]).to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, Xv.shape[0], AE_BATCH_SIZE):
                b = Xv[i : i + AE_BATCH_SIZE]
                r = model(b)
                preds.append(r.cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0)
        true_arr = Xv.cpu().numpy()

        # 可选：逆变换到原始尺度
        if tf is not None and DO_TRANSFORM:
            true_plot = inverse_transforms(true_arr, tf)
            pred_plot = inverse_transforms(pred_arr, tf)
        else:
            true_plot = true_arr
            pred_plot = pred_arr

        residuals = pred_plot - true_plot  # [Nv, D]

        # 逐维统计与阈值（使用配置的方法：kstd_abs 或 quantile_abs）
        mean_dim = np.nanmean(residuals, axis=0)
        std_dim = np.nanstd(residuals, axis=0, ddof=1)
        
        if TAU_METHOD == "kstd_abs":
            # 方法2：标准k-sigma原则，tau = k * std
            tau_dim = TAU_KSTD_K * std_dim
        else:
            # 方法1（默认旧逻辑）：按 |r| 的分位数
            tau_dim = np.nanquantile(np.abs(residuals), VAL_QUANTILE_BASE, axis=0)

        # 计算上下阈值边界：mean ± tau
        upper_threshold = mean_dim + tau_dim
        lower_threshold = mean_dim - tau_dim

        # 保存CSV
        pd.DataFrame(residuals, columns=[f"dim_{i}" for i in range(D)]).to_csv(
            os.path.join(OUTPUT_DIR, "val_residuals_by_dim.csv"), index=False
        )
        pd.DataFrame({
            "dim": np.arange(D),
            "mean": mean_dim,
            "std": std_dim,
            "var": std_dim ** 2,
            "tau": tau_dim,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
        }).to_csv(os.path.join(OUTPUT_DIR, "val_dim_stats.csv"), index=False)

        # 绘制随机25维（5x10）：左散点(样本索引-残差)，右密度(KDE), 标注 μ、σ²、τ
        def _plot_random25_dimensions(residuals: np.ndarray, mean: np.ndarray, std: np.ndarray, tau: np.ndarray, seed: int = 42):
            """随机选择25个维度，每个维度绘制两列：
            - 左列：所有验证样本在该维度的残差散点图 (sample_index vs residual)
            - 右列：该维度残差的KDE密度图 + 统计信息（μ, σ², τ）
            
            布局：5行×10列（5行，每行5个维度，每个维度占2列）
            """
            apply_style()
            
            Nv, Dlocal = residuals.shape
            rng = random.Random(seed)
            dims = sorted(rng.sample(range(Dlocal), min(25, Dlocal)))

            rows = 5
            cols_per_dim = 2  # 每个维度2列：散点+密度
            total_cols = 5 * cols_per_dim  # 每行5个维度
            
            fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 1.35, rows * 1.35), sharey=False)
            axes = np.atleast_2d(axes).reshape(rows, total_cols)

            global_scatter_ymin, global_scatter_ymax = np.inf, -np.inf
            global_density_ymax = 0.0
            scatter_axes = []
            density_axes = []

            for i, dim in enumerate(dims):
                row_idx = i // 5
                col_offset = (i % 5) * 2
                ax_scatter = axes[row_idx, col_offset]
                ax_density = axes[row_idx, col_offset + 1]

                r = residuals[:, dim]
                sample_indices = np.arange(Nv)
                
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
                        hist_vals, _, _ = ax_density.hist(
                            clean,
                            bins=PREPROCESS_HIST_BINS,
                            density=True,
                            color=PREPROCESS_DENSITY_COLOR,
                            alpha=PREPROCESS_HIST_ALPHA,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax_density.fill_between(xs, ys, color=PREPROCESS_DENSITY_COLOR, alpha=PREPROCESS_HIST_ALPHA)
                        ax_density.plot(xs, ys, color=PREPROCESS_DENSITY_COLOR, lw=PREPROCESS_KDE_LINEWIDTH)
                        ax_density.axhline(0, color="black", lw=0.5, alpha=0.4)
                        ax_density.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
                        ax_density.tick_params(axis="both", which="both", direction="in", labelsize=8)

                        mu = float(mean[dim])
                        sig = float(std[dim])
                        var = sig ** 2
                        
                        # 计算上下阈值的具体值：μ ± τ
                        upper_val = mu + tau[dim]
                        lower_val = mu - tau[dim]
                        
                        # 绘制统计线：均值 μ 与阈值边界 μ±τ（不带图例，统一放在文本框中）
                        ax_density.axvline(mu, color="tab:blue", linestyle="-", linewidth=0.4, alpha=0.8)
                        ax_density.axvline(upper_val, color="red", linestyle="--", linewidth=0.4, alpha=0.9)
                        ax_density.axvline(lower_val, color="red", linestyle=":", linewidth=0.4, alpha=0.9)
                        
                        # 统一的文本标注：μ、σ²、上下阈值合并在右上角
                        txt = f"μ={mu:.3g}\nσ²={var:.3g}\nUpper={upper_val:.3g}\nLower={lower_val:.3g}"
                        ax_density.text(0.98, 0.98, txt, transform=ax_density.transAxes, ha="right", va="top", fontsize=4,
                                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8, linewidth=0.4))

                        ymax_local = max((hist_vals.max() if hist_vals.size else 0.0), (ys.max() if ys is not None else 0.0))
                        global_density_ymax = max(global_density_ymax, ymax_local)
                        density_axes.append(ax_density)

                scatter_axes.append(ax_scatter)

            # 统一散点图Y轴范围
            if np.isfinite(global_scatter_ymin) and np.isfinite(global_scatter_ymax) and global_scatter_ymin < global_scatter_ymax:
                pad = 0.02 * (global_scatter_ymax - global_scatter_ymin)
                for ax in scatter_axes:
                    ax.set_ylim(global_scatter_ymin - pad, global_scatter_ymax + pad)
                    ax.set_xlim(-1, Nv)

            # 统一密度图Y轴范围
            if global_density_ymax > 0:
                for ax in density_axes:
                    ax.set_ylim(0, global_density_ymax * 1.05)

            # 隐藏除第一行外所有子图的刻度标签（减少拥挤）
            rows_total, cols_total = axes.shape
            for r in range(rows_total):
                for c in range(cols_total):
                    if r > 0:  # 只保留第一行的刻度标签
                        axes[r, c].set_xticklabels([])
                        axes[r, c].set_yticklabels([])

            fig.tight_layout(pad=0.6)
            fig.subplots_adjust(hspace=0.25, wspace=0.2)
            
            # 保存图形和数据
            save_data = {
                "dimension": np.array(dims),
                "mean": mean[dims],
                "std": std[dims],
                "tau": tau[dims]
            }
            save_figure(fig, name="val_residuals_random25_5x10", data=save_data)
            print("[评估] 已生成随机25维度残差可视化图 (5x10布局)")

        _plot_random25_dimensions(residuals, mean_dim, std_dim, tau_dim, seed=42)

        # τ汇总图 - 绘制上下阈值边界
        fig_tau, ax_tau = plt.subplots(figsize=(12, 4))
        ax_tau.plot(np.arange(D), upper_threshold, color="darkgreen", linewidth=1.5, label="Upper Threshold (μ+τ)")
        ax_tau.plot(np.arange(D), lower_threshold, color="green", linewidth=1.5, label="Lower Threshold (μ-τ)")
        ax_tau.fill_between(np.arange(D), lower_threshold, upper_threshold, alpha=0.15, color='tab:blue', label='Threshold Range')
        ax_tau.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax_tau.set_xlabel("Dimension Index")
        ax_tau.set_ylabel("Threshold Value")
        ax_tau.set_title("Per-Dimension Threshold Boundaries")
        ax_tau.legend(fontsize=9, loc='best')
        ax_tau.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        fig_tau.tight_layout()
        save_figure(fig_tau, name="val_tau_by_dimension", 
                    data={"dim": np.arange(D), "upper_threshold": upper_threshold, "lower_threshold": lower_threshold, "tau": tau_dim})
        print("[评估] 已生成逐维残差统计图与τ汇总图。")
    except Exception as e:
        print(f"[评估] 逐维残差评估出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"评估完成！结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


# ========================================
# 主程序
# ========================================

if __name__ == "__main__":
    # 支持命令行参数控制模式
    # python 04_train_model_main.py --mode=train  (仅训练)
    # python 04_train_model_main.py --mode=eval   (仅评估)
    # python 04_train_model_main.py --mode=both   (训练+评估)
    mode = "default"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--mode="):
                mode = arg.split("=")[1].lower()
    
    # 根据模式覆盖配置
    if mode == "train":
        do_train = True
        do_eval = False
    elif mode == "eval":
        do_train = False
        do_eval = True
    elif mode == "both":
        do_train = True
        do_eval = True
    else:
        # 使用配置文件的设置
        do_train = RUN_TRAIN
        do_eval = RUN_EVAL
    
    # 执行对应操作
    if do_train:
        train()
        if do_eval and not AUTO_TEST_AFTER_TRAIN:
            try:
                evaluate()
            except Exception as e:
                print(f"[错误] 评估失败: {e}")
                import traceback
                traceback.print_exc()
        elif AUTO_TEST_AFTER_TRAIN:
            try:
                evaluate()
            except Exception as e:
                print(f"[错误] 自动评估失败: {e}")
                import traceback
                traceback.print_exc()
    elif do_eval:
        # 仅测试：跳过训练直接评估
        try:
            evaluate()
        except Exception as e:
            print(f"[错误] 评估失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[Info] 训练与评估均未启用，脚本直接结束。")
