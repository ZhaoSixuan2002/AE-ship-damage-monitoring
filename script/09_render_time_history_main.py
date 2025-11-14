"""
流程09：时历动画渲染
功能：基于验证数据（crack/corrosion/multi/health）生成测点损伤可疑度随时间变化的热力图动画
依赖：流程04（模型训练）、流程07（验证数据预处理）
输入：
    - 04_train_model_output/autoencoder.pth
    - 07_preprocess_validation_data_output/{crack|corrosion|multi|health}/preprocessed_data_raw.npz
输出：
    - 09_render_time_history_output/
        - opacity_heatmap_combined_4types.png（四类数据横向合并的热力图）
        - opacity_animation_combined_2x2.gif（四类数据2x2布局合并的动画）
    - 09_render_time_history_output/{crack|corrosion|multi|health}/
        - opacity_data.csv（每个时间步每个测点的透明度数据）
        - exceed_history.csv（每个时间步每个测点的超阈值倍数历史）

说明：
    - 读取 crack/corrosion/multi/health 验证数据的所有样本（或前N个样本）
    - 对每个样本依次预测残差，计算超阈值倍数
    - 更新所有252个测点的透明度（滑动窗口平均方式）
    - 初始透明度100%，平均超阈值n倍则透明度减少n×衰减因子
    - 统一计算好所有数据后，直接生成合并的热力图和动画
    - 不再单独输出每种损伤类型的图片/GIF，只生成合并后的统一输出
    - 支持批量处理四种验证数据类型

工作流程：
    1. 加载模型和训练数据，计算阈值（共用）
    2. 计算所有数据类型的透明度时间序列（统一准备）
    3. 保存透明度CSV数据到各自子目录
    4. 直接生成合并的热力图（横向排列）
    5. 直接生成合并的动画（2x2布局）

使用方法：
    python 09_render_time_history_main.py                         # 渲染所有数据集
    python 09_render_time_history_main.py --datasets multi        # 只渲染multi数据集
    python 09_render_time_history_main.py --datasets crack multi  # 渲染crack和multi
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import argparse


# ========================================
# 参数配置区
# ========================================

# 路径配置（相对于script/目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_04 = os.path.join(SCRIPT_DIR, "04_train_model_output")  # 模型输入目录
OUTPUT_DIR_07 = os.path.join(SCRIPT_DIR, "07_preprocess_validation_data_output")  # 验证数据输入目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "09_render_time_history_output")  # 本脚本输出目录

# 输入文件
MODEL_PATH = os.path.join(OUTPUT_DIR_04, "autoencoder.pth")  # 模型文件
VALIDATION_DATA_TEMPLATE = os.path.join(OUTPUT_DIR_07, "{data_type}", "preprocessed_data_raw.npz")  # 验证数据模板

# 模型架构参数（必须与流程04一致）
AE_ENCODER_DIMS = [768, 384, 192]  # 编码器各隐藏层维度
AE_LATENT_DIM = 192  # 潜在空间维度
AE_DECODER_DIMS = [192, 384, 768]  # 解码器各隐藏层维度
AE_DROPOUT = 0.0  # Dropout概率
AE_ACTIVATION = "relu"  # 激活函数：'relu', 'elu', 'leaky_relu'

# 阈值计算参数（必须与流程08一致）
THRESHOLD_METHOD = "quantile_abs"  # 阈值方法：'quantile_abs', 'kstd_abs', 'mean_kstd'
THRESHOLD_QUANTILE = 0.95  # quantile_abs方法的分位数（0.90 ~ 0.99）
THRESHOLD_K_SIGMA = 3.0  # kstd_abs/mean_kstd方法的k倍标准差（1.0 ~ 5.0）

# 透明度计算参数
OPACITY_DECAY_FACTOR = 10.0  # 透明度衰减因子：平均超阈值1倍 = 减少10%透明度
OPACITY_WINDOW_SIZE = 10  # 滑动窗口大小（平滑超阈值倍数波动）

# 数据处理参数
MAX_SAMPLES = -1  # 最大处理样本数：-1表示全部，正整数表示处理前N个样本
# 注意：这些默认值会被命令行参数覆盖
PROCESS_DATA_TYPES = ["crack", "corrosion", "multi", "health"]  # 要处理的数据类型列表

# 热力图参数
GRID_ROWS = 12  # 热力图行数（252个测点 = 12行 × 21列）
GRID_COLS = 21  # 热力图列数
COLORMAP = "Reds"  # 颜色映射：'Reds', 'YlOrRd', 'OrRd'等

# 动画参数
ANIMATION_FPS = 10  # 帧率（每秒帧数）
SAVE_FORMAT = "gif"  # 保存格式：'mp4'（需要FFmpeg）或'gif'（通用，推荐用于合并）
ANIMATION_DPI = 100  # 动画分辨率

# 9x1子图热力图参数
HEATMAP_3X3_SAMPLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # 要展示的9个样本索引
HEATMAP_DPI = 300  # 热力图分辨率


# ========================================
# 内嵌工具函数
# ========================================

def apply_plot_style():
    """应用统一的绘图样式"""
    plt.rcParams.update({
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
    })


class Autoencoder(nn.Module):
    """自编码器模型（与流程04完全一致）"""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        latent_dim: int,
        decoder_dims: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 选择激活函数
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid
        elif activation == "elu":
            act_fn = nn.ELU
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建编码器
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(act_fn())
            if dropout and dropout > 0:  # 只有当dropout>0时才添加Dropout层
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 构建解码器
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in decoder_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(act_fn())
            if dropout and dropout > 0:  # 只有当dropout>0时才添加Dropout层
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def load_model(model_path: str, D: int, device: torch.device) -> Autoencoder:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        D: 输入维度
        device: 计算设备
        
    Returns:
        model: 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = Autoencoder(
        D, AE_ENCODER_DIMS, AE_LATENT_DIM, AE_DECODER_DIMS, AE_DROPOUT, AE_ACTIVATION
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Loaded] Model loaded from: {model_path}")
    return model


def load_validation_data(data_type: str) -> np.ndarray:
    """加载验证数据
    
    Args:
        data_type: 数据类型，'crack'|'corrosion'|'multi'|'health'
        
    Returns:
        V: 数据矩阵 [N, D]
    """
    data_type = data_type.lower().strip()
    if data_type not in {"crack", "corrosion", "multi", "health"}:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    data_path = VALIDATION_DATA_TEMPLATE.format(data_type=data_type)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation data not found for {data_type}: {data_path}")
    
    data = np.load(data_path)
    V = data["V"]
    print(f"[Loaded] {data_type} data shape: {V.shape}")
    return V


def compute_thresholds_from_validation(
    V_train: np.ndarray,
    model: Autoencoder,
    device: torch.device,
    method: str = "quantile_abs",
    quantile: float = 0.95,
    k: float = 3.0
) -> np.ndarray:
    """根据训练集验证样本计算每个维度的阈值
    
    Args:
        V_train: 训练数据 [N, D]
        model: 模型
        device: 设备
        method: 阈值方法
        quantile: quantile_abs方法的分位数
        k: kstd_abs/mean_kstd方法的k倍标准差
        
    Returns:
        thresholds: 阈值数组 [D]
    """
    # 批量预测
    V_tensor = torch.from_numpy(V_train.astype(np.float32)).to(device)
    with torch.no_grad():
        V_pred = model(V_tensor).cpu().numpy()
    
    # 计算残差
    residuals = V_pred - V_train  # [N, D]
    abs_residuals = np.abs(residuals)
    
    # 根据方法计算阈值
    if method == "quantile_abs":
        thresholds = np.quantile(abs_residuals, quantile, axis=0)
    elif method == "kstd_abs":
        thresholds = k * np.std(abs_residuals, axis=0)
    elif method == "mean_kstd":
        mean_res = np.mean(abs_residuals, axis=0)
        std_res = np.std(abs_residuals, axis=0)
        thresholds = mean_res + k * std_res
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
    
    print(f"[Computed] Thresholds using method '{method}': shape {thresholds.shape}")
    return thresholds


def compute_exceed_ratios(
    sample: np.ndarray,
    model: Autoencoder,
    device: torch.device,
    thresholds: np.ndarray
) -> np.ndarray:
    """计算单个样本每个维度超阈值的倍数
    
    Args:
        sample: 样本数据 [D]
        model: 模型
        device: 设备
        thresholds: 阈值 [D]
        
    Returns:
        exceed_ratios: 超阈值倍数 [D]，未超阈值的为0
    """
    x = torch.from_numpy(sample[None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(x)
        residuals = (pred - x).squeeze(0).cpu().numpy()
    
    # 计算超阈值倍数：|残差| / 阈值
    abs_residuals = np.abs(residuals)
    exceed_ratios = np.zeros_like(abs_residuals)
    
    # 只有超过阈值的才计算倍数
    exceed_mask = abs_residuals > thresholds
    exceed_ratios[exceed_mask] = abs_residuals[exceed_mask] / thresholds[exceed_mask]
    
    return exceed_ratios


def compute_opacity_time_series(
    V_val: np.ndarray,
    model: Autoencoder,
    device: torch.device,
    thresholds: np.ndarray,
    max_samples: int = -1,
    window_size: int = 10,
    decay_factor: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """计算所有样本的透明度时间序列
    
    Args:
        V_val: 验证数据 [N, D]
        model: 模型
        device: 设备
        thresholds: 阈值 [D]
        max_samples: 最大处理样本数，-1表示全部
        window_size: 滑动窗口大小
        decay_factor: 透明度衰减因子
        
    Returns:
        opacity_series: 透明度时间序列 [T, D]，T为时间步数
        exceed_history: 超阈值倍数历史 [T, D]
    """
    N, D = V_val.shape
    
    # 限制样本数
    if max_samples > 0 and max_samples < N:
        N = max_samples
        print(f"[Info] Processing first {N} samples (max_samples={max_samples})")
    
    # 初始化透明度和超阈值倍数历史
    opacity_series = []
    exceed_history = []
    
    # 初始透明度100%（全透明用1.0表示）
    current_opacity = np.ones(D) * 100.0
    
    # 存储最近window_size个样本的超阈值倍数（用于滑窗平均）
    exceed_window = []
    
    print(f"\n[Processing] Computing opacity time series for {N} samples with window size={window_size}...")
    
    for t in range(N):
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Processing sample {t+1}/{N}...")
        
        sample = V_val[t]
        
        # 计算当前样本的超阈值倍数
        exceed_ratios = compute_exceed_ratios(sample, model, device, thresholds)
        
        # 将当前超阈值倍数加入滑窗
        exceed_window.append(exceed_ratios)
        
        # 只保留最近window_size个样本
        if len(exceed_window) > window_size:
            exceed_window.pop(0)
        
        # 计算滑窗平均超阈值倍数
        avg_exceed_ratios = np.mean(exceed_window, axis=0)
        
        # 更新透明度：初始100% - 平均超阈值倍数 × 衰减因子
        # 透明度范围：0% (完全不透明，全红) ~ 100% (完全透明，不可见)
        current_opacity = 100.0 - avg_exceed_ratios * decay_factor
        
        # 限制透明度范围在 [0, 100]
        current_opacity = np.clip(current_opacity, 0.0, 100.0)
        
        # 记录当前时间步的透明度
        opacity_series.append(current_opacity.copy())
        exceed_history.append(avg_exceed_ratios.copy())
    
    opacity_series = np.array(opacity_series)  # [T, D]
    exceed_history = np.array(exceed_history)  # [T, D]
    
    print(f"[Completed] Opacity time series computed: shape {opacity_series.shape}")
    print(f"  - Window size: {window_size} samples")
    print(f"  - Min opacity: {opacity_series.min():.2f}%")
    print(f"  - Max opacity: {opacity_series.max():.2f}%")
    print(f"  - Mean opacity (final): {opacity_series[-1].mean():.2f}%")
    
    return opacity_series, exceed_history


def save_opacity_data(
    opacity_series: np.ndarray,
    exceed_history: np.ndarray,
    save_dir: str
):
    """保存透明度和超阈值倍数数据到CSV
    
    Args:
        opacity_series: [T, D]
        exceed_history: [T, D]
        save_dir: 保存目录
    """
    T, D = opacity_series.shape
    
    # 保存透明度数据
    opacity_df = pd.DataFrame(
        opacity_series,
        columns=[f"dim_{i}" for i in range(D)]
    )
    opacity_df.insert(0, "time_step", np.arange(T))
    
    opacity_csv_path = os.path.join(save_dir, "opacity_data.csv")
    opacity_df.to_csv(opacity_csv_path, index=False)
    print(f"[Saved] Opacity data saved to: {opacity_csv_path}")
    
    # 保存超阈值倍数历史
    exceed_df = pd.DataFrame(
        exceed_history,
        columns=[f"dim_{i}" for i in range(D)]
    )
    exceed_df.insert(0, "time_step", np.arange(T))
    
    exceed_csv_path = os.path.join(save_dir, "exceed_history.csv")
    exceed_df.to_csv(exceed_csv_path, index=False)
    print(f"[Saved] Exceed history saved to: {exceed_csv_path}")


def create_opacity_animation(
    opacity_series: np.ndarray,
    save_dir: str,
    title_prefix: str = "Time History Animation",
    fps: int = 10,
    save_format: str = "mp4",
    dpi: int = 100
):
    """创建透明度变化的热力图动画
    
    Args:
        opacity_series: 透明度时间序列 [T, D]
        save_dir: 保存目录
        title_prefix: 标题前缀
        fps: 帧率
        save_format: 保存格式 'mp4'或'gif'
        dpi: 分辨率
    """
    apply_plot_style()
    
    T, D = opacity_series.shape
    
    # 确保测点数量为252
    if D != 252:
        raise ValueError(f"Expected 252 dimensions, got {D}")
    
    # 将252个测点重塑为12×21网格
    opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    
    # 初始化热力图（使用第一帧）
    # 将透明度转换为显示值：opacity越低，颜色越深（越红）
    initial_display = 100.0 - opacity_grid[0]
    
    im = ax.imshow(
        initial_display,
        cmap=COLORMAP,
        aspect='equal',
        vmin=0,
        vmax=100,
        interpolation='nearest'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Damage Suspicion (100 - Opacity %)', fontsize=12)
    
    # 设置标题
    title = ax.set_title(f'{title_prefix} - Sample: 0/{T}', fontsize=14, fontweight='bold')
    
    # 移除刻度标签
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加网格线
    ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    def update(frame):
        """更新函数，用于动画"""
        display_value = 100.0 - opacity_grid[frame]
        im.set_array(display_value)
        title.set_text(f'{title_prefix} - Sample: {frame+1}/{T}')
        return [im, title]
    
    # 创建动画
    print(f"\n[Animation] Creating animation with {T} frames at {fps} FPS...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 / fps,
        blit=True,
        repeat=True
    )
    
    # 保存动画
    if save_format == 'mp4':
        output_path = os.path.join(save_dir, "opacity_animation.mp4")
        print(f"[Saving] Saving animation as MP4 (this may take a while)...")
        
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1800, codec='libx264')
            anim.save(output_path, writer=writer, dpi=dpi)
        except (RuntimeError, KeyError) as e:
            print(f"[Warning] FFmpeg not available: {e}")
            print(f"[Info] Falling back to GIF format...")
            
            output_path = os.path.join(save_dir, "opacity_animation.gif")
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
    
    elif save_format == 'gif':
        output_path = os.path.join(save_dir, "opacity_animation.gif")
        print(f"[Saving] Saving animation as GIF (this may take a while)...")
        
        Writer = animation.writers['pillow']
        writer = Writer(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    
    plt.close(fig)
    print(f"[Completed] Animation saved to: {output_path}")
    
    return output_path


def save_first_9_frames_grid(
    opacity_series: np.ndarray,
    save_dir: str
):
    """将前9帧组合成3行3列的大图并保存
    
    Args:
        opacity_series: 透明度时间序列 [T, D]
        save_dir: 保存目录
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端加速
    apply_plot_style()
    
    T, D = opacity_series.shape
    
    # 确保至少有9帧
    if T < 9:
        print(f"[Warning] Only {T} frames available, less than 9. Will use available frames.")
        num_frames = T
    else:
        num_frames = 9
    
    # 将252个测点重塑为12×21网格
    opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
    
    # 创建3行3列的子图
    fig = plt.figure(figsize=(28, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.15, left=0.05, right=0.92, top=0.94, bottom=0.06)
    
    print(f"[Info] Rendering {num_frames} subplots with ID annotations...")
    
    for idx in range(9):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        if idx < num_frames:
            print(f"  - Rendering subplot {idx+1}/9...")
            # 显示第idx帧
            display_value = 100.0 - opacity_grid[idx]
            
            im = ax.imshow(
                display_value,
                cmap=COLORMAP,
                aspect='equal',
                vmin=0,
                vmax=100,
                interpolation='nearest'
            )
            
            # 设置标题
            ax.set_title(f'Sample: {idx+1}/{T}', fontsize=14, fontweight='bold', pad=10)
            
            # 移除刻度标签
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加网格线
            ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
            ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 为每个格子添加ID文字标识
            for row_idx in range(GRID_ROWS):
                for col_idx in range(GRID_COLS):
                    cell_id = row_idx * GRID_COLS + col_idx
                    ax.text(col_idx, row_idx, str(cell_id),
                           ha='center', va='center',
                           fontsize=5.5, color='black', alpha=0.7,
                           family='Arial', weight='normal')
        else:
            # 如果不足9帧，隐藏多余的子图
            ax.axis('off')
    
    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Damage Suspicion (100 - Opacity %)', fontsize=12)
    
    # 添加总标题
    fig.suptitle('Time History Animation - First 9 Frames', fontsize=18, fontweight='bold')
    
    # 保存图片
    output_path = os.path.join(save_dir, "first_9_frames_grid.png")
    print(f"[Saving] Saving first 9 frames grid (300 dpi, this may take a minute)...")
    plt.savefig(output_path, dpi=300, format='png')
    plt.close(fig)
    
    print(f"[Saved] First 9 frames grid saved to: {output_path}")
    return output_path


def save_opacity_heatmap_9x1_samples(
    opacity_series: np.ndarray,
    save_dir: str,
    sample_indices: List[int],
    data_type: str = ""
):
    """生成9x1子图的透明度热力图（指定样本索引）
    
    Args:
        opacity_series: 透明度时间序列 [T, D]
        save_dir: 保存目录
        sample_indices: 要展示的9个样本索引
        data_type: 数据类型名称（用于标题）
    """
    apply_plot_style()
    
    T, D = opacity_series.shape
    
    # 检查样本索引是否有效
    valid_indices = [i for i in sample_indices if 0 <= i < T]
    if len(valid_indices) < len(sample_indices):
        print(f"[Warning] Some sample indices are out of range. Using valid indices only.")
    
    if len(valid_indices) == 0:
        print(f"[Warning] No valid sample indices. Skipping 9x1 heatmap.")
        return None
    
    # 确保有9个样本（不足则用现有的）
    if len(valid_indices) < 9:
        print(f"[Warning] Only {len(valid_indices)} valid samples, less than 9.")
    
    # 将252个测点重塑为12×21网格
    opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
    
    # 创建9行1列的子图（垂直排列）
    fig = plt.figure(figsize=(10.5, 54))
    gs = fig.add_gridspec(9, 1, hspace=0.3, wspace=0, left=0.08, right=0.92, top=0.98, bottom=0.02)
    
    print(f"[Info] Rendering 9x1 heatmap for {len(valid_indices)} samples...")
    
    for plot_idx, sample_idx in enumerate(valid_indices[:9]):  # 最多9个
        ax = fig.add_subplot(gs[plot_idx, 0])
        
        print(f"  - Rendering subplot {plot_idx+1}/{min(len(valid_indices), 9)} (sample {sample_idx})...")
        
        # 显示指定样本的透明度
        display_value = 100.0 - opacity_grid[sample_idx]
        
        im = ax.imshow(
            display_value,
            cmap=COLORMAP,
            aspect='equal',
            vmin=0,
            vmax=100,
            interpolation='nearest'
        )
        
        # 设置标题
        ax.set_title(f'Sample: {sample_idx+1}/{T}', fontsize=14, fontweight='bold', pad=10)
        
        # 移除刻度标签
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加网格线
        ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 隐藏多余的子图
    for plot_idx in range(len(valid_indices), 9):
        ax = fig.add_subplot(gs[plot_idx, 0])
        ax.axis('off')
    
    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Damage Suspicion (100 - Opacity %)', fontsize=12)
    
    # 添加总标题
    title_text = f'Opacity Heatmap - Selected Samples [{data_type}]' if data_type else 'Opacity Heatmap - Selected Samples'
    fig.suptitle(title_text, fontsize=18, fontweight='bold')
    
    # 保存图片
    output_path = os.path.join(save_dir, "opacity_heatmap_9x1_samples.png")
    print(f"[Saving] Saving 9x1 heatmap ({HEATMAP_DPI} dpi)...")
    plt.savefig(output_path, dpi=HEATMAP_DPI, format='png')
    plt.close(fig)
    
    print(f"[Saved] 9x1 heatmap saved to: {output_path}")
    
    # 保存对应的CSV数据
    csv_data = []
    for sample_idx in valid_indices[:9]:
        csv_data.append({
            "sample_index": sample_idx,
            "mean_opacity": opacity_series[sample_idx].mean(),
            "min_opacity": opacity_series[sample_idx].min(),
            "max_opacity": opacity_series[sample_idx].max(),
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(save_dir, "opacity_heatmap_9x1_samples.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"[Saved] 9x1 heatmap data saved to: {csv_path}")
    
    return output_path


def create_combined_heatmap_direct(
    opacity_data_dict: Dict[str, np.ndarray],
    output_dir: str,
    data_types: List[str] = ["corrosion", "crack", "multi", "health"]
):
    """直接生成四种数据类型横向合并的前10帧热力图
    
    Args:
        opacity_data_dict: 透明度数据字典 {data_type: opacity_series [T, D]}
        output_dir: 输出目录
        data_types: 数据类型列表，按顺序横向排列
    """
    apply_plot_style()
    
    print("\n" + "="*60)
    print("Creating Combined Heatmap (Horizontal Layout)")
    print("="*60 + "\n")
    
    # 检查数据
    available_types = [dt for dt in data_types if dt in opacity_data_dict]
    if len(available_types) == 0:
        print("[Error] No opacity data available to create combined heatmap.")
        return None
    
    print(f"[Info] Creating combined heatmap for: {available_types}")
    
    # 创建5行N列的子图（N = len(available_types)，横向排列）- 前5帧
    n_types = len(available_types)
    fig = plt.figure(figsize=(10.5 * n_types, 30))
    # 紧凑布局：hspace极小，wspace加大用于分隔，底部留出更多空间给色条标签
    gs = fig.add_gridspec(5, n_types, hspace=0.01, wspace=0.2, left=0.05, right=0.95, top=0.98, bottom=0.08)
    
    # 对每个数据类型绘制10x1热力图（前10帧）
    for type_idx, data_type in enumerate(available_types):
        opacity_series = opacity_data_dict[data_type]
        T, D = opacity_series.shape
        
        # 将252个测点重塑为12×21网格
        opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
        
        print(f"[Rendering] {data_type}: {T} samples...")
        
        # 绘制前5个样本（帧序数0-4）
        for plot_idx in range(5):
            if plot_idx >= T:
                print(f"  [Warning] Frame {plot_idx} out of range for {data_type} (max {T-1}), skipping...")
                continue
                
            ax = fig.add_subplot(gs[plot_idx, type_idx])
            
            # 显示第plot_idx帧的透明度
            display_value = 100.0 - opacity_grid[plot_idx]
            
            im = ax.imshow(
                display_value,
                cmap=COLORMAP,
                aspect='equal',
                vmin=0,
                vmax=100,
                interpolation='nearest'
            )
            
            # 移除刻度标签
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加网格线
            ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
            ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 添加横置在底端的色条，调整位置以确保标签完整显示
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Damage Suspicion (100 - Opacity %)', fontsize=40, labelpad=15)
    cbar.ax.tick_params(labelsize=40)
    
    # 保存图片（不添加总标题），使用透明背景，不裁剪以保留色条标签
    output_path = os.path.join(output_dir, "opacity_heatmap_combined_4types.png")
    print(f"\n[Saving] Saving combined heatmap ({HEATMAP_DPI} dpi) with transparent background...")
    plt.savefig(output_path, dpi=HEATMAP_DPI, format='png', transparent=True)
    plt.close(fig)
    
    print(f"[Saved] Combined heatmap saved to: {output_path}")
    
    print("="*60)
    print("Combined Heatmap Created Successfully!")
    print("="*60 + "\n")
    
    return output_path


def combine_four_types_heatmap(
    output_dir: str,
    data_types: List[str] = ["corrosion", "crack", "multi", "health"]
):
    """（已弃用）将四种数据类型的9x1热力图横向合并成一个大图
    
    此函数已被 create_combined_heatmap_direct 取代，保留仅为向后兼容
    
    Args:
        output_dir: 输出目录（09_render_time_history_output/）
        data_types: 数据类型列表，按顺序横向排列
    """
    from PIL import Image
    
    print("\n" + "="*60)
    print("Combining Four Types Heatmaps (Horizontal Layout)")
    print("="*60 + "\n")
    
    # 收集所有9x1热力图路径
    heatmap_paths = []
    for data_type in data_types:
        heatmap_path = os.path.join(output_dir, data_type, "opacity_heatmap_9x1_samples.png")
        if os.path.exists(heatmap_path):
            heatmap_paths.append((data_type, heatmap_path))
            print(f"[Found] {data_type}: {heatmap_path}")
        else:
            print(f"[Warning] Missing {data_type} heatmap: {heatmap_path}")
    
    if len(heatmap_paths) == 0:
        print("[Error] No heatmaps found to combine.")
        return None
    
    # 加载所有图片
    images = []
    for data_type, path in heatmap_paths:
        img = Image.open(path)
        images.append((data_type, img))
        print(f"[Loaded] {data_type} image size: {img.size}")
    
    # 获取图片尺寸（假设所有图片高度相同）
    max_height = max(img.size[1] for _, img in images)
    total_width = sum(img.size[0] for _, img in images)
    
    print(f"[Info] Combined image size: {total_width} x {max_height}")
    
    # 创建合并后的大图
    combined_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # 横向拼接
    x_offset = 0
    for data_type, img in images:
        combined_img.paste(img, (x_offset, 0))
        print(f"[Pasted] {data_type} at offset {x_offset}")
        x_offset += img.size[0]
    
    # 保存合并后的图片
    output_path = os.path.join(output_dir, "opacity_heatmap_combined_4types.png")
    combined_img.save(output_path, dpi=(HEATMAP_DPI, HEATMAP_DPI))
    print(f"\n[Saved] Combined heatmap saved to: {output_path}")
    
    # 关闭所有图片
    for _, img in images:
        img.close()
    
    print("="*60)
    print("Four Types Heatmaps Combined Successfully!")
    print("="*60 + "\n")
    
    return output_path


def create_combined_animation_direct(
    opacity_data_dict: Dict[str, np.ndarray],
    output_dir: str,
    data_types: List[str] = ["corrosion", "crack", "multi", "health"],
    fps: int = 10,
    dpi: int = 100
):
    """直接生成四种数据类型2x2布局合并的动画
    
    Args:
        opacity_data_dict: 透明度数据字典 {data_type: opacity_series [T, D]}
        output_dir: 输出目录
        data_types: 数据类型列表，按左上、右上、左下、右下顺序排列
        fps: 帧率
        dpi: 分辨率
    """
    apply_plot_style()
    
    print("\n" + "="*60)
    print("Creating Combined Animation (2x2 Layout)")
    print("="*60 + "\n")
    
    # 检查数据，确保有4种类型
    available_types = [dt for dt in data_types if dt in opacity_data_dict]
    if len(available_types) < 4:
        print(f"[Error] Need 4 data types for 2x2 layout, found only {len(available_types)}: {available_types}")
        return None
    
    # 只取前4个
    available_types = available_types[:4]
    print(f"[Info] Creating combined animation for: {available_types}")
    
    # 获取最小帧数
    frame_counts = [opacity_data_dict[dt].shape[0] for dt in available_types]
    min_frames = min(frame_counts)
    print(f"[Info] Frame counts: {dict(zip(available_types, frame_counts))}")
    print(f"[Info] Will create {min_frames} frames (minimum)")
    
    # 创建2x2布局的图形 - 增大纵向间距
    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.12)
    
    # 2x2布局位置
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (row, col)
    
    # 初始化4个子图
    axes = []
    ims = []
    titles = []
    
    for idx, (data_type, (row, col)) in enumerate(zip(available_types, positions)):
        ax = fig.add_subplot(gs[row, col])
        opacity_series = opacity_data_dict[data_type]
        T, D = opacity_series.shape
        
        # 将252个测点重塑为12×21网格
        opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
        
        # 初始化热力图（使用第一帧）
        initial_display = 100.0 - opacity_grid[0]
        
        im = ax.imshow(
            initial_display,
            cmap=COLORMAP,
            aspect='equal',
            vmin=0,
            vmax=100,
            interpolation='nearest'
        )
        
        # 设置标题（multi类型显示为Combined）
        title_prefix = 'Combined' if data_type.lower() == 'multi' else data_type.capitalize()
        title = ax.set_title(f'{title_prefix} - Sample: 1/{T}', fontsize=30, fontweight='bold', pad=2)
        
        # 移除刻度标签
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加网格线
        ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        axes.append(ax)
        ims.append(im)
        titles.append(title)
    
    # 添加横置在底端的色条，字号调整为30
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Damage Suspicion (100 - Opacity %)', fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    
    # 添加总标题，字号也调整为30
    main_title = fig.suptitle('Time History Animation', fontsize=30, fontweight='bold')
    
    def update(frame):
        """更新函数，用于动画"""
        artists = []
        for idx, data_type in enumerate(available_types):
            opacity_series = opacity_data_dict[data_type]
            T, D = opacity_series.shape
            opacity_grid = opacity_series.reshape(T, GRID_ROWS, GRID_COLS)
            
            if frame < T:
                display_value = 100.0 - opacity_grid[frame]
                ims[idx].set_array(display_value)
                # multi类型显示为Combined
                title_prefix = 'Combined' if data_type.lower() == 'multi' else data_type.capitalize()
                titles[idx].set_text(f'{title_prefix} - Sample: {frame+1}/{T}')
                artists.extend([ims[idx], titles[idx]])
        
        return artists
    
    # 创建动画
    print(f"\n[Animation] Creating animation with {min_frames} frames at {fps} FPS...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=min_frames,
        interval=1000 / fps,
        blit=True,
        repeat=True
    )
    
    # 保存为GIF
    output_path = os.path.join(output_dir, "opacity_animation_combined_2x2.gif")
    print(f"[Saving] Saving combined animation as GIF (this may take a while)...")
    
    Writer = animation.writers['pillow']
    writer = Writer(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    
    print(f"[Saved] Combined animation saved to: {output_path}")
    
    print("="*60)
    print("Combined Animation Created Successfully!")
    print("="*60 + "\n")
    
    return output_path


def combine_four_types_animation(
    output_dir: str,
    data_types: List[str] = ["corrosion", "crack", "multi", "health"],
    save_format: str = "gif"
):
    """（已弃用）将四种数据类型的动画合并成2x2布局的单个GIF
    
    此函数已被 create_combined_animation_direct 取代，保留仅为向后兼容
    
    Args:
        output_dir: 输出目录（09_render_time_history_output/）
        data_types: 数据类型列表，按左上、右上、左下、右下顺序排列
        save_format: 保存格式（只支持gif）
    """
    from PIL import Image
    import glob
    
    print("\n" + "="*60)
    print("Combining Four Types Animations (2x2 Layout)")
    print("="*60 + "\n")
    
    if save_format != "gif":
        print(f"[Warning] Only GIF format is supported for combined animation. Using GIF.")
        save_format = "gif"
    
    # 收集所有动画路径
    animation_paths = []
    for data_type in data_types:
        # 尝试找到gif或mp4文件
        gif_path = os.path.join(output_dir, data_type, "opacity_animation.gif")
        mp4_path = os.path.join(output_dir, data_type, "opacity_animation.mp4")
        
        if os.path.exists(gif_path):
            animation_paths.append((data_type, gif_path))
            print(f"[Found] {data_type}: {gif_path}")
        elif os.path.exists(mp4_path):
            print(f"[Warning] {data_type} has MP4 format, need to convert to GIF first")
            print(f"[Info] Skipping {data_type} - please ensure animations are saved as GIF")
        else:
            print(f"[Warning] Missing {data_type} animation")
    
    if len(animation_paths) < 4:
        print(f"[Error] Need 4 animations for 2x2 layout, found only {len(animation_paths)}")
        print("[Info] To fix: set SAVE_FORMAT = 'gif' in the script and re-run")
        return None
    
    # 只取前4个
    animation_paths = animation_paths[:4]
    
    # 加载所有GIF的帧
    print("\n[Loading] Loading animation frames...")
    all_frames = []
    frame_counts = []
    
    for data_type, path in animation_paths:
        gif = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        
        all_frames.append((data_type, frames))
        frame_counts.append(len(frames))
        print(f"[Loaded] {data_type}: {len(frames)} frames")
    
    # 取最小帧数作为合并后的帧数
    min_frames = min(frame_counts)
    print(f"\n[Info] Will create {min_frames} frames (minimum of all animations)")
    
    # 获取单个动画的尺寸
    sample_frame = all_frames[0][1][0]
    frame_width, frame_height = sample_frame.size
    
    # 创建2x2布局的合并帧
    combined_width = frame_width * 2
    combined_height = frame_height * 2
    
    print(f"[Info] Combined frame size: {combined_width} x {combined_height}")
    print(f"[Info] Creating combined frames...")
    
    combined_frames = []
    for frame_idx in range(min_frames):
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  Processing frame {frame_idx + 1}/{min_frames}...")
        
        # 创建空白画布
        combined_frame = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        
        # 2x2布局：左上、右上、左下、右下
        positions = [
            (0, 0),                          # 左上
            (frame_width, 0),                # 右上
            (0, frame_height),               # 左下
            (frame_width, frame_height)      # 右下
        ]
        
        for idx, (data_type, frames) in enumerate(all_frames):
            if frame_idx < len(frames):
                frame = frames[frame_idx].resize((frame_width, frame_height), Image.LANCZOS)
                combined_frame.paste(frame, positions[idx])
        
        combined_frames.append(combined_frame)
    
    # 保存合并后的GIF
    output_path = os.path.join(output_dir, "opacity_animation_combined_2x2.gif")
    print(f"\n[Saving] Saving combined GIF animation ({min_frames} frames)...")
    
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=1000 // ANIMATION_FPS,  # 毫秒
        loop=0
    )
    
    print(f"[Saved] Combined animation saved to: {output_path}")
    
    # 关闭所有图片
    for _, frames in all_frames:
        for frame in frames:
            frame.close()
    
    print("="*60)
    print("Four Types Animations Combined Successfully!")
    print("="*60 + "\n")
    
    return output_path


def compute_opacity_for_type(data_type: str, model: Autoencoder, device: torch.device, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算单一数据类型的透明度时间序列（仅计算，不输出）
    
    Args:
        data_type: 数据类型 'crack'|'corrosion'|'multi'|'health'
        model: 已加载的模型
        device: 计算设备
        thresholds: 预计算的阈值 [D]
        
    Returns:
        opacity_series: 透明度时间序列 [T, D]
        exceed_history: 超阈值倍数历史 [T, D]
    """
    print(f"\n[Processing] Computing opacity for {data_type}...")
    
    # 加载验证数据
    V_val = load_validation_data(data_type)
    
    # 计算透明度时间序列
    opacity_series, exceed_history = compute_opacity_time_series(
        V_val, model, device, thresholds,
        max_samples=MAX_SAMPLES,
        window_size=OPACITY_WINDOW_SIZE,
        decay_factor=OPACITY_DECAY_FACTOR
    )
    
    return opacity_series, exceed_history


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="生成验证数据集的时历透明度动画（crack/corrosion/multi/health）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 09_render_time_history_main.py                          # 渲染所有数据集
  python 09_render_time_history_main.py --datasets multi         # 只渲染multi数据集
  python 09_render_time_history_main.py --datasets crack multi   # 渲染crack和multi
  python 09_render_time_history_main.py --type all               # 渲染所有数据集（旧参数兼容）
        """
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["health", "crack", "corrosion", "multi"],
        help="指定要渲染的数据集类型（可指定多个）。不指定则渲染所有数据集。"
    )
    parser.add_argument(
        "--type",
        "-t",
        dest="types",
        nargs="+",
        default=None,
        help="（旧参数，保持兼容）数据类型，如 crack corrosion health multi；使用 'all' 处理所有",
    )
    args = parser.parse_args()
    
    # 确定要处理的数据类型（优先使用--datasets参数）
    if args.datasets:
        types_to_process = args.datasets
    elif args.types and not (len(args.types) == 1 and args.types[0].lower() == "all"):
        types_to_process = [x.lower() for x in args.types]
    else:
        types_to_process = PROCESS_DATA_TYPES
    
    print("\n" + "="*60)
    print("流程09：时历动画渲染")
    print("="*60)
    print(f"将渲染以下数据集: {', '.join(types_to_process)}")
    print("="*60)
    print(f"\n[Info] Script directory: {SCRIPT_DIR}")
    print(f"[Info] Output directory: {OUTPUT_DIR}")
    print(f"[Info] Data types to process: {types_to_process}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        print(f"[Error] Please run 04_train_model_main.py first")
        sys.exit(1)
    
    # 加载设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # 加载训练数据用于计算阈值（只需加载一次）
    print("\n" + "="*60)
    print("Step 0: 准备阈值")
    print("="*60)
    train_data_path = os.path.join(SCRIPT_DIR, "03_preprocess_training_data_output", "preprocessed_data_raw.npz")
    if not os.path.exists(train_data_path):
        print(f"[Error] Training data not found: {train_data_path}")
        print(f"[Error] Please run 03_preprocess_training_data_main.py first")
        sys.exit(1)
    
    train_data = np.load(train_data_path)
    V_train = train_data["V"]
    print(f"[Loaded] Training data shape: {V_train.shape}")
    
    # 加载模型用于计算阈值
    D = V_train.shape[1]
    model = load_model(MODEL_PATH, D, device)
    
    # 计算阈值（只计算一次，所有数据类型共用）
    print(f"[Computing] Thresholds using method '{THRESHOLD_METHOD}'...")
    thresholds = compute_thresholds_from_validation(
        V_train, model, device,
        method=THRESHOLD_METHOD,
        quantile=THRESHOLD_QUANTILE,
        k=THRESHOLD_K_SIGMA
    )
    
    # ========================================
    # Step 1: 计算所有数据类型的透明度时间序列
    # ========================================
    print("\n" + "="*60)
    print("Step 1: 计算所有数据类型的透明度数据")
    print("="*60)
    
    opacity_data_dict = {}  # {data_type: opacity_series}
    exceed_data_dict = {}   # {data_type: exceed_history}
    
    for data_type in types_to_process:
        try:
            opacity_series, exceed_history = compute_opacity_for_type(
                data_type, model, device, thresholds
            )
            opacity_data_dict[data_type] = opacity_series
            exceed_data_dict[data_type] = exceed_history
            
            print(f"[Success] {data_type}: {opacity_series.shape[0]} samples, {opacity_series.shape[1]} dimensions")
        except FileNotFoundError as e:
            print(f"[Warning] Skip {data_type}: {e}")
        except Exception as e:
            print(f"[Error] Failed for {data_type}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(opacity_data_dict) == 0:
        print("[Error] No valid data computed. Exiting.")
        sys.exit(1)
    
    # ========================================
    # Step 2: 保存CSV数据（可选）
    # ========================================
    print("\n" + "="*60)
    print("Step 2: 保存透明度数据到CSV")
    print("="*60)
    
    for data_type in opacity_data_dict.keys():
        save_dir = os.path.join(OUTPUT_DIR, data_type)
        os.makedirs(save_dir, exist_ok=True)
        
        opacity_series = opacity_data_dict[data_type]
        exceed_history = exceed_data_dict[data_type]
        
        save_opacity_data(opacity_series, exceed_history, save_dir)
    
    # ========================================
    # Step 3: 生成合并的热力图（横向排列）
    # ========================================
    print("\n" + "="*60)
    print("Step 3: 生成合并的热力图")
    print("="*60)
    
    try:
        combined_heatmap_path = create_combined_heatmap_direct(
            opacity_data_dict, OUTPUT_DIR, list(opacity_data_dict.keys())
        )
        if combined_heatmap_path:
            print(f"[Success] Combined heatmap: {combined_heatmap_path}")
    except Exception as e:
        print(f"[Error] Failed to create combined heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Step 4: 生成合并的动画（2x2布局）
    # ========================================
    # 只有当有4种数据类型时才生成2x2动画
    if len(opacity_data_dict) >= 4:
        print("\n" + "="*60)
        print("Step 4: 生成合并的动画 (2x2布局)")
        print("="*60)
        
        try:
            combined_animation_path = create_combined_animation_direct(
                opacity_data_dict, OUTPUT_DIR, list(opacity_data_dict.keys())[:4],
                fps=ANIMATION_FPS, dpi=ANIMATION_DPI
            )
            if combined_animation_path:
                print(f"[Success] Combined animation: {combined_animation_path}")
        except Exception as e:
            print(f"[Error] Failed to create combined animation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*60)
        print(f"Step 4: 跳过合并动画（需要4种数据类型，当前只有{len(opacity_data_dict)}种）")
        print("="*60)
    
    # ========================================
    # 最终总结
    # ========================================
    print("\n" + "="*60)
    print("流程09完成！")
    print("="*60)
    print(f"\n[Summary]")
    print(f"  - Processed data types: {list(opacity_data_dict.keys())}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Combined heatmap: opacity_heatmap_combined_4types.png")
    if len(opacity_data_dict) >= 4:
        print(f"  - Combined animation: opacity_animation_combined_2x2.gif")
    print()


if __name__ == "__main__":
    main()
