"""
流程08：损伤识别验证
功能：基于逐维度阈值，对四类验证数据（裂纹、腐蚀、多位置综合损伤、无损伤）进行损伤识别验证
依赖：流程03（训练数据预处理）、流程04（模型训练）、流程07（验证数据预处理）
输入：
    - 03_preprocess_training_data_output/preprocessed_data_raw.npz（训练集健康样本数据）
    - 04_train_model_output/validation_indices.csv（验证集索引）
    - 04_train_model_output/autoencoder.pth（训练好的模型）
    - 07_preprocess_validation_data_output/health/preprocessed_data_raw.npz（无损伤样本）
    - 07_preprocess_validation_data_output/crack/preprocessed_data_raw.npz（裂纹损伤样本）
    - 07_preprocess_validation_data_output/corrosion/preprocessed_data_raw.npz（腐蚀损伤样本）
    - 07_preprocess_validation_data_output/multi/preprocessed_data_raw.npz（多位置综合损伤样本）
输出：
    - 08_damage_validation_output/dimension_thresholds.csv（每个维度的阈值）
    - 08_damage_validation_output/combined_residuals_crack.png（裂纹样本残差组合图）
    - 08_damage_validation_output/crack_detection_summary.png（裂纹检测结果统计汇总）
    - 08_damage_validation_output/combined_residuals_corrosion.png（腐蚀样本残差组合图）
    - 08_damage_validation_output/corrosion_detection_summary.png（腐蚀检测结果统计汇总）
    - 08_damage_validation_output/combined_residuals_multi.png（多位置综合损伤样本残差组合图）
    - 08_damage_validation_output/multi_detection_summary.png（多位置综合损伤检测结果统计汇总）
    - 08_damage_validation_output/combined_residuals_health.png（健康样本残差组合图）
    - 08_damage_validation_output/health_detection_summary.png（健康样本检测结果统计汇总）
说明：
    - 基于验证集健康样本计算每个维度的阈值
    - 对四类验证数据（crack/corrosion/multi/health）的所有样本进行测试
    - 可视化每个维度的残差是否超过阈值
    - crack、corrosion 和 multi：期望检测到（超阈值），未检测到视为失败
    - health：期望不检测到（不超阈值），检测到视为误报

使用方法：
    python 08_damage_validation_main.py                           # 验证所有数据集
    python 08_damage_validation_main.py --datasets multi          # 只验证multi数据集
    python 08_damage_validation_main.py --datasets crack multi    # 验证crack和multi
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Tuple, Dict, List

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 路径配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PREPROCESS_OUTPUT = os.path.join(SCRIPT_DIR, "03_preprocess_training_data_output")
TRAIN_MODEL_OUTPUT = os.path.join(SCRIPT_DIR, "04_train_model_output")
VAL_PREPROCESS_OUTPUT = os.path.join(SCRIPT_DIR, "07_preprocess_validation_data_output")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "08_damage_validation_output")

# 输入文件路径
TRAIN_DATA_PATH = os.path.join(TRAIN_PREPROCESS_OUTPUT, "preprocessed_data_raw.npz")
MODEL_PATH = os.path.join(TRAIN_MODEL_OUTPUT, "autoencoder.pth")
VAL_INDICES_PATH = os.path.join(TRAIN_MODEL_OUTPUT, "validation_indices.csv")

# 验证数据路径
HEALTH_DATA_PATH = os.path.join(VAL_PREPROCESS_OUTPUT, "health", "preprocessed_data_raw.npz")
CRACK_DATA_PATH = os.path.join(VAL_PREPROCESS_OUTPUT, "crack", "preprocessed_data_raw.npz")
CORROSION_DATA_PATH = os.path.join(VAL_PREPROCESS_OUTPUT, "corrosion", "preprocessed_data_raw.npz")
MULTI_DATA_PATH = os.path.join(VAL_PREPROCESS_OUTPUT, "multi", "preprocessed_data_raw.npz")

# --- 模型结构配置（需与流程04一致）---
# 注意：这些参数必须与04_train_model_main.py中完全一致！
ENCODER_DIMS = [768, 384, 192]      # 编码器各层维度
LATENT_DIM = 192                    # 潜在空间维度
DECODER_DIMS = [192, 384, 768]      # 解码器各层维度
DROPOUT = 0.0                       # Dropout比例
ACTIVATION = "relu"                 # 激活函数（relu/elu/leaky_relu）

# --- 阈值计算方法配置 ---
# 方法选择：'quantile_abs' 或 'kstd_abs' 或 'mean_kstd'
TAU_METHOD = "kstd_abs"

# 方法1参数（quantile_abs）：基于验证集残差绝对值的分位数
VAL_QUANTILE_BASE = 0.995           # 99.5%分位数

# 方法2参数（kstd_abs）：基于残差绝对值的k倍标准差
TAU_KSTD_K = 3.0                    # k值（通常取2-3）

# 方法3参数（mean_kstd）：基于残差均值的k倍标准差
TAU_MEAN_KSTD_K = 3.0               # k值（通常取2-3）

# --- 可视化配置 ---
RANDOM_SEED = 42                    # 随机种子，用于选择可视化样本
NUM_VIS_SAMPLES = 4                 # 每类数据可视化样本数（2x2布局）
PLOT_DPI = 300                      # 图片分辨率

# 指定样本索引（如果为None则随机选择）
SPECIFIED_SAMPLES = {
    'corrosion': [15, 20, 54, 79],
    'crack': [44, 45, 5, 83],
    'multi': [25, 6, 7, 87],
    'health': [14, 28, 48, 2]
}

# --- 计算配置 ---
BATCH_SIZE = 512                    # 批处理大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 数据类型配置 ---
# 注意：这些默认值会被命令行参数覆盖
DATA_TYPES = ['crack', 'corrosion', 'multi', 'health']  # 要处理的验证数据类型


# ========================================
# 绘图样式配置（学术风格）
# ========================================
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": PLOT_DPI,
    "savefig.dpi": PLOT_DPI,
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

def apply_plot_style():
    """应用统一的绘图样式"""
    plt.rcParams.update(PLOT_STYLE)


def save_figure_with_data(fig, name: str, data: dict = None):
    """保存图片并保存对应的数据到CSV
    
    Args:
        fig: matplotlib图形对象
        name: 文件名（不含扩展名）
        data: 要保存的数据字典（可选）
    """
    # 保存图片（透明背景）
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches='tight', transparent=True)
    print(f"    - Saved figure: {png_path}")
    
    # 保存数据到CSV
    if data is not None:
        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"    - Saved data: {csv_path}")
    
    plt.close(fig)


class Autoencoder(nn.Module):
    """自编码器模型（需与流程04中的模型结构完全一致）
    
    注意：此模型结构必须与04_train_model_main.py中的Autoencoder完全一致！
    使用简单的 Linear + Activation + Dropout 结构，不使用BatchNorm。
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list,
        latent_dim: int,
        decoder_dims: list,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 激活函数选择
        act_cls_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "elu": nn.ELU,
            "leaky_relu": lambda: nn.LeakyReLU(0.2),
        }
        act_cls = act_cls_map.get(activation.lower(), nn.ReLU)
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(act_cls())
            if dropout and dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(act_cls())
            if dropout and dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Xavier 初始化
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(_init)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model(D: int, device: torch.device) -> Autoencoder:
    """加载训练好的模型
    
    Args:
        D: 输入维度
        device: 计算设备
    
    Returns:
        model: 加载好的模型
    """
    model = Autoencoder(
        D, ENCODER_DIMS, LATENT_DIM, DECODER_DIMS, DROPOUT, ACTIVATION
    ).to(device)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"  - Loaded model from: {MODEL_PATH}")
    return model


def compute_dimension_thresholds(
    V: np.ndarray, val_indices: List[int], model: Autoencoder, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个维度的阈值
    
    Args:
        V: 训练集数据矩阵 [N, D]
        val_indices: 验证集索引列表
        model: 训练好的自编码器模型
        device: 计算设备
    
    Returns:
        thresholds: 每个维度的阈值数组 [D]
        mean_residuals: 每个维度的残差均值 [D]
    """
    print(f"  - Computing thresholds using {len(val_indices)} validation samples...")
    
    # 提取验证集数据
    V_val = V[val_indices]
    X_val = torch.from_numpy(V_val.astype(np.float32)).to(device)
    
    # 批量预测
    all_residuals = []
    with torch.no_grad():
        for i in range(0, X_val.shape[0], BATCH_SIZE):
            batch = X_val[i : i + BATCH_SIZE]
            recon = model(batch)
            residuals = (recon - batch).cpu().numpy()
            all_residuals.append(residuals)
    
    residuals_all = np.concatenate(all_residuals, axis=0)  # [N_val, D]
    print(f"  - Residual matrix shape: {residuals_all.shape}")
    
    # 计算每个维度的阈值
    mean_residuals = np.mean(residuals_all, axis=0)
    std_residuals = np.std(residuals_all, axis=0, ddof=1)
    
    if TAU_METHOD == "kstd_abs":
        # 方法2：tau = |mean| + k * std
        thresholds = np.abs(mean_residuals) + TAU_KSTD_K * std_residuals
        print(f"  - Using method: kstd_abs (k={TAU_KSTD_K})")
    elif TAU_METHOD == "mean_kstd":
        # 方法3：tau = mean + k * std (保留符号)
        thresholds = mean_residuals + TAU_MEAN_KSTD_K * std_residuals
        print(f"  - Using method: mean_kstd (k={TAU_MEAN_KSTD_K})")
    else:
        # 方法1：基于残差绝对值的分位数
        abs_residuals = np.abs(residuals_all)
        thresholds = np.quantile(abs_residuals, VAL_QUANTILE_BASE, axis=0)
        print(f"  - Using method: quantile_abs (q={VAL_QUANTILE_BASE})")
    
    print(f"  - Threshold statistics:")
    print(f"    Min: {thresholds.min():.6f}, Max: {thresholds.max():.6f}")
    print(f"    Mean: {thresholds.mean():.6f}, Median: {np.median(thresholds):.6f}")
    
    return thresholds, mean_residuals


def load_validation_data(data_type: str) -> np.ndarray:
    """加载验证数据
    
    Args:
        data_type: 数据类型，'crack'|'corrosion'|'multi'|'health'
    
    Returns:
        V: 数据矩阵 [N, D]
    """
    path_mapping = {
        'health': HEALTH_DATA_PATH,
        'crack': CRACK_DATA_PATH,
        'corrosion': CORROSION_DATA_PATH,
        'multi': MULTI_DATA_PATH
    }
    
    data_path = path_mapping.get(data_type)
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation data not found: {data_path}")
    
    data = np.load(data_path)
    V = data["V"]
    print(f"  - Loaded {data_type} data: {V.shape}")
    return V


def predict_residuals(
    sample: np.ndarray, model: Autoencoder, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """预测单个样本的残差
    
    Args:
        sample: 样本数据 [D]
        model: 模型
        device: 设备
    
    Returns:
        residuals: 逐维度残差 [D]
        prediction: 预测值 [D]
    """
    x = torch.from_numpy(sample[None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(x)
        residuals = (pred - x).squeeze(0).cpu().numpy()
        prediction = pred.squeeze(0).cpu().numpy()
    
    return residuals, prediction


def plot_combined_residuals(
    samples_data: List[Dict],
    thresholds: np.ndarray,
    mean_residuals: np.ndarray,
    data_type: str
):
    """绘制多个样本的残差组合图（2x2子图矩阵）
    
    Args:
        samples_data: 样本数据列表，每个元素包含 {sample_idx, residuals}
        thresholds: 阈值 [D]
        mean_residuals: 验证集残差均值 [D]
        data_type: 数据类型（crack/corrosion/health）
    """
    apply_plot_style()
    
    D = len(thresholds)
    dims = np.arange(D)
    n_samples = len(samples_data)
    
    # 创建 2x2 子图矩阵，调整figsize使图更扁平
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    axes = axes.flatten()
    
    all_save_data = {}
    
    for idx, sample_info in enumerate(samples_data):
        sample_idx = sample_info['sample_idx']
        residuals = sample_info['residuals']
        
        abs_residuals = np.abs(residuals)
        exceed_mask = abs_residuals > thresholds
        n_exceed = exceed_mask.sum()
        
        ax = axes[idx]
        
        # 残差值（带正负）
        colors = ['red' if exceed else 'tab:blue' for exceed in exceed_mask]
        ax.bar(dims, residuals, color=colors, alpha=0.7, width=0.8, 
               edgecolor='black', linewidth=0.2)
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        
        # 绘制阈值线
        if TAU_METHOD == 'kstd_abs':
            upper_threshold = mean_residuals + thresholds
            lower_threshold = mean_residuals - thresholds
            label_pos = 'Upper Threshold'
            label_neg = 'Lower Threshold'
        else:
            upper_threshold = thresholds
            lower_threshold = -thresholds
            label_pos = '+τ'
            label_neg = '-τ'
        
        ax.plot(dims, upper_threshold, color='darkgreen', linewidth=1.5, 
                linestyle='--', label=label_pos, alpha=0.8)
        ax.plot(dims, lower_threshold, color='green', linewidth=1.5, 
                linestyle='--', label=label_neg, alpha=0.6)
        
        # 只在左下角子图（索引2）显示轴标题和图例
        if idx == 2:
            ax.set_xlabel('Dimension Index', fontsize=20)
            ax.set_ylabel('Residual', fontsize=20)
            # 图例：右上角，细边框，更小字体
            legend = ax.legend(fontsize=8, loc='upper right', frameon=True, 
                              fancybox=False, framealpha=1.0, shadow=False, 
                              edgecolor='black', borderpad=0.3)
            legend.get_frame().set_linewidth(0.4)
        
        # 不设置title，信息通过图例显示
        # ax.set_title(f'Sample {sample_idx} | Exceeded: {n_exceed}/{D} ({n_exceed/D*100:.1f}%)', 
        #              fontsize=20, fontweight='bold')
        
        ax.tick_params(labelsize=16)
        
        # 保存数据
        all_save_data[f'sample_{sample_idx}_dimension'] = dims
        all_save_data[f'sample_{sample_idx}_residual'] = residuals
        all_save_data[f'sample_{sample_idx}_exceeded'] = exceed_mask.astype(int)
    
    # 隐藏多余的子图
    for idx in range(n_samples, 4):
        axes[idx].axis('off')
    
    # 不添加总标题
    # fig.suptitle(f'{data_type.upper()} Samples - Residuals vs Thresholds', 
    #              fontsize=22, fontweight='bold', y=0.98)
    fig.tight_layout(h_pad=0.5, w_pad=2.0)
    
    all_save_data['threshold'] = thresholds
    save_figure_with_data(fig, f"combined_residuals_{data_type}", all_save_data)


def plot_detection_summary(results: List[Dict], data_type: str):
    """绘制样本检测结果统计图
    
    Args:
        results: 检测结果列表，每个元素包含 {type, sample_idx, n_exceed, total_dims, exceed_ratio}
        data_type: 数据类型（crack/corrosion/health）
    """
    apply_plot_style()
    
    sample_indices = [r['sample_idx'] for r in results]
    n_exceeds = [r['n_exceed'] for r in results]
    exceed_ratios = [r['exceed_ratio'] for r in results]
    total_dims = results[0]['total_dims']
    
    # 统计信息
    n_detected = sum(1 for n in n_exceeds if n > 0)
    detection_rate = n_detected / len(results) * 100
    avg_exceed = np.mean(n_exceeds)
    median_exceed = np.median(n_exceeds)
    max_exceed = np.max(n_exceeds)
    min_exceed = np.min(n_exceeds)
    
    # 创建单个图形布局（只保留左侧统计直方图，饼图叠加在右上角）
    fig = plt.figure(figsize=(10, 8))
    
    # 1. 主图：超阈值维度数分布直方图
    ax1 = fig.add_subplot(111)
    ax1.hist(n_exceeds, bins=30, color='tab:blue',  
             edgecolor='black', linewidth=1.0)
    # 移除竖直的均值和中位数线
    # ax1.axvline(avg_exceed, color='red', linestyle='--', linewidth=2.5)
    # ax1.axvline(median_exceed, color='green', linestyle=':', linewidth=2.5)
    ax1.set_xlabel('Number of Exceeded Dimensions', fontsize=20)
    ax1.set_ylabel('Frequency', fontsize=20)
    # 不设置title、图例和统计信息框
    ax1.tick_params(labelsize=16)
    
    # 2. 嵌入式饼图：叠加在主图的右上角位置
    # 使用inset_axes创建嵌入式子图（位置参数：[left, bottom, width, height]，相对于ax1）
    ax2 = ax1.inset_axes([0.4, 0.3, 0.45, 0.45])  # 调整位置往左移，尺寸略小
    
    if data_type == 'corrosion':
        # 腐蚀：4个类别
        # 类别1: 没检测出超阈值或超阈值维度数不足2
        # 类别2: 超阈值维度数恰好为2（准确检测）
        # 类别3: 超阈值维度数大于2（过检测）
        category_counts = [0, 0, 0]
        category_samples = [[], [], []]  # 记录每个类别的样本索引
        accurate_dims_info = []  # 记录准确检测的维度信息
        
        for r in results:
            n_exceed = r['n_exceed']
            if n_exceed < 2:
                category_counts[0] += 1
                category_samples[0].append(r['sample_idx'])
            elif n_exceed == 2:
                category_counts[1] += 1
                category_samples[1].append(r['sample_idx'])
                accurate_dims_info.append((r['sample_idx'], r['exceed_dims']))
            else:  # n_exceed > 2
                category_counts[2] += 1
                category_samples[2].append(r['sample_idx'])
        
        # 构建准确检测的详细信息：只显示维度ID（188、199）
        
        labels_pie = [
            f'Not Detected / < 2 dims\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
            f'Accurate (2 dims): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 188, 199',
            f'Over-detected (> 2 dims)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
        ]
        colors_pie = ['green', 'red', 'saddlebrown']
        
    elif data_type == 'crack':
        # 裂纹：3个类别
        # 类别1: 没检测出超阈值
        # 类别2: 超阈值维度数恰好为1（准确检测）
        # 类别3: 超阈值维度数大于1（过检测）
        category_counts = [0, 0, 0]
        category_samples = [[], [], []]
        accurate_dims_info = []
        
        for r in results:
            n_exceed = r['n_exceed']
            if n_exceed == 0:
                category_counts[0] += 1
                category_samples[0].append(r['sample_idx'])
            elif n_exceed == 1:
                category_counts[1] += 1
                category_samples[1].append(r['sample_idx'])
                accurate_dims_info.append((r['sample_idx'], r['exceed_dims']))
            else:  # n_exceed > 1
                category_counts[2] += 1
                category_samples[2].append(r['sample_idx'])
        
        # 构建准确检测的详细信息：只显示维度ID（208）
        
        labels_pie = [
            f'Not Detected\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
            f'Accurate (1 dim): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 208',
            f'Over-detected (> 1 dim)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
        ]
        colors_pie = ['green', 'red', 'saddlebrown']
        
    elif data_type == 'multi':
        # 多位置综合损伤：3个类别
        # 类别1: 没检测出超阈值或超阈值维度数不足2
        # 类别2: 超阈值维度数恰好为2（准确检测）
        # 类别3: 超阈值维度数大于2（过检测）
        category_counts = [0, 0, 0]
        category_samples = [[], [], []]  # 记录每个类别的样本索引
        accurate_dims_info = []  # 记录准确检测的维度信息
        
        for r in results:
            n_exceed = r['n_exceed']
            if n_exceed < 2:
                category_counts[0] += 1
                category_samples[0].append(r['sample_idx'])
            elif n_exceed == 2:
                category_counts[1] += 1
                category_samples[1].append(r['sample_idx'])
                accurate_dims_info.append((r['sample_idx'], r['exceed_dims']))
            else:  # n_exceed > 2
                category_counts[2] += 1
                category_samples[2].append(r['sample_idx'])
        
        # 构建准确检测的详细信息：只显示维度ID（2、19）
        
        labels_pie = [
            f'Not Detected / < 2 dims\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
            f'Accurate (2 dims): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 2, 19',
            f'Over-detected (> 2 dims)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
        ]
        colors_pie = ['green', 'red', 'saddlebrown']
        
    else:  # health
        # 健康：2个类别
        # 类别1: 没检测出超阈值（正确）
        # 类别2: 检测出超阈值（误报）
        category_counts = [0, 0]
        category_samples = [[], []]
        
        for r in results:
            n_exceed = r['n_exceed']
            if n_exceed == 0:
                category_counts[0] += 1
                category_samples[0].append(r['sample_idx'])
            else:
                category_counts[1] += 1
                category_samples[1].append(r['sample_idx'])
        
        labels_pie = [
            f'Not Detected (Correct)\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
            f'Detected (False Positive)\n{category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)'
        ]
        colors_pie = ['green', 'red']
    
    # 绘制嵌入式饼图 - 调整字体大小以适应较小尺寸
    wedges, texts, autotexts = ax2.pie(category_counts, labels=labels_pie, 
                                         colors=colors_pie, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 11})
    # 设置饼图透明度

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    # 不设置title
    # ax2.set_title(title_text, fontsize=20, fontweight='bold')
    
    fig.tight_layout()
    
    # 保存详细数据（包含超阈值维度信息）
    save_data = {
        "sample_idx": sample_indices,
        "n_exceed": n_exceeds,
        "exceed_ratio_%": exceed_ratios,
        "exceed_dims": [str(r['exceed_dims']) for r in results]  # 将列表转为字符串便于CSV保存
    }
    
    save_figure_with_data(fig, f"{data_type}_detection_summary", save_data)


def plot_combined_detection_summary(all_results: Dict[str, List[Dict]]):
    """绘制四个类别的检测结果统计合并图（2x2布局）
    
    Args:
        all_results: 字典，键为数据类型（corrosion/crack/multi/health），值为结果列表
    """
    apply_plot_style()
    
    # 标准科学绘图配色方案（鲜明、高对比度）
    morandi_colors = {
        'hist': '#1f77b4',      # 标准蓝色
        'mean': '#d62728',      # 标准红色
        'median': '#2ca02c',    # 标准绿色
        'pie_green': '#2ca02c', # 标准绿色
        'pie_red': '#d62728',   # 标准红色
        'pie_brown': '#ff7f0e', # 标准橙色
        'edge': '#000000'       # 黑色边框
    }
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 定义顺序和标题映射
    data_types = ['corrosion', 'crack', 'multi', 'health']
    titles = {
        'corrosion': 'Corrosion',
        'crack': 'Crack',
        'multi': 'Combined',
        'health': 'Health'
    }
    
    for idx, data_type in enumerate(data_types):
        if data_type not in all_results:
            continue
            
        results = all_results[data_type]
        ax_main = axes[idx]
        
        # 提取数据
        n_exceeds = [r['n_exceed'] for r in results]
        avg_exceed = np.mean(n_exceeds)
        median_exceed = np.median(n_exceeds)
        
        # 1. 主图：超阈值维度数分布直方图
        ax_main.hist(n_exceeds, bins=30, color=morandi_colors['hist'], 
                     edgecolor=morandi_colors['edge'], linewidth=1.0)
        # 移除竖直的均值和中位数线
        # ax_main.axvline(avg_exceed, color=morandi_colors['mean'], linestyle='--', linewidth=2.5)
        # ax_main.axvline(median_exceed, color=morandi_colors['median'], linestyle=':', linewidth=2.5)
        
        # 只在左下角（索引2）显示轴标签
        if idx == 2:
            ax_main.set_xlabel('Number of Exceeded Dimensions', fontsize=20)
            ax_main.set_ylabel('Frequency', fontsize=20)
        else:
            ax_main.set_xlabel('')
            ax_main.set_ylabel('')
        
        ax_main.tick_params(labelsize=16)
        ax_main.set_title(titles[data_type], fontsize=22, fontweight='bold', pad=15)
        
        # 2. 嵌入式饼图：叠加在主图的右上角
        ax_pie = ax_main.inset_axes([0.4, 0.3, 0.45, 0.45])
        
        # 根据数据类型计算分类统计
        if data_type == 'corrosion':
            category_counts = [0, 0, 0]
            for r in results:
                n_exceed = r['n_exceed']
                if n_exceed < 2:
                    category_counts[0] += 1
                elif n_exceed == 2:
                    category_counts[1] += 1
                else:
                    category_counts[2] += 1
            
            labels_pie = [
                f'Not Detected / < 2 dims\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
                f'Accurate (2 dims): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 188, 199',
                f'Over-detected (> 2 dims)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
            ]
            colors_pie = [morandi_colors['pie_green'], morandi_colors['pie_red'], morandi_colors['pie_brown']]
            
        elif data_type == 'crack':
            category_counts = [0, 0, 0]
            for r in results:
                n_exceed = r['n_exceed']
                if n_exceed == 0:
                    category_counts[0] += 1
                elif n_exceed == 1:
                    category_counts[1] += 1
                else:
                    category_counts[2] += 1
            
            labels_pie = [
                f'Not Detected\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
                f'Accurate (1 dim): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 208',
                f'Over-detected (> 1 dim)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
            ]
            colors_pie = [morandi_colors['pie_green'], morandi_colors['pie_red'], morandi_colors['pie_brown']]
            
        elif data_type == 'multi':
            category_counts = [0, 0, 0]
            for r in results:
                n_exceed = r['n_exceed']
                if n_exceed < 2:
                    category_counts[0] += 1
                elif n_exceed == 2:
                    category_counts[1] += 1
                else:
                    category_counts[2] += 1
            
            labels_pie = [
                f'Not Detected / < 2 dims\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
                f'Accurate (2 dims): {category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)\ndim ID: 2, 19',
                f'Over-detected (> 2 dims)\n{category_counts[2]} samples ({category_counts[2]/len(results)*100:.1f}%)'
            ]
            colors_pie = [morandi_colors['pie_green'], morandi_colors['pie_red'], morandi_colors['pie_brown']]
            
        else:  # health
            category_counts = [0, 0]
            for r in results:
                n_exceed = r['n_exceed']
                if n_exceed == 0:
                    category_counts[0] += 1
                else:
                    category_counts[1] += 1
            
            labels_pie = [
                f'Not Detected (Correct)\n{category_counts[0]} samples ({category_counts[0]/len(results)*100:.1f}%)',
                f'Detected (False Positive)\n{category_counts[1]} samples ({category_counts[1]/len(results)*100:.1f}%)'
            ]
            colors_pie = [morandi_colors['pie_green'], morandi_colors['pie_red']]
        
        # 绘制嵌入式饼图（增加透明度）
        wedges, texts, autotexts = ax_pie.pie(category_counts, labels=labels_pie, 
                                                colors=colors_pie, autopct='%1.1f%%', 
                                                startangle=90, textprops={'fontsize': 11})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
    
    fig.tight_layout()
    
    # 保存图形
    output_path = os.path.join(OUTPUT_DIR, "combined_detection_summary.png")
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] Combined detection summary: {output_path}")


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='损伤识别验证（crack/corrosion/multi/health）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 08_damage_validation_main.py                            # 验证所有数据集
  python 08_damage_validation_main.py --datasets multi           # 只验证multi数据集
  python 08_damage_validation_main.py --datasets crack multi     # 验证crack和multi
        """
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['health', 'crack', 'corrosion', 'multi'],
        help='指定要验证的数据集类型（可指定多个）。不指定则验证所有数据集。'
    )
    args = parser.parse_args()
    
    # 根据命令行参数设置要处理的数据类型
    global DATA_TYPES
    if args.datasets:
        DATA_TYPES = args.datasets
    else:
        DATA_TYPES = ['crack', 'corrosion', 'multi', 'health']
    
    print("\n" + "="*70)
    print("流程08：损伤识别验证")
    print("="*70)
    print(f"将验证以下数据集: {', '.join(DATA_TYPES)}")
    print("="*70 + "\n")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Output] Directory: {OUTPUT_DIR}\n")
    
    # 设置计算设备
    device = torch.device(DEVICE)
    print(f"[Device] Using: {device}\n")
    
    # 步骤1：加载训练集数据和验证集索引
    print("[Step 1/5] Loading training data and validation indices...")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")
    
    V_train = np.load(TRAIN_DATA_PATH)["V"]
    N, D = V_train.shape
    print(f"  - Training data shape: {V_train.shape}")
    
    if not os.path.exists(VAL_INDICES_PATH):
        raise FileNotFoundError(f"Validation indices not found: {VAL_INDICES_PATH}")
    
    val_indices = pd.read_csv(VAL_INDICES_PATH)["Validation Index"].tolist()
    print(f"  - Validation samples: {len(val_indices)}\n")
    
    # 步骤2：加载模型
    print("[Step 2/5] Loading trained model...")
    model = load_model(D, device)
    print()
    
    # 步骤3：计算每个维度的阈值
    print("[Step 3/5] Computing per-dimension thresholds...")
    thresholds, mean_residuals = compute_dimension_thresholds(
        V_train, val_indices, model, device
    )
    
    # 计算上下阈值边界
    upper_threshold = mean_residuals + thresholds
    lower_threshold = mean_residuals - thresholds
    
    # 保存阈值到CSV
    threshold_df = pd.DataFrame({
        "dimension": np.arange(D),
        "mean": mean_residuals,
        "tau": thresholds,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
    })
    threshold_csv_path = os.path.join(OUTPUT_DIR, "dimension_thresholds.csv")
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"  - Saved thresholds: {threshold_csv_path}\n")
    
    # 步骤4：对三类验证数据进行全样本测试
    print("[Step 4/5] Testing validation data for all damage types...")
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    
    all_stats = {}
    all_results = {}  # 新增：收集所有结果用于合并图
    
    for data_type in DATA_TYPES:
        print(f"\n{'='*70}")
        print(f"  Processing {data_type.upper()} data...")
        print('='*70)
        
        results = []
        
        # 加载验证数据
        try:
            V_val = load_validation_data(data_type)
        except FileNotFoundError as e:
            print(f"  [Warning] {data_type} data not found, skipping...\n")
            continue
        
        n_samples = V_val.shape[0]
        print(f"  - Total {n_samples} samples will be tested")
        
        # 选择样本进行可视化（优先使用指定样本，否则随机选择）
        if data_type in SPECIFIED_SAMPLES and SPECIFIED_SAMPLES[data_type]:
            vis_indices = np.array(SPECIFIED_SAMPLES[data_type])
            # 验证索引是否在范围内
            valid_indices = vis_indices[vis_indices < n_samples]
            if len(valid_indices) < len(vis_indices):
                invalid = vis_indices[vis_indices >= n_samples]
                print(f"  [Warning] Invalid sample indices {invalid.tolist()} for {data_type} (max={n_samples-1})")
            vis_indices = valid_indices
            print(f"  - Using specified samples for visualization: {sorted(vis_indices.tolist())}")
        else:
            vis_indices = np.random.choice(n_samples, size=min(NUM_VIS_SAMPLES, n_samples), 
                                           replace=False)
            print(f"  - Selected {len(vis_indices)} random samples for visualization: {sorted(vis_indices.tolist())}")
        
        # 收集可视化样本的数据
        vis_samples_data = []
        
        # 对所有样本进行检测
        for sample_idx in range(n_samples):
            sample = V_val[sample_idx]
            
            # 预测残差
            residuals, prediction = predict_residuals(sample, model, device)
            
            # 计算超阈值维度数
            abs_residuals = np.abs(residuals)
            exceed_mask = abs_residuals > thresholds
            n_exceed = exceed_mask.sum()
            exceed_dims = np.where(exceed_mask)[0].tolist()  # 超阈值维度的索引列表
            
            # 记录结果
            results.append({
                'type': data_type,
                'sample_idx': int(sample_idx),
                'n_exceed': n_exceed,
                'exceed_dims': exceed_dims,  # 新增：超阈值维度索引
                'total_dims': D,
                'exceed_ratio': n_exceed / D * 100
            })
            
            # 收集可视化样本
            if sample_idx in vis_indices:
                vis_samples_data.append({
                    'sample_idx': int(sample_idx),
                    'residuals': residuals
                })
        
        print(f"  - Completed: Processed {n_samples} samples")
        
        # 绘制样本组合图
        if len(vis_samples_data) > 0:
            print(f"\n  [Visualization] Creating combined plot for {len(vis_samples_data)} samples...")
            plot_combined_residuals(vis_samples_data, thresholds, mean_residuals, data_type)
        
        # 绘制检测结果统计汇总（单独图已不需要，只保留合并图）
        # print(f"\n  [Visualization] Generating {data_type.upper()} detection summary...")
        # plot_detection_summary(results, data_type)
        
        # 保存结果用于合并图
        all_results[data_type] = results
        
        # 统计分析
        n_exceeds = [r['n_exceed'] for r in results]
        exceed_ratios = [r['exceed_ratio'] for r in results]
        n_detected = sum(1 for n in n_exceeds if n > 0)
        detection_rate = n_detected / len(results) * 100
        
        # 保存统计信息
        all_stats[data_type] = {
            'total_samples': len(results),
            'n_detected': n_detected,
            'detection_rate': detection_rate,
            'mean_exceed': np.mean(n_exceeds),
            'median_exceed': np.median(n_exceeds),
            'min_exceed': np.min(n_exceeds),
            'max_exceed': np.max(n_exceeds),
            'std_exceed': np.std(n_exceeds),
            'mean_ratio': np.mean(exceed_ratios),
            'median_ratio': np.median(exceed_ratios),
            'max_ratio': np.max(exceed_ratios)
        }
        
        # 打印统计信息
        print(f"\n  [{data_type.upper()} Statistics]")
        print("  " + "-" * 66)
        print(f"    Total Samples:        {len(results)}")
        print(f"    Detected Samples:     {n_detected} ({detection_rate:.1f}%)")
        print(f"    Undetected Samples:   {len(results)-n_detected} ({100-detection_rate:.1f}%)")
        print("  " + "-" * 66)
        print(f"    Exceeded Dimensions:")
        print(f"      - Mean:             {np.mean(n_exceeds):.2f}")
        print(f"      - Median:           {np.median(n_exceeds):.1f}")
        print(f"      - Min:              {np.min(n_exceeds)}")
        print(f"      - Max:              {np.max(n_exceeds)}")
        print(f"      - Std:              {np.std(n_exceeds):.2f}")
        print("  " + "-" * 66)
        print(f"    Exceed Ratio:")
        print(f"      - Mean:             {np.mean(exceed_ratios):.2f}%")
        print(f"      - Median:           {np.median(exceed_ratios):.2f}%")
        print(f"      - Max:              {np.max(exceed_ratios):.2f}%")
        print("  " + "-" * 66)
    
    # 步骤5：打印最终统计总结
    print("\n" + "="*70)
    print("[Step 5/5] Overall Summary")
    print("="*70)
    
    for data_type in DATA_TYPES:
        if data_type not in all_stats:
            continue
        stats = all_stats[data_type]
        print(f"\n{data_type.upper()}:")
        print("-" * 70)
        print(f"  Detection Rate:       {stats['detection_rate']:.1f}%")
        print(f"  Mean Exceeded Dims:   {stats['mean_exceed']:.2f}")
        print(f"  Median Exceeded Dims: {stats['median_exceed']:.1f}")
        if data_type == 'health':
            print(f"  False Positives:      {stats['n_detected']} / {stats['total_samples']} ({stats['detection_rate']:.1f}%)")
        else:
            print(f"  True Positives:       {stats['n_detected']} / {stats['total_samples']} ({stats['detection_rate']:.1f}%)")
    
    # 生成合并检测结果图（如果所有四个类别都已处理）
    required_types = ['corrosion', 'crack', 'multi', 'health']
    if all(dt in all_results for dt in required_types):
        print("\n" + "="*70)
        print("[Generating Combined Detection Summary (2x2 layout)]")
        print("="*70)
        plot_combined_detection_summary(all_results)
        print()
    
    print("\n" + "="*70)
    print("流程08完成！")
    print("="*70)
    print(f"\n[Output Files]:")
    print(f"  - Threshold data: {threshold_csv_path}")
    print(f"  - All visualizations: {OUTPUT_DIR}/*.png")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
