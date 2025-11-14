"""
流程10：VTU 3D损伤动画渲染
功能：基于验证数据（crack/corrosion/multi/health）生成3D损伤可疑度动画
依赖：流程02（INP转VTU）、流程04（模型训练）、流程07（验证数据预处理）
输入：
    - 02_inp_to_vtu_output/whole_from_inp.vtu
    - 02_inp_to_vtu_output/measures_ID_auto.csv
    - 04_train_model_output/autoencoder.pth
    - 07_preprocess_validation_data_output/{crack|corrosion|multi|health}/preprocessed_data_raw.npz
    - camera_position.json（相机位置配置）
输出：
    - 10_render_vtu_animation_output/{crack|corrosion|multi|health}/
        - damage_suspicion_animation.gif 或 .mp4（3D动画）
        - damage_suspicion_timeline.csv（每个时间步每个测点的可疑度数据）

说明：
    - 读取 crack/corrosion/multi/health 验证数据的所有样本（或前N个样本）
    - 对每个样本依次预测残差，计算超阈值倍数
    - 使用滑动窗口平均超阈值倍数，映射为损伤可疑度（0-100）
    - 将252个测点的可疑度映射到VTU模型的所有单元（直接映射：VTU_Index = Abaqus_ID - 1）
    - 使用PyVista渲染3D动画，保存为GIF或MP4格式
    - 支持批量处理四种验证数据类型

使用方法：
    python 10_render_vtu_animation_main.py                         # 渲染所有数据集
    python 10_render_vtu_animation_main.py --datasets multi        # 只渲染multi数据集
    python 10_render_vtu_animation_main.py --datasets crack multi  # 渲染crack和multi
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import argparse

# PyVista导入（延迟到需要时导入，避免无头服务器问题）
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[Warning] PyVista not available. Please install: pip install pyvista")


# ========================================
# 参数配置区
# ========================================

# 路径配置（相对于script/目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_02 = os.path.join(SCRIPT_DIR, "02_inp_to_vtu_output")  # VTU输入目录
OUTPUT_DIR_04 = os.path.join(SCRIPT_DIR, "04_train_model_output")  # 模型输入目录
OUTPUT_DIR_07 = os.path.join(SCRIPT_DIR, "07_preprocess_validation_data_output")  # 验证数据输入目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "10_render_vtu_animation_output")  # 本脚本输出目录

# 输入文件
VTU_PATH = os.path.join(OUTPUT_DIR_02, "whole_from_inp.vtu")  # VTU模型文件
MEASURES_PATH = os.path.join(OUTPUT_DIR_02, "measures_ID_auto.csv")  # 测点ID映射
MODEL_PATH = os.path.join(OUTPUT_DIR_04, "autoencoder.pth")  # 模型文件
VALIDATION_DATA_TEMPLATE = os.path.join(OUTPUT_DIR_07, "{data_type}", "preprocessed_data_raw.npz")  # 验证数据模板
CAMERA_POSITION_FILE = os.path.join(SCRIPT_DIR, "camera_position.json")  # 相机位置配置

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

# 损伤可疑度计算参数
SUSPICION_WINDOW_SIZE = 10  # 滑动窗口大小（平滑超阈值倍数波动）
DAMAGE_SCALE_FACTOR = 10.0  # 损伤缩放因子：平均超阈值1倍 = 损伤可疑度10

# 数据处理参数
MAX_SAMPLES = -1  # 最大处理样本数：-1表示全部，正整数表示处理前N个样本
# 注意：这些默认值会被命令行参数覆盖
PROCESS_DATA_TYPES = ["crack", "corrosion", "multi", "health"]  # 要处理的数据类型列表

# 3D渲染参数
RENDER_FPS = 10  # 帧率（每秒帧数）
RENDER_FORMAT = "gif"  # 保存格式：'mp4'（需要FFmpeg）或'gif'（通用）
RENDER_DPI = 100  # 渲染分辨率
RENDER_WINDOW_WIDTH = 1920  # 窗口宽度
RENDER_WINDOW_HEIGHT = 1080  # 窗口高度
RENDER_BACKGROUND = "white"  # 背景颜色
RENDER_CMAP = "coolwarm"  # 颜色映射：蓝色=健康，红色=损伤
RENDER_CLIM_MIN = 0  # 颜色范围最小值（损伤可疑度）
RENDER_CLIM_MAX = 100  # 颜色范围最大值（损伤可疑度）
RENDER_OPACITY = 1.0  # 模型不透明度（1.0=完全不透明）
RENDER_SHOW_EDGES = False  # 是否显示所有网格边线（已弃用，改用特征边线）
RENDER_SAVE_VTU_FRAMES = False  # 是否保存每帧VTU文件（用于ParaView后期查看）

# 特征边线参数（用于显示模型几何结构）
FEATURE_EDGE_ANGLE = 30  # 特征角度阈值（度）：两个相邻面的夹角大于此值则视为特征边
                         # 值越小，提取的边线越多；值越大，提取的边线越少
                         # 推荐范围：20-45度
FEATURE_EDGE_COLOR = "black"  # 特征边线颜色
FEATURE_EDGE_WIDTH = 1.5  # 特征边线宽度


# ========================================
# 内嵌工具函数
# ========================================

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
            if dropout and dropout > 0:
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
            if dropout and dropout > 0:
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


def compute_suspicion_timeline(
    V: np.ndarray,
    thresholds: np.ndarray,
    model: Autoencoder,
    device: torch.device,
    max_samples: int = -1,
    window_size: int = 10,
    damage_scale_factor: float = 10.0
) -> Tuple[np.ndarray, List[int]]:
    """计算损伤可疑度时间序列（蓝-红渐变映射）
    
    采用滑动窗口平均超阈值倍数，直接映射到颜色值
    - 0（蓝色）: 健康状态，无损伤
    - 100（红色）: 严重损伤
    
    Args:
        V: 验证数据 [N, D]
        thresholds: 阈值 [D]
        model: 模型
        device: 设备
        max_samples: 最大处理样本数，-1表示全部
        window_size: 滑动窗口大小
        damage_scale_factor: 损伤缩放因子
        
    Returns:
        suspicion_timeline: 损伤可疑度时间序列 [T, D]
        processed_indices: 处理的样本索引列表
    """
    N, D = V.shape
    
    # 限制样本数
    if max_samples > 0 and max_samples < N:
        N_process = max_samples
        print(f"[Info] Processing first {N_process} samples (max_samples={max_samples})")
    else:
        N_process = N
    
    print(f"\n[Processing] Computing suspicion timeline for {N_process}/{N} samples...")
    print(f"  - Window size: {window_size} samples")
    print(f"  - Damage scale factor: {damage_scale_factor}")
    
    V_process = V[:N_process]
    
    # 初始化损伤可疑度时间序列
    suspicion_series = []
    
    # 滑动窗口存储最近的超阈值倍数
    exceed_window = []
    
    for t in range(N_process):
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Processing sample {t+1}/{N_process}...")
        
        sample = V_process[t]
        
        # 计算当前样本的残差和超阈值倍数
        x = torch.from_numpy(sample[None, :].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = model(x)
            residuals = (pred - x).squeeze(0).cpu().numpy()
        
        abs_residuals = np.abs(residuals)
        exceed_ratios = np.zeros_like(abs_residuals)
        
        # 只有超过阈值的才计算倍数
        exceed_mask = abs_residuals > thresholds
        exceed_ratios[exceed_mask] = abs_residuals[exceed_mask] / thresholds[exceed_mask]
        
        # 将当前超阈值倍数加入滑窗
        exceed_window.append(exceed_ratios)
        
        # 只保留最近window_size个样本
        if len(exceed_window) > window_size:
            exceed_window.pop(0)
        
        # 计算滑窗平均超阈值倍数
        avg_exceed_ratios = np.mean(exceed_window, axis=0)
        
        # 直接计算损伤可疑度：平均超阈值倍数 × 缩放因子
        # 0 = 蓝色（健康），100 = 红色（严重损伤）
        current_suspicion = avg_exceed_ratios * damage_scale_factor
        
        # 限制范围在 [0, 100]
        current_suspicion = np.clip(current_suspicion, 0.0, 100.0)
        
        # 记录当前时间步的损伤可疑度
        suspicion_series.append(current_suspicion.copy())
    
    suspicion_timeline = np.array(suspicion_series)  # [T, D]
    
    print(f"[Processing] Suspicion timeline computed: shape {suspicion_timeline.shape}")
    print(f"  - Min (damage suspicion): {suspicion_timeline.min():.2f}")
    print(f"  - Max (damage suspicion): {suspicion_timeline.max():.2f}")
    print(f"  - Mean (damage suspicion): {suspicion_timeline.mean():.2f}")
    
    processed_indices = list(range(N_process))
    return suspicion_timeline, processed_indices


def load_vtu_model(vtu_path: str) -> pv.UnstructuredGrid:
    """加载VTU模型文件
    
    Args:
        vtu_path: VTU文件路径
        
    Returns:
        mesh: VTU网格对象
    """
    if not os.path.exists(vtu_path):
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")
    
    mesh = pv.read(vtu_path)
    print(f"[Loaded] VTU model: {mesh.n_cells} cells, {mesh.n_points} points")
    return mesh


def load_measure_ids(measures_csv_path: str, group_name: str = "all_measures") -> np.ndarray:
    """加载测点ID列表
    
    Args:
        measures_csv_path: 测点ID文件路径
        group_name: 测点组名（列名）
        
    Returns:
        measure_ids: 测点ID数组（Abaqus单元编号）
    """
    if not os.path.exists(measures_csv_path):
        raise FileNotFoundError(f"Measures ID file not found: {measures_csv_path}")
    
    df = pd.read_csv(measures_csv_path)
    if group_name not in df.columns:
        raise ValueError(f"Group '{group_name}' not found in {measures_csv_path}")
    
    measure_ids = df[group_name].dropna().astype(int).values
    print(f"[Loaded] Measure IDs ({group_name}): {len(measure_ids)} points")
    print(f"[Debug] First 5 measure IDs: {measure_ids[:5]}")
    return measure_ids


def map_suspicion_to_vtu(
    suspicion_timeline: np.ndarray,
    measure_ids: np.ndarray,
    total_cells: int
) -> np.ndarray:
    """将252个测点的可疑度映射到VTU模型的所有单元
    
    使用直接映射：VTU_Cell_Index = Abaqus_Element_ID - 1
    
    Args:
        suspicion_timeline: 损伤可疑度时间序列 [T, D]
        measure_ids: 测点ID数组（Abaqus单元编号）
        total_cells: VTU模型总单元数
        
    Returns:
        vtu_suspicion_timeline: VTU可疑度时间序列 [T, total_cells]
    """
    T, D = suspicion_timeline.shape
    assert D == len(measure_ids), f"Dimension mismatch: {D} != {len(measure_ids)}"
    
    print(f"\n[Mapping] Mapping {D} measure points to {total_cells} VTU cells...")
    print("[Mapping] Using direct mapping: VTU_Index = Abaqus_ID - 1")
    
    vtu_suspicion_timeline = np.zeros((T, total_cells), dtype=np.float32)
    
    # 直接计算 VTU 索引（向量化操作）
    vtu_indices = measure_ids - 1
    
    # 验证索引范围
    valid_mask = (vtu_indices >= 0) & (vtu_indices < total_cells)
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        print(f"[Warning] {invalid_count} measure IDs out of range:")
        invalid_ids = measure_ids[~valid_mask]
        for inv_id in invalid_ids[:10]:
            print(f"  ID {inv_id} -> Index {inv_id-1}")
    
    # 批量赋值（仅处理有效的测点）
    valid_vtu_indices = vtu_indices[valid_mask]
    vtu_suspicion_timeline[:, valid_vtu_indices] = suspicion_timeline[:, valid_mask]
    
    mapped_count = valid_mask.sum()
    print(f"[Mapping] Successfully mapped {mapped_count}/{D} measure points")
    print("[Debug] First 5 direct mappings:")
    for i in range(min(5, len(measure_ids))):
        print(f"  Measure ID {measure_ids[i]} (dim {i}) -> VTU Cell {measure_ids[i]-1}")
    
    return vtu_suspicion_timeline


def load_camera_position(camera_file: str) -> List:
    """加载相机位置配置
    
    Args:
        camera_file: 相机位置JSON文件路径
        
    Returns:
        camera_position: 相机位置元组列表 [position, focal_point, view_up]
    """
    if not os.path.exists(camera_file):
        raise FileNotFoundError(
            f"Camera position file not found: {camera_file}\n"
            f"Please run 10_interactive_vtu_viewer_helper.py first to generate it.\n"
            f"Steps:\n"
            f"  1. Run: python script/10_interactive_vtu_viewer_helper.py --data crack\n"
            f"  2. Adjust camera to desired view in the interactive window\n"
            f"  3. Close the window (camera position will be saved automatically)\n"
            f"  4. Then run this script again"
        )
    
    import json
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)
    
    # 构建相机位置元组
    camera_position = [
        tuple(camera_data['camera_position']),
        tuple(camera_data['focal_point']),
        tuple(camera_data['view_up'])
    ]
    
    print(f"[Loaded] Camera position from: {camera_file}")
    print(f"  Camera: {camera_position[0]}")
    print(f"  Focal:  {camera_position[1]}")
    print(f"  ViewUp: {camera_position[2]}")
    print(f"  Saved at: {camera_data.get('timestamp', 'unknown')}")
    
    return camera_position


def render_animation(
    base_mesh: pv.UnstructuredGrid,
    vtu_suspicion_timeline: np.ndarray,
    output_dir: str,
    camera_position: List,
    fps: int = 10,
    save_format: str = "gif",
    window_size: Tuple[int, int] = (1920, 1080),
    background_color: str = "white",
    cmap: str = "coolwarm",
    clim: Tuple[float, float] = (0, 100),
    opacity: float = 1.0,
    show_edges: bool = False,
    save_vtu_frames: bool = False,
    dpi: int = 100
) -> str:
    """渲染3D动画（蓝-红渐变，蓝色=健康，红色=损伤）
    
    Args:
        base_mesh: VTU网格对象
        vtu_suspicion_timeline: VTU可疑度时间序列 [T, total_cells]
        output_dir: 输出目录
        camera_position: 相机位置
        fps: 帧率
        save_format: 保存格式 'mp4'或'gif'
        window_size: 窗口大小
        background_color: 背景颜色
        cmap: 颜色映射
        clim: 颜色范围
        opacity: 不透明度
        show_edges: 是否显示边缘
        save_vtu_frames: 是否保存VTU帧文件
        dpi: 分辨率
        
    Returns:
        output_path: 动画文件路径
    """
    T, total_cells = vtu_suspicion_timeline.shape
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建静态图片输出目录
    static_frames_dir = os.path.join(output_dir, "static_frames")
    os.makedirs(static_frames_dir, exist_ok=True)
    
    if save_format == "mp4":
        output_path = os.path.join(output_dir, "damage_suspicion_animation.mp4")
    else:
        output_path = os.path.join(output_dir, "damage_suspicion_animation.gif")
    
    print(f"\n[Rendering] Generating {T} frames at {fps} FPS...")
    print(f"  - Format: {save_format}")
    print(f"  - Output: {output_path}")
    print(f"  - Colormap: {cmap}")
    print(f"  - Color limits: {clim}")
    print(f"  - Static frames dir: {static_frames_dir}")
    print(f"  - Will save first 10 frames as PNG images")
    
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background_color)
    
    # 打开视频或GIF文件
    if save_format == "mp4":
        try:
            plotter.open_movie(output_path, framerate=fps)
        except (RuntimeError, KeyError) as e:
            print(f"[Warning] FFmpeg not available: {e}")
            print(f"[Info] Falling back to GIF format...")
            save_format = "gif"
            output_path = os.path.join(output_dir, "damage_suspicion_animation.gif")
            plotter.open_gif(output_path, fps=fps)
    else:
        plotter.open_gif(output_path, fps=fps)
    
    # 提取特征边线（只需计算一次）
    print(f"[Rendering] Extracting feature edges (angle={FEATURE_EDGE_ANGLE}°)...")
    feature_edges = base_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=FEATURE_EDGE_ANGLE
    )
    print(f"[Rendering] Feature edges extracted: {feature_edges.n_points} points, {feature_edges.n_cells} lines")
    
    for frame_idx in range(T):
        mesh = base_mesh.copy()
        mesh.cell_data['damage_suspicion'] = vtu_suspicion_timeline[frame_idx]
        
        plotter.clear()
        
        # 添加主网格（带颜色映射）
        plotter.add_mesh(
            mesh,
            scalars='damage_suspicion',
            cmap=cmap,
            clim=clim,
            show_edges=False,  # 关闭所有网格边线
            opacity=opacity,
            scalar_bar_args={
                'title': 'Damage Suspicion (0=Healthy, 100=Severe)',
                'n_labels': 11,  # 增加标签数量，使刻度更精细
                'fmt': '%.0f',
                'font_family': 'arial',
                'label_font_size': 16,  # 增大字体
                'title_font_size': 18,  # 增大标题字体
                'color': 'black',  # 字体颜色
                'width': 0.6,  # 横向宽度
                'height': 0.06,  # 横向高度
                'position_x': 0.20,  # 居中偏左
                'position_y': 0.02,  # 底部位置
                'vertical': False,  # 横向显示
                'shadow': True,  # 添加阴影效果
                'italic': False,
                'bold': False
            }
        )
        
        # 添加特征边线（黑色线条，便于识别结构）
        plotter.add_mesh(
            feature_edges,
            color=FEATURE_EDGE_COLOR,
            line_width=FEATURE_EDGE_WIDTH,
            render_lines_as_tubes=False
        )
        
        plotter.add_text(
            f'Frame: {frame_idx + 1}/{T}',
            position='upper_left',
            font_size=12,
            color='black'
        )
        
        if frame_idx == 0:
            # 设置相机位置
            plotter.camera_position = camera_position
        
        plotter.write_frame()
        
        # 保存前10帧为PNG静态图片
        if frame_idx < 10:
            static_image_path = os.path.join(static_frames_dir, f"frame_{frame_idx:02d}.png")
            plotter.screenshot(static_image_path)
            if frame_idx == 0:
                print(f"  - Saving static frames to: {static_frames_dir}")
        
        if save_vtu_frames:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.vtu")
            mesh.save(frame_path)
        
        if (frame_idx + 1) % max(1, T // 10) == 0 or frame_idx == T - 1:
            print(f"  - Rendered {frame_idx + 1}/{T} frames ({100 * (frame_idx + 1) / T:.1f}%)")
    
    plotter.close()
    print(f"\n[Success] Animation saved to: {output_path}")
    print(f"[Success] First 10 static frames saved to: {static_frames_dir}")
    
    return output_path


def save_suspicion_data(
    suspicion_timeline: np.ndarray,
    processed_indices: List[int],
    output_path: str
) -> None:
    """保存损伤可疑度时间序列数据到CSV
    
    Args:
        suspicion_timeline: 损伤可疑度时间序列 [T, D]
        processed_indices: 处理的样本索引列表
        output_path: 输出CSV路径
    """
    T, D = suspicion_timeline.shape
    
    data = {
        'Frame': list(range(T)),
        'Sample_Index': processed_indices
    }
    
    for dim_idx in range(D):
        data[f'Dim_{dim_idx:03d}'] = suspicion_timeline[:, dim_idx]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"[Saved] Suspicion timeline data: {output_path}")


def run_vtu_animation_for_type(data_type: str, thresholds: np.ndarray, camera_position: List) -> dict:
    """执行单一数据类型的VTU动画渲染
    
    Args:
        data_type: 数据类型 'crack'|'corrosion'|'multi'|'health'
        thresholds: 预计算的阈值 [D]
        camera_position: 相机位置
        
    Returns:
        结果字典
    """
    print("\n" + "="*60)
    print(f"Start VTU Animation Rendering [{data_type}]")
    print("="*60 + "\n")
    
    # 创建输出目录
    save_dir = os.path.join(OUTPUT_DIR, data_type)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Output] Output directory: {save_dir}")
    
    # 加载设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # 1. 加载验证数据
    print(f"\n[Step 1] Loading {data_type} validation data...")
    V_val = load_validation_data(data_type)
    N, D = V_val.shape
    
    # 2. 加载模型
    print("\n[Step 2] Loading trained model...")
    model = load_model(MODEL_PATH, D, device)
    
    # 3. 计算损伤可疑度时间序列
    print("\n[Step 3] Computing damage suspicion timeline...")
    suspicion_timeline, processed_indices = compute_suspicion_timeline(
        V_val, thresholds, model, device,
        max_samples=MAX_SAMPLES,
        window_size=SUSPICION_WINDOW_SIZE,
        damage_scale_factor=DAMAGE_SCALE_FACTOR
    )
    
    # 4. 保存可疑度数据
    print("\n[Step 4] Saving suspicion timeline data...")
    suspicion_csv_path = os.path.join(save_dir, "damage_suspicion_timeline.csv")
    save_suspicion_data(suspicion_timeline, processed_indices, suspicion_csv_path)
    
    # 5. 加载VTU模型和测点映射
    print("\n[Step 5] Loading VTU model and measure IDs...")
    base_mesh = load_vtu_model(VTU_PATH)
    measure_ids = load_measure_ids(MEASURES_PATH, group_name="all_measures")
    
    # 6. 映射可疑度到VTU模型
    print("\n[Step 6] Mapping suspicion to VTU cells...")
    vtu_suspicion_timeline = map_suspicion_to_vtu(
        suspicion_timeline,
        measure_ids,
        base_mesh.n_cells
    )
    
    # 7. 渲染3D动画
    print("\n[Step 7] Rendering 3D animation...")
    try:
        animation_path = render_animation(
            base_mesh,
            vtu_suspicion_timeline,
            save_dir,
            camera_position,
            fps=RENDER_FPS,
            save_format=RENDER_FORMAT,
            window_size=(RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT),
            background_color=RENDER_BACKGROUND,
            cmap=RENDER_CMAP,
            clim=(RENDER_CLIM_MIN, RENDER_CLIM_MAX),
            opacity=RENDER_OPACITY,
            show_edges=RENDER_SHOW_EDGES,
            save_vtu_frames=RENDER_SAVE_VTU_FRAMES,
            dpi=RENDER_DPI
        )
    except KeyboardInterrupt:
        print(f"[Warning] Animation rendering interrupted by user")
        animation_path = "Animation rendering interrupted"
    except Exception as e:
        print(f"[Warning] Animation rendering failed: {e}")
        import traceback
        traceback.print_exc()
        animation_path = "Animation rendering failed"
    
    # 打印总结
    print("\n" + "="*60)
    print(f"VTU Animation Rendering Completed [{data_type}]!")
    print("="*60)
    print(f"\n[Summary]")
    print(f"  - Type: {data_type}")
    print(f"  - Processed samples: {suspicion_timeline.shape[0]}")
    print(f"  - Dimensions: {suspicion_timeline.shape[1]}")
    print(f"  - VTU cells: {base_mesh.n_cells}")
    print(f"  - Animation format: {RENDER_FORMAT}")
    print(f"  - Frame rate: {RENDER_FPS} FPS")
    print(f"  - Output directory: {save_dir}")
    print(f"  - Suspicion data: {suspicion_csv_path}")
    print(f"  - Animation file: {animation_path}")
    print()
    
    return {
        "type": data_type,
        "samples": int(suspicion_timeline.shape[0]),
        "dims": int(suspicion_timeline.shape[1]),
        "vtu_cells": int(base_mesh.n_cells),
        "save_dir": save_dir,
        "suspicion_csv": suspicion_csv_path,
        "animation": animation_path,
    }


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主函数"""
    # 检查PyVista
    if not HAS_PYVISTA:
        print("\n[Error] PyVista is not installed!")
        print("[Error] Please install it: pip install pyvista")
        sys.exit(1)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="生成验证数据集的3D VTU损伤动画（crack/corrosion/multi/health）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 10_render_vtu_animation_main.py                         # 渲染所有数据集
  python 10_render_vtu_animation_main.py --datasets multi        # 只渲染multi数据集
  python 10_render_vtu_animation_main.py --datasets crack multi  # 渲染crack和multi
  python 10_render_vtu_animation_main.py --type all              # 渲染所有数据集（旧参数兼容）
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
    print("流程10：VTU 3D损伤动画渲染")
    print("="*60)
    print(f"将渲染以下数据集: {', '.join(types_to_process)}")
    print("="*60)
    print(f"\n[Info] Script directory: {SCRIPT_DIR}")
    print(f"[Info] Output directory: {OUTPUT_DIR}")
    print(f"[Info] Data types to process: {types_to_process}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查必需文件
    print("\n[Check] Verifying required files...")
    
    if not os.path.exists(VTU_PATH):
        print(f"[Error] VTU file not found: {VTU_PATH}")
        print(f"[Error] Please run 02_inp_to_vtu_main.py first")
        sys.exit(1)
    
    if not os.path.exists(MEASURES_PATH):
        print(f"[Error] Measures file not found: {MEASURES_PATH}")
        print(f"[Error] Please run 02_inp_to_vtu_main.py first")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found: {MODEL_PATH}")
        print(f"[Error] Please run 04_train_model_main.py first")
        sys.exit(1)
    
    print("[Check] All required files found!")
    
    # 加载设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # 加载训练数据用于计算阈值（只需加载一次）
    print("\n[Step 0] Loading training data for threshold calculation...")
    train_data_path = os.path.join(SCRIPT_DIR, "03_preprocess_training_data_output", "preprocessed_data_raw.npz")
    if not os.path.exists(train_data_path):
        print(f"[Error] Training data not found: {train_data_path}")
        print(f"[Error] Please run 03_preprocess_training_data_main.py first")
        sys.exit(1)
    
    train_data = np.load(train_data_path)
    V_train = train_data["V"]
    print(f"[Loaded] Training data shape: {V_train.shape}")
    
    # 加载模型用于计算阈值
    print("\n[Step 0] Loading model for threshold calculation...")
    D = V_train.shape[1]
    model = load_model(MODEL_PATH, D, device)
    
    # 计算阈值（只计算一次，所有数据类型共用）
    print(f"\n[Step 0] Computing thresholds using method '{THRESHOLD_METHOD}'...")
    thresholds = compute_thresholds_from_validation(
        V_train, model, device,
        method=THRESHOLD_METHOD,
        quantile=THRESHOLD_QUANTILE,
        k=THRESHOLD_K_SIGMA
    )
    
    # 保存阈值到输出目录
    thresholds_df = pd.DataFrame({
        "dimension": np.arange(len(thresholds)),
        "tau": thresholds
    })
    thresholds_path = os.path.join(OUTPUT_DIR, "dimension_thresholds.csv")
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"[Saved] Thresholds saved to: {thresholds_path}")
    
    # 加载相机位置
    print("\n[Step 0] Loading camera position...")
    camera_position = load_camera_position(CAMERA_POSITION_FILE)
    
    # 批量处理各数据类型
    results = []
    for data_type in types_to_process:
        try:
            result = run_vtu_animation_for_type(data_type, thresholds, camera_position)
            results.append(result)
        except FileNotFoundError as e:
            print(f"[Warning] Skip {data_type}: {e}")
        except Exception as e:
            print(f"[Error] Failed for {data_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    print("\n" + "="*60)
    print("All VTU Animations Completed")
    print("="*60)
    for r in results:
        print(f"- {r['type']}: samples={r['samples']}, dims={r['dims']}, vtu_cells={r['vtu_cells']}, out={r['save_dir']}")
    if not results:
        print("No animations generated.")
    
    print("\n[Info] All outputs saved to: " + OUTPUT_DIR)
    print("\n" + "="*60)
    print("流程10完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
