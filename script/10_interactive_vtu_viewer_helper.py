"""
流程10辅助工具：交互式VTU查看器
功能：在PyVista交互窗口中显示损伤可疑度动画，用于调整相机视角
依赖：流程02（INP转VTU）、流程04（模型训练）、流程07（验证数据预处理）
输入：
    - 02_inp_to_vtu_output/whole_from_inp.vtu
    - 02_inp_to_vtu_output/measures_ID_auto.csv
    - 04_train_model_output/autoencoder.pth
    - 07_preprocess_validation_data_output/{crack|corrosion|health}/preprocessed_data_raw.npz
输出：
    - camera_position.json（相机位置配置，供流程10使用）

说明：
    - 这是一个分支工具（helper），不参与主流程
    - 主要用途：调整并保存相机视角，供 10_render_vtu_animation_main.py 使用
    - 支持旋转、缩放、平移等所有交互操作
    - 支持暂停/播放、快进/后退、调整速度
    - 关闭窗口时自动保存当前相机位置
    
使用方法：
    python 10_interactive_vtu_viewer_helper.py --data crack
    # 在交互窗口中调整视角后关闭窗口
    # 相机位置自动保存到 camera_position.json
    # 然后运行 10_render_vtu_animation_main.py 即可使用该视角
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import argparse
import json

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
AE_ACTIVATION = "relu"  # 激活函数

# 阈值计算参数（必须与流程08一致）
THRESHOLD_METHOD = "quantile_abs"  # 阈值方法
THRESHOLD_QUANTILE = 0.95  # 分位数
THRESHOLD_K_SIGMA = 3.0  # k倍标准差

# 损伤可疑度计算参数
SUSPICION_WINDOW_SIZE = 10  # 滑动窗口大小
DAMAGE_SCALE_FACTOR = 10.0  # 损伤缩放因子

# 查看器参数
MAX_SAMPLES = 50  # 最大样本数（帧数），减少加载时间
VIEWER_FPS = 10  # 帧率
VIEWER_WINDOW_WIDTH = 1920  # 窗口宽度
VIEWER_WINDOW_HEIGHT = 1080  # 窗口高度
VIEWER_BACKGROUND = "white"  # 背景颜色
VIEWER_CMAP = "coolwarm"  # 颜色映射
VIEWER_CLIM_MIN = 0  # 颜色范围最小值
VIEWER_CLIM_MAX = 100  # 颜色范围最大值

# 特征边线参数
FEATURE_EDGE_ANGLE = 30  # 特征角度阈值（度）
FEATURE_EDGE_COLOR = "black"  # 特征边线颜色
FEATURE_EDGE_WIDTH = 1.5  # 特征边线宽度

# 默认相机位置（初始视角，如果没有保存的位置）
DEFAULT_CAMERA_POSITION = [
    (-118250.9077302128, 99019.47724416424, 130225.53976681917),  # camera position
    (16817.598535918256, 18266.62363928763, 31742.253660874267),   # focal point
    (0.4214857390997515, 0.8935556469878645, -0.1546223705407302)  # view up
]


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
    """加载训练好的模型"""
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
    """加载验证数据"""
    data_type = data_type.lower().strip()
    if data_type not in {"crack", "corrosion", "health"}:
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
    """根据训练集验证样本计算每个维度的阈值"""
    # 批量预测
    V_tensor = torch.from_numpy(V_train.astype(np.float32)).to(device)
    with torch.no_grad():
        V_pred = model(V_tensor).cpu().numpy()
    
    # 计算残差
    residuals = V_pred - V_train
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
    """计算损伤可疑度时间序列"""
    N, D = V.shape
    
    if max_samples > 0 and max_samples < N:
        N_process = max_samples
        print(f"[Info] Processing first {N_process} samples (max_samples={max_samples})")
    else:
        N_process = N
    
    print(f"\n[Processing] Computing suspicion timeline for {N_process}/{N} samples...")
    
    V_process = V[:N_process]
    suspicion_series = []
    exceed_window = []
    
    for t in range(N_process):
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Processing sample {t+1}/{N_process}...")
        
        sample = V_process[t]
        
        x = torch.from_numpy(sample[None, :].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = model(x)
            residuals = (pred - x).squeeze(0).cpu().numpy()
        
        abs_residuals = np.abs(residuals)
        exceed_ratios = np.zeros_like(abs_residuals)
        
        exceed_mask = abs_residuals > thresholds
        exceed_ratios[exceed_mask] = abs_residuals[exceed_mask] / thresholds[exceed_mask]
        
        exceed_window.append(exceed_ratios)
        
        if len(exceed_window) > window_size:
            exceed_window.pop(0)
        
        avg_exceed_ratios = np.mean(exceed_window, axis=0)
        current_suspicion = avg_exceed_ratios * damage_scale_factor
        current_suspicion = np.clip(current_suspicion, 0.0, 100.0)
        
        suspicion_series.append(current_suspicion.copy())
    
    suspicion_timeline = np.array(suspicion_series)
    
    print(f"[Processing] Suspicion timeline computed: shape {suspicion_timeline.shape}")
    
    processed_indices = list(range(N_process))
    return suspicion_timeline, processed_indices


def load_vtu_model(vtu_path: str) -> pv.UnstructuredGrid:
    """加载VTU模型文件"""
    if not os.path.exists(vtu_path):
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")
    
    mesh = pv.read(vtu_path)
    print(f"[Loaded] VTU model: {mesh.n_cells} cells, {mesh.n_points} points")
    return mesh


def load_measure_ids(measures_csv_path: str, group_name: str = "all_measures") -> np.ndarray:
    """加载测点ID列表"""
    if not os.path.exists(measures_csv_path):
        raise FileNotFoundError(f"Measures ID file not found: {measures_csv_path}")
    
    df = pd.read_csv(measures_csv_path)
    if group_name not in df.columns:
        raise ValueError(f"Group '{group_name}' not found in {measures_csv_path}")
    
    measure_ids = df[group_name].dropna().astype(int).values
    print(f"[Loaded] Measure IDs ({group_name}): {len(measure_ids)} points")
    return measure_ids


def map_suspicion_to_vtu(
    suspicion_timeline: np.ndarray,
    measure_ids: np.ndarray,
    total_cells: int
) -> np.ndarray:
    """将测点可疑度映射到VTU模型（直接映射）"""
    T, D = suspicion_timeline.shape
    assert D == len(measure_ids), f"Dimension mismatch: {D} != {len(measure_ids)}"
    
    print(f"\n[Mapping] Mapping {D} measure points to {total_cells} VTU cells...")
    print("[Mapping] Using direct mapping: VTU_Index = Abaqus_ID - 1")
    
    vtu_suspicion_timeline = np.zeros((T, total_cells), dtype=np.float32)
    
    vtu_indices = measure_ids - 1
    valid_mask = (vtu_indices >= 0) & (vtu_indices < total_cells)
    valid_vtu_indices = vtu_indices[valid_mask]
    vtu_suspicion_timeline[:, valid_vtu_indices] = suspicion_timeline[:, valid_mask]
    
    mapped_count = valid_mask.sum()
    print(f"[Mapping] Successfully mapped {mapped_count}/{D} measure points")
    
    return vtu_suspicion_timeline


class InteractiveAnimationViewer:
    """交互式动画查看器"""
    
    def __init__(
        self,
        base_mesh: pv.UnstructuredGrid,
        vtu_suspicion_timeline: np.ndarray,
        fps: int = 10,
        cmap: str = "coolwarm",
        clim: Tuple[float, float] = (0, 100),
        window_size: Tuple[int, int] = (1920, 1080),
        background_color: str = "white",
        initial_camera_position = None
    ):
        self.base_mesh = base_mesh
        self.vtu_suspicion_timeline = vtu_suspicion_timeline
        self.fps = fps
        self.cmap = cmap
        self.clim = clim
        self.window_size = window_size
        self.background_color = background_color
        self.initial_camera_position = initial_camera_position
        
        self.T = vtu_suspicion_timeline.shape[0]
        self.current_frame = 0
        self.is_playing = True
        self.speed_multiplier = 1.0
        self.plotter = None
        self.current_actor = None
        self.info_actor = None
        
        print(f"\n[Viewer] Initializing interactive viewer...")
        print(f"  - Total frames: {self.T}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Window size: {self.window_size}")
    
    def update_info_text(self):
        """更新信息文本"""
        if self.info_actor is not None:
            self.plotter.remove_actor(self.info_actor)
        
        info_text = (
            f'Frame: {self.current_frame + 1}/{self.T}\n'
            f'Speed: {self.speed_multiplier:.1f}x\n'
            f'Status: {"Playing" if self.is_playing else "Paused"}\n\n'
            f'Controls:\n'
            f'  Space - Play/Pause\n'
            f'  Right - Next Frame\n'
            f'  Left - Previous Frame\n'
            f'  + - Speed Up\n'
            f'  - - Slow Down\n'
            f'  R - Reset View\n'
            f'  C - Print Camera\n'
            f'  Q - Quit'
        )
        
        self.info_actor = self.plotter.add_text(
            info_text,
            position='upper_left',
            font_size=11,
            color='black',
            font='arial'
        )
    
    def update_mesh_and_text(self):
        """更新网格和文本"""
        # 直接更新cell_data
        self.base_mesh.cell_data['damage_suspicion'] = self.vtu_suspicion_timeline[self.current_frame]
        
        # 更新文本
        self.update_info_text()
        
        # 强制渲染
        if self.plotter is not None:
            self.plotter.render()
    
    def timer_callback(self):
        """定时器回调函数"""
        if not self.is_playing:
            return
        
        # 计算应该跳过多少帧（基于速度倍数）
        frames_to_advance = max(1, int(self.fps * self.speed_multiplier / 10))
        
        self.current_frame = (self.current_frame + frames_to_advance) % self.T
        self.update_mesh_and_text()
    
    def toggle_play_pause(self):
        """切换播放/暂停"""
        self.is_playing = not self.is_playing
        status = "Playing" if self.is_playing else "Paused"
        print(f"[Viewer] {status}")
    
    def next_frame(self):
        """下一帧"""
        self.current_frame = (self.current_frame + 1) % self.T
        print(f"[Viewer] Frame: {self.current_frame + 1}/{self.T}")
    
    def previous_frame(self):
        """上一帧"""
        self.current_frame = (self.current_frame - 1) % self.T
        print(f"[Viewer] Frame: {self.current_frame + 1}/{self.T}")
    
    def speed_up(self):
        """加速"""
        self.speed_multiplier = min(5.0, self.speed_multiplier + 0.5)
        print(f"[Viewer] Speed: {self.speed_multiplier:.1f}x")
    
    def slow_down(self):
        """减速"""
        self.speed_multiplier = max(0.1, self.speed_multiplier - 0.5)
        print(f"[Viewer] Speed: {self.speed_multiplier:.1f}x")
    
    def print_camera_position(self, plotter):
        """打印当前相机位置"""
        print("\n[Viewer] Current camera position:")
        print(f"  {plotter.camera_position}")
        print("\nYou can use this position in 10_render_vtu_animation_main.py")
    
    def run(self):
        """运行交互式查看器"""
        plotter = pv.Plotter(window_size=self.window_size)
        plotter.set_background(self.background_color)
        
        # 保存plotter的引用
        self.plotter = plotter
        
        # 设置初始mesh数据
        self.base_mesh.cell_data['damage_suspicion'] = self.vtu_suspicion_timeline[0]
        
        # 提取特征边线（便于看清模型结构）
        print(f"[Viewer] Extracting feature edges (angle={FEATURE_EDGE_ANGLE}°)...")
        feature_edges = self.base_mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False,
            feature_angle=FEATURE_EDGE_ANGLE
        )
        print(f"[Viewer] Feature edges extracted: {feature_edges.n_points} points, {feature_edges.n_cells} lines")
        
        # 添加主网格到场景
        self.current_actor = plotter.add_mesh(
            self.base_mesh,
            scalars='damage_suspicion',
            cmap=self.cmap,
            clim=self.clim,
            show_edges=False,  # 关闭所有网格边线
            opacity=1.0,
            scalar_bar_args={
                'title': 'Damage Suspicion\n(0=Healthy, 100=Severe)',
                'n_labels': 6,
                'fmt': '%.1f',
                'font_family': 'arial',
                'label_font_size': 14,
                'title_font_size': 16,
                'width': 0.08,
                'height': 0.6,
                'position_x': 0.90,
                'position_y': 0.20
            }
        )
        
        # 添加特征边线（黑色线条）
        plotter.add_mesh(
            feature_edges,
            color=FEATURE_EDGE_COLOR,
            line_width=FEATURE_EDGE_WIDTH,
            render_lines_as_tubes=False
        )
        
        # 添加初始文本
        self.info_actor = None
        self.update_info_text()
        
        # 设置默认视角
        if self.initial_camera_position is not None:
            plotter.camera_position = self.initial_camera_position
        else:
            plotter.camera_position = 'iso'
        
        # 添加键盘控制
        def key_press_callback(key):
            if key == ' ':
                self.toggle_play_pause()
                self.update_info_text()
            elif key == 'Right':
                self.is_playing = False
                self.next_frame()
                self.update_mesh_and_text()
            elif key == 'Left':
                self.is_playing = False
                self.previous_frame()
                self.update_mesh_and_text()
            elif key == 'plus' or key == '=':
                self.speed_up()
                self.update_info_text()
            elif key == 'minus' or key == '_':
                self.slow_down()
                self.update_info_text()
            elif key == 'r' or key == 'R':
                plotter.reset_camera()
                print("[Viewer] Camera reset")
            elif key == 'c' or key == 'C':
                self.print_camera_position(plotter)
        
        plotter.add_key_event('space', lambda: key_press_callback(' '))
        plotter.add_key_event('Right', lambda: key_press_callback('Right'))
        plotter.add_key_event('Left', lambda: key_press_callback('Left'))
        plotter.add_key_event('plus', lambda: key_press_callback('plus'))
        plotter.add_key_event('equal', lambda: key_press_callback('='))
        plotter.add_key_event('minus', lambda: key_press_callback('minus'))
        plotter.add_key_event('underscore', lambda: key_press_callback('_'))
        plotter.add_key_event('r', lambda: key_press_callback('r'))
        plotter.add_key_event('R', lambda: key_press_callback('R'))
        plotter.add_key_event('c', lambda: key_press_callback('c'))
        plotter.add_key_event('C', lambda: key_press_callback('C'))
        
        # 添加定时器来更新动画 (固定100ms间隔)
        plotter.add_timer_event(max_steps=None, duration=100, callback=self.timer_callback)
        
        print("\n" + "=" * 60)
        print("INTERACTIVE VIEWER CONTROLS")
        print("=" * 60)
        print("Mouse:")
        print("  - Left Click + Drag: Rotate")
        print("  - Middle Click + Drag: Pan")
        print("  - Scroll Wheel: Zoom")
        print("  - Right Click + Drag: Zoom (alternative)")
        print("\nKeyboard:")
        print("  - Space: Play/Pause")
        print("  - Right Arrow: Next Frame")
        print("  - Left Arrow: Previous Frame")
        print("  - + (Plus): Speed Up")
        print("  - - (Minus): Slow Down")
        print("  - R: Reset Camera View")
        print("  - C: Print Current Camera Position")
        print("  - Q: Quit")
        print("=" * 60)
        print("\n✨ Tip: Adjust the camera to your desired view, then close the window.")
        print("       The camera position will be saved automatically!\n")
        
        plotter.show()


# ========================================
# 主程序逻辑
# ========================================

def main():
    """主函数"""
    print("\n" + "="*60)
    print("流程10辅助工具：交互式VTU查看器")
    print("="*60)
    print(f"\n[Info] Script directory: {SCRIPT_DIR}")
    
    # 检查PyVista
    if not HAS_PYVISTA:
        print("\n[Error] PyVista is not installed!")
        print("[Error] Please install it: pip install pyvista")
        sys.exit(1)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Interactive VTU viewer for adjusting camera position"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="crack",
        choices=["crack", "corrosion", "health"],
        help="Data type to visualize (default: crack)",
    )
    args = parser.parse_args()
    
    data_type = args.data.lower()
    print(f"[Info] Data type: {data_type}")
    
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
    
    # 1. 加载训练数据用于计算阈值
    print("\n[Step 1] Loading training data for threshold calculation...")
    train_data_path = os.path.join(SCRIPT_DIR, "03_preprocess_training_data_output", "preprocessed_data_raw.npz")
    if not os.path.exists(train_data_path):
        print(f"[Error] Training data not found: {train_data_path}")
        print(f"[Error] Please run 03_preprocess_training_data_main.py first")
        sys.exit(1)
    
    train_data = np.load(train_data_path)
    V_train = train_data["V"]
    print(f"[Loaded] Training data shape: {V_train.shape}")
    
    D = V_train.shape[1]
    model = load_model(MODEL_PATH, D, device)
    
    # 计算阈值
    print(f"\n[Step 2] Computing thresholds using method '{THRESHOLD_METHOD}'...")
    thresholds = compute_thresholds_from_validation(
        V_train, model, device,
        method=THRESHOLD_METHOD,
        quantile=THRESHOLD_QUANTILE,
        k=THRESHOLD_K_SIGMA
    )
    
    # 2. 加载验证数据
    print(f"\n[Step 3] Loading {data_type} validation data...")
    V = load_validation_data(data_type)
    
    # 3. 计算损伤可疑度时间序列
    print("\n[Step 4] Computing damage suspicion timeline...")
    suspicion_timeline, processed_indices = compute_suspicion_timeline(
        V, thresholds, model, device,
        max_samples=MAX_SAMPLES,
        window_size=SUSPICION_WINDOW_SIZE,
        damage_scale_factor=DAMAGE_SCALE_FACTOR
    )
    
    # 4. 加载VTU模型和测点映射
    print("\n[Step 5] Loading VTU model and measure IDs...")
    base_mesh = load_vtu_model(VTU_PATH)
    measure_ids = load_measure_ids(MEASURES_PATH, group_name="all_measures")
    
    # 5. 映射可疑度到VTU模型
    print("\n[Step 6] Mapping suspicion to VTU cells...")
    vtu_suspicion_timeline = map_suspicion_to_vtu(
        suspicion_timeline,
        measure_ids,
        base_mesh.n_cells
    )
    
    # 6. 加载初始相机位置（如果存在）
    print("\n[Step 7] Loading initial camera position...")
    if os.path.exists(CAMERA_POSITION_FILE):
        with open(CAMERA_POSITION_FILE, 'r') as f:
            camera_data = json.load(f)
        
        initial_camera_position = [
            tuple(camera_data['camera_position']),
            tuple(camera_data['focal_point']),
            tuple(camera_data['view_up'])
        ]
        
        print(f"[Loaded] Using saved camera position from: {CAMERA_POSITION_FILE}")
    else:
        initial_camera_position = DEFAULT_CAMERA_POSITION
        print(f"[Info] Using default camera position")
    
    # 7. 启动交互式查看器
    print("\n[Step 8] Starting interactive viewer...")
    
    viewer = InteractiveAnimationViewer(
        base_mesh,
        vtu_suspicion_timeline,
        fps=VIEWER_FPS,
        cmap=VIEWER_CMAP,
        clim=(VIEWER_CLIM_MIN, VIEWER_CLIM_MAX),
        window_size=(VIEWER_WINDOW_WIDTH, VIEWER_WINDOW_HEIGHT),
        background_color=VIEWER_BACKGROUND,
        initial_camera_position=initial_camera_position
    )
    
    viewer.run()
    
    # 8. 保存最终相机位置（用于后续 GIF 生成）
    print("\n" + "=" * 60)
    print("Saving Camera Position")
    print("=" * 60)
    
    if viewer.plotter is not None:
        camera_pos = viewer.plotter.camera_position
        print(f"\nFinal Camera Position: {camera_pos}")
        
        # 自动保存相机位置到文件
        camera_data = {
            "camera_position": list(camera_pos[0]),
            "focal_point": list(camera_pos[1]),
            "view_up": list(camera_pos[2]),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(CAMERA_POSITION_FILE, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        print(f"\n✅ Camera position saved to: {CAMERA_POSITION_FILE}")
        print(f"   This will be automatically used by 10_render_vtu_animation_main.py")
        print(f"\n💡 Next step: Run 10_render_vtu_animation_main.py to render animations with this camera view!")
    
    print("\n" + "=" * 60)
    print("Viewer closed. Thank you!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
