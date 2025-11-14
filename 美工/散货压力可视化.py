import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- 全局设置 ---
# 确保保存图片的文件夹存在
output_dir = "plot_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置英文字体
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_bulk_cargo_pressure(X, Y, Z, x0, y0, z0, theta, Lx, Ly, rho, g=9.81):
    """
    计算散货的三维抛物面压力分布。
    """
    r_c = np.sqrt(Lx**2 + Ly**2)
    a = np.tan(theta) / (2 * r_c)
    z_surface = z0 - a * ((X - x0) ** 2 + (Y - y0) ** 2)
    h_mm = z_surface - Z
    h_meters = h_mm / 1000.0
    pressure_pa = rho * g * np.maximum(0, h_meters)
    return pressure_pa, h_mm


def plot_beautiful_3d_surface():
    """绘制精美的3D散货表面（纯几何形状）"""
    x0, y0, z0 = 70000, 0, 25000
    Lx, Ly = 33000, 32000
    theta = np.radians(35)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(x0 - 50000, x0 + 50000, 120)
    y = np.linspace(-50000, 50000, 120)
    X, Y = np.meshgrid(x, y)

    r_c = np.sqrt(Lx**2 + Ly**2)
    a = np.tan(theta) / (2 * r_c)
    Z_surface = z0 - a * ((X - x0) ** 2 + (Y - y0) ** 2)
    mask = Z_surface > 5000
    Z_surface_masked = np.where(mask, Z_surface, np.nan)

    norm = plt.Normalize(vmin=np.nanmin(Z_surface_masked), vmax=np.nanmax(Z_surface_masked))
    ax.plot_surface(
        X,
        Y,
        Z_surface_masked,
        facecolors=plt.cm.copper(norm(Z_surface_masked)),
        alpha=0.9,
        linewidth=0.1,
        antialiased=True,
        shade=True,
    )

    ax.set_xlim([20000, 120000])
    ax.set_ylim([-50000, 50000])
    ax.set_zlim([5000, 30000])
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(Z_surface_masked[~np.isnan(Z_surface_masked)])))

    ax.set_xlabel("X Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_ylabel("Y Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_zlabel("Z Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_title(
        "Bulk Cargo 3D Paraboloid Surface Distribution", fontsize=18, pad=20, fontweight="bold"
    )
    ax.view_init(elev=25, azim=-135)

    m = plt.cm.ScalarMappable(cmap=plt.cm.copper, norm=norm)
    cbar = fig.colorbar(m, ax=ax, shrink=0.6, aspect=20, pad=0.12)
    cbar.set_label("Surface Height (mm)", fontsize=12, labelpad=15)

    plt.tight_layout()

    # 保存高清图
    save_path = os.path.join(output_dir, "1_surface_distribution_3d.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {save_path}")

    plt.show()


def plot_pressure_slices(rho):
    """
    【新增】绘制 X=70000 和 Y=0 两个截面的压力分布图。
    """
    x0, y0, z0 = 70000, 0, 25000
    Lx, Ly = 33000, 32000
    theta = np.radians(35)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"2D Pressure Slices (Cargo Density: {rho} kg/m³)", fontsize=20, fontweight="bold")

    # --- 切片 1: X = 70000 mm ---
    y_slice = np.linspace(-40000, 40000, 100)
    z_slice = np.linspace(5000, 25000, 100)
    Y_grid, Z_grid = np.meshgrid(y_slice, z_slice)
    X_grid = np.full_like(Y_grid, x0)  # X坐标是常数

    pressure_pa, _ = calculate_bulk_cargo_pressure(
        X_grid, Y_grid, Z_grid, x0, y0, z0, theta, Lx, Ly, rho
    )

    contour = ax1.contourf(Y_grid, Z_grid, pressure_pa, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax1, label="Pressure (Pa)")

    # 绘制自由表面轮廓线
    r_c = np.sqrt(Lx**2 + Ly**2)
    a = np.tan(theta) / (2 * r_c)
    z_surface_line = z0 - a * ((x0 - x0) ** 2 + (y_slice - y0) ** 2)
    ax1.plot(y_slice, z_surface_line, "r--", linewidth=2.5, label="Free Surface")

    ax1.set_title("Pressure Distribution at X = 70000 mm", fontsize=16)
    ax1.set_xlabel("Y Coordinate (mm)", fontsize=12)
    ax1.set_ylabel("Z Coordinate (mm)", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_aspect("equal", "box")

    # --- 切片 2: Y = 0 mm ---
    x_slice = np.linspace(20000, 120000, 100)
    # z_slice 保持不变
    X_grid, Z_grid = np.meshgrid(x_slice, z_slice)
    Y_grid = np.zeros_like(X_grid)  # Y坐标是常数

    pressure_pa, _ = calculate_bulk_cargo_pressure(
        X_grid, Y_grid, Z_grid, x0, y0, z0, theta, Lx, Ly, rho
    )

    contour = ax2.contourf(X_grid, Z_grid, pressure_pa, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax2, label="Pressure (Pa)")

    # 绘制自由表面轮廓线
    z_surface_line = z0 - a * ((x_slice - x0) ** 2 + (0 - y0) ** 2)
    ax2.plot(x_slice, z_surface_line, "r--", linewidth=2.5, label="Free Surface")

    ax2.set_title("Pressure Distribution at Y = 0 mm", fontsize=16)
    ax2.set_xlabel("X Coordinate (mm)", fontsize=12)
    ax2.set_ylabel("Z Coordinate (mm)", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_aspect("equal", "box")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局为总标题留出空间

    # 保存高清图
    save_path = os.path.join(output_dir, "2_pressure_slices_2d.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {save_path}")

    plt.show()


def plot_enhanced_surface_with_pressure(rho):
    """绘制带压力可视化的3D表面"""
    x0, y0, z0 = 70000, 0, 25000
    Lx, Ly = 33000, 32000
    theta = np.radians(35)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(x0 - 45000, x0 + 45000, 100)
    y = np.linspace(-45000, 45000, 100)
    X, Y = np.meshgrid(x, y)

    r_c = np.sqrt(Lx**2 + Ly**2)
    a = np.tan(theta) / (2 * r_c)
    Z_surface = z0 - a * ((X - x0) ** 2 + (Y - y0) ** 2)
    mask = Z_surface > 8000
    Z_surface_masked = np.where(mask, Z_surface, np.nan)

    pressure_surface, _ = calculate_bulk_cargo_pressure(
        X, Y, Z_surface_masked - 1000, x0, y0, z0, theta, Lx, Ly, rho
    )

    colors_list = ["#FFEFD5", "#F4A460", "#CD853F", "#8B4513"]
    custom_cmap = LinearSegmentedColormap.from_list("cargo_pressure", colors_list)

    norm = plt.Normalize(vmin=np.nanmin(pressure_surface), vmax=np.nanmax(pressure_surface))
    ax.plot_surface(
        X,
        Y,
        Z_surface_masked,
        facecolors=custom_cmap(norm(pressure_surface)),
        alpha=0.85,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.plot_wireframe(
        X[::8, ::8],
        Y[::8, ::8],
        Z_surface_masked[::8, ::8],
        color="saddlebrown",
        alpha=0.3,
        linewidth=0.5,
    )

    ax.set_xlim([25000, 115000])
    ax.set_ylim([-45000, 45000])
    ax.set_zlim([8000, 26000])
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(Z_surface_masked[~np.isnan(Z_surface_masked)])))

    ax.set_xlabel("X Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_ylabel("Y Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_zlabel("Z Coordinate (mm)", fontsize=14, labelpad=15)
    ax.set_title(
        "Bulk Cargo Surface with Pressure Distribution", fontsize=18, pad=25, fontweight="bold"
    )
    ax.view_init(elev=20, azim=135)

    m = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(m, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label("Pressure (Pa)", fontsize=12, labelpad=15)

    plt.tight_layout()

    # 保存高清图
    save_path = os.path.join(output_dir, "3_pressure_surface_3d.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {save_path}")

    plt.show()


def plot_interactive_3d(rho):
    """绘制带压力向量的3D交互图，并调整Z轴比例"""
    x0, y0, z0 = 70000, 0, 25000
    Lx, Ly = 33000, 32000
    theta = np.radians(35)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    x_surf = np.linspace(x0 - 40000, x0 + 40000, 80)
    y_surf = np.linspace(-40000, 40000, 80)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

    r_c = np.sqrt(Lx**2 + Ly**2)
    a = np.tan(theta) / (2 * r_c)
    Z_surface = z0 - a * ((X_surf - x0) ** 2 + (Y_surf - y0) ** 2)

    norm = plt.Normalize(Z_surface.min(), Z_surface.max())
    ax.plot_surface(
        X_surf,
        Y_surf,
        Z_surface,
        facecolors=plt.cm.viridis(norm(Z_surface)),
        alpha=0.6,
        linewidth=0,
        antialiased=True,
    )

    x_vec, y_vec = np.meshgrid(
        np.linspace(x0 - Lx // 2, x0 + Lx // 2, 7), np.linspace(-Ly // 2, Ly // 2, 7)
    )
    Z_vec = z0 - a * ((x_vec - x0) ** 2 + (y_vec - y0) ** 2) - 5000
    pressure_vec, _ = calculate_bulk_cargo_pressure(
        x_vec, y_vec, Z_vec, x0, y0, z0, theta, Lx, Ly, rho
    )

    ax.quiver(
        x_vec,
        y_vec,
        Z_vec,
        0,
        0,
        -pressure_vec,
        length=2e-2,
        color="red",
        alpha=0.8,
        arrow_length_ratio=0.3,
    )

    ax.set_title(f"Interactive 3D Pressure (Density: {rho} kg/m³)", fontsize=16, pad=20)
    ax.set_xlabel("X Coordinate (mm)", fontsize=12, labelpad=10)
    ax.set_ylabel("Y Coordinate (mm)", fontsize=12, labelpad=10)
    ax.set_zlabel("Z Coordinate (mm)", fontsize=12, labelpad=10)

    # 【修改】调整Z轴视觉比例，使其看起来更扁平 (0.6倍)
    z_range = np.ptp(Z_surface[~np.isnan(Z_surface)])
    ax.set_box_aspect((np.ptp(x_surf), np.ptp(y_surf), z_range * 0.6))

    ax.view_init(elev=25, azim=45)

    m = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cbar = plt.colorbar(m, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label("Free Surface Height (mm)", fontsize=12)

    plt.tight_layout()

    # 保存高清图
    save_path = os.path.join(output_dir, "4_pressure_interactive_3d.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图片已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    # --- 在这里设置货物密度 ---
    # 例如：煤炭密度约 800-900 kg/m³, 铁矿石约 2200-3200 kg/m³
    cargo_density = 1500  # 单位: kg/m³

    print("--- 开始生成可视化图表 ---")

    print("\n[1/4] 正在绘制精美的3D散货表面图...")
    plot_beautiful_3d_surface()

    print(f"\n[2/4] 正在使用货物密度 {cargo_density} kg/m³ 绘制2D压力切片图...")
    plot_pressure_slices(rho=cargo_density)

    print(f"\n[3/4] 正在使用货物密度 {cargo_density} kg/m³ 绘制带压力的3D表面图...")
    plot_enhanced_surface_with_pressure(rho=cargo_density)

    print(f"\n[4/4] 正在使用货物密度 {cargo_density} kg/m³ 绘制交互式3D压力示意图...")
    plot_interactive_3d(rho=cargo_density)

    print("\n--- 所有可视化完成！图片已保存在 'plot_outputs' 文件夹中。 ---")
