"""
交互式批量图像裁剪工具
用于统一裁剪 10_render_vtu_animation_output 中的所有静帧图像

使用方法:
1. 运行脚本
2. 在参考图像上拖动鼠标选择裁剪区域
3. 按 'c' 确认裁剪区域
4. 按 'r' 重新选择
5. 按 'q' 退出
6. 确认后会批量处理所有子文件夹中的静帧图像
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional, List
import shutil
from datetime import datetime


class InteractiveCropper:
    """交互式图像裁剪器"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.display_image = self.image.copy()
        self.original_image = self.image.copy()
        
        # 裁剪区域 (x1, y1, x2, y2)
        self.crop_rect: Optional[Tuple[int, int, int, int]] = None
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.drawing = False
        
        # 窗口名称
        self.window_name = "图像裁剪 - 拖动鼠标选择区域"
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新结束点
                self.end_point = (x, y)
                # 重绘图像
                self.display_image = self.original_image.copy()
                cv2.rectangle(
                    self.display_image,
                    self.start_point,
                    self.end_point,
                    (0, 255, 0),
                    2
                )
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            self.drawing = False
            self.end_point = (x, y)
            
            # 确保坐标顺序正确
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            # 确保区域有效
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.crop_rect = (x1, y1, x2, y2)
                # 绘制最终矩形
                self.display_image = self.original_image.copy()
                cv2.rectangle(
                    self.display_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                # 添加文本提示
                cv2.putText(
                    self.display_image,
                    f"裁剪区域: ({x2-x1}x{y2-y1})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    
    def select_crop_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        交互式选择裁剪区域
        
        Returns:
            (x1, y1, x2, y2) 或 None（如果取消）
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 添加操作提示
        info_image = self.display_image.copy()
        instructions = [
            "操作说明:",
            "- 拖动鼠标选择裁剪区域",
            "- 按 'c' 确认裁剪",
            "- 按 'r' 重新选择",
            "- 按 'q' 退出"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(
                info_image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        
        print("\n=== 交互式裁剪工具 ===")
        print("操作说明:")
        print("  - 在图像上拖动鼠标选择裁剪区域")
        print("  - 按 'c' 确认当前选择")
        print("  - 按 'r' 重新选择区域")
        print("  - 按 'q' 退出不保存")
        print()
        
        while True:
            cv2.imshow(self.window_name, info_image if self.crop_rect is None else self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # 确认裁剪
                if self.crop_rect is not None:
                    print(f"✓ 已确认裁剪区域: {self.crop_rect}")
                    cv2.destroyAllWindows()
                    return self.crop_rect
                else:
                    print("! 请先选择裁剪区域")
                    
            elif key == ord('r'):
                # 重置
                print("重新选择裁剪区域...")
                self.crop_rect = None
                self.display_image = self.original_image.copy()
                info_image = self.display_image.copy()
                y_offset = 30
                for i, text in enumerate(instructions):
                    cv2.putText(
                        info_image,
                        text,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                    
            elif key == ord('q'):
                # 退出
                print("已取消")
                cv2.destroyAllWindows()
                return None
            
            # 更新显示（如果正在绘制）
            if self.drawing:
                info_image = self.display_image.copy()
        
        cv2.destroyAllWindows()
        return None


def crop_image(image_path: Path, crop_rect: Tuple[int, int, int, int], output_path: Path):
    """
    裁剪单个图像
    
    Args:
        image_path: 输入图像路径
        crop_rect: 裁剪区域 (x1, y1, x2, y2)
        output_path: 输出图像路径
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ! 警告: 无法读取图像 {image_path}")
        return False
    
    x1, y1, x2, y2 = crop_rect
    
    # 确保裁剪区域在图像范围内
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        print(f"  ! 警告: 裁剪区域无效 {image_path}")
        return False
    
    # 裁剪
    cropped = image[y1:y2, x1:x2]
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped)
    return True


def batch_crop_images(
    base_dir: Path,
    crop_rect: Tuple[int, int, int, int],
    backup: bool = True
):
    """
    批量裁剪所有静帧图像
    
    Args:
        base_dir: 基础目录 (10_render_vtu_animation_output)
        crop_rect: 裁剪区域 (x1, y1, x2, y2)
        backup: 是否备份原始图像
    """
    # 查找所有类型的输出目录
    damage_types = ['health', 'crack', 'corrosion', 'multi']
    
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    print(f"\n=== 开始批量裁剪 ===")
    print(f"裁剪区域: {crop_rect}")
    print(f"基础目录: {base_dir}")
    print()
    
    # 创建备份目录
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = base_dir / f"backup_before_crop_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"备份目录: {backup_dir}\n")
    
    for damage_type in damage_types:
        type_dir = base_dir / damage_type / "static_frames"
        
        if not type_dir.exists():
            print(f"跳过 {damage_type}: 目录不存在")
            continue
        
        # 查找所有图像
        image_files = list(type_dir.glob("*.png"))
        if not image_files:
            print(f"跳过 {damage_type}: 没有找到图像")
            continue
        
        print(f"处理 {damage_type}: 找到 {len(image_files)} 张图像")
        total_images += len(image_files)
        
        for image_file in sorted(image_files):
            # 备份
            if backup:
                backup_file = backup_dir / damage_type / "static_frames" / image_file.name
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_file, backup_file)
            
            # 裁剪（直接覆盖原文件）
            temp_output = image_file.parent / f"temp_{image_file.name}"
            if crop_image(image_file, crop_rect, temp_output):
                # 替换原文件
                shutil.move(str(temp_output), str(image_file))
                processed_images += 1
            else:
                failed_images += 1
                if temp_output.exists():
                    temp_output.unlink()
        
        print(f"  ✓ 完成 {damage_type}")
    
    # 保存裁剪参数
    crop_config = {
        "crop_rect": crop_rect,
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "processed_images": processed_images,
        "failed_images": failed_images
    }
    
    config_file = base_dir / "crop_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(crop_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 批量裁剪完成 ===")
    print(f"总图像数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"失败数量: {failed_images}")
    print(f"裁剪配置已保存到: {config_file}")
    
    if backup:
        print(f"原始图像已备份到: {backup_dir}")


def main():
    """主函数"""
    # 设置路径
    script_dir = Path(__file__).parent
    output_dir = script_dir / "10_render_vtu_animation_output"
    
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return
    
    # 查找第一张参考图像
    reference_image = None
    for damage_type in ['health', 'crack', 'corrosion', 'multi']:
        static_frames_dir = output_dir / damage_type / "static_frames"
        if static_frames_dir.exists():
            images = list(static_frames_dir.glob("*.png"))
            if images:
                reference_image = sorted(images)[0]
                break
    
    if reference_image is None:
        print("错误: 未找到任何图像文件")
        return
    
    print(f"使用参考图像: {reference_image}")
    
    # 交互式选择裁剪区域
    cropper = InteractiveCropper(str(reference_image))
    crop_rect = cropper.select_crop_region()
    
    if crop_rect is None:
        print("\n已取消裁剪操作")
        return
    
    # 确认操作
    print(f"\n裁剪区域: {crop_rect}")
    x1, y1, x2, y2 = crop_rect
    print(f"裁剪尺寸: {x2-x1} x {y2-y1}")
    print("\n将对所有静帧图像应用此裁剪")
    
    response = input("是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 询问是否备份
    backup_response = input("是否备份原始图像? (y/n, 默认: y): ").strip().lower()
    backup = backup_response != 'n'
    
    # 批量裁剪
    batch_crop_images(output_dir, crop_rect, backup=backup)
    
    print("\n✓ 所有操作完成!")


if __name__ == "__main__":
    main()
