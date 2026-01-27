#!/usr/bin/env python3
"""
Nikon RAW (NEF) 去色罩处理脚本
适用于处理负片扫描的NEF文件，自动去除色罩

使用方法:
    python remove_color_cast.py input.NEF output.jpg --ratio 0.01
"""

import argparse
import numpy as np
import rawpy
import cv2
from pathlib import Path


def remove_color_cast(image, ratio=0.01):
    """
    去色罩处理

    步骤:
    1. 反相图像
    2. 分别调整RGB通道的色阶，使最大值/最小值接近目标范围

    Args:
        image: 输入的RGB图像 (H, W, 3)
        ratio: 色阶调整的超参数，控制最大值最小值的调整范围

    Returns:
        处理后的图像
    """
    # 步骤1: 反相
    print("步骤1: 反相图像...")
    inverted = 1.0 - image

    # 步骤2: 分别调整RGB通道的色阶
    print("步骤2: 调整RGB通道色阶...")

    # 获取每个通道的当前最大值和最小值
    channel_names = ['R', 'G', 'B']
    adjusted_channels = []

    for c in range(3):
        channel = inverted[:, :, c]
        current_min = channel.min()
        current_max = channel.max()

        print(f"  通道{channel_names[c]}: 当前范围 [{current_min:.4f}, {current_max:.4f}]")

        # 计算目标最大值和最小值
        # real_max 是理论最大值1.0，但允许一定的调整范围
        target_max = 1.0 - ratio
        target_min = 0.0 + ratio

        # 如果当前范围已经包含在目标范围内，则不需要调整
        if current_max >= target_max and current_min <= target_min:
            print(f"    通道{channel_names[c]}已在目标范围内，无需调整")
            adjusted_channels.append(channel)
            continue

        # 计算缩放因子，使当前范围映射到目标范围
        current_range = current_max - current_min
        target_range = target_max - target_min

        if current_range > 0:
            # 先归一化到[0, 1]
            normalized = (channel - current_min) / current_range
            # 缩放到目标范围
            scaled = normalized * target_range
            # 平移到目标位置
            adjusted = scaled + target_min
            # 限制在[0, 1]范围内
            adjusted = np.clip(adjusted, 0.0, 1.0)

            print(f"    通道{channel_names[c]}: 调整后范围 [{adjusted.min():.4f}, {adjusted.max():.4f}]")
            adjusted_channels.append(adjusted)
        else:
            print(f"    通道{channel_names[c]}: 范围为0，无法调整")
            adjusted_channels.append(channel)

    # 合并通道
    result = np.stack(adjusted_channels, axis=2)

    return result


def process_nef_file(input_path, output_path, ratio=0.01, use_camera_wb=False):
    """
    处理Nikon NEF文件

    Args:
        input_path: 输入NEF文件路径
        output_path: 输出文件路径
        ratio: 色阶调整的超参数
        use_camera_wb: 是否使用相机的白平衡
    """
    print(f"\n处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"调整比率: {ratio}")
    print(f"使用相机白平衡: {use_camera_wb}")

    # 读取RAW文件
    with rawpy.imread(input_path) as raw:
        print(f"\nRAW文件信息:")
        print(f"  图像尺寸: {raw.raw_image.shape[1]} x {raw.raw_image.shape[0]}")
        print(f"  颜色描述: {raw.color_desc}")
        print(f"  白平衡: {raw.camera_whitebalance}")

        # 使用rawpy进行基本的后处理
        # use_camera_wb=False 以保留原始颜色信息用于去色罩
        params = rawpy.Params(
            use_camera_wb=use_camera_wb,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=False,
        )

        # 获取处理后的RGB图像
        rgb = raw.postprocess(params=params)

        # 归一化到[0, 1]范围
        rgb_normalized = rgb.astype(np.float64) / 65535.0

        print(f"\n归一化后图像范围: [{rgb_normalized.min():.4f}, {rgb_normalized.max():.4f}]")

        # 应用去色罩处理
        result = remove_color_cast(rgb_normalized, ratio=ratio)

        # 转换回16位范围
        result_16bit = np.clip(result * 65535, 0, 65535).astype(np.uint16)

        # 保存结果
        # 根据输出文件扩展名选择保存格式
        output_ext = Path(output_path).suffix.lower()

        if output_ext in ['.jpg', '.jpeg']:
            # JPEG需要8位
            result_8bit = (result_16bit // 256).astype(np.uint8)
            # OpenCV使用BGR顺序，需要转换
            result_bgr = cv2.cvtColor(result_8bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif output_ext == '.png':
            # PNG支持16位
            # OpenCV使用BGR顺序，需要转换
            result_bgr = cv2.cvtColor(result_16bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif output_ext in ['.tif', '.tiff']:
            # TIFF支持16位
            result_bgr = cv2.cvtColor(result_16bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
        else:
            # 默认保存为16位PNG
            result_bgr = cv2.cvtColor(result_16bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        print(f"\n处理完成! 结果已保存到: {output_path}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description='Nikon RAW (NEF) 去色罩处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 基本使用
    python remove_color_cast.py input.NEF output.jpg

    # 自定义调整比率
    python remove_color_cast.py input.NEF output.jpg --ratio 0.02

    # 使用相机白平衡
    python remove_color_cast.py input.NEF output.jpg --use-camera-wb

    # 保存为16位TIFF
    python remove_color_cast.py input.NEF output.tif
        '''
    )

    parser.add_argument('input', help='输入NEF文件路径')
    parser.add_argument('output', help='输出文件路径 (支持.jpg, .png, .tif)')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.01,
        help='色阶调整比率 (默认: 0.01)。较大的值会进行更激进的调整。'
    )
    parser.add_argument(
        '--use-camera-wb',
        action='store_true',
        help='使用相机的白平衡设置'
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 处理文件
    try:
        process_nef_file(
            args.input,
            args.output,
            ratio=args.ratio,
            use_camera_wb=args.use_camera_wb
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
