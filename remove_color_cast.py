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


def analyze_color_cast(image, ratio=0.01):
    """
    分析图像的色罩参数

    在指定区域内分析每个RGB通道的范围，计算需要的色阶调整参数

    Args:
        image: 输入的RGB图像 (H, W, 3)
        ratio: 色阶调整的超参数

    Returns:
        (inverted, adjust_params): 反相后的图像和每个通道的调整参数
    """
    # 步骤1: 反相
    print("步骤1: 反相图像...")
    inverted = 1.0 - image

    # 步骤2: 分析每个通道的范围
    print("步骤2: 分析RGB通道色阶...")

    channel_names = ['R', 'G', 'B']
    adjust_params = []

    for c in range(3):
        channel = inverted[:, :, c]
        current_min = channel.min()
        current_max = channel.max()

        print(f"  通道{channel_names[c]}: 当前范围 [{current_min:.4f}, {current_max:.4f}]")

        # 计算目标最大值和最小值
        target_max = 1.0 - ratio
        target_min = 0.0 + ratio

        # 计算调整参数
        current_range = current_max - current_min
        target_range = target_max - target_min

        if current_range > 0:
            # 保存调整参数
            params = {
                'min': current_min,
                'max': current_max,
                'range': current_range,
                'target_min': target_min,
                'target_max': target_max,
                'target_range': target_range
            }
            adjust_params.append(params)
            print(f"    通道{channel_names[c]}: 将调整到 [{target_min:.4f}, {target_max:.4f}]")
        else:
            # 无法调整
            adjust_params.append(None)
            print(f"    通道{channel_names[c]}: 范围为0，无法调整")

    return inverted, adjust_params


def apply_color_cast_adjustment(inverted, adjust_params):
    """
    应用色罩调整参数

    使用分析得到的调整参数对图像进行色阶调整

    Args:
        inverted: 反相后的图像
        adjust_params: 每个通道的调整参数

    Returns:
        调整后的图像
    """
    print("步骤3: 应用色阶调整...")

    channel_names = ['R', 'G', 'B']
    adjusted_channels = []

    for c in range(3):
        channel = inverted[:, :, c]
        params = adjust_params[c]

        if params is None:
            print(f"  通道{channel_names[c]}: 无调整参数，跳过")
            adjusted_channels.append(channel)
            continue

        # 应用调整
        # 先归一化到[0, 1]
        normalized = (channel - params['min']) / params['range']
        # 缩放到目标范围
        scaled = normalized * params['target_range']
        # 平移到目标位置
        adjusted = scaled + params['target_min']
        # 限制在[0, 1]范围内
        adjusted = np.clip(adjusted, 0.0, 1.0)

        print(f"  通道{channel_names[c]}: 调整后范围 [{adjusted.min():.4f}, {adjusted.max():.4f}]")
        adjusted_channels.append(adjusted)

    # 合并通道
    result = np.stack(adjusted_channels, axis=2)

    return result


def remove_color_cast(image, ratio=0.01):
    """
    去色罩处理（完整流程）

    步骤:
    1. 反相图像
    2. 分析RGB通道的色阶参数
    3. 应用色阶调整

    Args:
        image: 输入的RGB图像 (H, W, 3)
        ratio: 色阶调整的超参数，控制最大值最小值的调整范围

    Returns:
        处理后的图像
    """
    inverted, adjust_params = analyze_color_cast(image, ratio)
    result = apply_color_cast_adjustment(inverted, adjust_params)
    return result


def extract_roi(image, roi_config='center'):
    """
    提取感兴趣区域(ROI)用于分析

    Args:
        image: 输入图像 (H, W, C)
        roi_config: ROI配置
            - 'center': 中心80%区域
            - 'center-60': 中心60%区域
            - 'center-50': 中心50%区域
            - 'full': 全图
            - (x1, y1, x2, y2): 自定义矩形区域 (像素坐标)

    Returns:
        ROI图像
    """
    h, w = image.shape[:2]

    if roi_config == 'full':
        return image
    elif roi_config == 'center':
        # 默认使用中心80%
        margin_h = int(h * 0.1)
        margin_w = int(w * 0.1)
        return image[margin_h:h-margin_h, margin_w:w-margin_w]
    elif roi_config == 'center-60':
        margin_h = int(h * 0.2)
        margin_w = int(w * 0.2)
        return image[margin_h:h-margin_h, margin_w:w-margin_w]
    elif roi_config == 'center-50':
        margin_h = int(h * 0.25)
        margin_w = int(w * 0.25)
        return image[margin_h:h-margin_h, margin_w:w-margin_w]
    elif isinstance(roi_config, (tuple, list)) and len(roi_config) == 4:
        x1, y1, x2, y2 = roi_config
        return image[y1:y2, x1:x2]
    else:
        print(f"警告: 未知的ROI配置 '{roi_config}'，使用全图")
        return image


def process_nef_file(input_path, output_path, ratio=0.01, use_camera_wb=False,
                     roi='center', show_roi=False):
    """
    处理Nikon NEF文件

    Args:
        input_path: 输入NEF文件路径
        output_path: 输出文件路径
        ratio: 色阶调整的超参数
        use_camera_wb: 是否使用相机的白平衡
        roi: 感兴趣区域配置，用于分析色罩
            - 'center': 中心80%区域（默认）
            - 'center-60': 中心60%区域
            - 'center-50': 中心50%区域
            - 'full': 全图
            - (x1, y1, x2, y2): 自定义矩形区域（像素坐标）
        show_roi: 是否显示并保存ROI区域的可视化图像
    """
    print(f"\n处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"调整比率: {ratio}")
    print(f"使用相机白平衡: {use_camera_wb}")
    print(f"分析区域: {roi}")

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

        # 提取ROI用于分析
        print(f"\n提取分析区域...")
        roi_image = extract_roi(rgb_normalized, roi)
        print(f"  ROI尺寸: {roi_image.shape[1]} x {roi_image.shape[0]}")

        # 可视化ROI（如果需要）
        if show_roi:
            roi_vis = rgb_normalized.copy()
            h, w = rgb_normalized.shape[:2]

            # 绘制ROI边框
            if roi == 'center':
                margin_h = int(h * 0.1)
                margin_w = int(w * 0.1)
                roi_vis = cv2.rectangle(roi_vis, (margin_w, margin_h),
                                       (w-margin_w, h-margin_h), (1, 0, 0), 10)
            elif roi == 'center-60':
                margin_h = int(h * 0.2)
                margin_w = int(w * 0.2)
                roi_vis = cv2.rectangle(roi_vis, (margin_w, margin_h),
                                       (w-margin_w, h-margin_h), (1, 0, 0), 10)
            elif roi == 'center-50':
                margin_h = int(h * 0.25)
                margin_w = int(w * 0.25)
                roi_vis = cv2.rectangle(roi_vis, (margin_w, margin_h),
                                       (w-margin_w, h-margin_h), (1, 0, 0), 10)

            # 保存ROI可视化
            roi_vis_path = Path(output_path).parent / f"{Path(output_path).stem}_roi_vis.jpg"
            roi_vis_8bit = (roi_vis * 255).astype(np.uint8)
            roi_vis_bgr = cv2.cvtColor(roi_vis_8bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(roi_vis_path), roi_vis_bgr)
            print(f"  ROI可视化已保存: {roi_vis_path}")

        # 在ROI区域分析色罩参数
        print(f"\n在ROI区域分析色罩...")
        inverted_roi, adjust_params = analyze_color_cast(roi_image, ratio=ratio)

        # 对整张图应用调整
        print(f"\n对整张图应用色罩调整...")
        inverted_full = 1.0 - rgb_normalized
        result = apply_color_cast_adjustment(inverted_full, adjust_params)

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
    # 基本使用（使用中心80%区域分析）
    python remove_color_cast.py input.NEF output.jpg

    # 使用更小的中心区域分析（60%）
    python remove_color_cast.py input.NEF output.jpg --roi center-60

    # 使用全图分析
    python remove_color_cast.py input.NEF output.jpg --roi full

    # 自定义分析区域 (x1, y1, x2, y2)
    python remove_color_cast.py input.NEF output.jpg --roi 1000,500,4000,3000

    # 显示ROI可视化
    python remove_color_cast.py input.NEF output.jpg --show-roi

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
    parser.add_argument(
        '--roi',
        default='center',
        help='色罩分析区域 (默认: center)。可选值: center, center-60, center-50, full, 或自定义坐标 x1,y1,x2,y2'
    )
    parser.add_argument(
        '--show-roi',
        action='store_true',
        help='显示并保存ROI区域的可视化图像'
    )

    args = parser.parse_args()

    # 解析ROI参数
    roi = args.roi
    try:
        # 尝试解析为自定义坐标
        if ',' in roi:
            coords = [int(x.strip()) for x in roi.split(',')]
            if len(coords) == 4:
                roi = tuple(coords)
                print(f"使用自定义ROI区域: {roi}")
    except ValueError:
        pass

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
            use_camera_wb=args.use_camera_wb,
            roi=roi,
            show_roi=args.show_roi
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
