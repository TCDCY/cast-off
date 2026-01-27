#!/usr/bin/env python3
"""
Nikon RAW (NEF) 去色罩处理脚本
适用于处理负片扫描的NEF文件，自动去除色罩

使用方法:
    python remove_color_cast.py input.NEF output.jpg --ratio 0.01
"""

import argparse
import json
import numpy as np
import rawpy
import cv2
from pathlib import Path


# 默认预设配置
DEFAULT_PRESET = {
    'ratio': 0.01,
    'use_camera_wb': False,
    'roi': 'center',
    'white_balance': 'none',
    'wb_roi': None,
    'exposure': 0.0,
    'brightness': 0.0,
    'contrast': 1.0,
    'highlights': 0.0,
    'shadows': 0.0,
    'temperature': 0.0,  # 色温：负数偏冷(蓝)，正数偏暖(黄)，范围 [-100, 100]
    'tint': 0.0,  # 色调：负数偏绿，正数偏品红，范围 [-100, 100]
}


def temperature_tint_to_rgb(temperature, tint):
    """
    将色温和色调转换为RGB白平衡增益

    Args:
        temperature: 色温值，负数偏冷(蓝)，正数偏暖(黄)，范围 [-100, 100]
        tint: 色调值，负数偏绿，正数偏品红，范围 [-100, 100]

    Returns:
        [R, G, B] 增益系数
    """
    # 色温调整：蓝色通道 <-> 黄色(R+G)
    # temperature < 0: 增加蓝，减少红和绿
    # temperature > 0: 减少蓝，增加红和绿

    temp_factor = temperature / 100.0  # 归一化到 [-1, 1]

    if temp_factor < 0:
        # 偏冷：增加蓝色
        blue_gain = 1.0 + abs(temp_factor) * 0.5
        red_gain = 1.0 - abs(temp_factor) * 0.3
        green_gain = 1.0 - abs(temp_factor) * 0.2
    else:
        # 偏暖：增加红色和绿色
        blue_gain = 1.0 - temp_factor * 0.4
        red_gain = 1.0 + temp_factor * 0.4
        green_gain = 1.0 + temp_factor * 0.2

    # 色调调整：绿色 <-> 品红(R+B)
    # tint < 0: 增加绿色
    # tint > 0: 增加品红(红+蓝)

    tint_factor = tint / 100.0  # 归一化到 [-1, 1]

    if tint_factor < 0:
        # 偏绿
        green_gain = green_gain * (1.0 + abs(tint_factor) * 0.3)
        red_gain = red_gain * (1.0 - abs(tint_factor) * 0.2)
        blue_gain = blue_gain * (1.0 - abs(tint_factor) * 0.2)
    else:
        # 偏品红
        red_gain = red_gain * (1.0 + tint_factor * 0.2)
        blue_gain = blue_gain * (1.0 + tint_factor * 0.2)
        green_gain = green_gain * (1.0 - tint_factor * 0.3)

    # 限制增益范围
    red_gain = max(0.5, min(2.0, red_gain))
    green_gain = max(0.5, min(2.0, green_gain))
    blue_gain = max(0.5, min(2.0, blue_gain))

    return [red_gain, green_gain, blue_gain]


def save_preset(filepath, params):
    """保存预设到JSON文件"""
    preset = {}
    for key, value in params.items():
        if value is not None and value != DEFAULT_PRESET.get(key):
            preset[key] = value
    preset['_version'] = '1.0'
    preset['_description'] = 'Nikon RAW 负片处理预设'

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(preset, f, indent=2, ensure_ascii=False)
    print(f"预设已保存到: {filepath}")


def load_preset(filepath):
    """从JSON文件加载预设"""
    with open(filepath, 'r', encoding='utf-8') as f:
        preset = json.load(f)

    # 移除元数据
    preset.pop('_version', None)
    preset.pop('_description', None)

    # 合并默认值
    for key, value in DEFAULT_PRESET.items():
        if key not in preset:
            preset[key] = value

    return preset


def apply_exposure(image, ev):
    """
    应用曝光调整

    Args:
        image: 输入图像 (H, W, 3)
        ev: 曝光值（EV），正数增加曝光，负数减少曝光

    Returns:
        调整后的图像
    """
    if ev == 0:
        return image

    factor = 2.0 ** ev
    result = np.clip(image * factor, 0.0, 1.0)
    return result


def apply_brightness_contrast(image, brightness=0.0, contrast=1.0):
    """
    应用亮度和对比度调整

    Args:
        image: 输入图像 (H, W, 3)
        brightness: 亮度调整，范围 [-1, 1]，0为不调整
        contrast: 对比度调整，范围 [0, 3+]，1为不调整

    Returns:
        调整后的图像
    """
    if brightness == 0 and contrast == 1.0:
        return image

    # 应用对比度
    if contrast != 1.0:
        result = (image - 0.5) * contrast + 0.5
    else:
        result = image

    # 应用亮度
    if brightness != 0:
        result = result + brightness

    return np.clip(result, 0.0, 1.0)


def apply_highlights_shadows(image, highlights=0.0, shadows=0.0):
    """
    应用高光和阴影调整

    Args:
        image: 输入图像 (H, W, 3)
        highlights: 高光调整，范围 [-100, 100]，0为不调整
                   负数降低高光，正数提亮高光
        shadows: 阴影调整，范围 [-100, 100]，0为不调整
                负数压暗阴影，正数提亮阴影

    Returns:
        调整后的图像
    """
    if highlights == 0 and shadows == 0:
        return image

    result = image.copy()

    # 高光调整
    if highlights != 0:
        highlight_threshold = 0.6
        highlight_mask = (image > highlight_threshold).astype(np.float32)
        highlight_adjustment = (image - highlight_threshold) * (highlights / 100.0)
        result = result + highlight_mask * highlight_adjustment

    # 阴影调整
    if shadows != 0:
        shadow_threshold = 0.4
        shadow_mask = (image < shadow_threshold).astype(np.float32)
        shadow_adjustment = (shadow_threshold - image) * (shadows / 100.0) * shadow_mask
        result = result - shadow_adjustment

    return np.clip(result, 0.0, 1.0)


def apply_manual_white_balance(image, temperature=0.0, tint=0.0):
    """
    应用手动白平衡（使用色温和色调）

    Args:
        image: 输入图像 (H, W, 3)
        temperature: 色温值，负数偏冷(蓝)，正数偏暖(黄)
        tint: 色调值，负数偏绿，正数偏品红

    Returns:
        调整后的图像
    """
    if temperature == 0.0 and tint == 0.0:
        return image

    # 转换为RGB增益
    multipliers = temperature_tint_to_rgb(temperature, tint)

    print(f"  色温调整: {temperature:+.1f} (冷←→暖)")
    print(f"  色调调整: {tint:+.1f} (绿←→品红)")
    print(f"  RGB增益: R={multipliers[0]:.3f}, G={multipliers[1]:.3f}, B={multipliers[2]:.3f}")

    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] * multipliers[c], 0.0, 1.0)

    return result


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


def apply_white_balance(image, method='gray-world', roi=None):
    """
    应用白平衡矫正

    Args:
        image: 输入的RGB图像 (H, W, 3)
        method: 白平衡方法
            - 'gray-world': 灰度世界假设（默认）
            - 'perfect-reflector': 完美反射假设
            - 'auto': 自动选择（结合gray-world和perfect-reflector）
        roi: 用于分析白平衡的区域，如果为None则使用全图

    Returns:
        白平衡后的图像
    """
    print(f"\n应用白平衡矫正 (方法: {method})...")

    # 如果指定了ROI，在ROI区域分析白平衡
    if roi is not None:
        if isinstance(roi, str):
            analysis_region = extract_roi(image, roi)
        else:
            analysis_region = image[roi[1]:roi[3], roi[0]:roi[2]]
    else:
        analysis_region = image

    channel_names = ['R', 'G', 'B']

    if method == 'gray-world':
        # 灰度世界假设：假设图像的平均颜色是灰色
        print("  使用灰度世界假设")

        # 计算每个通道的平均值
        means = [
            analysis_region[:, :, c].mean()
            for c in range(3)
        ]

        print(f"  通道平均值: R={means[0]:.4f}, G={means[1]:.4f}, B={means[2]:.4f}")

        # 以绿色通道为基准（或使用最大值）
        # 这里使用最大值作为基准，避免某个通道过曝
        max_mean = max(means)

        # 计算增益系数
        gains = [max_mean / m if m > 0 else 1.0 for m in means]

        print(f"  增益系数: R={gains[0]:.4f}, G={gains[1]:.4f}, B={gains[2]:.4f}")

    elif method == 'perfect-reflector':
        # 完美反射假设：假设最亮的像素应该是白色
        print("  使用完美反射假设")

        # 使用percentile来避免极端值的影响
        percentile = 99.5

        # 计算每个通道的99.5分位数
        brights = [
            np.percentile(analysis_region[:, :, c], percentile)
            for c in range(3)
        ]

        print(f"  通道{percentile}%分位数: R={brights[0]:.4f}, G={brights[1]:.4f}, B={brights[2]:.4f}")

        # 以最小值作为基准（让最亮的通道保持不变）
        min_bright = min(brights)

        # 计算增益系数
        gains = [min_bright / b if b > 0 else 1.0 for b in brights]

        print(f"  增益系数: R={gains[0]:.4f}, G={gains[1]:.4f}, B={gains[2]:.4f}")

    elif method == 'auto':
        # 自动选择：结合两种方法
        print("  使用自动白平衡（结合灰度世界和完美反射）")

        # 灰度世界
        means = [analysis_region[:, :, c].mean() for c in range(3)]
        max_mean = max(means)
        gains_gray = [max_mean / m if m > 0 else 1.0 for m in means]

        # 完美反射
        percentile = 99.5
        brights = [np.percentile(analysis_region[:, :, c], percentile) for c in range(3)]
        min_bright = min(brights)
        gains_reflector = [min_bright / b if b > 0 else 1.0 for b in brights]

        # 取平均
        gains = [(g1 + g2) / 2 for g1, g2 in zip(gains_gray, gains_reflector)]

        print(f"  增益系数: R={gains[0]:.4f}, G={gains[1]:.4f}, B={gains[2]:.4f}")

    else:
        print(f"  警告: 未知的白平衡方法 '{method}'，跳过白平衡")
        return image

    # 应用增益到全图
    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] * gains[c], 0.0, 1.0)

    print(f"  白平衡后范围: [{result.min():.4f}, {result.max():.4f}]")

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
                     roi='center', show_roi=False, white_balance='none',
                     wb_roi=None, exposure=0.0, brightness=0.0, contrast=1.0,
                     highlights=0.0, shadows=0.0, temperature=0.0, tint=0.0,
                     save_preset_path=None, load_preset_path=None,
                     relative_adjust=False):
    """
    处理Nikon NEF文件

    Args:
        input_path: 输入NEF文件路径
        output_path: 输出文件路径
        ratio: 色阶调整的超参数
        use_camera_wb: 是否使用相机的白平衡
        roi: 感兴趣区域配置，用于分析色罩
        show_roi: 是否显示并保存ROI区域的可视化图像
        white_balance: 白平衡方法
        wb_roi: 白平衡分析区域
        exposure: 曝光调整（EV）
        brightness: 亮度调整 [-1, 1]
        contrast: 对比度调整 [0, 3+]
        highlights: 高光调整 [-100, 100]
        shadows: 阴影调整 [-100, 100]
        temperature: 色温调整 [-100, 100]，负数偏冷，正数偏暖
        tint: 色调调整 [-100, 100]，负数偏绿，正数偏品红
        save_preset_path: 保存预设到文件
        load_preset_path: 从文件加载预设
        relative_adjust: 是否使用相对值调整（在预设基础上微调）
    """
    # 如果指定了预设文件，先加载预设
    if load_preset_path:
        print(f"\n加载预设: {load_preset_path}")
        preset = load_preset(load_preset_path)

        # 更新参数
        if relative_adjust:
            # 相对值调整：命令行参数作为增量叠加到预设值
            print("使用相对值调整模式")
            ratio = preset.get('ratio', ratio)
            use_camera_wb = preset.get('use_camera_wb', use_camera_wb)
            roi = preset.get('roi', roi)
            white_balance = preset.get('white_balance', white_balance)
            wb_roi = preset.get('wb_roi', wb_roi)
            exposure = preset.get('exposure', 0.0) + exposure
            brightness = preset.get('brightness', 0.0) + brightness
            contrast = preset.get('contrast', 1.0) + (contrast - 1.0)
            highlights = preset.get('highlights', 0.0) + highlights
            shadows = preset.get('shadows', 0.0) + shadows
            temperature = preset.get('temperature', 0.0) + temperature
            tint = preset.get('tint', 0.0) + tint
            print(f"在预设基础上微调")
        else:
            # 绝对值调整：命令行参数覆盖预设值（仅当使用默认值时）
            if ratio == DEFAULT_PRESET['ratio']: ratio = preset.get('ratio', ratio)
            if use_camera_wb == DEFAULT_PRESET['use_camera_wb']: use_camera_wb = preset.get('use_camera_wb', use_camera_wb)
            if roi == DEFAULT_PRESET['roi']: roi = preset.get('roi', roi)
            if white_balance == DEFAULT_PRESET['white_balance']: white_balance = preset.get('white_balance', white_balance)
            if wb_roi == DEFAULT_PRESET['wb_roi']: wb_roi = preset.get('wb_roi', wb_roi)
            if exposure == DEFAULT_PRESET['exposure']: exposure = preset.get('exposure', exposure)
            if brightness == DEFAULT_PRESET['brightness']: brightness = preset.get('brightness', brightness)
            if contrast == DEFAULT_PRESET['contrast']: contrast = preset.get('contrast', contrast)
            if highlights == DEFAULT_PRESET['highlights']: highlights = preset.get('highlights', highlights)
            if shadows == DEFAULT_PRESET['shadows']: shadows = preset.get('shadows', shadows)
            if temperature == DEFAULT_PRESET['temperature']: temperature = preset.get('temperature', temperature)
            if tint == DEFAULT_PRESET['tint']: tint = preset.get('tint', tint)

        print(f"预设参数已加载")

    print(f"\n处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"调整比率: {ratio}")
    print(f"使用相机白平衡: {use_camera_wb}")
    print(f"分析区域: {roi}")
    print(f"白平衡方法: {white_balance}")
    if exposure != 0:
        print(f"曝光调整: {exposure:+.2f} EV")
    if brightness != 0:
        print(f"亮度调整: {brightness:+.2f}")
    if contrast != 1.0:
        print(f"对比度调整: {contrast:.2f}")
    if highlights != 0:
        print(f"高光调整: {highlights:+.1f}")
    if shadows != 0:
        print(f"阴影调整: {shadows:+.1f}")
    if temperature != 0:
        print(f"色温调整: {temperature:+.1f}")
    if tint != 0:
        print(f"色调调整: {tint:+.1f}")

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

        # 应用白平衡（如果启用）
        if white_balance != 'none':
            # 如果没有指定白平衡ROI，使用与色罩相同的ROI
            wb_analysis_roi = wb_roi if wb_roi is not None else roi
            result = apply_white_balance(result, method=white_balance, roi=wb_analysis_roi)

        # 应用手动白平衡（如果指定了temperature或tint）
        if temperature != 0.0 or tint != 0.0:
            print(f"\n应用手动白平衡...")
            result = apply_manual_white_balance(result, temperature, tint)

        # 应用曝光调整
        if exposure != 0:
            print(f"\n应用曝光调整...")
            result = apply_exposure(result, exposure)
            print(f"  曝光调整后范围: [{result.min():.4f}, {result.max():.4f}]")

        # 应用亮度/对比度调整
        if brightness != 0 or contrast != 1.0:
            print(f"\n应用亮度/对比度调整...")
            result = apply_brightness_contrast(result, brightness, contrast)
            print(f"  调整后范围: [{result.min():.4f}, {result.max():.4f}]")

        # 应用高光/阴影调整
        if highlights != 0 or shadows != 0:
            print(f"\n应用高光/阴影调整...")
            result = apply_highlights_shadows(result, highlights, shadows)
            print(f"  调整后范围: [{result.min():.4f}, {result.max():.4f}]")

        # 保存预设（如果指定）
        if save_preset_path:
            print(f"\n保存预设...")
            preset_params = {
                'ratio': ratio,
                'use_camera_wb': use_camera_wb,
                'roi': roi if isinstance(roi, str) else list(roi) if isinstance(roi, tuple) else roi,
                'white_balance': white_balance,
                'wb_roi': wb_roi if isinstance(wb_roi, str) else list(wb_roi) if isinstance(wb_roi, tuple) else wb_roi,
                'exposure': exposure,
                'brightness': brightness,
                'contrast': contrast,
                'highlights': highlights,
                'shadows': shadows,
                'temperature': temperature,
                'tint': tint,
            }
            save_preset(save_preset_path, preset_params)

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

    # 色温调整（正数偏暖，负数偏冷）
    python remove_color_cast.py input.NEF output.jpg --temperature +20

    # 色调调整（正数偏品红，负数偏绿）
    python remove_color_cast.py input.NEF output.jpg --tint -10

    # 曝光调整
    python remove_color_cast.py input.NEF output.jpg --exposure +0.5

    # 保存预设
    python remove_color_cast.py input.NEF output.jpg --exposure +0.3 --temperature +10 --save-preset my_settings.json

    # 使用预设（绝对值模式，命令行参数覆盖预设）
    python remove_color_cast.py input.NEF output.jpg --load-preset my_settings.json

    # 使用预设并微调（相对值模式）
    python remove_color_cast.py input.NEF output.jpg --load-preset my_settings.json --relative-adjust --temperature +5 --exposure +0.1

    # 完整示例
    python remove_color_cast.py input.NEF output.jpg \\
        --roi center-60 \\
        --exposure +0.3 \\
        --contrast 1.1 \\
        --temperature +15 \\
        --tint -5 \\
        --save-preset negative_film.json
        '''
    )

    parser.add_argument('input', help='输入NEF文件路径')
    parser.add_argument('output', help='输出文件路径 (支持.jpg, .png, .tif)')

    # 基础参数
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.01,
        help='色阶调整比率 (默认: 0.01)'
    )
    parser.add_argument(
        '--use-camera-wb',
        action='store_true',
        help='使用相机的白平衡设置'
    )
    parser.add_argument(
        '--roi',
        default='center',
        help='色罩分析区域 (默认: center)。可选值: center, center-60, center-50, full, 或 x1,y1,x2,y2'
    )
    parser.add_argument(
        '--show-roi',
        action='store_true',
        help='显示并保存ROI区域的可视化图像'
    )

    # 白平衡参数
    parser.add_argument(
        '--white-balance',
        choices=['none', 'gray-world', 'perfect-reflector', 'auto'],
        default='none',
        help='白平衡方法 (默认: none)'
    )
    parser.add_argument(
        '--wb-roi',
        default=None,
        help='白平衡分析区域 (默认: 与色罩ROI相同)'
    )

    # 色温和色调调整（手动白平衡）
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        metavar='VAL',
        help='色温调整，范围 [-100, 100]。负数偏冷(蓝)，正数偏暖(黄)'
    )
    parser.add_argument(
        '--tint',
        type=float,
        default=0.0,
        metavar='VAL',
        help='色调调整，范围 [-100, 100]。负数偏绿，正数偏品红'
    )

    # 亮度调整参数
    parser.add_argument(
        '--exposure',
        type=float,
        default=0.0,
        metavar='EV',
        help='曝光调整，例如: +0.5, -0.3 (单位: EV)'
    )
    parser.add_argument(
        '--brightness',
        type=float,
        default=0.0,
        metavar='VAL',
        help='亮度调整，范围 [-1, 1]，例如: 0.1, -0.2'
    )
    parser.add_argument(
        '--contrast',
        type=float,
        default=1.0,
        metavar='VAL',
        help='对比度调整，正常值1.0，例如: 1.2, 0.9'
    )

    # 高光/阴影调整
    parser.add_argument(
        '--highlights',
        type=float,
        default=0.0,
        metavar='VAL',
        help='高光调整，范围 [-100, 100]，负数降低高光'
    )
    parser.add_argument(
        '--shadows',
        type=float,
        default=0.0,
        metavar='VAL',
        help='阴影调整，范围 [-100, 100]，正数提亮阴影'
    )

    # 预设参数
    parser.add_argument(
        '--save-preset',
        metavar='PATH',
        default=None,
        help='保存当前设置到预设文件'
    )
    parser.add_argument(
        '--load-preset',
        metavar='PATH',
        default=None,
        help='从预设文件加载设置'
    )
    parser.add_argument(
        '--relative-adjust',
        action='store_true',
        help='使用相对值调整模式（在预设基础上微调，需要配合--load-preset使用）'
    )

    args = parser.parse_args()

    # 解析ROI参数
    roi = args.roi
    try:
        if ',' in roi:
            coords = [int(x.strip()) for x in roi.split(',')]
            if len(coords) == 4:
                roi = tuple(coords)
                print(f"使用自定义ROI区域: {roi}")
    except ValueError:
        pass

    # 解析白平衡ROI参数
    wb_roi = args.wb_roi
    if wb_roi is not None:
        try:
            if ',' in wb_roi:
                coords = [int(x.strip()) for x in wb_roi.split(',')]
                if len(coords) == 4:
                    wb_roi = tuple(coords)
                    print(f"使用自定义白平衡ROI区域: {wb_roi}")
        except ValueError:
            pass

    # 检查相对值调整模式
    if args.relative_adjust and not args.load_preset:
        print("警告: --relative-adjust 需要配合 --load-preset 使用，将忽略此选项")

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
            show_roi=args.show_roi,
            white_balance=args.white_balance,
            wb_roi=wb_roi,
            exposure=args.exposure,
            brightness=args.brightness,
            contrast=args.contrast,
            highlights=args.highlights,
            shadows=args.shadows,
            temperature=args.temperature,
            tint=args.tint,
            save_preset_path=args.save_preset,
            load_preset_path=args.load_preset,
            relative_adjust=args.relative_adjust
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
