#!/usr/bin/env python3
"""
Nikon RAW Film Mask Removal Script
Process: Base white balance -> Invert -> RGB level adjustment

Refactored with Stage-based Pipeline architecture.
"""

import numpy as np
import rawpy
import cv2
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

# Color map for visualization (BGR format for OpenCV)
CLUSTER_COLORS = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
]


def extract_region_pixels(
    image: np.ndarray,
    region: str = 'border',
    border_specs: Optional[Dict[str, float]] = None,
    center_ratio: float = 0.5,
) -> np.ndarray:
    """Extract pixels from specified region of image.

    Args:
        image: Input image (H, W, C)
        region: Region type - 'border', 'center', or 'manual'
        border_specs: Border ratios for 'border' mode: {'u': 0.05, 'd': 0.05, 'l': 0.05, 'r': 0.05}
        center_ratio: Center rectangle ratio for 'center' mode (0.0-1.0)

    Returns:
        Extracted pixels as (N, C) array
    """
    h, w = image.shape[:2]

    if region == 'border':
        # Extract borders
        if border_specs is None:
            border_specs = {'u': 0.05, 'd': 0.05, 'l': 0.05, 'r': 0.05}

        u_ratio = border_specs.get('u', 0.05)
        d_ratio = border_specs.get('d', 0.05)
        l_ratio = border_specs.get('l', 0.05)
        r_ratio = border_specs.get('r', 0.05)

        border_h_top = int(h * u_ratio)
        border_h_bottom = int(h * d_ratio)
        border_w_left = int(w * l_ratio)
        border_w_right = int(w * r_ratio)

        border_pixels_list = []

        if border_h_top > 0:
            top = image[:border_h_top, :, :]
            border_pixels_list.append(top.reshape(-1, 3))

        if border_h_bottom > 0:
            bottom = image[-border_h_bottom:, :, :]
            border_pixels_list.append(bottom.reshape(-1, 3))

        if border_w_left > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                left = image[v_start:v_end, :border_w_left, :]
                border_pixels_list.append(left.reshape(-1, 3))

        if border_w_right > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                right = image[v_start:v_end, -border_w_right:, :]
                border_pixels_list.append(right.reshape(-1, 3))

        if border_pixels_list:
            return np.vstack(border_pixels_list)
        else:
            return np.empty((0, 3), dtype=image.dtype)

    elif region == 'center':
        # Extract center rectangle
        h_start = int(h * (1 - center_ratio) / 2)
        h_end = int(h * (1 + center_ratio) / 2)
        w_start = int(w * (1 - center_ratio) / 2)
        w_end = int(w * (1 + center_ratio) / 2)

        center_region = image[h_start:h_end, w_start:w_end, :]
        return center_region.reshape(-1, 3)

    else:
        # Custom region: parse as "x,y,w,h" format (ratios 0-1)
        try:
            parts = [float(x) for x in region.split(',')]
            if len(parts) == 4:
                x, y, rw, rh = parts
                h_start = int(h * y)
                h_end = int(h * (y + rh))
                w_start = int(w * x)
                w_end = int(w * (x + rw))

                custom_region = image[h_start:h_end, w_start:w_end, :]
                return custom_region.reshape(-1, 3)
        except:
            pass

        # Fallback to entire image
        return image.reshape(-1, 3)


def find_level_point(
    pixels: np.ndarray,
    level_type: str,
    threshold: float,
    max_val: int = 65535,
    method: str = 'cumulative'
) -> float:
    """Find black or white point from histogram using specified method.

    Args:
        pixels: 1D array of pixel values
        level_type: 'black' or 'white'
        threshold: Pixel count threshold (as fraction of total pixels)
        max_val: Maximum pixel value (e.g., 65535 for 16-bit)
        method: 'cumulative', 'peak', or 'first_peak'

    Returns:
        Detected level point value
    """
    hist, bin_edges = np.histogram(pixels, bins=512, range=(0, max_val))
    total_pixels = pixels.size

    if method == 'cumulative':
        # Cumulative method: accumulate from edge until threshold
        target_count = total_pixels * threshold

        if level_type == 'black':
            # From left, accumulate until reaching threshold
            cumulative = 0
            for i in range(len(hist)):
                cumulative += hist[i]
                if cumulative >= target_count:
                    return bin_edges[i + 1] if i < len(hist) - 1 else bin_edges[i]
            return bin_edges[-1]
        else:
            # From right, accumulate until reaching threshold
            cumulative = 0
            for i in range(len(hist) - 1, -1, -1):
                cumulative += hist[i]
                if cumulative >= target_count:
                    return bin_edges[i]
            return bin_edges[0]

    elif method == 'peak':
        # Peak method: find histogram peak and extend outward
        level_threshold = total_pixels * threshold
        max_bin = np.argmax(hist)

        if level_type == 'black':
            # From peak, go left until count drops below threshold
            for i in range(max_bin, -1, -1):
                if hist[i] < level_threshold:
                    return bin_edges[i + 1] if i < max_bin else bin_edges[i]
            return bin_edges[0]
        else:
            # From peak, go right until count drops below threshold
            for i in range(max_bin, len(hist)):
                if hist[i] < level_threshold:
                    return bin_edges[i]
            return bin_edges[-1]

    else:  # method == 'first_peak'
        # First peak method: find first significant peak from edge
        level_threshold = total_pixels * threshold

        if level_type == 'black':
            # From left, find first peak, then go to its left edge
            # Find first bin where count exceeds threshold
            start_bin = 0
            for i in range(len(hist)):
                if hist[i] > level_threshold:
                    start_bin = i
                    break
            else:
                return bin_edges[0]

            # Find local maximum around this region
            peak_bin = start_bin
            max_count = hist[start_bin]
            search_range = 20  # bins to search for local peak
            for i in range(start_bin, min(start_bin + search_range, len(hist))):
                if hist[i] > max_count:
                    max_count = hist[i]
                    peak_bin = i

            # From peak, go left until count drops below threshold
            for i in range(peak_bin, -1, -1):
                if hist[i] < level_threshold:
                    return bin_edges[i + 1] if i < peak_bin else bin_edges[i]
            return bin_edges[0]
        else:
            # From right, find first peak, then go to its right edge
            # Find first bin where count exceeds threshold
            start_bin = len(hist) - 1
            for i in range(len(hist) - 1, -1, -1):
                if hist[i] > level_threshold:
                    start_bin = i
                    break
            else:
                return bin_edges[-1]

            # Find local maximum around this region
            peak_bin = start_bin
            max_count = hist[start_bin]
            search_range = 20  # bins to search for local peak
            for i in range(start_bin, max(start_bin - search_range, -1), -1):
                if hist[i] > max_count:
                    max_count = hist[i]
                    peak_bin = i

            # From peak, go right until count drops below threshold
            for i in range(peak_bin, len(hist)):
                if hist[i] < level_threshold:
                    return bin_edges[i]
            return bin_edges[-1]


def create_region_mask(
    h: int,
    w: int,
    region: str = 'border',
    border_specs: Optional[Dict[str, float]] = None,
    center_ratio: float = 0.5,
) -> np.ndarray:
    """Create a boolean mask for the specified region.

    Args:
        h: Image height
        w: Image width
        region: Region type - 'border', 'center', or 'manual'
        border_specs: Border ratios for 'border' mode
        center_ratio: Center rectangle ratio for 'center' mode

    Returns:
        Boolean mask (H, W)
    """
    mask = np.zeros((h, w), dtype=bool)

    if region == 'border':
        if border_specs is None:
            border_specs = {'u': 0.05, 'd': 0.05, 'l': 0.05, 'r': 0.05}

        u_ratio = border_specs.get('u', 0.05)
        d_ratio = border_specs.get('d', 0.05)
        l_ratio = border_specs.get('l', 0.05)
        r_ratio = border_specs.get('r', 0.05)

        border_h_top = int(h * u_ratio)
        border_h_bottom = int(h * d_ratio)
        border_w_left = int(w * l_ratio)
        border_w_right = int(w * r_ratio)

        if border_h_top > 0:
            mask[:border_h_top, :] = True
        if border_h_bottom > 0:
            mask[-border_h_bottom:, :] = True
        if border_w_left > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                mask[v_start:v_end, :border_w_left] = True
        if border_w_right > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                mask[v_start:v_end, -border_w_right:] = True

    elif region == 'center':
        h_start = int(h * (1 - center_ratio) / 2)
        h_end = int(h * (1 + center_ratio) / 2)
        w_start = int(w * (1 - center_ratio) / 2)
        w_end = int(w * (1 + center_ratio) / 2)
        mask[h_start:h_end, w_start:w_end] = True

    else:
        # Custom region: parse as "x,y,w,h" format
        try:
            parts = [float(x) for x in region.split(',')]
            if len(parts) == 4:
                x, y, rw, rh = parts
                h_start = int(h * y)
                h_end = int(h * (y + rh))
                w_start = int(w * x)
                w_end = int(w * (x + rw))
                mask[h_start:h_end, w_start:w_end] = True
        except:
            pass

    return mask


def create_histogram_panel(
    histograms: List[np.ndarray],
    panel_height: int,
    panel_width: int,
    title: str = "Histogram",
    channel_names: Optional[List[str]] = None,
    colors: Optional[List[tuple]] = None,
    max_val: int = 65535,
    detected_levels: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """Create a histogram visualization panel.

    Args:
        histograms: List of histogram arrays (one per channel)
        panel_height: Height of the panel in pixels
        panel_width: Width of the panel in pixels
        title: Title for the histogram
        channel_names: Optional list of channel names (e.g., ['R', 'G', 'B'])
        colors: Optional list of BGR colors for each channel
        max_val: Maximum pixel value (e.g., 65535 for 16-bit)
        detected_levels: Optional dict of detected levels {'R': {'black': x, 'white': y}, ...}

    Returns:
        Panel image as numpy array
    """
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel.fill(30)  # Dark gray background

    # Histogram margins
    margin_left = 60
    margin_right = 20
    margin_top = 25
    margin_bottom = 35
    hist_width = panel_width - margin_left - margin_right
    hist_height = panel_height - margin_top - margin_bottom

    # Find max count across all histograms for normalization
    max_count = 0
    for hist in histograms:
        max_count = max(max_count, np.max(hist))

    # Default colors
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][:len(histograms)]

    # Draw title
    cv2.putText(panel, title, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw y-axis grid lines and labels (actual pixel counts)
    for y_ratio in [1.0, 0.5, 0.0]:
        y = margin_top + int((1.0 - y_ratio) * hist_height * 0.85)
        # Draw horizontal grid line
        cv2.line(panel, (margin_left, y), (margin_left + hist_width, y), (80, 80, 80), 1)
        # Draw label with actual pixel count
        pixel_count = int(max_count * y_ratio)
        if pixel_count >= 1000000:
            label = f'{pixel_count/1000000:.1f}M'
        elif pixel_count >= 1000:
            label = f'{pixel_count/1000:.1f}K'
        else:
            label = f'{pixel_count}'
        cv2.putText(panel, label, (5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    # Draw channel labels on the left (if provided)
    if channel_names:
        for i, name in enumerate(channel_names):
            y_pos = margin_top + 20 + i * 15
            color = colors[i] if i < len(colors) else (200, 200, 200)
            cv2.putText(panel, name, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Draw histograms
    line_thickness = 5
    for i, (hist, color) in enumerate(zip(histograms, colors)):
        points = []
        for j in range(len(hist)):
            x = margin_left + int(j / len(hist) * hist_width)
            y = margin_top + hist_height - int(hist[j] / max_count * hist_height * 0.85)
            points.append((x, y))

        points = np.array(points, dtype=np.int32)
        cv2.polylines(panel, [points], False, color, line_thickness)

    # Draw detected level markers (triangles) - if provided
    marker_thickness = 10
    if detected_levels:
        for color, color_name in zip(colors, channel_names if channel_names else []):
            if color_name not in detected_levels:
                continue

            levels = detected_levels[color_name]
            if 'black' in levels and 'white' in levels:
                black_point = levels['black']
                white_point = levels['white']

                # Draw black point marker
                black_x = margin_left + int(black_point / max_val * hist_width)
                triangle_pts = [
                    (black_x, margin_top + hist_height),
                    (black_x - 6, margin_top + hist_height - 10),
                    (black_x + 6, margin_top + hist_height - 10)
                ]
                pts = np.array(triangle_pts, dtype=np.int32)
                cv2.polylines(panel, [pts], True, color, marker_thickness)

                # Draw white point marker
                white_x = margin_left + int(white_point / max_val * hist_width)
                triangle_pts = [
                    (white_x, margin_top + hist_height),
                    (white_x - 6, margin_top + hist_height - 10),
                    (white_x + 6, margin_top + hist_height - 10)
                ]
                pts = np.array(triangle_pts, dtype=np.int32)
                cv2.polylines(panel, [pts], True, color, marker_thickness)

    # Draw x-axis
    cv2.line(panel, (margin_left, margin_top + hist_height),
             (margin_left + hist_width, margin_top + hist_height), (150, 150, 150), 1)

    # Draw x-axis labels
    for val in [0, 16384, 32768, 49152, 65535]:
        x = margin_left + int(val / max_val * hist_width)
        cv2.line(panel, (x, margin_top + hist_height),
                 (x, margin_top + hist_height + 3), (150, 150, 150), 1)
        label = str(val) if val > 0 else '0'
        cv2.putText(panel, label, (x - 15, margin_top + hist_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)

    return panel


class Metadata:
    """Container for all intermediate data and results."""

    def __init__(self):
        # Input/Output
        self.raw_path: Optional[str] = None
        self.output_path: Optional[str] = None

        # Images
        self.raw_image: Optional[np.ndarray] = None      # Original RAW (uint16)
        self.current_image: Optional[np.ndarray] = None  # Current working image

        # Border info
        self.border_specs: Optional[Dict[str, float]] = None  # {'u': 0.05, 'd': 0.05, ...}
        self.border_mask: Optional[np.ndarray] = None
        self.border_pixels: Optional[np.ndarray] = None

        # Classification
        self.n_clusters: int = 3
        self.wb_classes: List[int] = [0]
        self.cluster_labels: Optional[np.ndarray] = None
        self.clusters: Optional[List[np.ndarray]] = None

        # White balance
        self.wb_gains: Optional[np.ndarray] = None

        # Parameters
        self.wb_threshold: float = 30
        self.level_threshold: float = 0.99
        self.level_pixel_threshold: float = 0.0001  # Threshold for level detection (0.01% by default)

        # Level detection regions
        self.black_level_region: str = 'border'  # 'border' or 'center' or custom
        self.white_level_region: str = 'center'  # 'border' or 'center' or custom
        self.center_rect_ratio: float = 0.5  # Center region ratio (for 'center' detection)

        # Tone adjustment parameters
        self.tone_region: str = 'center'  # Region for tone detection ('border', 'center', 'manual')
        self.tone_black_point: Optional[float] = None  # Auto-detect if None
        self.tone_white_point: Optional[float] = None  # Auto-detect if None
        self.tone_gamma: float = 1.0  # Gamma correction (1.0 = no correction)
        self.tone_pixel_threshold: float = 0.001  # Threshold for tone detection (0.1% by default)

        # Detected level points (for visualization)
        self.detected_levels: Optional[Dict[str, Dict[str, float]]] = None  # {'R': {'black': x, 'white': y}, ...}
        self.detected_tone_levels: Optional[Dict[str, float]] = None  # {'black': x, 'white': y} for luminance

        # Visualization
        self.vis_image: Optional[np.ndarray] = None
        self.vis_path: Optional[str] = None
        self.vis_callbacks: List[callable] = []  # Visualization callbacks registered by stages

    def register_vis(self, callback: callable):
        """Register a visualization callback.

        Callback should be a function that takes metadata and returns visualization image.
        """
        self.vis_callbacks.append(callback)

    def __repr__(self):
        items = []
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                if isinstance(value, np.ndarray):
                    items.append(f"{key}=array({value.shape})")
                elif isinstance(value, list) and key == 'vis_callbacks':
                    items.append(f"{key}=[{len(value)} callbacks]")
                elif isinstance(value, list):
                    items.append(f"{key}=list({len(value)})")
                else:
                    items.append(f"{key}={value}")
        return f"Metadata({', '.join(items)})"


class Stage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, metadata: Metadata) -> Metadata:
        """Execute the stage processing."""
        pass

    def vis(self, metadata: Metadata) -> Optional[np.ndarray]:
        """Generate visualization for this stage. Optional."""
        return None


class RawLoadStage(Stage):
    """Load RAW file from disk."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Loading RAW file: {metadata.raw_path}")

        with rawpy.imread(metadata.raw_path) as raw:
            metadata.raw_image = raw.postprocess(
                use_camera_wb=False,
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB
            )
            metadata.current_image = metadata.raw_image.copy()

        print(f"       Image size: {metadata.raw_image.shape}")
        return metadata


class BorderExtractStage(Stage):
    """Extract border region from image."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Extracting border region...")

        h, w = metadata.current_image.shape[:2]

        # Get border ratios
        border_specs = metadata.border_specs or {'u': 0.05, 'd': 0.05, 'l': 0.05, 'r': 0.05}
        metadata.border_specs = border_specs

        # Use common function to create mask
        metadata.border_mask = create_region_mask(h, w, 'border', border_specs)

        # Use common function to extract pixels
        metadata.border_pixels = extract_region_pixels(
            metadata.current_image,
            region='border',
            border_specs=border_specs
        )

        print(f"       Border pixels: {len(metadata.border_pixels)}")

        # Register visualization callback with current image snapshot
        img_snapshot = metadata.current_image.copy()
        metadata.register_vis(lambda md: self._vis_with_image(img_snapshot, md))

        return metadata

    def _vis_with_image(self, img: np.ndarray, metadata: Metadata) -> Optional[np.ndarray]:
        """Create visualization with specific image as background."""
        img_8bit = (img / 256).astype(np.uint8)

        overlay = np.zeros_like(img_8bit)
        overlay[metadata.border_mask] = [0, 255, 0]
        result = cv2.addWeighted(img_8bit, 0.7, overlay, 0.3, 0)

        # Add label
        cv2.putText(result, 'Border Selection (Green)', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return result

    def vis(self, metadata: Metadata) -> Optional[np.ndarray]:
        """Visualize border selection."""
        img_8bit = (metadata.current_image / 256).astype(np.uint8)

        overlay = np.zeros_like(img_8bit)
        overlay[metadata.border_mask] = [0, 255, 0]
        result = cv2.addWeighted(img_8bit, 0.7, overlay, 0.3, 0)

        # Add label
        cv2.putText(result, 'Border Selection (Green)', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return result


class ColorClassifyStage(Stage):
    """Classify border pixels by color characteristics."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Classifying border pixels...")

        pixels = metadata.border_pixels

        if len(pixels) == 0:
            metadata.clusters = [np.empty((0, 3), dtype=pixels.dtype)] * metadata.n_clusters
            metadata.cluster_labels = np.array([], dtype=int)
            return metadata

        # Calculate brightness
        if pixels.dtype == np.uint16:
            gray = pixels.astype(np.float32).mean(axis=1)
        else:
            gray = pixels.mean(axis=1)

        # Normalize
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

        # Divide into clusters by brightness
        brightness_bins = np.linspace(0, 1, metadata.n_clusters + 1)
        labels = np.zeros(len(pixels), dtype=int)
        clusters = []

        for i in range(metadata.n_clusters):
            mask = (gray_norm >= brightness_bins[i]) & (gray_norm < brightness_bins[i + 1])
            labels[mask] = i
            cluster_pixels = pixels[mask]
            clusters.append(cluster_pixels)

        metadata.cluster_labels = labels
        metadata.clusters = clusters

        print(f"       Clusters: {metadata.n_clusters}, using {metadata.wb_classes} for WB")

        # Register visualization callback with current image snapshot
        img_snapshot = metadata.current_image.copy()
        metadata.register_vis(lambda md: self._vis_with_image(img_snapshot, md))

        return metadata

    def _vis_with_image(self, img: np.ndarray, metadata: Metadata) -> Optional[np.ndarray]:
        """Create visualization with specific image as background."""
        if metadata.cluster_labels is None:
            return None

        h, w = img.shape[:2]
        img_8bit = (img / 256).astype(np.uint8)

        # Create label_map following extraction order
        label_map = np.full((h, w), -1, dtype=int)

        border_specs = metadata.border_specs
        u_ratio = border_specs.get('u', 0.05)
        d_ratio = border_specs.get('d', 0.05)
        l_ratio = border_specs.get('l', 0.05)
        r_ratio = border_specs.get('r', 0.05)

        border_h_top = int(h * u_ratio)
        border_h_bottom = int(h * d_ratio)
        border_w_left = int(w * l_ratio)
        border_w_right = int(w * r_ratio)

        label_idx = 0

        if border_h_top > 0:
            for y in range(border_h_top):
                for x in range(w):
                    label_map[y, x] = metadata.cluster_labels[label_idx]
                    label_idx += 1

        if border_h_bottom > 0:
            for y in range(h - border_h_bottom, h):
                for x in range(w):
                    label_map[y, x] = metadata.cluster_labels[label_idx]
                    label_idx += 1

        if border_w_left > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            for y in range(v_start, v_end):
                for x in range(border_w_left):
                    label_map[y, x] = metadata.cluster_labels[label_idx]
                    label_idx += 1

        if border_w_right > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            for y in range(v_start, v_end):
                for x in range(w - border_w_right, w):
                    label_map[y, x] = metadata.cluster_labels[label_idx]
                    label_idx += 1

        # Create color overlay
        color_overlay = np.zeros_like(img_8bit)

        for i in range(min(metadata.n_clusters, len(CLUSTER_COLORS))):
            cluster_mask = label_map == i
            color_overlay[cluster_mask] = CLUSTER_COLORS[i]

        # Blend with original
        result = cv2.addWeighted(img_8bit, 0.5, color_overlay, 0.5, 0)

        # Highlight selected clusters
        for class_idx in metadata.wb_classes:
            if class_idx < metadata.n_clusters:
                cluster_mask = label_map == class_idx
                for c in range(3):
                    result[cluster_mask, c] = np.minimum(255, result[cluster_mask, c] + 80)

        # Add label
        cv2.putText(result, f'Color Clusters (WB: {metadata.wb_classes})', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add legend
        for i in range(min(metadata.n_clusters, len(CLUSTER_COLORS))):
            y_pos = 80 + i * 40
            color = CLUSTER_COLORS[i]
            label_text = f'Cluster {i}'
            if i in metadata.wb_classes:
                label_text += ' (WB)'

            cv2.rectangle(result, (10, y_pos - 20), (40, y_pos + 10), color, -1)
            cv2.putText(result, label_text, (50, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        return result


class WhiteBalanceComputeStage(Stage):
    """Compute white balance gains from classified clusters."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Computing white balance...")

        # Collect pixels from selected classes
        target_pixels_list = []
        for class_idx in metadata.wb_classes:
            if class_idx < len(metadata.clusters) and len(metadata.clusters[class_idx]) > 0:
                target_pixels_list.append(metadata.clusters[class_idx])

        if not target_pixels_list or len(np.vstack(target_pixels_list)) < 100:
            target_pixels = metadata.border_pixels
        else:
            target_pixels = np.vstack(target_pixels_list)

        # Calculate RGB channel means
        means = target_pixels.mean(axis=0)

        # Compute gains (green as reference)
        wb_gains = means[1] / means
        wb_gains = wb_gains / wb_gains[1]  # Normalize G to 1.0

        metadata.wb_gains = wb_gains

        print(f"       WB gains R={wb_gains[0]:.3f} G={wb_gains[1]:.3f} B={wb_gains[2]:.3f}")
        return metadata


class WhiteBalanceApplyStage(Stage):
    """Apply white balance to image."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Applying white balance...")

        float_img = metadata.current_image.astype(np.float32)
        balanced = float_img * metadata.wb_gains[np.newaxis, np.newaxis, :]
        metadata.current_image = np.clip(balanced, 0, 65535).astype(metadata.current_image.dtype)

        return metadata


class InvertStage(Stage):
    """Invert image (negative to positive)."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Inverting image...")
        max_val = np.iinfo(metadata.current_image.dtype).max
        metadata.current_image = max_val - metadata.current_image
        return metadata


class LevelRegionSelectStage(Stage):
    """Visualize level detection regions."""

    def __call__(self, metadata: Metadata) -> Metadata:
        # Register visualization callback with current image snapshot
        img_snapshot = metadata.current_image.copy()
        metadata.register_vis(lambda md: self._vis_with_image(img_snapshot, md))
        return metadata

    def _vis_with_image(self, img: np.ndarray, metadata: Metadata) -> Optional[np.ndarray]:
        """Create visualization with specific image as background and RGB histograms."""
        if img is None:
            return None

        img_8bit = (img / 256).astype(np.uint8)
        h, w = img_8bit.shape[:2]

        # Calculate histogram height (20% of image height)
        hist_height = int(h * 0.2)

        # Create main image visualization
        vis = img_8bit.copy()

        # Create masks for visualization
        black_region = self._get_region_mask(metadata, 'black', h, w)
        white_region = self._get_region_mask(metadata, 'white', h, w)

        # Overlay black region (red)
        if black_region is not None:
            overlay = np.zeros_like(vis)
            overlay[black_region] = [0, 0, 255]  # Red for black level region
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Overlay white region (blue)
        if white_region is not None:
            overlay = np.zeros_like(vis)
            overlay[white_region] = [255, 0, 0]  # Blue for white level region
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Add labels
        cv2.putText(vis, f'Black: {metadata.black_level_region}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f'White: {metadata.white_level_region}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Add pixel count info
        if black_region is not None:
            black_count = np.sum(black_region)
            cv2.putText(vis, f'{black_count:,} px', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        if white_region is not None:
            white_count = np.sum(white_region)
            cv2.putText(vis, f'{white_count:,} px', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Create histogram panel (at the bottom, 20% height)
        hist_panel = self._create_histogram_panel(img, metadata, hist_height, w)

        # Combine image and histogram vertically
        combined = np.vstack([vis, hist_panel])

        return combined

    def _create_histogram_panel(self, img: np.ndarray, metadata: Metadata, panel_height: int, panel_width: int) -> np.ndarray:
        """Create RGB histogram visualization panel."""
        max_val = 65535  # 16-bit

        # Calculate histograms for each channel
        histograms = []
        for c in range(3):
            hist, _ = np.histogram(img[:, :, c], bins=512, range=(0, max_val))
            histograms.append(hist)

        # Use common histogram drawing function
        return create_histogram_panel(
            histograms=histograms,
            panel_height=panel_height,
            panel_width=panel_width,
            title='RGB Histograms',
            channel_names=['R', 'G', 'B'],
            colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # BGR format
            max_val=max_val,
            detected_levels=metadata.detected_levels,
        )

    def _get_region_mask(self, metadata: Metadata, level_type: str, h: int, w: int) -> np.ndarray:
        """Get mask for level detection region."""
        region = metadata.black_level_region if level_type == 'black' else metadata.white_level_region

        # Use existing border mask for efficiency
        if region == 'border' and metadata.border_mask is not None:
            return metadata.border_mask

        # Use common function to create mask
        return create_region_mask(h, w, region, metadata.border_specs, metadata.center_rect_ratio)


class LevelAdjustStage(Stage):
    """Adjust RGB channel levels using histogram-based detection from different regions."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Adjusting levels...")
        print(f"       Black level region: {metadata.black_level_region}")
        print(f"       White level region: {metadata.white_level_region}")
        print(f"       Pixel threshold: {metadata.level_pixel_threshold*100:.3f}%")

        result = np.empty_like(metadata.current_image, dtype=np.float32)
        max_val = np.iinfo(metadata.current_image.dtype).max

        # Extract pixels for black level detection
        black_pixels = self._extract_region_pixels(metadata, 'black')
        print(f"       Black region pixels: {black_pixels.shape[0] if black_pixels is not None else 'N/A'}")

        # Extract pixels for white level detection
        white_pixels = self._extract_region_pixels(metadata, 'white')
        print(f"       White region pixels: {white_pixels.shape[0] if white_pixels is not None else 'N/A'}")

        # Store detected levels for visualization
        metadata.detected_levels = {}

        for c, name in enumerate(['R', 'G', 'B']):
            channel = metadata.current_image[:, :, c].astype(np.float32)

            # Calculate black point from black region
            black_point = self._compute_level_point(
                black_pixels[:, c] if black_pixels is not None else channel.flatten(),
                'black',
                metadata.level_pixel_threshold,
                max_val
            )

            # Calculate white point from white region
            white_point = self._compute_level_point(
                white_pixels[:, c] if white_pixels is not None else channel.flatten(),
                'white',
                metadata.level_pixel_threshold,
                max_val
            )

            # Ensure valid range
            if white_point - black_point < max_val * 0.02:
                black_point = np.percentile(channel, 0.01)
                white_point = np.percentile(channel, 99.99)

            print(f"       {name}: black={black_point:.0f} white={white_point:.0f} range={white_point-black_point:.0f}")

            # Save detected levels
            metadata.detected_levels[name] = {'black': black_point, 'white': white_point}

            # Stretch to full range
            stretched = (channel - black_point) / (white_point - black_point) * max_val
            result[:, :, c] = np.clip(stretched, 0, max_val)

        metadata.current_image = result.astype(metadata.current_image.dtype)
        return metadata

    def _extract_region_pixels(self, metadata: Metadata, level_type: str) -> np.ndarray:
        """Extract pixels from specified region for level detection."""
        region = metadata.black_level_region if level_type == 'black' else metadata.white_level_region

        # Use common function to extract pixels
        if region == 'border' and metadata.border_mask is not None:
            # Use existing border mask for efficiency
            return metadata.current_image[metadata.border_mask]

        return extract_region_pixels(
            metadata.current_image,
            region=region,
            border_specs=metadata.border_specs if region == 'border' else None,
            center_ratio=metadata.center_rect_ratio
        )

    def _compute_level_point(self, pixels: np.ndarray, level_type: str, threshold: float, max_val: float) -> float:
        """Compute black or white point from histogram using utility function.

        Uses the peak method: find histogram peak and extend outward.

        Args:
            pixels: 1D array of pixel values
            level_type: 'black' or 'white'
            threshold: Pixel count threshold (as fraction of total pixels)
            max_val: Maximum pixel value (e.g., 65535 for 16-bit)
        """
        return find_level_point(pixels, level_type, threshold, max_val, method='peak')


class ToneAdjustStage(Stage):
    """Adjust image tone using luminance level adjustment (after RGB level adjustment)."""

    def __call__(self, metadata: Metadata) -> Metadata:
        print(f"  [{self.name}] Adjusting tone (luminance)...")
        print(f"       Tone detection region: {metadata.tone_region}")
        print(f"       Pixel threshold: {metadata.tone_pixel_threshold*100:.3f}%")

        img = metadata.current_image.astype(np.float32)
        max_val = 65535  # 16-bit

        # Extract pixels from specified region for detection
        region_pixels = extract_region_pixels(
            img,
            region=metadata.tone_region,
            border_specs=metadata.border_specs if metadata.tone_region == 'border' else None,
            center_ratio=metadata.center_rect_ratio
        )

        print(f"       Region pixels: {region_pixels.shape[0]}")

        # Calculate luminance from region pixels
        lum_region = 0.299 * region_pixels[:, 0] + 0.587 * region_pixels[:, 1] + 0.114 * region_pixels[:, 2]

        # Detect black and white points using histogram method
        if metadata.tone_black_point is None:
            black_point = self._compute_level_point(lum_region, 'black', metadata.tone_pixel_threshold, max_val)
        else:
            black_point = metadata.tone_black_point

        if metadata.tone_white_point is None:
            white_point = self._compute_level_point(lum_region, 'white', metadata.tone_pixel_threshold, max_val)
        else:
            white_point = metadata.tone_white_point

        print(f"       Luminance: black={black_point:.0f} white={white_point:.0f} gamma={metadata.tone_gamma:.2f}")

        # Store detected levels for visualization
        metadata.detected_tone_levels = {'black': black_point, 'white': white_point}

        # Apply tone adjustment to ENTIRE image by clipping and remapping to full range
        # For each channel: clip and stretch
        for c in range(3):
            channel = img[:, :, c]

            # Clip to detected range
            channel_clipped = np.clip(channel, black_point, white_point)

            # Normalize to [0, 1]
            channel_norm = (channel_clipped - black_point) / (white_point - black_point)

            # Apply gamma correction
            if metadata.tone_gamma != 1.0:
                channel_norm = np.power(channel_norm, 1.0 / metadata.tone_gamma)

            # Remap to full range [0, max_val]
            result_channel = channel_norm * max_val

            img[:, :, c] = result_channel

        metadata.current_image = np.clip(img, 0, max_val).astype(metadata.current_image.dtype)

        # Register visualization callback
        img_snapshot = metadata.current_image.copy()
        metadata.register_vis(lambda md: self._vis_with_image(img_snapshot, md))

        return metadata

    def _vis_with_image(self, img: np.ndarray, metadata: Metadata) -> Optional[np.ndarray]:
        """Create tone adjustment visualization."""
        if img is None or metadata.detected_tone_levels is None:
            return None

        img_8bit = (img / 256).astype(np.uint8)
        h, w = img_8bit.shape[:2]

        # Calculate histogram height (25% of image height)
        hist_height = int(h * 0.25)

        # Create main image visualization
        vis = img_8bit.copy()

        # Create mask for tone detection region
        tone_region = self._get_region_mask(metadata, h, w)

        # Overlay tone region (green)
        if tone_region is not None:
            overlay = np.zeros_like(vis)
            overlay[tone_region] = [0, 255, 0]  # Green for tone detection region
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Add title
        cv2.putText(vis, 'Tone Adjustment (Luminance)', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add region label
        cv2.putText(vis, f'Region: {metadata.tone_region}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Add pixel count info
        if tone_region is not None:
            tone_count = np.sum(tone_region)
            cv2.putText(vis, f'{tone_count:,} px', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Add parameters
        black = metadata.detected_tone_levels['black']
        white = metadata.detected_tone_levels['white']
        gamma = metadata.tone_gamma
        cv2.putText(vis, f'Black: {black:.0f} White: {white:.0f} Gamma: {gamma:.2f}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Create luminance histogram panel
        hist_panel = self._create_luminance_histogram_panel(img, metadata, hist_height, w)

        # Combine image and histogram vertically
        combined = np.vstack([vis, hist_panel])

        return combined

    def _get_region_mask(self, metadata: Metadata, h: int, w: int) -> np.ndarray:
        """Get mask for tone detection region."""
        region = metadata.tone_region

        # Use existing border mask for efficiency
        if region == 'border' and metadata.border_mask is not None:
            return metadata.border_mask

        # Use common function to create mask
        return create_region_mask(h, w, region, metadata.border_specs, metadata.center_rect_ratio)

    def _create_luminance_histogram_panel(self, img: np.ndarray, metadata: Metadata,
                                          panel_height: int, panel_width: int) -> np.ndarray:
        """Create luminance histogram visualization panel."""
        max_val = 65535

        # Calculate luminance
        img_float = img.astype(np.float32)
        luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]

        # Calculate histogram
        hist, _ = np.histogram(luminance, bins=512, range=(0, max_val))

        # Convert detected_tone_levels to format expected by create_histogram_panel
        # detected_tone_levels = {'black': x, 'white': y}
        # Need: {'L': {'black': x, 'white': y}}
        detected_levels = None
        if metadata.detected_tone_levels is not None:
            detected_levels = {
                'L': {
                    'black': metadata.detected_tone_levels['black'],
                    'white': metadata.detected_tone_levels['white']
                }
            }

        # Use common histogram drawing function
        return create_histogram_panel(
            histograms=[hist],
            panel_height=panel_height,
            panel_width=panel_width,
            title='Luminance Histogram',
            channel_names=['L'],
            colors=[(128, 128, 128)],  # Gray for luminance
            max_val=max_val,
            detected_levels=detected_levels,
        )

    def _compute_level_point(self, pixels: np.ndarray, level_type: str, threshold: float, max_val: float) -> float:
        """Compute black or white point from histogram using utility function.

        Uses the cumulative method: accumulate from edge until threshold.

        Args:
            pixels: 1D array of pixel values
            level_type: 'black' or 'white'
            threshold: Pixel count threshold (as fraction of total pixels)
            max_val: Maximum pixel value (e.g., 65535 for 16-bit)
        """
        return find_level_point(pixels, level_type, threshold, int(max_val), method='cumulative')


class SaveStage(Stage):
    """Save current image."""

    def __call__(self, metadata: Metadata) -> Metadata:
        if metadata.output_path is None:
            return metadata

        print(f"  [{self.name}] Saving result...")
        img_8bit = (metadata.current_image / 256).astype(np.uint8)
        cv2.imwrite(metadata.output_path, cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
        print(f"       Saved: {metadata.output_path}")
        return metadata


class VisualizeStage(Stage):
    """Generate combined visualization from registered callbacks."""

    def __call__(self, metadata: Metadata) -> Metadata:
        if metadata.vis_path is None:
            return metadata

        print(f"  [{self.name}] Creating visualization...")

        # Collect visualizations from registered callbacks
        vis_images = []
        for callback in metadata.vis_callbacks:
            vis_img = callback(metadata)
            if vis_img is not None:
                vis_images.append(vis_img)

        if len(vis_images) < 1:
            print(f"       No visualizations to display")
            return metadata

        if len(vis_images) == 1:
            # Single visualization, save directly
            combined_bgr = cv2.cvtColor(vis_images[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(metadata.vis_path, combined_bgr)
            print(f"       Visualization saved: {metadata.vis_path}")
            metadata.vis_image = vis_images[0]
            return metadata

        # Resize and combine multiple visualizations
        h, w = vis_images[0].shape[:2]
        max_width = 2000
        scale = min(1.0, max_width / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        vis_images_resized = [cv2.resize(img, (new_w, new_h)) for img in vis_images]
        combined = np.hstack(vis_images_resized)

        # Save
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(metadata.vis_path, combined_bgr)
        print(f"       Visualization saved: {metadata.vis_path}")

        metadata.vis_image = combined
        return metadata


class Pipeline:
    """Processing pipeline that chains multiple stages."""

    def __init__(self, stages: List[Stage], name: str = "Pipeline"):
        self.stages = stages
        self.name = name

    def __call__(self, metadata: Metadata) -> Metadata:
        """Execute all stages in sequence."""
        for stage in self.stages:
            metadata = stage(metadata)
        return metadata


def parse_border_specs(border_str: str) -> Dict[str, float]:
    """Parse border specification string."""
    border_specs = {}

    # TODO: refact: split?

    if ',' in border_str or any(c in border_str for c in 'udlr'):
        for part in border_str.replace(' ', ',').split(','):
            if part:
                direction = part[0].lower()
                if direction in 'udlr':
                    value = float(part[1:])
                    border_specs[direction] = value

        # Fill missing directions with 0
        for direction in 'udlr':
            if direction not in border_specs:
                border_specs[direction] = 0.0
    else:
        ratio = float(border_str)
        border_specs = {'u': ratio, 'd': ratio, 'l': ratio, 'r': ratio}

    return border_specs


def parse_wb_ix(wb_ix_str: str) -> tuple:
    """Parse white balance index specification."""
    parts = [int(x.strip()) for x in wb_ix_str.split(',')]
    n_clusters = parts[0]
    wb_classes = parts[1:] if len(parts) > 1 else [0]
    return n_clusters, wb_classes


def main():
    parser = argparse.ArgumentParser(
        description='Nikon RAW Film Mask Removal Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python film_mask_removal.py input.NEF -o output.jpg

  # Custom borders and clusters
  python film_mask_removal.py input.NEF -o output.jpg \\
      --border u0.03,d0.03,l0,r0 --wb-ix 4,0,1

  # With visualization
  python film_mask_removal.py input.NEF -o output.jpg --visualize --debug

  # Custom processing pipeline
  python film_mask_removal.py input.NEF -o output.jpg \\
      --stages load,border,classify,wb_compute,wb_apply,invert,level,save
        """
    )
    parser.add_argument('input', help='Input RAW file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-b', '--border', type=str, default='0.05',
                        help='Border ratios (default: 0.05, or u0.05,d0.05,l0.05,r0.05)')
    parser.add_argument('--wb-ix', type=str, default='3,0',
                        help='White balance clusters: n_clusters,class_idx1,... (default: 3,0)')
    parser.add_argument('-w', '--wb-threshold', type=float, default=30,
                        help='White balance color threshold (default: 30)')
    parser.add_argument('-l', '--level-threshold', type=float, default=0.99,
                        help='Level threshold percentile (default: 0.99)')
    parser.add_argument('--level-pixel-threshold', type=float, default=0.0001,
                        help='Level detection threshold (0.0001=0.01%%, 0.001=0.1%%, 0.01=1%%) (default: 0.0001)')
    parser.add_argument('--black-level-region', type=str, default='border',
                        help='Black level detection region: border, center, or x,y,w,h (default: border)')
    parser.add_argument('--white-level-region', type=str, default='center',
                        help='White level detection region: border, center, or x,y,w,h (default: center)')
    parser.add_argument('--center-ratio', type=float, default=0.5,
                        help='Center region ratio for level detection (0.0-1.0, default: 0.5)')
    parser.add_argument('--tone-black', type=float, default=None,
                        help='Tone adjustment black point (auto-detect if not specified)')
    parser.add_argument('--tone-white', type=float, default=None,
                        help='Tone adjustment white point (auto-detect if not specified)')
    parser.add_argument('--tone-gamma', type=float, default=1.0,
                        help='Tone adjustment gamma correction (default: 1.0, <1.0 darker, >1.0 lighter)')
    parser.add_argument('--tone-region', type=str, default='center',
                        choices=['border', 'center', 'manual'],
                        help='Region for tone detection (default: center)')
    parser.add_argument('--tone-pixel-threshold', type=float, default=None,
                        help='Tone detection threshold (0.0001=0.01%%, 0.001=0.1%%, 0.01=1%%)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information')
    parser.add_argument('--visualize', action='store_true',
                        help='Create classification visualization')

    args = parser.parse_args()

    # Create metadata
    metadata = Metadata()
    metadata.raw_path = args.input
    metadata.output_path = args.output
    metadata.border_specs = parse_border_specs(args.border)
    metadata.wb_threshold = args.wb_threshold
    metadata.level_threshold = args.level_threshold
    metadata.level_pixel_threshold = args.level_pixel_threshold
    metadata.black_level_region = args.black_level_region
    metadata.white_level_region = args.white_level_region
    metadata.center_rect_ratio = args.center_ratio
    metadata.tone_black_point = args.tone_black
    metadata.tone_white_point = args.tone_white
    metadata.tone_gamma = args.tone_gamma
    metadata.tone_region = args.tone_region
    metadata.tone_pixel_threshold = args.tone_pixel_threshold

    n_clusters, wb_classes = parse_wb_ix(args.wb_ix)
    metadata.n_clusters = n_clusters
    metadata.wb_classes = wb_classes

    assert bool(metadata.tone_white_point is not None or metadata.tone_black_point is not None) ^ bool(metadata.tone_pixel_threshold is not None), "Manual point and threshold should not be set at the same time."


    # Setup visualization path
    if args.visualize and args.output:
        metadata.vis_path = args.output.rsplit('.', 1)[0] + '_vis.jpg'

    # Build pipeline
    stages = [
        RawLoadStage("LoadRAW"),
        BorderExtractStage("BorderExtract"),
        ColorClassifyStage("ColorClassify"),
        WhiteBalanceComputeStage("WBCompute"),
        WhiteBalanceApplyStage("WBApply"),
        InvertStage("Invert"),
        LevelRegionSelectStage("LevelRegionSelect"),
        LevelAdjustStage("LevelAdjust"),
        ToneAdjustStage("ToneAdjust"),
    ]

    # Add visualization stage if requested
    if args.visualize:
        stages.append(VisualizeStage("Visualize"))

    stages.append(SaveStage("Save"))

    pipeline = Pipeline(stages, "FilmMaskRemoval")

    # Execute
    print(f"Processing: {args.input}")
    metadata = pipeline(metadata)
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
