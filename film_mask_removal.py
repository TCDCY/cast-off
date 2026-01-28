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

        u_ratio = border_specs.get('u', 0.05)
        d_ratio = border_specs.get('d', 0.05)
        l_ratio = border_specs.get('l', 0.05)
        r_ratio = border_specs.get('r', 0.05)

        border_h_top = int(h * u_ratio)
        border_h_bottom = int(h * d_ratio)
        border_w_left = int(w * l_ratio)
        border_w_right = int(w * r_ratio)

        # Create mask
        mask = np.zeros((h, w), dtype=bool)

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

        metadata.border_mask = mask

        # Extract pixels
        border_pixels_list = []

        if border_h_top > 0:
            top = metadata.current_image[:border_h_top, :, :]
            border_pixels_list.append(top.reshape(-1, 3))

        if border_h_bottom > 0:
            bottom = metadata.current_image[-border_h_bottom:, :, :]
            border_pixels_list.append(bottom.reshape(-1, 3))

        if border_w_left > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                left = metadata.current_image[v_start:v_end, :border_w_left, :]
                border_pixels_list.append(left.reshape(-1, 3))

        if border_w_right > 0:
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                right = metadata.current_image[v_start:v_end, -border_w_right:, :]
                border_pixels_list.append(right.reshape(-1, 3))

        if border_pixels_list:
            metadata.border_pixels = np.vstack(border_pixels_list)
        else:
            metadata.border_pixels = np.empty((0, 3), dtype=metadata.current_image.dtype)

        print(f"       Border pixels: {len(metadata.border_pixels)}")

        # Register visualization callback
        metadata.register_vis(self.vis)

        return metadata

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

        # Register visualization callback
        metadata.register_vis(self.vis)

        return metadata

    def vis(self, metadata: Metadata) -> Optional[np.ndarray]:
        """Visualize color classification."""
        if metadata.cluster_labels is None:
            return None

        h, w = metadata.current_image.shape[:2]
        img_8bit = (metadata.current_image / 256).astype(np.uint8)

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
        # Register visualization callback
        metadata.register_vis(self.vis)
        return metadata

    def vis(self, metadata: Metadata) -> Optional[np.ndarray]:
        """Visualize the regions used for black and white level detection."""
        if metadata.current_image is None:
            return None

        img_8bit = (metadata.current_image / 256).astype(np.uint8)
        h, w = img_8bit.shape[:2]

        # Create visualization image
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
        cv2.putText(vis, f'Black Level: {metadata.black_level_region}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f'White Level: {metadata.white_level_region}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Add pixel count info
        if black_region is not None:
            black_count = np.sum(black_region)
            cv2.putText(vis, f'Black: {black_count:,} pixels', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        if white_region is not None:
            white_count = np.sum(white_region)
            cv2.putText(vis, f'White: {white_count:,} pixels', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    def _get_region_mask(self, metadata: Metadata, level_type: str, h: int, w: int) -> np.ndarray:
        """Get mask for level detection region."""
        region = metadata.black_level_region if level_type == 'black' else metadata.white_level_region

        if region == 'border':
            return metadata.border_mask

        elif region == 'center':
            ratio = metadata.center_rect_ratio
            h_start = int(h * (1 - ratio) / 2)
            h_end = int(h * (1 + ratio) / 2)
            w_start = int(w * (1 - ratio) / 2)
            w_end = int(w * (1 + ratio) / 2)
            mask = np.zeros((h, w), dtype=bool)
            mask[h_start:h_end, w_start:w_end] = True
            return mask

        else:
            # Custom region
            try:
                parts = [float(x) for x in region.split(',')]
                if len(parts) == 4:
                    x, y, rw, rh = parts
                    h_start = int(h * y)
                    h_end = int(h * (y + rh))
                    w_start = int(w * x)
                    w_end = int(w * (x + rw))
                    mask = np.zeros((h, w), dtype=bool)
                    mask[h_start:h_end, w_start:w_end] = True
                    return mask
            except:
                pass

            return None


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

            # Stretch to full range
            stretched = (channel - black_point) / (white_point - black_point) * max_val
            result[:, :, c] = np.clip(stretched, 0, max_val)

        metadata.current_image = result.astype(metadata.current_image.dtype)
        return metadata

    def _extract_region_pixels(self, metadata: Metadata, level_type: str) -> np.ndarray:
        """Extract pixels from specified region for level detection."""
        region = metadata.black_level_region if level_type == 'black' else metadata.white_level_region
        h, w = metadata.current_image.shape[:2]

        if region == 'border':
            # Use existing border mask
            if metadata.border_mask is None:
                # Fallback to entire image
                return metadata.current_image.reshape(-1, 3)
            return metadata.current_image[metadata.border_mask]

        elif region == 'center':
            # Use center rectangle
            ratio = metadata.center_rect_ratio
            h_start = int(h * (1 - ratio) / 2)
            h_end = int(h * (1 + ratio) / 2)
            w_start = int(w * (1 - ratio) / 2)
            w_end = int(w * (1 + ratio) / 2)

            center_region = metadata.current_image[h_start:h_end, w_start:w_end, :]
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

                    custom_region = metadata.current_image[h_start:h_end, w_start:w_end, :]
                    return custom_region.reshape(-1, 3)
            except:
                pass

            # Fallback to entire image
            return metadata.current_image.reshape(-1, 3)

    def _compute_level_point(self, pixels: np.ndarray, level_type: str, threshold: float, max_val: float) -> float:
        """Compute black or white point from histogram.

        Uses a robust approach:
        1. Find the histogram peak (mode)
        2. From the peak, extend outward until finding where count drops below threshold

        Args:
            pixels: 1D array of pixel values
            level_type: 'black' or 'white'
            threshold: Pixel count threshold (as fraction of total pixels)
            max_val: Maximum pixel value (e.g., 65535 for 16-bit)
        """
        hist, bin_edges = np.histogram(pixels, bins=512, range=(0, max_val))
        total_pixels = pixels.size
        level_threshold = total_pixels * threshold

        # Find histogram peak
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

    n_clusters, wb_classes = parse_wb_ix(args.wb_ix)
    metadata.n_clusters = n_clusters
    metadata.wb_classes = wb_classes

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
    ]

    # Add visualize stage at the end if requested
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
