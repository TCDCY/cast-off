#!/usr/bin/env python3
"""
Nikon RAW Film Mask Removal Script
Process: Base white balance -> Invert -> RGB level adjustment
"""

import numpy as np
import rawpy
import cv2
import argparse
from pathlib import Path


class FilmMaskRemoval:
    """Film mask removal processor"""

    def __init__(self,
                 border_ratio: float = 0.05,
                 wb_threshold: float = 30,
                 level_threshold: float = 0.99,
                 border_specs: dict = None,
                 n_clusters: int = 3,
                 wb_classes: list = None):
        """
        Initialize processor

        Args:
            border_ratio: Default border area ratio (used if border_specs not provided)
            wb_threshold: White balance color classification threshold (0-255)
            level_threshold: Level threshold (percentile, 0-1)
            border_specs: Custom border specs {'u': 0.05, 'd': 0.05, 'l': 0.05, 'r': 0.05}
            n_clusters: Number of color clusters for classification
            wb_classes: List of cluster indices to use for white balance (e.g., [0, 1])
        """
        self.border_ratio = border_ratio
        self.wb_threshold = wb_threshold
        self.level_threshold = level_threshold
        self.border_specs = border_specs or {'u': border_ratio, 'd': border_ratio,
                                              'l': border_ratio, 'r': border_ratio}
        self.n_clusters = n_clusters
        self.wb_classes = wb_classes if wb_classes is not None else [0]

    def load_raw(self, raw_path: str) -> np.ndarray:
        """
        Load Nikon RAW file

        Args:
            raw_path: RAW file path

        Returns:
            RGB image array (uint16)
        """
        with rawpy.imread(raw_path) as raw:
            # Use camera parameters, output 16-bit RGB
            rgb = raw.postprocess(
                use_camera_wb=False,     # Don't use camera white balance
                output_bps=16,            # 16-bit output
                gamma=(1, 1),             # Linear gamma, no curve adjustment
                no_auto_bright=True,      # No auto brightness
                output_color=rawpy.ColorSpace.sRGB  # sRGB color space
            )
        return rgb

    def extract_border(self, image: np.ndarray) -> np.ndarray:
        """
        Extract border region with custom specs for each direction

        Args:
            image: Input image

        Returns:
            Border pixel collection (N, 3)
        """
        h, w = image.shape[:2]

        # Get custom border ratios
        u_ratio = self.border_specs.get('u', self.border_ratio)
        d_ratio = self.border_specs.get('d', self.border_ratio)
        l_ratio = self.border_specs.get('l', self.border_ratio)
        r_ratio = self.border_specs.get('r', self.border_ratio)

        border_h_top = int(h * u_ratio)
        border_h_bottom = int(h * d_ratio)
        border_w_left = int(w * l_ratio)
        border_w_right = int(w * r_ratio)

        # Extract four borders with custom sizes
        border_pixels_list = []

        if border_h_top > 0:
            top = image[:border_h_top, :, :]
            border_pixels_list.append(top.reshape(-1, 3))

        if border_h_bottom > 0:
            bottom = image[-border_h_bottom:, :, :]
            border_pixels_list.append(bottom.reshape(-1, 3))

        if border_w_left > 0:
            # Exclude corner areas already covered by top/bottom
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                left = image[v_start:v_end, :border_w_left, :]
                border_pixels_list.append(left.reshape(-1, 3))

        if border_w_right > 0:
            # Exclude corner areas already covered by top/bottom
            v_start = border_h_top if border_h_top > 0 else 0
            v_end = h - border_h_bottom if border_h_bottom > 0 else h
            if v_end > v_start:
                right = image[v_start:v_end, -border_w_right:, :]
                border_pixels_list.append(right.reshape(-1, 3))

        # Merge all border pixels
        if border_pixels_list:
            border_pixels = np.vstack(border_pixels_list)
        else:
            border_pixels = np.empty((0, 3), dtype=image.dtype)

        return border_pixels

    def classify_by_color(self, pixels: np.ndarray) -> list:
        """
        Classify border pixels by color characteristics into clusters

        Args:
            pixels: Border pixels (N, 3)

        Returns:
            List of clusters, each containing pixels belonging to that cluster
        """
        if len(pixels) == 0:
            return [np.empty((0, 3), dtype=pixels.dtype)] * self.n_clusters

        # Calculate brightness
        if pixels.dtype == np.uint16:
            gray = pixels.astype(np.float32).mean(axis=1)
        else:
            gray = pixels.mean(axis=1)

        # Calculate color consistency (standard deviation)
        color_std = np.std(pixels, axis=1)

        # Normalize features for clustering
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
        std_norm = (color_std - color_std.min()) / (color_std.max() - color_std.min() + 1e-6)

        # Simple clustering: divide brightness range into equal intervals
        # and filter by color consistency
        clusters = []
        brightness_bins = np.linspace(0, 1, self.n_clusters + 1)

        for i in range(self.n_clusters):
            # Pixels in this brightness range
            mask = (gray_norm >= brightness_bins[i]) & (gray_norm < brightness_bins[i + 1])
            cluster_pixels = pixels[mask]
            clusters.append(cluster_pixels)

        return clusters

    def classify_by_color_old(self, pixels: np.ndarray) -> dict:
        """
        Old classification method (kept for reference)

        Args:
            pixels: Border pixels (N, 3)

        Returns:
            Dictionary with 'base'(film base), 'text'(edge text), 'backlight'(backlight)
        """
        # Calculate brightness
        if pixels.dtype == np.uint16:
            gray = pixels.astype(np.float32).mean(axis=1)
        else:
            gray = pixels.mean(axis=1)

        # Calculate color consistency (standard deviation)
        color_std = np.std(pixels, axis=1)

        # Classification logic:
        # Base: medium brightness, consistent color (low std)
        # Backlight: high brightness
        # Text: low brightness

        base_mask = (gray > 50) & (gray < 250) & (color_std < self.wb_threshold)
        backlight_mask = (gray >= 250) & (color_std < self.wb_threshold * 1.5)
        text_mask = (gray <= 50)

        base = pixels[base_mask]
        backlight = pixels[backlight_mask]
        text = pixels[text_mask]

        return {
            'base': base,
            'backlight': backlight,
            'text': text
        }

    def compute_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Compute white balance from border using custom classification

        Uses specified cluster classes for white balance calculation

        Args:
            image: Input image

        Returns:
            White balance gains (R, G, B)
        """
        border_pixels = self.extract_border(image)
        clusters = self.classify_by_color(border_pixels)

        # Collect pixels from selected classes
        target_pixels_list = []
        for class_idx in self.wb_classes:
            if class_idx < len(clusters) and len(clusters[class_idx]) > 0:
                target_pixels_list.append(clusters[class_idx])

        if not target_pixels_list:
            # Fallback to all border pixels
            target_pixels = border_pixels
        else:
            # Merge pixels from selected classes
            target_pixels = np.vstack(target_pixels_list)

        if len(target_pixels) < 100:
            # Not enough pixels, use all border pixels
            target_pixels = border_pixels

        # Calculate RGB channel means
        means = target_pixels.mean(axis=0)

        # Use green channel as reference, compute white balance gains
        # gain = green_mean / channel_mean
        wb_gains = means[1] / means

        # Normalize, keep overall brightness
        wb_gains = wb_gains / wb_gains[1]  # G channel gain = 1

        return wb_gains

    def apply_white_balance(self, image: np.ndarray, wb_gains: np.ndarray) -> np.ndarray:
        """
        Apply white balance

        Args:
            image: Input image
            wb_gains: White balance gains (R, G, B)

        Returns:
            White balanced image
        """
        # Convert to float for calculation
        float_img = image.astype(np.float32)

        # Apply gains
        balanced = float_img * wb_gains[np.newaxis, np.newaxis, :]

        # Clip to uint16 range
        balanced = np.clip(balanced, 0, 65535)

        return balanced.astype(image.dtype)

    def invert(self, image: np.ndarray) -> np.ndarray:
        """
        Invert image (negative to positive)

        Args:
            image: Input image

        Returns:
            Inverted image
        """
        max_val = np.iinfo(image.dtype).max
        return max_val - image

    def adjust_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust RGB channel levels, redefine black and white points

        Args:
            image: Input image

        Returns:
            Level adjusted image
        """
        h, w = image.shape[:2]

        # Process each channel separately
        result = np.empty_like(image, dtype=np.float32)

        for c in range(3):
            channel = image[:, :, c].astype(np.float32)

            # Calculate percentile thresholds as new black/white points
            black_point = np.percentile(channel, (1 - self.level_threshold) * 100)
            white_point = np.percentile(channel, self.level_threshold * 100)

            # Prevent division by zero
            if white_point - black_point < 1:
                white_point = black_point + 1

            # Linear stretch: map [black_point, white_point] to [0, max_val]
            max_val = np.iinfo(image.dtype).max
            stretched = (channel - black_point) / (white_point - black_point) * max_val

            # Clip
            result[:, :, c] = np.clip(stretched, 0, max_val)

        return result.astype(image.dtype)

    def process(self, raw_path: str, output_path: str = None, debug: bool = False) -> np.ndarray:
        """
        Complete processing pipeline

        Args:
            raw_path: RAW file path
            output_path: Output file path (optional)
            debug: Show debug information

        Returns:
            Processed image
        """
        print(f"Processing: {raw_path}")

        # 1. Load RAW
        print("  [1/4] Loading RAW file...")
        image = self.load_raw(raw_path)
        print(f"       Image size: {image.shape}")

        # Debug: show border specs and cluster info
        if debug:
            print(f"       Border specs: U={self.border_specs.get('u', self.border_ratio):.3f} "
                  f"D={self.border_specs.get('d', self.border_ratio):.3f} "
                  f"L={self.border_specs.get('l', self.border_ratio):.3f} "
                  f"R={self.border_specs.get('r', self.border_ratio):.3f}")
            border_pixels = self.extract_border(image)
            print(f"       Border pixels: {len(border_pixels)}")

        # 2. White balance
        print("  [2/4] Computing and applying white balance...")
        wb_gains = self.compute_white_balance(image)
        print(f"       WB gains R={wb_gains[0]:.3f} G={wb_gains[1]:.3f} B={wb_gains[2]:.3f}")
        if debug:
            print(f"       Using clusters: {self.wb_classes} (total {self.n_clusters} clusters)")
        image = self.apply_white_balance(image, wb_gains)

        # 3. Invert
        print("  [3/4] Inverting image...")
        image = self.invert(image)

        # 4. Level adjustment
        print("  [4/4] Adjusting levels...")
        image = self.adjust_levels(image)

        # Save
        if output_path:
            # Convert to 8-bit for saving
            img_8bit = (image / 256).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {output_path}")

        return image


def main():
    parser = argparse.ArgumentParser(
        description='Nikon RAW Film Mask Removal Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python film_mask_removal.py input.NEF -o output.jpg

  # Custom border ratios for each direction
  python film_mask_removal.py input.NEF -o output.jpg --border u0.03,d0.05,l0.04,r0.06

  # Custom white balance: 4 clusters, use clusters 0 and 1
  python film_mask_removal.py input.NEF -o output.jpg --wb-ix 4,0,1

  # Debug mode to see cluster information
  python film_mask_removal.py input.NEF -o output.jpg --debug
        """
    )
    parser.add_argument('input', help='Input RAW file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-b', '--border', type=str, default='0.05',
                        help='Border ratios (default: 0.05, or u0.05,d0.05,l0.05,r0.05)')
    parser.add_argument('--wb-ix', type=str, default='3,0',
                        help='White balance clusters: n_clusters,class_idx1,class_idx2,... (default: 3,0)')
    parser.add_argument('-w', '--wb-threshold', type=float, default=30,
                        help='White balance color threshold (default: 30)')
    parser.add_argument('-l', '--level-threshold', type=float, default=0.99,
                        help='Level threshold (default: 0.99)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information')

    args = parser.parse_args()

    # Parse border specs
    border_specs = {}
    if ',' in args.border or any(c in args.border for c in 'udlr'):
        # Format: u0.05,d0.03,l0.04,r0.06 or u0.05 d0.03 l0.04 r0.06
        for part in args.border.replace(' ', ',').split(','):
            if part:
                direction = part[0].lower()
                if direction in 'udlr':
                    value = float(part[1:])
                    border_specs[direction] = value

        # Fill in missing directions with 0 (exclude those borders)
        for direction in 'udlr':
            if direction not in border_specs:
                border_specs[direction] = 0.0
    else:
        # Single value applies to all directions
        border_ratio = float(args.border)
        border_specs = {'u': border_ratio, 'd': border_ratio,
                        'l': border_ratio, 'r': border_ratio}

    # Parse wb-ix: n_clusters, class_idx1, class_idx2, ...
    wb_ix_parts = [int(x.strip()) for x in args.wb_ix.split(',')]
    n_clusters = wb_ix_parts[0]
    wb_classes = wb_ix_parts[1:] if len(wb_ix_parts) > 1 else [0]

    # Create processor
    processor = FilmMaskRemoval(
        border_ratio=0.05,  # Default fallback
        wb_threshold=args.wb_threshold,
        level_threshold=args.level_threshold,
        border_specs=border_specs,
        n_clusters=n_clusters,
        wb_classes=wb_classes
    )

    # Process
    result = processor.process(args.input, args.output, debug=args.debug)

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
