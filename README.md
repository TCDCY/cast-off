# Nikon RAW Film Mask Removal Tool

Automatically remove color mask from Nikon RAW (.NEF) format film scans.

## Processing Pipeline

1. **White Balance**: Compute white balance from border area, apply to entire image
   - Classify border pixels into clusters by brightness
   - Select specific clusters for white balance calculation

2. **Invert**: Invert all 3 RGB channels (negative to positive)

3. **Level Adjustment**: Redefine black and white points
   - Automatically calculate percentile threshold for each channel

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python film_mask_removal.py ZFC_5837.NEF -o output.jpg
```

### Custom Border Ratios

Customize border extraction area for each direction:

```bash
# Different ratios for each direction
python film_mask_removal.py input.NEF -o output.jpg \
    --border u0.03,d0.05,l0.04,r0.06

# u=up, d=down, l=left, r=right (values are ratios of image dimension)
```

### Custom White Balance Clusters

Control how border pixels are classified and which clusters to use:

```bash
# Divide into 4 clusters, use clusters 0 and 1 for white balance
python film_mask_removal.py input.NEF -o output.jpg \
    --wb-ix 4,0,1

# 3 clusters, use only cluster 0 (default)
python film_mask_removal.py input.NEF -o output.jpg \
    --wb-ix 3,0
```

### Debug Mode

Show detailed processing information:

```bash
python film_mask_removal.py input.NEF -o output.jpg --debug
```

### Visualization

Create a visualization showing border selection and color classification:

```bash
python film_mask_removal.py input.NEF -o output.jpg --visualize
```

This creates a side-by-side view:
- **Left**: Original image with green overlay showing selected border regions
- **Right**: Color classification with each cluster colored differently
  - Selected clusters for white balance are highlighted with "(WB)"
  - Clusters are indexed by brightness (0=darkest, N-1=brightest)

The visualization is saved as `output_vis.jpg` (appends `_vis` to output filename).

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--border` | Border ratios: `u0.05,d0.05,l0.05,r0.05` or single value | 0.05 |
| `--wb-ix` | Clusters: `n_clusters,class1,class2,...` | 3,0 |
| `--wb-threshold` | White balance color threshold (0-255) | 30 |
| `--level-threshold` | Level threshold percentile (0-1) | 0.99 |
| `--debug` | Show debug information | false |
| `--visualize` | Create classification visualization | false |

## Examples

```bash
# Basic processing
python film_mask_removal.py ZFC_5837.NEF -o result.jpg

# Custom borders and clusters
python film_mask_removal.py ZFC_5837.NEF -o result.jpg \
    --border u0.02,d0.02,l0.03,r0.03 \
    --wb-ix 4,0,1

# Debug mode to see what's happening
python film_mask_removal.py ZFC_5837.NEF -o result.jpg --debug

# Create visualization to check border selection and classification
python film_mask_removal.py ZFC_5837.NEF -o result.jpg \
    --visualize --border u0.03,d0.03 --wb-ix 4,0,1

# Batch processing
for file in *.NEF; do
    python film_mask_removal.py "$file" -o "processed_${file%.NEF}.jpg" \
        --border u0.03,d0.02 \
        --wb-ix 4,0,1
done
```

## Algorithm Details

### Border Extraction

Four directional borders can be independently specified:
- `u` (up): Top border ratio
- `d` (down): Bottom border ratio
- `l` (left): Left border ratio
- `r` (right): Right border ratio

Corner areas are excluded from left/right borders to avoid overlap.

### Color Clustering

Border pixels are classified into N clusters by brightness:
- Brightness range is divided into equal intervals
- Each cluster contains pixels in a specific brightness range
- Clusters are indexed from 0 (darkest) to N-1 (brightest)

### White Balance Calculation

1. Extract border pixels using custom ratios
2. Classify pixels into N clusters by brightness
3. Select pixels from specified cluster indices
4. Calculate RGB mean values from selected clusters
5. Compute gains: `gain = green_mean / channel_mean`
6. Normalize with G channel = 1.0

### Level Adjustment

For each RGB channel:
- Calculate percentile thresholds (default: 1% and 99%)
- Linear stretch: map [black_point, white_point] to [0, 65535]
