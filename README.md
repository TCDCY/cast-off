# Negative Film Mask Removal Tool

Automatically remove color mask from camera raw film scans.

## Quick Start

Process order.

```python
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
```

1. black/white point auto tone by threshold first
2. overwrite with specific black/white if --tone-white, --tone-black, --level-white-point is not None


```bash
python film_mask_removal.py ./ZFC_5837.NEF --border l0.1,r0.1 --level-pixel-threshold r0.001,g0.001,b0.03  --wb-ix 4,1 --debug --visualize  --tone-pixel-threshold 0.03  --tone-white 64000 --level-white-point "b61000"
```


## Processing Pipeline

1. **White Balance**: Compute white balance from border area, apply to entire image
   - Classify border pixels into clusters by brightness
   - Select specific clusters for white balance calculation
   - `--wb-ix 4,0,1` means 4 clusters, select `[0, 1]` to compute wb.
2. **Invert**: Invert all 3 RGB channels (negative to positive)
3. **Level Adjustment**: Redefine black and white points
   - Automatically calculate percentile threshold for each channel
4. **Luminance Adjustment**
    - Set black/white point from [0, 65535]
