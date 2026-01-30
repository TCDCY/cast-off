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

1. black/white point auto tune by threshold first
2. overwrite with specific black/white if --tone-white, --tone-black, --level-white-point is not None

- [download a camera raw](https://forum.affinity.serif.com/index.php?/topic/129312-raw-dng-negative-film-scanned-from-plustek-optic-film/)

<img width="889" height="602" alt="image" src="https://github.com/user-attachments/assets/b7f127ff-28b4-400a-b9a4-b69b07e19415" />


- run:

```bash
python cast_off.py ./RAW.dng --border u0.02 --level-pixel-threshold 0.0001  --wb-ix 4,3 --debug --visualize  --tone-pixel-threshold 0.02  --center-ratio 0.8 --tone-black 37000
```

- get vis like:

![RAW_1769685698_vis](https://github.com/user-attachments/assets/d5d87478-97ca-4fdf-84b3-d12d6b88b4ba)

- output:

![RAW_1769685698](https://github.com/user-attachments/assets/980cabe6-c39c-43a7-9a31-15b63273631a)

### Save preset

```bash
python cast_off.py ./ZFC_5837.NEF --border l0.1,r0.1 --level-pixel-threshold r0.001,g0.001,b0.03  --wb-ix 4,1 --debug --visualize  --tone-pixel-threshold 0.03  --tone-white 64000 --level-white-point "b61000" --save-preset ./preset.pkl
```

### Load preset

```bash
python cast_off.py ./ZFC_5837.NEF --load-preset ./preset.pkl -o "new_{name}.png"
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
