# Nikon RAW 负片去色罩工具

用于处理Nikon RAW (NEF格式) 负片扫描的Python脚本，可以自动去除色罩、调整色阶并矫正白平衡。

## 功能特点

- 支持Nikon NEF格式的RAW文件
- 自动反相处理
- 智能色罩检测和去除
- 支持在指定区域（如胶片主体部分）分析色罩，避免边框干扰
- **多种白平衡矫正方法**：
  - 灰度世界假设（Gray World）
  - 完美反射假设（Perfect Reflector）
  - 自动白平衡（结合两种方法）
- 支持多种输出格式（JPEG、PNG、TIFF）
- 可保存ROI可视化图像，方便查看分析区域

## 安装依赖

```bash
pip install rawpy numpy opencv-python
```

## 使用方法

### 基本使用

使用中心80%区域分析色罩（推荐）：

```bash
python remove_color_cast.py input.NEF output.jpg
```

### 白平衡矫正

#### 灰度世界假设

假设图像的平均颜色应该是灰色的，适合大多数场景：

```bash
python remove_color_cast.py input.NEF output.jpg --white-balance gray-world
```

#### 完美反射假设

假设图像中最亮的像素应该是白色，适合有高光区域的图像：

```bash
python remove_color_cast.py input.NEF output.jpg --white-balance perfect-reflector
```

#### 自动白平衡

结合灰度世界和完美反射两种方法：

```bash
python remove_color_cast.py input.NEF output.jpg --white-balance auto
```

#### 使用不同的白平衡分析区域

如果需要使用与色罩分析不同的区域进行白平衡：

```bash
python remove_color_cast.py input.NEF output.jpg --white-balance gray-world --wb-roi center-50
```

### 选择不同的分析区域

使用更小的中心区域（60%）：

```bash
python remove_color_cast.py input.NEF output.jpg --roi center-60
```

使用中心50%区域：

```bash
python remove_color_cast.py input.NEF output.jpg --roi center-50
```

使用全图分析（不推荐，可能会被边框干扰）：

```bash
python remove_color_cast.py input.NEF output.jpg --roi full
```

使用自定义矩形区域（像素坐标 x1,y1,x2,y2）：

```bash
python remove_color_cast.py input.NEF output.jpg --roi 1000,500,4000,3000
```

### 显示分析区域

生成带有ROI边框的可视化图像：

```bash
python remove_color_cast.py input.NEF output.jpg --show-roi
```

这会额外生成一个 `output_roi_vis.jpg` 文件，显示分析区域的位置。

### 调整参数

自定义色阶调整比率（默认0.01）：

```bash
python remove_color_cast.py input.NEF output.jpg --ratio 0.02
```

使用相机白平衡：

```bash
python remove_color_cast.py input.NEF output.jpg --use-camera-wb
```

保存为16位TIFF：

```bash
python remove_color_cast.py input.NEF output.tif
```

### 完整示例

使用中心60%区域分析色罩，应用灰度世界白平衡，并显示ROI：

```bash
python remove_color_cast.py input.NEF output.jpg --roi center-60 --white-balance gray-world --show-roi
```

## 算法原理

1. **读取RAW文件**：使用rawpy库读取NEF文件，进行基本的demosaicing处理
2. **提取分析区域**：从图像中提取指定区域（默认为中心80%），避免胶片边框干扰
3. **反相**：将图像反相（负片转正片）
4. **分析色罩**：在分析区域内检测每个RGB通道的最大值和最小值
5. **计算调整参数**：根据检测到的范围和目标范围，计算色阶调整参数
6. **应用色阶调整**：将计算得到的调整参数应用到整张图像
7. **白平衡矫正**（可选）：
   - **灰度世界**：计算各通道平均值，调整使其相等
   - **完美反射**：基于最亮像素（99.5分位数）进行白平衡
   - **自动**：结合两种方法，取平均增益

## 参数说明

### 色罩分析参数

- `--ratio`: 色阶调整比率，控制最大值最小值的调整范围（默认0.01）
  - 较大的值会进行更激进的调整
  - 目标最大值 = 1.0 - ratio
  - 目标最小值 = 0.0 + ratio

- `--roi`: 色罩分析区域（默认center）
  - `center`: 中心80%区域
  - `center-60`: 中心60%区域
  - `center-50`: 中心50%区域
  - `full`: 全图
  - `x1,y1,x2,y2`: 自定义矩形区域坐标

### 白平衡参数

- `--white-balance`: 白平衡方法（默认none）
  - `none`: 不进行白平衡
  - `gray-world`: 灰度世界假设
  - `perfect-reflector`: 完美反射假设
  - `auto`: 自动选择

- `--wb-roi`: 白平衡分析区域（默认与色罩ROI相同）
  - 可使用与 `--roi` 相同的选项
  - 如果不指定，将使用与色罩相同的区域

### 其他参数

- `--show-roi`: 显示并保存ROI区域的可视化图像
- `--use-camera-wb`: 使用相机的白平衡设置（RAW解码阶段）

## 白平衡方法选择建议

- **灰度世界假设**：适合色彩分布均匀的场景，大多数情况下效果良好
- **完美反射假设**：适合有明显高光或白色物体的场景
- **自动白平衡**：综合两种方法，适合不确定使用哪种方法时
- **不使用白平衡**：如果已经在拍摄时正确设置了白平衡，或后续会手动调整

## 注意事项

- 对于有胶片边框的负片扫描，建议使用 `center` 或 `center-60` 选项
- 如果边框很宽，可以使用 `center-50` 或自定义坐标
- 分析区域应该只包含负片的主体内容，避免包含边框或黑色区域
- 如果处理结果不理想，可以尝试调整 `--ratio` 参数
- 白平衡应该在色阶调整之后进行，脚本会自动按正确顺序执行
- 不同的白平衡方法适用于不同的场景，建议多尝试几种方法找到最佳效果
