# Nikon RAW 负片去色罩工具

用于处理Nikon RAW (NEF格式) 负片扫描的Python脚本，可以自动去除色罩、调整色阶、矫正白平衡，并提供丰富的手动调整选项。

## 功能特点

- ✅ 支持Nikon NEF格式的RAW文件
- ✅ 自动反相处理和色罩去除
- ✅ 智能ROI分析，避免胶片边框干扰
- ✅ **多种白平衡方法**：灰度世界、完美反射、自动白平衡
- ✅ **手动白平衡**：直接指定RGB增益
- ✅ **亮度调整**：曝光、亮度、对比度
- ✅ **高光/阴影调整**：独立控制高光和阴影
- ✅ **预设系统**：保存和加载处理配置
- ✅ **16-bit处理**：保持最大图像质量
- ✅ 支持多种输出格式（JPEG、PNG、TIFF）

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

### 白平衡调整

#### 自动白平衡方法

```bash
# 灰度世界假设（适合大多数场景）
python remove_color_cast.py input.NEF output.jpg --white-balance gray-world

# 完美反射假设（适合有高光的场景）
python remove_color_cast.py input.NEF output.jpg --white-balance perfect-reflector

# 自动白平衡（结合两种方法）
python remove_color_cast.py input.NEF output.jpg --white-balance auto
```

#### 手动白平衡

直接指定RGB增益系数：

```bash
python remove_color_cast.py input.NEF output.jpg --manual-wb 1.2,1.0,0.9
```

### 亮度调整

#### 曝光调整（EV单位）

```bash
# 增加曝光
python remove_color_cast.py input.NEF output.jpg --exposure +0.5

# 减少曝光
python remove_color_cast.py input.NEF output.jpg --exposure -0.3
```

#### 亮度和对比度

```bash
python remove_color_cast.py input.NEF output.jpg --brightness 0.1 --contrast 1.2
```

### 高光和阴影调整

```bash
# 降低高光，提亮阴影
python remove_color_cast.py input.NEF output.jpg --highlights -20 --shadows +30
```

### 预设系统

#### 保存预设

处理完成后保存当前设置：

```bash
python remove_color_cast.py input.NEF output.jpg \
    --exposure +0.3 \
    --contrast 1.1 \
    --shadows +20 \
    --white-balance auto \
    --save-preset my_settings.json
```

#### 使用预设

从预设文件加载设置：

```bash
python remove_color_cast.py input.NEF output.jpg --load-preset my_settings.json
```

预设文件也可以手动编辑，JSON格式：

```json
{
  "white_balance": "auto",
  "exposure": 0.3,
  "brightness": 0.1,
  "contrast": 1.1,
  "shadows": 20.0,
  "roi": "center-60"
}
```

### 完整工作流程示例

```bash
# 1. 使用所有功能处理负片
python remove_color_cast.py input.NEF output.jpg \
    --roi center-60 \
    --exposure +0.3 \
    --brightness 0.05 \
    --contrast 1.1 \
    --highlights -10 \
    --shadows +20 \
    --white-balance gray-world \
    --save-preset negative_film.json

# 2. 使用预设处理其他照片
python remove_color_cast.py input2.NEF output2.jpg --load-preset negative_film.json

# 3. 微调预设（命令行参数会覆盖预设）
python remove_color_cast.py input3.NEF output3.jpg \
    --load-preset negative_film.json \
    --exposure +0.4  # 覆盖预设中的曝光值
```

## 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ratio` | 0.01 | 色阶调整比率 |
| `--roi` | center | 分析区域：center, center-60, center-50, full, 或自定义坐标 |
| `--show-roi` | - | 显示分析区域的可视化 |

### 白平衡参数

| 参数 | 说明 |
|------|------|
| `--white-balance` | 白平衡方法：none, gray-world, perfect-reflector, auto |
| `--wb-roi` | 白平衡分析区域（默认与色罩ROI相同） |
| `--manual-wb` | 手动白平衡RGB增益，例如：1.2,1.0,0.9 |

### 亮度调整参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `--exposure` | ±EV | 曝光调整，例如：+0.5, -0.3 |
| `--brightness` | [-1, 1] | 亮度调整，例如：0.1, -0.2 |
| `--contrast` | [0, 3+] | 对比度调整，1.0为正常 |

### 高光/阴影参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `--highlights` | [-100, 100] | 高光调整，负数降低高光 |
| `--shadows` | [-100, 100] | 阴影调整，正数提亮阴影 |

### 预设参数

| 参数 | 说明 |
|------|------|
| `--save-preset PATH` | 保存当前设置到指定文件 |
| `--load-preset PATH` | 从指定文件加载设置 |

## 处理流程

所有处理都在16-bit精度下进行，保证最佳图像质量：

1. **RAW解码**：使用rawpy读取NEF文件
2. **ROI分析**：在指定区域分析色罩参数
3. **反相**：负片转正片
4. **色阶调整**：应用自动分析的色阶参数
5. **白平衡**：应用自动或手动白平衡
6. **曝光调整**：应用曝光补偿
7. **亮度/对比度**：应用亮度和对比度调整
8. **高光/阴影**：应用高光和阴影调整
9. **保存输出**：根据格式保存（JPEG 8-bit, PNG/TIFF 16-bit）

## 预设文件格式

预设文件是JSON格式，包含以下字段：

```json
{
  "_version": "1.0",
  "_description": "Nikon RAW 负片处理预设",
  "ratio": 0.01,
  "use_camera_wb": false,
  "roi": "center-60",
  "white_balance": "gray-world",
  "wb_roi": null,
  "exposure": 0.3,
  "brightness": 0.1,
  "contrast": 1.1,
  "highlights": -10.0,
  "shadows": 20.0,
  "manual_wb_multipliers": null
}
```

## 白平衡方法选择

- **gray-world**：适合色彩分布均匀的场景，大多数情况下效果良好
- **perfect-reflector**：适合有明显高光或白色物体的场景
- **auto**：综合两种方法，适合不确定使用哪种方法时
- **manual-wb**：当自动方法不准确时，可以手动指定RGB增益

## 典型应用场景

### 场景1：一般负片扫描

```bash
python remove_color_cast.py input.NEF output.jpg \
    --roi center-60 \
    --white-balance gray-world \
    --exposure +0.2
```

### 场景2：过曝的负片

```bash
python remove_color_cast.py input.NEF output.jpg \
    --roi center-60 \
    --exposure -0.3 \
    --highlights -30 \
    --shadows +20
```

### 场景3：需要精确色彩控制

```bash
# 第一步：尝试自动白平衡
python remove_color_cast.py input.NEF output.jpg \
    --white-balance auto

# 第二步：根据结果手动调整RGB增益
python remove_color_cast.py input.NEF output.jpg \
    --manual-wb 1.15,1.0,0.95

# 保存为预设供后续使用
python remove_color_cast.py input.NEF output.jpg \
    --manual-wb 1.15,1.0,0.95 \
    --save-preset color_correct.json
```

### 场景4：批量处理

```bash
# 创建预设
python remove_color_cast.py sample.NEF output.jpg \
    --roi center-60 \
    --exposure +0.3 \
    --contrast 1.1 \
    --white-balance auto \
    --save-preset batch_preset.json

# 批量处理其他照片
for nef in *.NEF; do
    python remove_color_cast.py "$nef" "${nef%.NEF}.jpg" --load-preset batch_preset.json
done
```

## 注意事项

- 对于有胶片边框的负片，建议使用 `center-60` 或 `center-50` ROI
- 分析区域应只包含负片主体，避免边框和黑色区域
- 所有处理都在高精度下进行，输出TIFF/PNG可保留16-bit
- 预设文件中的命令行参数会覆盖预设值
- 建议先用一张照片测试参数，然后保存预设用于批量处理
