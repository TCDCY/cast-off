# Nikon RAW 负片去色罩工具

用于处理Nikon RAW (NEF格式) 负片扫描的Python脚本，可以自动去除色罩、调整色阶、矫正白平衡，并提供丰富的手动调整选项。

## 功能特点

- ✅ 支持Nikon NEF格式的RAW文件
- ✅ 自动反相处理和色罩去除
- ✅ 智能ROI分析，避免胶片边框干扰
- ✅ **多种白平衡方法**：灰度世界、完美反射、自动白平衡
- ✅ **色温/色调调整**：直观的冷暖和绿品调整
- ✅ **亮度调整**：曝光、亮度、对比度
- ✅ **高光/阴影调整**：独立控制高光和阴影
- ✅ **预设系统**：保存和加载处理配置
- ✅ **相对值调整**：在预设基础上微调
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

- --exposure: 曝光（EV）
- --brightness: 亮度 [-1, 1]
- --contrast: 对比度 [0, 3+]
- --highlights: 高光 [-100, 100]
- --shadows: 阴影 [-100, 100]
- --temperature: 色温 [-100, 100]
- --tint: 色调 [-100, 100]

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

#### 色温和色调调整（推荐）

使用直观的色温和色调参数：

```bash
# 色温调整：正数偏暖(黄)，负数偏冷(蓝)
python remove_color_cast.py input.NEF output.jpg --temperature +20

# 色调调整：正数偏品红，负数偏绿
python remove_color_cast.py input.NEF output.jpg --tint -10

# 同时调整色温和色调
python remove_color_cast.py input.NEF output.jpg --temperature +15 --tint -5
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
    --temperature +15 \
    --tint -5 \
    --save-preset my_settings.json
```

#### 使用预设（绝对值模式）

从预设文件加载设置，命令行参数会覆盖预设值：

```bash
python remove_color_cast.py input.NEF output.jpg --load-preset my_settings.json
```

#### 使用预设并微调（相对值模式）⭐

在预设基础上进行相对值调整，这是推荐的工作流程：

```bash
# 第一步：创建基础预设
python remove_color_cast.py sample.NEF output.jpg \
    --exposure +0.3 \
    --temperature +15 \
    --tint -5 \
    --save-preset base_settings.json

# 第二步：在预设基础上微调
python remove_color_cast.py input2.NEF output2.jpg \
    --load-preset base_settings.json \
    --relative-adjust \
    --temperature +5 \
    --exposure +0.1

# 结果：exposure = 0.3 + 0.1 = 0.4
#      temperature = 15 + 5 = 20
#      tint = -5 (保持不变)
```

预设文件格式（JSON）：

```json
{
  "exposure": 0.3,
  "temperature": 15.0,
  "tint": -5.0,
  "contrast": 1.1
}
```

### 完整工作流程示例

```bash
# 1. 创建基础预设
python remove_color_cast.py sample.NEF output.jpg \
    --roi center-60 \
    --exposure +0.3 \
    --brightness 0.05 \
    --contrast 1.1 \
    --highlights -10 \
    --shadows +20 \
    --temperature +15 \
    --tint -5 \
    --save-preset negative_film.json

# 2. 批量处理其他照片（使用预设）
python remove_color_cast.py input2.NEF output2.jpg --load-preset negative_film.json

# 3. 在预设基础上微调某张照片
python remove_color_cast.py input3.NEF output3.jpg \
    --load-preset negative_film.json \
    --relative-adjust \
    --temperature +10 \
    --exposure +0.2
```

## 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ratio` | 0.01 | 色阶调整比率 |
| `--roi` | center | 分析区域：center, center-60, center-50, full, 或自定义坐标 |
| `--show-roi` | - | 显示分析区域的可视化 |

### 白平衡参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `--white-balance` | - | 白平衡方法：none, gray-world, perfect-reflector, auto |
| `--temperature` | [-100, 100] | 色温：负数偏冷(蓝)，正数偏暖(黄) |
| `--tint` | [-100, 100] | 色调：负数偏绿，正数偏品红 |

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
| `--relative-adjust` | 使用相对值调整模式（在预设基础上微调） |

## 白平衡系统说明

### 色温（Temperature）

控制图像的冷暖色调：
- **负值**：偏冷，增加蓝色，减少红绿色
- **正值**：偏暖，增加红黄色，减少蓝色
- **范围**：-100 (最冷) 到 +100 (最暖)
- **0**：中性，不调整

### 色调（Tint）

控制图像的绿色/品红偏移：
- **负值**：偏绿，增加绿色，减少红蓝色
- **正值**：偏品红，增加红蓝色，减少绿色
- **范围**：-100 (最绿) 到 +100 (最品红)
- **0**：中性，不调整

### 白平衡工作流程建议

1. **第一步**：使用自动白平衡（`--white-balance auto`）
2. **第二步**：根据结果使用色温/色调微调
3. **第三步**：保存满意的设置为预设
4. **第四步**：使用相对值模式在其他照片上微调

## 相对值调整模式

相对值调整模式允许你在预设基础上进行增量调整，这是处理多张相似负片时的最佳方式。

### 工作原理

```bash
# 预设文件内容：exposure=0.3, temperature=15, tint=-5

# 使用相对值模式
python remove_color_cast.py input.NEF output.jpg \
    --load-preset preset.json \
    --relative-adjust \
    --exposure +0.1 \
    --temperature +5

# 实际应用：exposure=0.4, temperature=20, tint=-5
# (预设值 + 命令行增量)
```

### 适用场景

- ✅ 同一卷胶片的不同照片
- ✅ 相同拍摄条件的多张负片
- ✅ 需要保持一致性的批量处理
- ✅ 逐步微调找到最佳参数

### 绝对值 vs 相对值

| 模式 | 命令行参数含义 | 用途 |
|------|--------------|------|
| 绝对值（默认） | 直接指定最终值 | 完全替换预设参数 |
| 相对值（--relative-adjust） | 在预设基础上增量调整 | 在预设基础上微调 |

## 处理流程

所有处理都在16-bit精度下进行，保证最佳图像质量：

1. **RAW解码**：使用rawpy读取NEF文件
2. **ROI分析**：在指定区域分析色罩参数
3. **反相**：负片转正片
4. **色阶调整**：应用自动分析的色阶参数
5. **白平衡**：应用自动或手动白平衡（色温/色调）
6. **曝光调整**：应用曝光补偿
7. **亮度/对比度**：应用亮度和对比度调整
8. **高光/阴影**：应用高光和阴影调整
9. **保存输出**：根据格式保存（JPEG 8-bit, PNG/TIFF 16-bit）

## 典型应用场景

### 场景1：单张负片处理

```bash
python remove_color_cast.py input.NEF output.jpg \
    --roi center-60 \
    --white-balance auto \
    --temperature +10 \
    --exposure +0.2
```

### 场景2：批量处理同一卷胶片

```bash
# 1. 用一张照片创建预设
python remove_color_cast.py sample.NEF sample.jpg \
    --roi center-60 \
    --exposure +0.3 \
    --temperature +15 \
    --tint -5 \
    --save-preset roll1.json

# 2. 批量应用预设到其他照片
for nef in roll1_*.NEF; do
    python remove_color_cast.py "$nef" "${nef%.NEF}.jpg" --load-preset roll1.json
done
```

### 场景3：精细调整工作流

```bash
# 1. 创建基础预设
python remove_color_cast.py sample.NEF out.jpg \
    --exposure +0.3 --temperature +15 --save-preset base.json

# 2. 逐步微调
# 第一次微调：稍微暖一点
python remove_color_cast.py input2.NEF out2.jpg \
    --load-preset base.json --relative-adjust --temperature +5

# 第二次微调：再暖一点，曝光增加
python remove_color_cast.py input3.NEF out3.jpg \
    --load-preset base.json --relative-adjust --temperature +10 --exposure +0.1

# 保存新的预设
python remove_color_cast.py input3.NEF out3.jpg \
    --exposure +0.4 --temperature +25 --save-preset warmer.json
```

### 场景4：处理不同批次的负片

```bash
# 批次1：色温偏冷
python remove_color_cast.py batch1_sample.NEF out.jpg \
    --temperature -10 --save-preset cool_batch.json

# 批次2：在批次1基础上调整为中性
python remove_color_cast.py batch2_sample.NEF out.jpg \
    --load-preset cool_batch.json \
    --relative-adjust \
    --temperature +10  # -10 + 10 = 0 (中性)
```

## 注意事项

- 对于有胶片边框的负片，建议使用 `center-60` 或 `center-50` ROI
- 分析区域应只包含负片主体，避免边框和黑色区域
- 所有处理都在高精度下进行，输出TIFF/PNG可保留16-bit
- 相对值模式需要配合 `--load-preset` 使用
- 色温和色调是直观的调整方式，比RGB更容易理解和使用
- 建议先用一张照片创建基础预设，然后用相对值模式微调其他照片

