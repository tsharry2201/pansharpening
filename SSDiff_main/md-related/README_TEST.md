# 测试和评估指南

## 📋 完整流程

### 1. 运行测试生成结果

```bash
cd /data2/user/zelilin/ARConv_SSDiff/SSDiff_main

# 使用EMA模型测试（推荐）
python image_sample.py
```

**配置说明：**
- 模型路径：在 `configs/option_DPM_pansharpening.py` 第33行设置
- 测试数据集：默认使用 `dataset/WV3/test_wv3_multiExm1.h5`
- 采样步数：第99行 `timestep_respacing="ddim1000"` (可改为ddim10/ddim50加速)

**输出文件：**
```
logs/samp_reduced_20_256_MM-DD-HH-MM.mat
```

包含两个字段：
- `sr`: 模型融合结果 (Super-Resolution)
- `gt`: 真实标签 (Ground Truth)

---

### 2. 评估生成的结果

使用Python评估脚本：

```bash
# 基本用法
python evaluate_results.py --mat_file logs/samp_reduced_20_256_10-14-19-50.mat

# 完整参数
python evaluate_results.py \
    --mat_file logs/samp_reduced_20_256_10-14-19-50.mat \
    --dynamic_range 2047 \
    --scale 4 \
    --block_size 8
```

**参数说明：**
- `--mat_file`: 测试生成的.mat文件路径（必需）
- `--dynamic_range`: 动态范围，WV3用2047，GF2用1023（默认2047）
- `--scale`: 下采样倍数（默认4）
- `--block_size`: Q-index计算的块大小（默认8）

**评估指标：**
- **PSNR** (Peak Signal-to-Noise Ratio): 越高越好，单位dB
- **SSIM** (Structural Similarity): 越高越好，最佳值1
- **SAM** (Spectral Angle Mapper): 越低越好，最佳值0
- **ERGAS**: 越低越好，最佳值0
- **Q8**: 越高越好，最佳值1

---

### 3. 示例输出

```
Loading results from: logs/samp_reduced_20_256_10-14-19-50.mat
SR shape: (20, 8, 256, 256)
GT shape: (20, 8, 256, 256)

Evaluating 20 images...
================================================================================

Image 1/20:
  PSNR:  42.3456 dB
  SSIM:  0.9823
  SAM:   0.0234
  ERGAS: 1.2345
  Q8:    0.9654

...

================================================================================
Average Metrics:
================================================================================
PSNR  : 42.1234 ± 0.5678
SSIM  : 0.9801 ± 0.0023
SAM   : 0.0245 ± 0.0012
ERGAS : 1.2567 ± 0.0345
Q8    : 0.9632 ± 0.0045
```

---

## 🎯 快速开始

### 完整测试流程（一键式）

```bash
cd /data2/user/zelilin/ARConv_SSDiff/SSDiff_main

# 1. 修改配置文件中的模型路径
# configs/option_DPM_pansharpening.py 第33行

# 2. 运行测试
python image_sample.py

# 3. 评估结果（替换为实际生成的文件名）
python evaluate_results.py --mat_file logs/samp_reduced_20_256_*.mat
```

---

## 📊 不同模型对比

### 测试不同checkpoint

```bash
# 修改配置文件，测试10000步模型
# test_model_path = "results/10-14-17-18/ema_0.9999_010000.pt"
python image_sample.py
python evaluate_results.py --mat_file logs/samp_reduced_20_256_*.mat

# 修改配置文件，测试16000步模型
# test_model_path = "results/10-14-17-18/ema_0.9999_016000.pt"
python image_sample.py
python evaluate_results.py --mat_file logs/samp_reduced_20_256_*.mat
```

### EMA vs 普通模型对比

```bash
# EMA模型
# test_model_path = "results/10-14-17-18/ema_0.9999_010000.pt"
python image_sample.py
mv logs/samp_reduced_20_256_*.mat logs/result_ema.mat

# 普通模型
# test_model_path = "results/10-14-17-18/model010000.pt"
python image_sample.py
mv logs/samp_reduced_20_256_*.mat logs/result_normal.mat

# 对比评估
echo "EMA Model:"
python evaluate_results.py --mat_file logs/result_ema.mat
echo ""
echo "Normal Model:"
python evaluate_results.py --mat_file logs/result_normal.mat
```

---

## 🔧 常见问题

### 1. 找不到测试数据集

确保软链接创建成功：
```bash
ls -lh dataset/PanCollection/test_data/test_wv3_multiExm1.h5
```

### 2. 内存不足

减少测试图像数量（修改image_sample.py第72行）：
```python
image_num = 5  # 从20改为5
```

### 3. 加速测试

使用更少的采样步数（修改configs/option_DPM_pansharpening.py第99行）：
```python
parser.add_argument('--timestep_respacing', default="ddim10")  # 从ddim1000改为ddim10
```

---

## 📈 期望的指标范围

基于WV3数据集的典型值：

| 指标 | 优秀 | 良好 | 一般 |
|------|------|------|------|
| PSNR | > 40 dB | 35-40 dB | < 35 dB |
| SSIM | > 0.95 | 0.90-0.95 | < 0.90 |
| SAM | < 2.0 | 2.0-3.0 | > 3.0 |
| ERGAS | < 2.0 | 2.0-3.0 | > 3.0 |
| Q8 | > 0.95 | 0.90-0.95 | < 0.90 |

---

## 💡 提示

1. **优先使用EMA模型**：通常比普通模型性能更好
2. **多个checkpoint测试**：选择性能最好的模型
3. **保存结果**：将评估结果重定向到文件
   ```bash
   python evaluate_results.py --mat_file xxx.mat > results.txt
   ```

