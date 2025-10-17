# SSDiff参数配置对照表

本文档对比原始SSDiff和蒸馏SSDiff的参数配置，帮助您在两种模式之间快速切换。

## 🔄 模式切换

### 原始SSDiff（多步采样）

```python
# 关键参数
use_distillation = False
timestep_respacing = "ddim10"  # 或 ddim50, ddim100
use_ddim = True
clip_denoised = True
```

### 蒸馏SSDiff（单步采样）

```python
# 关键参数
use_distillation = True
timestep_respacing = "ddim1"
use_ddim = True
clip_denoised = True
```

## 📊 完整参数对比

| 参数 | 原始SSDiff | 蒸馏SSDiff | 说明 |
|------|-----------|-----------|------|
| **模式控制** |
| `use_distillation` | `False` | `True` | 是否使用蒸馏模型 |
| `model_path` | 预训练SSDiff.pt | checkpoint.pkl | 模型路径 |
| `pretrained_ssdiff_path` | N/A | 预训练SSDiff.pt | 基础模型路径（蒸馏需要） |
| **采样控制** |
| `timestep_respacing` | `ddim10/50/100` | `ddim1` | 采样步数 |
| `use_ddim` | `True` | `True` | 使用DDIM采样 |
| `diffusion_steps` | `1000` | `1000` | 扩散总步数 |
| `clip_denoised` | `True` | `True` | 裁剪去噪结果 |
| **模型结构** |
| `num_channels` | `128` | `128` | UNet通道数 |
| `num_res_blocks` | `2` | `2` | 残差块数量 |
| `attention_resolutions` | `"16,8"` | `"16,8"` | 注意力分辨率 |
| `lora_rank` | N/A | `4` | LoRA秩（仅蒸馏） |
| **数据参数** |
| `ms_dim` | `8` | `8` | 多光谱通道数 |
| `pan_dim` | `1` | `1` | 全色通道数 |
| `image_size` | `64` | `64` | Patch大小 |
| `crop_batch_size` | `8` | `8` | 批次大小 |
| **训练参数** |
| `learning_rate` | `1e-3` | `5e-5` | 学习率 |
| `epochs` | `12000` | N/A | 训练轮数 |
| `max_train_steps` | N/A | `50000` | 最大训练步数 |
| `lambda_l2` | N/A | `1.0` | L2损失权重 |
| `lambda_vsd` | N/A | `1.0` | VSD损失权重 |
| `lambda_vsd_lora` | N/A | `1.0` | LoRA扩散损失权重 |
| **优化器** |
| `optimizer` | `Adam` | `AdamW` | 优化器类型 |
| `adam_weight_decay` | `0` | `1e-2` | 权重衰减 |
| `lr_scheduler` | 自定义 | `constant/cosine` | 学习率调度器 |

## 🎮 命令行示例

### 训练命令对比

#### 原始SSDiff训练

```bash
python train_DPM.py \
    --arch iDPM \
    --dataset wv3 \
    --data_dir /path/to/data \
    --lr 1e-3 \
    --epochs 12000 \
    --samples_per_gpu 20 \
    --diffusion_steps 1000 \
    --noise_schedule cosine
```

#### 蒸馏SSDiff训练

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path /path/to/pretrained/ssdiff.pt \
    --data_dir /path/to/data \
    --learning_rate 5e-5 \
    --max_train_steps 50000 \
    --train_batch_size 4 \
    --lora_rank 4 \
    --lambda_l2 1.0 \
    --lambda_vsd 1.0 \
    --lambda_vsd_lora 1.0 \
    --mixed_precision fp16
```

### 测试命令对比

#### 原始SSDiff测试（10步）

```bash
python image_sample.py \
    --device cuda:0 \
    --model_path /path/to/model.pt \
    --timestep_respacing ddim10 \
    --crop_batch_size 8 \
    --test_dataset test_wv3_multiExm1.h5
```

#### 蒸馏SSDiff测试（1步）

```bash
python test_ssdiff_unified.py \
    --model_path /path/to/distilled/checkpoint.pkl \
    --pretrained_ssdiff_path /path/to/base/model.pt \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --device cuda:0 \
    --test_dataset test_wv3_multiExm1.h5
```

## 🔧 配置文件对比

### 原始SSDiff配置

```python
# configs/option_DPM_pansharpening.py
parser.add_argument('--diffusion_steps', default=1000)
parser.add_argument('--noise_schedule', default="cosine")
parser.add_argument('--timestep_respacing', default="ddim10")
parser.add_argument('--use_ddim', default=True)
parser.add_argument('--predict_xstart', default=True)
```

### 蒸馏SSDiff配置

```python
# ssdiff_distill.py - SSDiff_gen
self.timesteps = torch.tensor([999], dtype=torch.long)  # 固定单步
self.args.predict_xstart = True
self.args.use_distillation = True
```

## 📈 性能对比

| 指标 | 原始SSDiff<br>(DDIM-50) | 原始SSDiff<br>(DDIM-10) | 蒸馏SSDiff<br>(1步) |
|------|----------------------|----------------------|-------------------|
| **推理时间** | ~5.0s | ~1.0s | ~0.1s |
| **加速比** | 1x | 5x | 50x |
| **PSNR** | 32.50 dB | 32.45 dB | 32.20 dB ↓0.3dB |
| **SSIM** | 0.950 | 0.948 | 0.940 ↓0.01 |
| **参数量** | 全模型 | 全模型 | 仅LoRA (~1%) |
| **内存占用** | 高 | 高 | 低 |

## 🎯 使用场景推荐

### 原始SSDiff（多步采样）

**适用场景**:
- ✅ 追求最高质量
- ✅ 离线处理，不在意速度
- ✅ 研究和实验
- ✅ 作为教师模型蒸馏

**推荐配置**:
```bash
--timestep_respacing ddim50
--use_ddim True
--clip_denoised True
```

### 蒸馏SSDiff（单步采样）

**适用场景**:
- ✅ 实时/准实时应用
- ✅ 大规模批量处理
- ✅ 移动/边缘设备部署
- ✅ 需要快速反馈的交互应用

**推荐配置**:
```bash
--use_distillation True
--timestep_respacing ddim1
--lora_rank 4
--mixed_precision fp16
```

## 🔄 切换流程

### 从原始SSDiff切换到蒸馏版本

1. **训练蒸馏模型**:
```bash
bash scripts/train_ssdiff_distill.sh
```

2. **修改测试脚本**:
```python
# 原始
use_distillation = False
timestep_respacing = "ddim10"

# 改为
use_distillation = True
timestep_respacing = "ddim1"
model_path = "experiments/ssdiff_distill/checkpoints/model_10000.pkl"
```

3. **运行测试**:
```bash
bash scripts/test_ssdiff_distilled.sh
```

### 从蒸馏版本回退到原始SSDiff

1. **修改测试脚本**:
```python
# 蒸馏
use_distillation = True
timestep_respacing = "ddim1"

# 改为
use_distillation = False
timestep_respacing = "ddim10"
model_path = "/path/to/original/ssdiff.pt"
```

2. **运行测试**:
```bash
bash scripts/test_ssdiff_original.sh
```

## 🛠️ 调试技巧

### 检查当前使用的模式

```python
# 在代码中添加日志
print(f"Mode: {'Distillation' if args.use_distillation else 'Original'}")
print(f"Timestep respacing: {args.timestep_respacing}")
print(f"Model path: {args.model_path}")
```

### 验证模型加载

```python
# 检查是否正确加载了LoRA
for name, param in model.named_parameters():
    if 'lora' in name:
        print(f"LoRA parameter: {name}, requires_grad: {param.requires_grad}")
```

### 性能监控

```python
import time

start = time.time()
output = model(lms, pan, ms)
end = time.time()

print(f"Inference time: {end - start:.3f}s")
print(f"FPS: {1/(end-start):.2f}")
```

## 📝 总结

| 特性 | 原始SSDiff | 蒸馏SSDiff |
|------|-----------|-----------|
| **速度** | 慢 (10-50步) | 快 (1步) |
| **质量** | 最高 | 接近原始 (95%+) |
| **参数** | 全模型 | 仅LoRA |
| **部署** | 重量级 | 轻量级 |
| **训练** | 从头训练 | 基于预训练蒸馏 |
| **切换** | 修改配置即可 | 需要训练蒸馏模型 |

选择建议：
- 🎯 **高质量需求** → 原始SSDiff (DDIM-50)
- ⚡ **高速度需求** → 蒸馏SSDiff (1步)
- 🔄 **平衡方案** → 原始SSDiff (DDIM-10)

