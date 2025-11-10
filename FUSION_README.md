# OTPNet-SSDiff 后融合模型

将预训练的OTPNet和SSDiff（带ARConv版）模型输出进行后融合，提升全色锐化性能。

## 📋 模型架构

```
输入: pan [B,1,H,W], lms [B,8,H,W], ms [B,8,H/4,W/4]
  │
  ├─→ OTPNet (冻结)  → output1 [B,8,H,W]
  │
  └─→ SSDiff (冻结)  → output2 [B,8,H,W]
        │
        ├─→ Concat [B,16,H,W]
        │
        └─→ 融合网络 (可训练)
             ├─ FusionBlock × N
             ├─ OutputConv
             └─→ 残差学习 → 最终输出 [B,8,H,W]
```

### 数据说明
- **ms**: 原始低分辨率多光谱 `[B, 8, H/4, W/4]` （从数据集获取）
- **lms**: 上采样后的多光谱 `[B, 8, H, W]` （由ms双线性上采样得到）
- **pan**: 高分辨率全色图像 `[B, 1, H, W]`

## 📦 文件说明

- `fusion_model.py`: 融合模型定义
- `train_fusion.py`: 训练脚本
- `test_fusion.py`: 测试脚本
- `run_fusion_train.sh`: 训练示例脚本
- `run_fusion_test.sh`: 测试示例脚本

## 🚀 使用方法

### 1. 训练融合模型

```bash
python train_fusion.py \
    --otpnet_checkpoint /path/to/otpnet_best.pth \
    --ssdiff_checkpoint /path/to/ssdiff_distilled.pth \
    --pretrained_ssdiff_path /path/to/pretrained_ssdiff_base.pth \
    --train_dataset train_wv3.h5 \
    --val_dataset valid_wv3.h5 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --fusion_channels 64 \
    --num_fusion_blocks 3 \
    --device cuda:0 \
    --save_dir checkpoints/fusion
```

**训练参数说明：**
- `--fusion_channels`: 融合网络隐藏通道数 (默认64)
- `--num_fusion_blocks`: 融合块数量 (默认3)
- `--loss_l1_weight`: L1损失权重 (默认1.0)
- `--loss_l2_weight`: L2损失权重 (默认1.0)
- `--loss_consistency_weight`: 一致性损失权重 (默认0.1，防止融合结果过度偏离基础模型)

### 2. 测试融合模型

**普通测试：**
```bash
python test_fusion.py \
    --otpnet_checkpoint /path/to/otpnet_best.pth \
    --ssdiff_checkpoint /path/to/ssdiff_distilled.pth \
    --pretrained_ssdiff_path /path/to/pretrained_ssdiff_base.pth \
    --fusion_weights checkpoints/fusion/fusion_best.pth \
    --test_dataset test_wv3_multiExm1.h5 \
    --num_test_images 20 \
    --device cuda:0 \
    --save_individual_outputs
```

**使用Self-Ensemble：**
```bash
python test_fusion.py \
    --otpnet_checkpoint /path/to/otpnet_best.pth \
    --ssdiff_checkpoint /path/to/ssdiff_distilled.pth \
    --pretrained_ssdiff_path /path/to/pretrained_ssdiff_base.pth \
    --fusion_weights checkpoints/fusion/fusion_best.pth \
    --test_dataset test_wv3_multiExm1.h5 \
    --use_self_ensemble \
    --device cuda:0
```

## 🔧 核心特性

### 1. 只训练融合层
- OTPNet和SSDiff的权重完全冻结
- 仅训练轻量级融合网络（通常<1M参数）
- 快速训练，避免破坏预训练模型

### 2. 多重损失函数
- **L1损失**: 主要重建损失
- **L2损失**: 辅助平滑损失
- **一致性损失**: 保持与基础模型输出的一致性，避免过拟合

### 3. 残差学习
融合网络学习残差：
```python
base = (otpnet_out + ssdiff_out) / 2.0
output = base + residual
```

### 4. Self-Ensemble支持
测试时支持8种几何变换集成：
1. 原图
2. 水平翻转
3. 垂直翻转
4. 旋转90°
5. 旋转180°
6. 旋转270°
7. 水平翻转+旋转90°
8. 垂直翻转+旋转90°

## 📊 输出格式

测试脚本输出`.mat`文件，包含：
- `sr`: 融合模型输出 `[N, C, H, W]`
- `gt`: Ground truth `[N, C, H, W]`
- `sr_otpnet`: OTPNet单独输出 (可选)
- `sr_ssdiff`: SSDiff单独输出 (可选)
- `inference_time`: 推理总时间
- `avg_time_per_image`: 平均每张图像时间

## 🎯 预期效果

融合模型通过结合两个模型的优势：
- OTPNet: 快速、结构保持好
- SSDiff: 细节丰富、光谱保真度高

融合后预期提升：
- PSNR: +0.5~1.5 dB
- SAM: -5%~10%
- 更好的视觉质量

## 📝 注意事项

1. **模型配置一致性**: 测试时的`fusion_channels`和`num_fusion_blocks`必须与训练时一致

2. **数据归一化**: 所有输入数据应归一化到[0,1]范围

3. **显存要求**: 同时加载两个模型，显存需求约为单模型的2倍+融合网络

4. **Self-ensemble**: 使用self-ensemble会增加8倍推理时间，但通常能提升0.2~0.5dB

## 🛠️ 故障排查

**问题1: OTPNet加载失败**
- 检查checkpoint中是否包含`model_kwargs`
- 确认OTPNet配置（num_stages, hidden_channels等）

**问题2: SSDiff加载失败**
- 确认同时提供了`ssdiff_checkpoint`和`pretrained_ssdiff_path`
- 检查SSDiff配置中的`use_distillation=True`

**问题3: 训练loss不下降**
- 降低学习率（如1e-5）
- 增加`consistency_weight`权重
- 检查数据归一化是否正确

## 📚 相关文件

- SSDiff蒸馏模型: `ssdiff_distill.py`
- SSDiff测试（带self-ensemble）: `test_ssdiff_distilled.py`
- OTPNet测试: `OTPNet/test.py`

