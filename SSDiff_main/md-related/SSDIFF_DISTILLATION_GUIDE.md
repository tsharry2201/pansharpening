# SSDiff一步蒸馏训练指南

本指南说明如何将预训练的SSDiff模型进行一步蒸馏，模仿OSEDiff的训练范式。

## 🎯 核心思想

将SSDiff从多步扩散过程（如DDIM 10/50/100步）蒸馏为单步生成模型，提高推理速度的同时保持质量。

## 📐 技术方案

### 1. 架构设计

```
SSDiff_gen (生成器):
├── UNet (带LoRA)      # 单步去噪网络
└── timestep = 999      # 固定在最后一步

SSDiff_reg (正则化器):
├── UNet_fix (固定)     # 原始预训练SSDiff，多步去噪
└── UNet_update (更新)  # 带LoRA的正则化网络
```

### 2. 损失函数

```python
Total Loss = L2_loss + Distribution_Matching_loss + Diff_loss

其中：
- L2_loss: 重建损失，确保生成结果与GT接近
- Distribution_Matching_loss: 让单步模型接近多步SSDiff的分布
- Diff_loss: 让LoRA学习扩散先验
```

### 3. 训练流程

**阶段1: 准备预训练模型**
- 加载预训练的SSDiff checkpoint
- 冻结主干网络参数
- 只训练LoRA层

**阶段2: 单步蒸馏**
- 固定timestep=999，将多步扩散过程压缩到单步
- 使用原始SSDiff作为教师模型（多步采样）
- 训练学生模型（单步采样）逼近教师模型输出

**阶段3: 分布匹配**
- 在潜在空间进行KL散度优化
- 确保单步模型遵循预训练的扩散分布

## 🔧 实现步骤

### Step 1: 创建蒸馏模型类 (ssdiff_distill.py)

创建类似OSEDiff的模型结构：
- `SSDiff_gen`: 单步生成器
- `SSDiff_reg`: 正则化器（包含固定和可更新的UNet）

### Step 2: 创建训练脚本 (train_ssdiff_distill.py)

参照`train_osediff.py`的训练流程：
1. 加载预训练SSDiff模型
2. 初始化LoRA层
3. 设置优化器和学习率调度器
4. 训练循环：
   - 前向传播（单步去噪）
   - 计算损失
   - 反向传播更新LoRA参数

### Step 3: 创建推理脚本 (test_ssdiff_distill.py)

支持两种模式：
```python
# 模式1: 原始SSDiff多步采样
--use_distillation False --timestep_respacing ddim10

# 模式2: 蒸馏后的单步采样
--use_distillation True --timestep_respacing ddim1
```

## 📊 与OSEDiff的对比

| 特性 | OSEDiff | SSDiff蒸馏 |
|------|---------|------------|
| 任务 | 真实图像超分辨率 | 全景锐化 |
| 输入 | 低分辨率RGB图像(3通道) | MS+PAN图像(8+1通道) |
| 输出 | 高分辨率RGB图像(3通道) | 锐化MS图像(8通道) |
| VAE | 使用SD的VAE | 不使用VAE（直接在像素空间） |
| 文本条件 | 使用RAM生成caption | 无文本条件 |
| 预训练基座 | Stable Diffusion 2.1 | 自训练的DPM模型 |

## 💡 关键差异处理

### 1. 无VAE编码
OSEDiff在潜在空间工作，SSDiff直接在像素空间：
```python
# OSEDiff
latents = vae.encode(x).latent_dist.sample()

# SSDiff蒸馏（直接使用图像）
x_input = torch.cat([lms, pan], dim=1)  # 9通道输入
```

### 2. 无文本条件
移除text_encoder和prompt相关代码：
```python
# OSEDiff需要prompt
model_pred = unet(latents, t, encoder_hidden_states=prompt_embeds)

# SSDiff蒸馏不需要
model_pred = unet(x_noisy, t, lms=lms, pan=pan, ms=ms)
```

### 3. 条件输入
SSDiff使用特殊的条件机制：
```python
# 在UNet forward中传入条件数据
kwargs = {"lms": lms, "pan": pan, "ms": ms}
sample = diffusion.ddim_sample_loop(model, shape, model_kwargs=kwargs)
```

## 🎮 训练命令示例

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path /path/to/pretrained/ssdiff/model.pt \
    --data_dir /path/to/wv3/dataset \
    --learning_rate 5e-5 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --mixed_precision fp16 \
    --max_train_steps 50000 \
    --checkpointing_steps 500 \
    --output_dir experiments/ssdiff_distill \
    --lambda_l2 1.0 \
    --lambda_vsd 1.0 \
    --lambda_vsd_lora 1.0 \
    --lora_rank 4
```

## 🧪 测试命令示例

```bash
# 原始SSDiff多步采样
python test_ssdiff_unified.py \
    --model_path /path/to/pretrained/ssdiff/model.pt \
    --use_distillation False \
    --timestep_respacing ddim10

# 蒸馏后单步采样
python test_ssdiff_unified.py \
    --model_path experiments/ssdiff_distill/checkpoints/model_10000.pkl \
    --pretrained_ssdiff_path /path/to/pretrained/ssdiff/model.pt \
    --use_distillation True \
    --timestep_respacing ddim1
```

## 📈 预期效果

- **速度提升**: 10-50倍（取决于原始采样步数）
- **质量保持**: 95%+ 的原始指标（PSNR/SSIM）
- **推理时间**: ~0.1秒/图像 (256x256)

## 🔍 调试技巧

1. **检查数据范围**: SSDiff使用[0, 2047]范围，确保归一化正确
2. **监控损失**: 初期L2 loss应该快速下降
3. **渐进式训练**: 先只用L2 loss训练，稳定后加入KL loss
4. **学习率调整**: 如果loss震荡，降低学习率到1e-5

## 📚 参考资料

- [OSEDiff Paper](https://arxiv.org/pdf/2406.08177)
- [Diffusion Distillation](https://arxiv.org/abs/2202.00512)
- [LoRA Fine-tuning](https://arxiv.org/abs/2106.09685)

