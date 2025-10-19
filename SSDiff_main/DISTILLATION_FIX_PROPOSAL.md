# SSDiff 蒸馏修复方案

## 当前问题诊断

### 1. 核心问题：不是蒸馏，是直接训练
当前代码：
```python
gt_residual = gt - lms
loss = F.l1_loss(residual_pred, gt_residual)
```

问题：
- ❌ 没有teacher模型
- ❌ 直接让student在t=999预测GT，这是不可能的任务
- ❌ 训练越久，模型越过拟合到错误的目标

### 2. 实验证据
- Model 20001: 残差MSE=0.005024
- Model 59501: 残差MSE=0.015001 (更差！)
- 预测残差最大值只有实际的60%
- PSNR只有16dB（应该>30dB）

### 3. 根本原因
单步在t=999直接预测GT是一个**病态问题**：
- 扩散模型的t=999是噪声最大的时刻
- 需要多步去噪才能得到清晰结果
- 单步跳跃太大，无法学习

## 修复方案

### 方案A：真正的知识蒸馏（推荐）

#### 步骤1：使用teacher模型生成伪标签
```python
# 加载teacher（原始预训练的SSDiff）
teacher_model = load_pretrained_ssdiff(args.pretrained_ssdiff_path)
teacher_model.eval()

# 训练循环
for batch in dataloader:
    lms, pan, ms, gt = batch
    
    # 1. Teacher生成高质量输出（10步DDIM）
    with torch.no_grad():
        teacher_output = teacher_model.ddim_sample_loop(
            lms, pan, ms, 
            steps=10,  # 使用ddim10
            clip_denoised=True
        )
    
    # 2. Student在t=999单步预测
    student_output, student_residual = student_model(lms, pan, ms)
    
    # 3. 蒸馏损失：学习teacher的输出
    loss_distill = F.l1_loss(student_output, teacher_output)
    
    # 4. 可选：添加GT损失作为辅助
    loss_gt = F.l1_loss(student_output, gt) * 0.1
    
    loss = loss_distill + loss_gt
```

#### 步骤2：渐进式蒸馏
```python
# 不要直接从t=999开始，而是从较小的t逐步增加
current_t = 100  # 从t=100开始
for epoch in range(num_epochs):
    if epoch % 10 == 0 and current_t < 999:
        current_t = min(current_t + 100, 999)
    
    # 使用当前的timestep训练
    student_model.timesteps = torch.tensor([current_t])
```

### 方案B：多步蒸馏（更稳定）

不要追求单步，而是从10步蒸馏到2-3步：

```python
# 步骤1：Teacher 10步 -> Student 5步
# 步骤2：Teacher 5步 -> Student 3步
# 步骤3：Teacher 3步 -> Student 2步
```

### 方案C：使用一致性蒸馏（Consistency Distillation）

这是最新的方法，专门用于单步生成：

```python
# 1. Teacher在t和t-1时刻的输出应该一致
with torch.no_grad():
    # 从t-1开始，用teacher走一步到t
    x_t_minus_1 = add_noise(gt, t-1)
    teacher_pred_t = teacher_model(x_t_minus_1, t-1)

# 2. Student直接从t预测
x_t = add_noise(gt, t)
student_pred_t = student_model(x_t, t)

# 3. 一致性损失
loss = F.l1_loss(student_pred_t, teacher_pred_t)
```

## 立即可以尝试的快速修复

### 修复1：添加teacher模型

在`train_ssdiff_distill.py`中：

```python
# 加载teacher
teacher_unet, _ = initialize_ssdiff_unet_with_lora(
    args, 
    pretrained_path=args.pretrained_ssdiff_path,
    add_lora=False  # teacher不需要LoRA
)
teacher_unet.eval()
for p in teacher_unet.parameters():
    p.requires_grad = False

# 训练时
with torch.no_grad():
    # Teacher多步采样
    teacher_residual = teacher_unet.forward_impl(lms, pan, ms, x_t, timesteps)
    # 可能需要多步，这里简化为单步但用较小的t
    teacher_output = lms + teacher_residual

# Student单步
student_output, student_residual = model_gen(lms, pan, ms, gt)

# 蒸馏损失
loss_distill = F.l1_loss(student_output, teacher_output)
loss_gt = F.l1_loss(student_output, gt) * 0.1

loss = loss_distill + loss_gt
```

### 修复2：使用更小的timestep

将t=999改为t=100或t=200：

```python
# 在SSDiff_gen.__init__中
self.timesteps = torch.tensor([200], dtype=torch.long)  # 改为200
```

### 修复3：添加噪声schedule

```python
# 渐进式降低噪声
noise_scale = 1.0 - (global_step / total_steps) * 0.5
x_t = lms + torch.randn_like(lms) * noise_scale * 0.1
```

## 验证步骤

1. 首先测试原始SSDiff（ddim10）的性能基线
2. 测试teacher在单步（t=999）的性能上限
3. 对比student是否接近teacher的单步性能

## 参考文献

- Progressive Distillation for Fast Sampling of Diffusion Models (Salimans & Ho, 2022)
- Consistency Models (Song et al., 2023)
- On Distillation of Guided Diffusion Models (Meng et al., 2023)

