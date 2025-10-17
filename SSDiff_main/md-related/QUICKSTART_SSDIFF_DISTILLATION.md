# SSDiff一步蒸馏快速开始指南

本指南帮助您快速上手SSDiff的一步蒸馏训练和测试。

## 📋 目录结构

```
OSEDiff/
├── ssdiff_distill.py              # 蒸馏模型定义
├── train_ssdiff_distill.py        # 训练脚本
├── test_ssdiff_unified.py         # 统一测试脚本
├── scripts/
│   ├── train_ssdiff_distill.sh    # 训练脚本示例
│   ├── test_ssdiff_original.sh    # 测试原始SSDiff
│   ├── test_ssdiff_distilled.sh   # 测试蒸馏SSDiff
│   └── compare_speed.sh           # 速度对比
├── SSDIFF_DISTILLATION_GUIDE.md   # 详细技术指南
└── QUICKSTART_SSDIFF_DISTILLATION.md  # 本文件
```

## 🚀 快速开始

### 步骤1: 环境准备

确保您已经安装了必要的依赖：

```bash
# 使用OSEDiff的环境
conda activate OSEDiff

# 或者安装必要的包
pip install torch accelerate diffusers peft transformers wandb scipy
```

### 步骤2: 修改配置路径

编辑 `scripts/train_ssdiff_distill.sh`，修改以下路径：

```bash
# 预训练SSDiff模型路径
PRETRAINED_SSDIFF="/your/path/to/ssdiff/model.pt"

# 数据集路径
DATA_DIR="/your/path/to/wv3/dataset"
```

### 步骤3: 开始训练

```bash
# 方式1: 使用脚本
bash scripts/train_ssdiff_distill.sh

# 方式2: 直接运行命令
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path /path/to/pretrained/ssdiff.pt \
    --data_dir /path/to/wv3/dataset \
    --output_dir experiments/ssdiff_distill \
    --learning_rate 5e-5 \
    --train_batch_size 4 \
    --max_train_steps 50000 \
    --lora_rank 4
```

### 步骤4: 监控训练

训练过程中可以通过TensorBoard监控：

```bash
tensorboard --logdir experiments/ssdiff_distill/logs
```

或者查看wandb日志（需要设置wandb账号）。

### 步骤5: 测试模型

#### 5.1 测试原始SSDiff（多步采样）

```bash
bash scripts/test_ssdiff_original.sh
```

或手动运行：

```bash
python test_ssdiff_unified.py \
    --model_path /path/to/original/ssdiff.pt \
    --use_distillation False \
    --timestep_respacing ddim10 \
    --test_dataset test_wv3_multiExm1.h5 \
    --num_test_images 20
```

#### 5.2 测试蒸馏后的SSDiff（单步采样）

```bash
bash scripts/test_ssdiff_distilled.sh
```

或手动运行：

```bash
python test_ssdiff_unified.py \
    --model_path experiments/ssdiff_distill/checkpoints/model_10000.pkl \
    --pretrained_ssdiff_path /path/to/original/ssdiff.pt \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset test_wv3_multiExm1.h5 \
    --num_test_images 20
```

### 步骤6: 速度对比

运行速度对比脚本：

```bash
bash scripts/compare_speed.sh
```

这将测试原始SSDiff和蒸馏SSDiff的推理速度，结果保存在 `test_results/comparison/` 目录。

## 📊 预期结果

### 训练损失

正常训练时，您应该看到以下损失变化：

- **loss_l2**: 从 0.01-0.1 逐渐下降到 0.001-0.01
- **loss_vsd**: 从 0.1-0.5 逐渐下降到 0.01-0.1
- **loss_diff**: 保持在 0.01-0.1 范围

如果损失不下降或震荡，尝试：
1. 降低学习率（如 1e-5）
2. 增加warmup步数
3. 减小batch size

### 推理速度

在典型的GPU上（如NVIDIA A100），您应该看到：

| 模式 | 采样步数 | 速度 (img/s) | 加速比 |
|------|----------|--------------|--------|
| 原始SSDiff | DDIM-50 | ~1-2 | 1x |
| 原始SSDiff | DDIM-10 | ~5-8 | 5x |
| 蒸馏SSDiff | 1步 | ~50-100 | 50x |

### 质量指标

蒸馏后的模型应该保持原始模型 95%+ 的质量：

- **PSNR**: 下降 < 0.5 dB
- **SSIM**: 下降 < 0.02
- **视觉质量**: 接近原始多步采样

## 🔧 常见问题

### Q1: 导入错误 "ModuleNotFoundError: No module named 'improved_diffusion'"

**解决方案**: 确保SSDiff路径已添加到Python路径：

```python
import sys
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')
```

### Q2: CUDA内存不足

**解决方案**:
1. 减小batch size: `--train_batch_size 2`
2. 使用gradient accumulation: `--gradient_accumulation_steps 2`
3. 启用gradient checkpointing: `--gradient_checkpointing`

### Q3: 训练损失不下降

**解决方案**:
1. 检查数据归一化是否正确（SSDiff使用[0, 2047]范围）
2. 降低学习率
3. 增加warmup步数
4. 只用L2 loss训练几千步后再加入其他损失

### Q4: 蒸馏后质量下降严重

**可能原因**:
1. 训练步数不够（至少需要10k-50k步）
2. 损失权重不合适（尝试调整lambda参数）
3. LoRA rank太小（尝试增加到8或16）

**解决方案**:
```bash
# 增加训练步数和LoRA rank
--max_train_steps 100000 \
--lora_rank 8 \
--lambda_vsd 2.0
```

## 📈 进阶配置

### 调整损失权重

根据您的需求调整损失权重：

```bash
# 更注重重建质量
--lambda_l2 2.0 --lambda_vsd 0.5 --lambda_vsd_lora 0.5

# 更注重分布匹配
--lambda_l2 0.5 --lambda_vsd 2.0 --lambda_vsd_lora 1.0

# 平衡配置（推荐）
--lambda_l2 1.0 --lambda_vsd 1.0 --lambda_vsd_lora 1.0
```

### 调整LoRA配置

```bash
# 较小的LoRA (更快，更少参数)
--lora_rank 2

# 标准LoRA (推荐)
--lora_rank 4

# 较大的LoRA (更高质量，但更慢)
--lora_rank 8
```

### 多分辨率测试

```bash
# Reduced-Resolution (256x256)
--test_dataset test_wv3_multiExm1.h5

# Full-Resolution (512x512)
--test_dataset test_wv3_OrigScale_multiExm1.h5
```

## 🎯 下一步

1. **评估质量**: 使用 `test_wv3_metrics.py` 计算PSNR/SSIM等指标
2. **可视化结果**: 使用MATLAB或Python查看 `.mat` 文件中的结果
3. **优化模型**: 尝试不同的超参数组合
4. **部署应用**: 将蒸馏后的模型集成到您的应用中

## 📚 相关资源

- [详细技术指南](SSDIFF_DISTILLATION_GUIDE.md)
- [OSEDiff论文](https://arxiv.org/pdf/2406.08177)
- [WV3测试指南](WV3_TESTING_GUIDE.md)

## 💬 获取帮助

如果遇到问题：

1. 查看详细的技术指南: `SSDIFF_DISTILLATION_GUIDE.md`
2. 检查日志文件: `experiments/ssdiff_distill/logs/`
3. 查看TensorBoard: `tensorboard --logdir experiments/ssdiff_distill/logs`

## 🎉 完成！

恭喜！您已经成功完成了SSDiff的一步蒸馏训练和测试。

如果一切顺利，您应该得到：
- ✅ 速度提升 10-50倍
- ✅ 质量保持 95%+
- ✅ 模型大小减小（只包含LoRA权重）

祝您使用愉快！🚀

