# Loss调试指南

## 🔍 Loss值分析

### 当前情况
- **观察到的Loss**: 31.8 - 32.4
- **训练模式**: Mixed (L1 + VSD)
- **条件模块**: 全部关闭 (VAE/ControlNet/CLIP/RAM/ARConv = False)

### Loss值判断标准

#### 1. **数据范围在 [0, 1]**
- ✅ **合理Loss范围**:
  - L1 Loss: 0.001 - 0.1
  - VSD Loss: 0.01 - 1.0
  - Mixed Loss: 0.05 - 2.0
  
- ❌ **异常Loss (>10)**: 可能原因
  - 数据未归一化
  - 学习率过大
  - 模型初始化问题

#### 2. **数据范围在 [0, 2047]** (未归一化)
- Loss会放大2000+倍
- 31.8的loss ÷ 2047 ≈ 0.015 (实际上是合理的！)

## 📊 数据范围检查结果

运行训练后，会在第一个batch输出：
```
======================================================================
📊 数据范围检查
======================================================================
PAN   - min: X.XXXX, max: X.XXXX, mean: X.XXXX
LMS   - min: X.XXXX, max: X.XXXX, mean: X.XXXX
MS    - min: X.XXXX, max: X.XXXX, mean: X.XXXX
GT    - min: X.XXXX, max: X.XXXX, mean: X.XXXX
GT-LMS - min: X.XXXX, max: X.XXXX, mean: X.XXXX, std: X.XXXX
======================================================================
```

### 如何判断：
1. **如果数据在 [0, 1]**: Loss=31是异常的
2. **如果数据在 [0, 2047]**: Loss=31是**正常的**！

## 🎯 解决方案

### 情况A: 数据在[0, 1]，Loss仍然>10

#### 方案1: 降低学习率
```bash
LEARNING_RATE=1e-5  # 从5e-5降到1e-5
```

#### 方案2: 先用L1模式训练
```bash
LOSS_MODE="l1"      # 从mixed改为l1
LAMBDA_L1=1.0
LAMBDA_VSD=0.0      # 先关闭VSD
```

#### 方案3: 启用条件模块（提供更多信息）
```bash
USE_VAE="True"      # 启用VAE提供全局场景特征
TRAIN_VAE="True"
```

### 情况B: 数据在[0, 2047]，Loss=31正常

**这是正常现象！** 因为：
```
实际误差 = 31.8 / 2047 ≈ 0.0155
```

换算到图像空间：
- 每个像素平均误差约 31.8 (0-2047范围)
- 相对误差约 1.5%
- 这在遥感图像中是**合理的**！

#### 如何验证训练是否正常：
1. **看Loss是否下降**：持续训练，Loss应该逐步降低
2. **看相对变化**：不是看绝对值，而是看相对下降趋势
3. **定期验证**：每500步保存checkpoint并验证PSNR/SAM

## 📈 训练监控建议

### 正常训练的特征：
```
Step 0:     loss=32.0
Step 500:   loss=28.5  ✅ 下降了
Step 1000:  loss=25.2  ✅ 持续下降
Step 2000:  loss=20.1  ✅ 趋势正确
```

### 异常训练的特征：
```
Step 0:     loss=32.0
Step 500:   loss=45.2  ❌ 上升了（梯度爆炸）
Step 1000:  loss=nan   ❌ 出现NaN
```

或
```
Step 0:     loss=32.0
Step 500:   loss=32.0  ❌ 完全不变（学习率太小/梯度消失）
Step 1000:  loss=32.0
```

## 🔧 快速诊断命令

重新运行训练，观察第一个batch的输出：
```bash
bash train_ssdiff_unified.sh
```

在输出中找到：
```
📊 数据范围检查
```

根据显示的数据范围判断：
- **[0, 1]**: 使用 `train_ssdiff_unified_optimized.sh` (降低学习率)
- **[0, 2047]**: 当前loss正常，继续训练即可

## 💡 推荐策略

### 阶段1: Warm-up (0-5000步)
```bash
LOSS_MODE="l1"
LEARNING_RATE=1e-5
USE_VAE="False"
```

### 阶段2: 添加VAE (5000-20000步)
```bash
LOSS_MODE="l1"
LEARNING_RATE=5e-6
USE_VAE="True"
TRAIN_VAE="True"
```

### 阶段3: 混合训练 (20000+步)
```bash
LOSS_MODE="mixed"
LAMBDA_L1=1.0
LAMBDA_VSD=0.5
USE_VAE="True"
```

## 📝 总结

**最可能的情况**: 你的数据在[0, 2047]范围，Loss=31实际上是**正常的**！

**验证方法**: 重新运行训练，查看数据范围检查的输出。

**如果Loss确实异常**: 使用 `train_ssdiff_unified_optimized.sh` 脚本。



