# 🎯 训练决策总结（一页版）

## 📊 当前状态

| 指标 | 值 | 状态 |
|------|-----|------|
| 训练步数 | 45501 | - |
| 测试SSIM | **0.16** | ❌ 应该>0.8 |
| 测试PSNR | 18.23 | ❌ 应该>25 |
| VSD Loss占比 | 96% | ⚠️ 过高 |
| VSD波动率 | 39.5% | ⚠️ 极不稳定 |

**结论**: 训练策略严重失败，VSD损失权重过高导致模型学偏了 🚨

---

## 💡 问题根源

1. **VSD损失权重=1.0 太高** → 占总损失96%
2. **L1损失权重=1.0 太低** → 像素级监督不足
3. **固定学习率=2e-5** → 无法精细调整

**核心问题**: 模型过度关注噪声分布匹配，忽略了图像重建质量

---

## 🔥 推荐方案：方案A（激进修正）

### 调整策略
```
L1权重:      1.0  → 10.0   (提高10倍) ⬆️
VSD权重:     1.0  → 0.05   (降低20倍) ⬇️
VSD Lora权重: 1.0  → 0.5    (降低2倍)  ⬇️
学习率:      2e-5 → 1e-5   (降低2倍)  ⬇️
调度器:      constant → cosine
```

### 立即执行
```bash
cd /data2/user/zelilin/ARConv_SSDiff/SSDiff_main
bash resume_training_plan_A.sh
```

### 预期效果
- ✅ SSIM 从 0.16 提升到 0.4-0.6
- ✅ L1 Loss 成为主导（占60%+）
- ✅ 训练更稳定（波动率<20%）

---

## 📋 执行清单

### 步骤1: 启动训练
```bash
bash resume_training_plan_A.sh
```

### 步骤2: 训练500步后测试
```bash
# 训练到46001步时测试
bash quick_test_checkpoint.sh 46001
```

### 步骤3: 检查改善
观察以下指标：
- [ ] SSIM > 0.3（目标 0.4+）
- [ ] L1 Loss 占比 > 50%
- [ ] VSD Loss波动率 < 25%

### 步骤4: 决定下一步
- ✅ **如果SSIM>0.4**: 继续训练到收敛
- ⚠️  **如果0.2<SSIM<0.4**: 尝试方案B（更保守）
- 🔴 **如果SSIM<0.2**: 使用方案C从头训练

---

## 🛠️ 可用脚本

| 脚本 | 用途 | 推荐度 |
|------|------|--------|
| `resume_training_plan_A.sh` | 激进修正（推荐） | ⭐⭐⭐⭐⭐ |
| `resume_training_plan_B.sh` | 保守调整 | ⭐⭐⭐ |
| `train_from_scratch_plan_C.sh` | 从头训练 | ⭐⭐⭐⭐ |
| `quick_test_checkpoint.sh` | 快速测试 | ⭐⭐⭐⭐⭐ |

---

## 🔍 修复的WandB记录

现在 `train_ssdiff_distill.py` 已更新，会记录：

✅ **新增记录**:
- `train/learning_rate` - 实际学习率（动态）
- `train/learning_rate_reg` - 正则化器学习率
- `train/lambda_l2` - L1损失权重
- `train/lambda_vsd` - VSD损失权重
- `train/lambda_vsd_lora` - VSD Lora损失权重

✅ **已有记录**:
- `train/loss_total`
- `train/loss_l2`
- `train/loss_vsd_teacher`
- `train/loss_vsd_reg`
- `train/loss_diff`

---

## 📈 监控要点

训练时重点观察：

1. **损失占比变化**
   ```
   之前: VSD(96%) + L1(4%)
   目标: L1(60-70%) + VSD(30-40%)
   ```

2. **L1 Loss绝对值**
   ```
   之前: 0.011
   可能上升到: 0.05-0.1（正常，因为权重提高了）
   ```

3. **VSD Teacher Loss**
   ```
   之前: 0.195 (波动39.5%)
   目标: 0.01-0.05 (波动<20%)
   ```

---

## ⚡ 快速命令

```bash
# 1. 查看完整分析
cat COMPREHENSIVE_ANALYSIS_AND_DECISION.md

# 2. 启动方案A训练（推荐）
bash resume_training_plan_A.sh

# 3. 查看训练日志
tail -f experiments/ssdiff_distill_20251017_232238/logs/ssdiff_distill/*.log

# 4. 测试最新checkpoint
bash quick_test_checkpoint.sh 46001

# 5. 对比多个checkpoints
bash quick_test_checkpoint.sh 15501  # 旧的
bash quick_test_checkpoint.sh 46001  # 新的
```

---

## 🎬 立即行动

**第一优先级**: 执行方案A
```bash
bash resume_training_plan_A.sh
```

**第二优先级**: 500步后评估
```bash
# 等训练500步后
bash quick_test_checkpoint.sh 46001
```

---

**最后更新**: 2025-10-18  
**状态**: 🔴 紧急 - 需要立即调整  
**预计改善时间**: 500-1000步后可见明显提升

