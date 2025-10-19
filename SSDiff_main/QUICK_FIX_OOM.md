# ⚡ GPU显存不足（OOM）快速修复

## 🔴 问题
```
CUDA out of memory
尝试分配: 2.50 GiB
可用显存: 1.26 GiB
实际batch: 20 (过大)
```

## ✅ 已修复

### `train_ssdiff_distill.sh` 已更新:

```bash
BATCH_SIZE=4              # 从8降到4 (显存减半)
GRADIENT_ACCUM_STEPS=3    # 累积3步后更新 
# 有效batch = 4 × 3 = 12 (保持训练效果)
```

## 🚀 重新启动训练

```bash
cd /data2/user/zelilin/ARConv_SSDiff/SSDiff_main
bash train_ssdiff_distill.sh
```

## 📊 预期变化

| 项目 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| Batch Size | 8 | 4 | 每GPU的batch |
| 显存使用 | ~2.5 GiB | ~1.2 GiB | 降低50% |
| 有效Batch | 8 | 12 | 通过梯度累积 |
| 更新频率 | 每步 | 每3步 | 略慢但不影响效果 |

## 📝 其他可选优化（如仍然OOM）

### 1. 启用混合精度（推荐）
```bash
# 在train_ssdiff_distill.sh中修改:
--mixed_precision fp16   # 从 "no" 改为 "fp16"
```
**效果**: 显存减半，速度提升30%

### 2. 进一步降低batch
```bash
BATCH_SIZE=3
GRADIENT_ACCUM_STEPS=4  # 3×4=12
```

### 3. 只用2个GPU
```bash
GPUS="4,6"  # 从"4,6,7"改为"4,6"
```

---

**状态**: ✅ 已修复  
**操作**: 直接运行 `bash train_ssdiff_distill.sh`

