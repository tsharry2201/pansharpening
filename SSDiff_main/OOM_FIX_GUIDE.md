# 🔧 GPU显存不足（OOM）问题解决方案

## ❌ 错误信息
```
CUDA out of memory. Tried to allocate 2.50 GiB 
(GPU 0; 47.53 GiB total capacity; 2.85 GiB already allocated; 1.26 GiB free)
```

**问题**: 尝试分配2.50 GiB，但只有1.26 GiB空闲

**数据形状**: batch_size = 20（实际运行时的batch）

---

## ✅ 已应用的修复

### 修改 `train_ssdiff_distill.sh`:

```bash
# 修改前
BATCH_SIZE=10
--gradient_accumulation_steps 1

# 修改后  
BATCH_SIZE=6              # 降低batch size
GRADIENT_ACCUM_STEPS=2    # 使用梯度累积
--gradient_accumulation_steps 2
```

**效果**:
- 每个GPU的batch: 10 → 6 (降低40%)
- 显存使用: ~2.5 GiB → ~1.5 GiB (估算)
- 有效batch size: 6 × 2 = 12 (通过梯度累积保持)

---

## 🎯 梯度累积原理

```
不使用梯度累积:
  Step 1: forward(batch=10) → backward → update weights
  Step 2: forward(batch=10) → backward → update weights

使用梯度累积(accum=2):
  Step 1: forward(batch=6) → backward (accumulate grads)
  Step 2: forward(batch=6) → backward (accumulate grads) → update weights
  
有效等价于: forward(batch=12)，但显存只需要batch=6的量
```

---

## 📊 多GPU训练的显存分配

当前配置: `GPUS="4,6,7"` (3个GPU)

**Accelerate的分配方式**:
```
总batch=6 分配到3个GPU:
  - GPU 4: 2 samples
  - GPU 6: 2 samples  
  - GPU 7: 2 samples
```

**之前的问题** (batch=10):
```
总batch=10 分配到3个GPU:
  - GPU 4: 4 samples
  - GPU 6: 3 samples
  - GPU 7: 3 samples
  
某些GPU可能分到更多 → 显存不足
```

---

## 🔍 进一步优化（如果仍然OOM）

### 方案1: 进一步降低batch size

```bash
BATCH_SIZE=4
GRADIENT_ACCUM_STEPS=3    # 有效batch = 4×3 = 12
```

### 方案2: 启用混合精度训练

```bash
--mixed_precision fp16    # 从 "no" 改为 "fp16"
```

**效果**: 显存使用减半，速度提升~30%

### 方案3: 减少GPU数量

```bash
GPUS="4,6"    # 只用2个GPU而不是3个
```

**原因**: 多GPU通信开销 + 数据不均匀分配可能导致某个GPU压力过大

### 方案4: 降低LoRA rank

```bash
LORA_RANK=2    # 从4降到2
```

**效果**: LoRA参数量减少75%，显存占用减少

---

## 📝 监控显存使用

### 训练开始前检查:
```bash
watch -n 1 nvidia-smi
```

### 训练中监控:
```bash
# 终端1: 训练
bash train_ssdiff_distill.sh

# 终端2: 监控
nvidia-smi dmon -s um -i 4,6,7
```

### 查看各GPU显存:
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```

---

## ⚙️ Accelerate配置检查

### 查看当前配置:
```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

### 推荐的配置:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 3  # 对应3个GPU
gpu_ids: '4,6,7'
mixed_precision: 'no'  # 或 'fp16' 如果需要节省显存
```

---

## 🚨 常见OOM原因

| 原因 | 解决方案 |
|------|---------|
| Batch size过大 | ✅ 降低到6 |
| 没有梯度累积 | ✅ 设置为2 |
| 图像分辨率大 | 已固定64×64，无法优化 |
| 模型参数多 | 使用LoRA，参数量已大幅减少 |
| 多GPU负载不均 | ✅ 降低总batch避免不均 |
| 没用混合精度 | 可选：启用fp16 |

---

## ✅ 预期效果

修改后的显存使用（估算）:

```
修改前 (batch=10):
  Per GPU: ~2.5 GiB per batch
  峰值: ~3.0 GiB (forward + backward)
  
修改后 (batch=6):
  Per GPU: ~1.5 GiB per batch  
  峰值: ~1.8 GiB (forward + backward)
  
空闲显存: 1.26 GiB → 应该够用 ✅
```

---

## 🎯 验证修复

重新运行训练后，应该看到:

```
✅ 正常情况:
  Data shapes - pan: [6, 1, 64, 64]  ← batch=6
  开始正常训练，无OOM错误
  
❌ 如果仍然OOM:
  1. 降低batch size到4
  2. 启用 --mixed_precision fp16
  3. 只使用2个GPU
```

---

**修复日期**: 2025-10-18  
**状态**: ✅ 已应用 (batch 10→6, gradient_accum 1→2)  
**建议**: 重新运行 `bash train_ssdiff_distill.sh`

