#!/bin/bash
# SSDiff一步蒸馏训练脚本 - 优化版
# 使用方法: bash train_ssdiff_distill.sh

# ============ 配置参数 ============
# 预训练SSDiff模型路径
PRETRAINED_SSDIFF="/home/zelilin/data/pansharpening/SSDiff_main/results/10-18-13-11/model120000.pt"

# 数据集路径
DATA_DIR="/home/zelilin/data/pansharpening/SSDiff_main/dataset"

# 输出目录
OUTPUT_DIR="experiments/ssdiff_distill_$(date +%Y%m%d_%H%M%S)"

# GPU设置
GPUS="3,4,5"  # 使用的GPU编号

# ============ 训练参数 ============
BATCH_SIZE=24
LEARNING_RATE=5e-5
MAX_STEPS=60000
CHECKPOINT_STEPS=500
LORA_RANK=4
GRADIENT_ACCUMULATION_STEPS=4

# 🔥 平衡的损失权重配置
LAMBDA_L2=2.0             # 强化L1重建损失
LAMBDA_VSD=0.5            # 适度降低VSD损失
LAMBDA_VSD_LORA=0.5       # 适度降低Lora损失

# ============ 开始训练 ============
echo "========================================"
echo "🚀 优化的SSDiff蒸馏训练"
echo "========================================"
echo "📦 预训练模型: $PRETRAINED_SSDIFF"
echo "📁 数据目录: $DATA_DIR"
echo "📂 输出目录: $OUTPUT_DIR"
echo "🎮 GPUs: $GPUS"
echo ""
echo "📊 优化的训练配置:"
echo "   - Batch Size: $BATCH_SIZE (per GPU)"
echo "   - 梯度累积: $GRADIENT_ACCUM_STEPS 步"
echo "   - 有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUM_STEPS))"
echo "   - 学习率: $LEARNING_RATE (cosine余弦衰减)"
echo "   - L1损失权重: $LAMBDA_L2"
echo "   - VSD损失权重: $LAMBDA_VSD"
echo "   - VSD Lora权重: $LAMBDA_VSD_LORA"
echo "   - Checkpoint间隔: 每${CHECKPOINT_STEPS}步"
echo "   - 学习率调度器: cosine (余弦衰减)"
echo "========================================"

# 检查预训练模型是否存在
if [ ! -f "$PRETRAINED_SSDIFF" ]; then
    echo "❌ Error: Pretrained model not found at $PRETRAINED_SSDIFF"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found at $DATA_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed 123 \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_STEPS \
    --checkpointing_steps $CHECKPOINT_STEPS \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill_optimized" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64

echo "========================================"
echo "✅ Training completed!"
echo "📂 Results saved to: $OUTPUT_DIR"
echo "========================================"

