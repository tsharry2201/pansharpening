#!/bin/bash
# SSDiff一步蒸馏训练脚本示例
# 使用方法: bash scripts/train_ssdiff_distill.sh

# ============ 配置参数 ============
# 🔧 请根据您的环境修改以下路径

# 预训练SSDiff模型路径
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"

# 数据集路径
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"

# 输出目录
OUTPUT_DIR="experiments/ssdiff_distill_$(date +%Y%m%d_%H%M%S)"

# GPU设置
GPUS="4"  # 使用的GPU编号

# ============ 训练参数 ============
BATCH_SIZE=4
LEARNING_RATE=5e-5
MAX_STEPS=60000
CHECKPOINT_STEPS=500
LORA_RANK=4

# 损失权重
LAMBDA_L2=1.0
LAMBDA_VSD=1.0
LAMBDA_VSD_LORA=1.0

# ============ 开始训练 ============
echo "========================================"
echo "🚀 Starting SSDiff Distillation Training"
echo "========================================"
echo "📦 Pretrained Model: $PRETRAINED_SSDIFF"
echo "📁 Data Directory: $DATA_DIR"
echo "📂 Output Directory: $OUTPUT_DIR"
echo "🎮 GPUs: $GPUS"
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
    --gradient_accumulation_steps 1 \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_STEPS \
    --checkpointing_steps $CHECKPOINT_STEPS \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64

echo "========================================"
echo "✅ Training completed!"
echo "📂 Results saved to: $OUTPUT_DIR"
echo "========================================"

