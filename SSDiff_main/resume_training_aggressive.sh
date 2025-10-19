#!/bin/bash
# 激进的优化方案：更低学习率 + 更小VSD权重
# 使用方法: bash resume_training_aggressive.sh

# ============ 配置参数 ============
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"
OUTPUT_DIR="experiments/ssdiff_distill_20251017_232238"
RESUME_CHECKPOINT="./experiments/ssdiff_distill_20251017_232238/checkpoints/model_45001.pkl"
RESUME_STEP=45001
GPUS="4,6,7"

# ============ 激进优化参数 ============
BATCH_SIZE=12
LEARNING_RATE=1e-6        # 🔥 更低的学习率
MAX_STEPS=80000
CHECKPOINT_STEPS=500
LORA_RANK=4

# 🔥 更激进的损失权重调整
LAMBDA_L2=2.0             # 增加L2权重，强化像素级监督
LAMBDA_VSD=0.1            # 大幅降低VSD权重
LAMBDA_VSD_LORA=0.5       # 降低Lora损失权重

echo "========================================"
echo "🚀 Aggressive Optimization Training"
echo "========================================"
echo "⚠️  使用激进参数:"
echo "   - Learning Rate: $LEARNING_RATE (更低)"
echo "   - VSD Weight: $LAMBDA_VSD (更小)"
echo "   - L2 Weight: $LAMBDA_L2 (更大)"
echo "========================================"

if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found"
    exit 1
fi

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
    --lr_scheduler cosine \
    --lr_warmup_steps 50 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill_aggressive" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --resume_step $RESUME_STEP

echo "✅ Aggressive training completed!"

