#!/bin/bash
# 方案B: 保守调整 - 适度调整损失权重
# 适用场景: 想逐步测试效果

# ============ 配置参数 ============
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"
OUTPUT_DIR="experiments/ssdiff_distill_20251017_232238"
RESUME_CHECKPOINT="./experiments/ssdiff_distill_20251017_232238/checkpoints/model_45001.pkl"
RESUME_STEP=45001
GPUS="4,6,7"

# ============ 保守调整参数 ============
BATCH_SIZE=12
LEARNING_RATE=5e-6        # 降低学习率
MAX_STEPS=80000
CHECKPOINT_STEPS=500
LORA_RANK=4

# 保守的损失权重调整
LAMBDA_L2=5.0             # 适度提高L1损失权重（从1.0到5.0）
LAMBDA_VSD=0.1            # 适度降低VSD损失权重（从1.0到0.1）
LAMBDA_VSD_LORA=0.8       # 小幅降低Lora损失权重

echo "========================================"
echo "🔧 方案B: 保守调整训练"
echo "========================================"
echo "📊 调整策略:"
echo "   - L1权重: 1.0 → 5.0 (提高5倍)"
echo "   - VSD权重: 1.0 → 0.1 (降低10倍)"
echo "   - 学习率: 2e-5 → 5e-6"
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
    --lr_warmup_steps 100 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill_planB" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --resume_step $RESUME_STEP

echo "✅ 方案B训练完成！"

