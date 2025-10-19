#!/bin/bash
# 从checkpoint恢复训练，使用优化的参数
# 使用方法: bash resume_training_optimized.sh

# ============ 配置参数 ============
# 预训练SSDiff模型路径
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"

# 数据集路径
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"

# 输出目录（复用原来的目录）
OUTPUT_DIR="experiments/ssdiff_distill_20251017_232238"

# 恢复的checkpoint
RESUME_CHECKPOINT="./experiments/ssdiff_distill_20251017_232238/checkpoints/model_45001.pkl"
RESUME_STEP=45001

# GPU设置
GPUS="4,6,7"

# ============ 优化的训练参数 ============
BATCH_SIZE=12
LEARNING_RATE=5e-6        # 🔥 降低学习率（从2e-5到5e-6）
MAX_STEPS=80000           # 继续训练到80000步
CHECKPOINT_STEPS=500
LORA_RANK=4

# 🔥 调整损失权重
LAMBDA_L2=1.0
LAMBDA_VSD=0.3            # 🔥 大幅降低VSD权重（从1.0到0.3）
LAMBDA_VSD_LORA=1.0

# ============ 开始训练 ============
echo "========================================"
echo "🔄 Resuming SSDiff Distillation Training"
echo "========================================"
echo "📦 Pretrained Model: $PRETRAINED_SSDIFF"
echo "📁 Data Directory: $DATA_DIR"
echo "📂 Output Directory: $OUTPUT_DIR"
echo "🔄 Resume from: $RESUME_CHECKPOINT (step $RESUME_STEP)"
echo "🎮 GPUs: $GPUS"
echo "📊 New Learning Rate: $LEARNING_RATE"
echo "📊 New VSD Weight: $LAMBDA_VSD"
echo "========================================"

# 检查checkpoint是否存在
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $RESUME_CHECKPOINT"
    exit 1
fi

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
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill_resume" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --resume_step $RESUME_STEP

echo "========================================"
echo "✅ Training completed!"
echo "📂 Results saved to: $OUTPUT_DIR"
echo "========================================"

