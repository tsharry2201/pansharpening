#!/bin/bash
# 方案A: 激进修正 - 强化L1损失，大幅降低VSD损失
# 适用场景: 当前模型质量极差（SSIM=0.16），需要彻底改变训练策略

# ============ 配置参数 ============
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"
OUTPUT_DIR="experiments/ssdiff_distill_20251017_232238"
RESUME_CHECKPOINT="./experiments/ssdiff_distill_20251017_232238/checkpoints/model_45001.pkl"
RESUME_STEP=45001
GPUS="4,6,7"

# ============ 激进调整参数 ============
BATCH_SIZE=12
LEARNING_RATE=1e-5        # 🔥 降低学习率（从2e-5到1e-5）
MAX_STEPS=80000
CHECKPOINT_STEPS=500
LORA_RANK=4

# 🔥 重新平衡损失权重 - 核心修改！
LAMBDA_L2=10.0            # 🔥 大幅提高L1损失权重（从1.0提高到10.0）
LAMBDA_VSD=0.05           # 🔥 大幅降低VSD损失权重（从1.0降低到0.05）
LAMBDA_VSD_LORA=0.5       # 🔥 降低Lora损失权重（从1.0降低到0.5）

# ============ 开始训练 ============
echo "========================================"
echo "🔥 方案A: 激进修正训练"
echo "========================================"
echo "📊 当前问题诊断:"
echo "   - 测试SSIM: 0.16（应该>0.8）"
echo "   - VSD Loss占比: 96%（过高）"
echo "   - VSD Loss波动: 39.5%（极不稳定）"
echo ""
echo "📊 关键调整:"
echo "   - L1权重: 1.0 → 10.0 (提高10倍) ⬆️"
echo "   - VSD权重: 1.0 → 0.05 (降低20倍) ⬇️"
echo "   - VSD Lora权重: 1.0 → 0.5 ⬇️"
echo "   - 学习率: 2e-5 → 1e-5 ⬇️"
echo "   - LR调度器: constant → cosine"
echo ""
echo "🎯 预期效果:"
echo "   - SSIM提升到 0.4-0.6"
echo "   - L1 Loss占主导地位"
echo "   - 训练更稳定"
echo "========================================"

# 检查checkpoint
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
    --lr_warmup_steps 50 \
    --mixed_precision no \
    --lora_rank $LORA_RANK \
    --lambda_l2 $LAMBDA_L2 \
    --lambda_vsd $LAMBDA_VSD \
    --lambda_vsd_lora $LAMBDA_VSD_LORA \
    --report_to tensorboard \
    --tracker_project_name "ssdiff_distill_planA" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --resume_step $RESUME_STEP

echo "========================================"
echo "✅ 方案A训练完成！"
echo "📊 下一步: 测试model_46001.pkl并检查SSIM是否提升"
echo "========================================"

