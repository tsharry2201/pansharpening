#!/bin/bash
# 方案C: 从头开始训练 - 使用修正的损失权重
# 适用场景: 当前checkpoint学偏了，恢复训练也无法修正

# ============ 配置参数 ============
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"
DATA_DIR="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/dataset"
OUTPUT_DIR="experiments/ssdiff_distill_planC_$(date +%Y%m%d_%H%M%S)"
GPUS="4,6,7"

# ============ 从零开始的优化参数 ============
BATCH_SIZE=12
LEARNING_RATE=1e-4        # 更高的初始学习率（从头开始）
MAX_STEPS=30000           # 减少训练步数
CHECKPOINT_STEPS=500
LORA_RANK=4

# 修正的损失权重（一开始就使用正确的配比）
LAMBDA_L2=10.0            # 强调像素级监督
LAMBDA_VSD=0.1            # 弱化VSD损失
LAMBDA_VSD_LORA=0.5       # 适度的Lora损失

echo "========================================"
echo "🆕 方案C: 从头开始训练（修正版）"
echo "========================================"
echo "📦 输出目录: $OUTPUT_DIR"
echo ""
echo "📊 优化的损失权重配置:"
echo "   - L1权重: 10.0（强调像素重建）"
echo "   - VSD权重: 0.1（弱化分布匹配）"
echo "   - VSD Lora权重: 0.5"
echo ""
echo "🎯 训练策略:"
echo "   - 初始学习率: 1e-4（较高）"
echo "   - LR调度器: cosine"
echo "   - 最大步数: 30000（更快验证）"
echo "========================================"

# 检查预训练模型
if [ ! -f "$PRETRAINED_SSDIFF" ]; then
    echo "❌ Error: Pretrained model not found"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed 456 \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
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
    --tracker_project_name "ssdiff_distill_planC_scratch" \
    --dataloader_num_workers 0 \
    --ms_dim 8 \
    --pan_dim 1 \
    --image_size 64

echo "========================================"
echo "✅ 方案C训练完成！"
echo "📂 结果保存在: $OUTPUT_DIR"
echo "========================================"

