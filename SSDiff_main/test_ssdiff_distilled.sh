#!/bin/bash
# 测试蒸馏后的SSDiff（单步采样）
# 使用方法: bash scripts/test_ssdiff_distilled.sh

# ============ 配置参数 ============
# 蒸馏后的模型checkpoint
DISTILLED_MODEL="experiments/ssdiff_distill/checkpoints/model_10000.pkl"

# 原始预训练SSDiff（用于加载基础权重）
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-15-22-29/model065000.pt"

# 测试配置
TEST_DATASET="test_wv3_multiExm1.h5"  # 或 test_wv3_OrigScale_multiExm1.h5
NUM_IMAGES=20
DEVICE="cuda:0"
OUTPUT_DIR="test_results/distilled"

# ============ 开始测试 ============
echo "========================================"
echo "🟢 Testing Distilled SSDiff (One-step)"
echo "========================================"
echo "📦 Distilled Model: $DISTILLED_MODEL"
echo "📦 Base Model: $PRETRAINED_SSDIFF"
echo "🖼️  Dataset: $TEST_DATASET"
echo "========================================"

python test_ssdiff_unified.py \
    --model_path "$DISTILLED_MODEL" \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision fp16

echo "========================================"
echo "✅ Testing completed!"
echo "========================================"

