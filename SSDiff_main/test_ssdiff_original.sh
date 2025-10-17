#!/bin/bash
# 测试原始SSDiff（多步采样）
# 使用方法: bash scripts/test_ssdiff_original.sh

# ============ 配置参数 ============
MODEL_PATH="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-15-22-29/model065000.pt"
TIMESTEP_RESPACING="ddim10"  # ddim1, ddim10, ddim50, ddim100等
TEST_DATASET="test_wv3_multiExm1.h5"  # 或 test_wv3_OrigScale_multiExm1.h5
NUM_IMAGES=20
DEVICE="cuda:0"
OUTPUT_DIR="test_results/original"

# ============ 开始测试 ============
echo "========================================"
echo "🔵 Testing Original SSDiff"
echo "========================================"
echo "📦 Model: $MODEL_PATH"
echo "🔢 Sampling: $TIMESTEP_RESPACING"
echo "🖼️  Dataset: $TEST_DATASET"
echo "========================================"

python test_ssdiff_unified.py \
    --model_path "$MODEL_PATH" \
    --use_distillation False \
    --timestep_respacing "$TIMESTEP_RESPACING" \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision fp16

echo "========================================"
echo "✅ Testing completed!"
echo "========================================"

