#!/bin/bash
# 比较原始SSDiff和蒸馏SSDiff的速度
# 使用方法: bash scripts/compare_speed.sh

echo "========================================"
echo "⚡ Speed Comparison: Original vs Distilled"
echo "========================================"

# 原始模型路径
ORIGINAL_MODEL="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-15-22-29/model065000.pt"
DISTILLED_MODEL="experiments/ssdiff_distill/checkpoints/model_10000.pkl"

TEST_DATASET="test_wv3_multiExm1.h5"
NUM_IMAGES=20
DEVICE="cuda:0"

echo ""
echo "1️⃣  Testing Original SSDiff (DDIM-10 steps)..."
echo "----------------------------------------"
python test_ssdiff_unified.py \
    --model_path "$ORIGINAL_MODEL" \
    --use_distillation False \
    --timestep_respacing ddim10 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "test_results/comparison/original_ddim10"

echo ""
echo "2️⃣  Testing Distilled SSDiff (1 step)..."
echo "----------------------------------------"
python test_ssdiff_unified.py \
    --model_path "$DISTILLED_MODEL" \
    --pretrained_ssdiff_path "$ORIGINAL_MODEL" \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "test_results/comparison/distilled_1step"

echo ""
echo "========================================"
echo "✅ Comparison completed!"
echo "📊 Check test_results/comparison/ for results"
echo "========================================"

