#!/bin/bash
# 快速测试指定checkpoint的脚本
# 使用方法: bash quick_test_checkpoint.sh <checkpoint_step>
# 例如: bash quick_test_checkpoint.sh 46001

if [ -z "$1" ]; then
    echo "❌ 请提供checkpoint步数"
    echo "用法: bash quick_test_checkpoint.sh <step_number>"
    echo "例如: bash quick_test_checkpoint.sh 46001"
    exit 1
fi

STEP=$1
CHECKPOINT="./experiments/ssdiff_distill_20251017_232238/checkpoints/model_${STEP}.pkl"
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-14-22-28/model065000.pt"
TEST_DATASET="test_wv3_multiExm1.h5"
NUM_IMAGES=20
DEVICE="cuda:4"
OUTPUT_DIR="test_results/checkpoint_${STEP}"

echo "========================================"
echo "🧪 快速测试 Checkpoint ${STEP}"
echo "========================================"
echo "📦 模型: $CHECKPOINT"
echo "📊 测试数据集: $TEST_DATASET"
echo "🖼️  测试图像数: $NUM_IMAGES"
echo "========================================"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "可用的checkpoints:"
    ls -lh ./experiments/ssdiff_distill_20251017_232238/checkpoints/ | tail -10
    exit 1
fi

# 运行测试
python test_ssdiff_unified.py \
    --model_path "$CHECKPOINT" \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision no

echo "========================================"
echo "✅ 测试完成！结果保存在: $OUTPUT_DIR"
echo "========================================"

