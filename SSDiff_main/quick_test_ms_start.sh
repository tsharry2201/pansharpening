#!/bin/bash
# 快速测试MS起点改进的效果
# 使用方法: bash scripts/quick_test_ms_start.sh

echo "========================================"
echo "🧪 快速测试：MS图像作为扩散起点"
echo "========================================"

# ============ 配置参数 ============
# 蒸馏模型路径（如果已有）
DISTILLED_MODEL="experiments/ssdiff_distill/checkpoints/model_10000.pkl"

# 原始预训练SSDiff
PRETRAINED_SSDIFF="/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/results/10-15-22-29/model065000.pt"

# 测试配置
TEST_DATASET="test_wv3_multiExm1.h5"
NUM_IMAGES=5  # 快速测试，只用5张图
DEVICE="cuda:0"
OUTPUT_DIR="test_results/ms_start_comparison"

# ============ 检查文件 ============
echo ""
echo "📋 检查文件..."

if [ ! -f "$PRETRAINED_SSDIFF" ]; then
    echo "❌ 未找到预训练SSDiff: $PRETRAINED_SSDIFF"
    exit 1
fi

if [ ! -f "$DISTILLED_MODEL" ]; then
    echo "⚠️  未找到蒸馏模型: $DISTILLED_MODEL"
    echo "💡 提示：如果还没训练，可以先测试原始SSDiff"
    echo "或者先运行训练脚本："
    echo "   bash scripts/train_ssdiff_distill.sh"
    exit 1
fi

echo "✅ 所有文件检查通过"

# ============ 测试蒸馏模型（MS起点） ============
echo ""
echo "========================================"
echo "1️⃣  测试改进后的蒸馏模型（MS起点）"
echo "========================================"

python test_ssdiff_unified.py \
    --model_path "$DISTILLED_MODEL" \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR/ms_start" \
    --mixed_precision fp16

RESULT1=$?

# ============ 测试原始SSDiff（对比基准） ============
echo ""
echo "========================================"
echo "2️⃣  测试原始SSDiff（DDIM-10，基准对比）"
echo "========================================"

python test_ssdiff_unified.py \
    --model_path "$PRETRAINED_SSDIFF" \
    --use_distillation False \
    --timestep_respacing ddim10 \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR/original_ddim10" \
    --mixed_precision fp16

RESULT2=$?

# ============ 结果总结 ============
echo ""
echo "========================================"
echo "✅ 测试完成！"
echo "========================================"

if [ $RESULT1 -eq 0 ] && [ $RESULT2 -eq 0 ]; then
    echo "✅ 两个测试都成功完成"
    echo ""
    echo "📊 结果保存在: $OUTPUT_DIR/"
    echo "   - ms_start/         (改进版：MS起点)"
    echo "   - original_ddim10/  (基准：原始10步)"
    echo ""
    echo "📈 下一步："
    echo "   1. 查看.mat文件对比结果"
    echo "   2. 使用test_wv3_metrics.py计算指标"
    echo "   3. 如果效果好，考虑重新训练模型"
    echo ""
    echo "💡 计算指标示例："
    echo "   python test_wv3_metrics.py \\"
    echo "       --result_path $OUTPUT_DIR/ms_start/*.mat"
else
    echo "❌ 部分测试失败"
    if [ $RESULT1 -ne 0 ]; then
        echo "   - 蒸馏模型测试失败"
    fi
    if [ $RESULT2 -ne 0 ]; then
        echo "   - 原始模型测试失败"
    fi
    echo ""
    echo "💡 请检查错误信息并重试"
fi

echo "========================================"

