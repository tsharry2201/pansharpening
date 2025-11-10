#!/bin/bash
# 测试蒸馏后的SSDiff（单步采样）
# 使用方法: bash scripts/test_ssdiff_distilled.sh

# ============ 配置参数 ============
# 蒸馏后的模型checkpoint (使用最新的checkpoint)
DISTILLED_MODEL="./experiments/ssdiff_distill_20251103_010245/checkpoints/model_12000.pkl"
# 232238是改进后的蒸馏模型，但出现loss一直没降下来  第二次修改版是165605  第三次修改了加噪模式，但未使用 microbarch 221733
#1103_010245是SSDiff+ARConv蒸馏
# 原始预训练SSDiff（用于加载基础权重）
#PRETRAINED_SSDIFF="/home/zelilin/data/pansharpening/SSDiff_main/results/10-18-13-11/model120000.pt"
PRETRAINED_SSDIFF="/data2/user/zelilin/pansharpening/SSDiff_main/results/10-18-13-11/model120000.pt"
# 测试配置
TEST_DATASET="test_wv3_multiExm1.h5"  # 或 test_wv3_OrigScale_multiExm1.h5
NUM_IMAGES=20
DEVICE="cuda:4"
OUTPUT_DIR="test_results/distilled"

#ensemble推理（8种几何变换）效果不好

#*** 注意两个版本的skip_c0不一样

#  实验配置：是否使用带噪声的 x_t（更接近训练分布）
# True: 使用 q_sample_xt(0, t=999, noise) 生成带噪声的 x_t
# False (或留空): 使用 x_t=0（稳定、可复现）
USE_NOISE="False"  # 改为 "True" 来测试带噪版本

# ============ 开始测试 ============
echo "========================================"
echo "🟢 Testing Distilled SSDiff (One-step)"
echo "========================================"
echo "📦 Distilled Model: $DISTILLED_MODEL"
echo "📦 Base Model: $PRETRAINED_SSDIFF"
echo "🖼️  Dataset: $TEST_DATASET"
echo "🔬 Use Noise x_t: $USE_NOISE"
echo "========================================"

# 构建命令
CMD="python test_ssdiff_distilled.py \
    --model_path \"$DISTILLED_MODEL\" \
    --pretrained_ssdiff_path \"$PRETRAINED_SSDIFF\" \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --test_dataset \"$TEST_DATASET\" \
    --num_test_images $NUM_IMAGES \
    --device \"$DEVICE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --mixed_precision no \
    --crop_batch_size 1"

# 如果启用噪声，添加参数
if [ "$USE_NOISE" = "True" ]; then
    CMD="$CMD --test_with_noise True"
    echo "⚠️  注意：使用带噪声的 x_t（更接近训练分布，但有随机性）"
else
    echo "✓ 使用 x_t=0（稳定、可复现）"
fi

# 执行命令
eval $CMD

echo "========================================"
echo "✅ Testing completed!"
echo "========================================"
