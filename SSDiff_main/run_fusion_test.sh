#!/bin/bash

# OTPNet-SSDiff融合模型测试脚本示例
# 请根据实际路径修改以下变量

# ============== 配置路径 ==============
OTPNET_CKPT="/data2/user/zelilin/OTPNet/checkpoints/best.pth"
SSDIFF_CKPT="./experiments/ssdiff_distill_20251103_010245/checkpoints/model_12000.pkl"
PRETRAINED_SSDIFF="/data2/user/zelilin/pansharpening/SSDiff_main/results/10-18-13-11/model120000.pt"
FUSION_WEIGHTS="checkpoints/fusion/run_1109_2338/fusion_epoch40.pth"

TEST_DATASET="test_wv3_multiExm1.h5"
NUM_TEST_IMAGES=20

# ============== 融合网络配置（需与训练时一致）==============
FUSION_CHANNELS=64
NUM_FUSION_BLOCKS=3

# ============== 其他设置 ==============
DEVICE="cuda:4"
OUTPUT_DIR="test_results"

echo "========================================"
echo "OTPNet-SSDiff 融合模型测试"
echo "========================================"
echo "OTPNet: ${OTPNET_CKPT}"
echo "SSDiff: ${SSDIFF_CKPT}"
echo "融合权重: ${FUSION_WEIGHTS}"
echo "测试集: ${TEST_DATASET}"
echo "========================================"

# 测试1: 普通推理
echo ""
echo "测试1: 普通推理（不使用self-ensemble）"
python test_fusion.py \
    --otpnet_checkpoint ${OTPNET_CKPT} \
    --ssdiff_checkpoint ${SSDIFF_CKPT} \
    --pretrained_ssdiff_path ${PRETRAINED_SSDIFF} \
    --fusion_weights ${FUSION_WEIGHTS} \
    --test_dataset ${TEST_DATASET} \
    --num_test_images ${NUM_TEST_IMAGES} \
    --fusion_channels ${FUSION_CHANNELS} \
    --num_fusion_blocks ${NUM_FUSION_BLOCKS} \
    --device ${DEVICE} \
    --output_dir ${OUTPUT_DIR} \
    --save_individual_outputs


