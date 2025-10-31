#!/bin/bash

# SSDiff统一模型测试脚本
# 支持测试用unified训练的模型（可以直接使用VAE的测试脚本）

# 模型配置
MODEL_PATH="./experiments/ssdiff_unified/checkpoints/model_10000.pkl"
PRETRAINED_PATH="./ckpt/ssdiff.pt"

# 测试配置
TEST_DATASET="test_wv3_multiExm1.h5"
NUM_IMAGES=20
DEVICE="cuda:0"

# 模块配置（需要与训练时保持一致）
USE_VAE="true"
VAE_LATENT_DIM=256

USE_CONTROLNET="true"
MS_CHANNELS=8

USE_CLIP="true"
CLIP_MODEL_PATH="./model/SD21Base"

USE_RAM="true"
RAM_MODEL_PATH="./ram/pretrained/ram_swin_large_14m.pth"

# ARConv配置
USE_ARCONV="true"
ARCONV_HW_RANGE="[1,9]"

# 输出目录
OUTPUT_DIR="./test_results_unified"

echo "=========================================="
echo "SSDiff统一模型测试"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "VAE: $USE_VAE"
echo "ControlNet: $USE_CONTROLNET"
echo "CLIP: $USE_CLIP"
echo "RAM: $USE_RAM"
echo "ARConv: $USE_ARCONV"
echo "=========================================="

# 运行测试（使用 test_ssdiff_vae.py，因为权重格式兼容）
python3 test_ssdiff_vae.py \
    --model_path "$MODEL_PATH" \
    --pretrained_ssdiff_path "$PRETRAINED_PATH" \
    --use_distillation true \
    --test_dataset "$TEST_DATASET" \
    --num_test_images $NUM_IMAGES \
    --device "$DEVICE" \
    --use_vae "$USE_VAE" \
    --vae_latent_dim $VAE_LATENT_DIM \
    --use_controlnet "$USE_CONTROLNET" \
    --ms_channels $MS_CHANNELS \
    --use_clip "$USE_CLIP" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --use_ram "$USE_RAM" \
    --ram_model_path "$RAM_MODEL_PATH" \
    --use_arconv "$USE_ARCONV" \
    --arconv_hw_range "$ARCONV_HW_RANGE" \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision "fp16" \
    --crop_batch_size 8

echo "=========================================="
echo "测试完成!"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

