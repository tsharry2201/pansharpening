#!/bin/bash

# OTPNet-SSDiff融合模型训练脚本示例
# 请根据实际路径修改以下变量

# ============== 配置路径 ==============
OTPNET_CKPT="/data2/user/zelilin/OTPNet/checkpoints/best.pth"
SSDIFF_CKPT="./experiments/ssdiff_distill_20251103_010245/checkpoints/model_12000.pkl"
PRETRAINED_SSDIFF="/data2/user/zelilin/pansharpening/SSDiff_main/results/10-18-13-11/model120000.pt"

TRAIN_DATASET="WV3/train_wv3.h5"
VAL_DATASET="WV3/valid_wv3.h5"

# ============== 训练参数 ==============
BATCH_SIZE=16
NUM_EPOCHS=100
LR=1e-4
FUSION_CHANNELS=64
NUM_FUSION_BLOCKS=3

# ============== 损失权重 ==============
L1_WEIGHT=1.0
L2_WEIGHT=0
CONSISTENCY_WEIGHT=0 # 0.1

# ============== 其他设置 ==============
DEVICE="cuda:4"
SAVE_DIR="checkpoints/fusion"

echo "========================================"
echo "OTPNet-SSDiff 融合模型训练"
echo "========================================"
echo "OTPNet: ${OTPNET_CKPT}"
echo "SSDiff: ${SSDIFF_CKPT}"
echo "训练集: ${TRAIN_DATASET}"
echo "验证集: ${VAL_DATASET}"
echo "========================================"

python train_fusion.py \
    --otpnet_checkpoint ${OTPNET_CKPT} \
    --ssdiff_checkpoint ${SSDIFF_CKPT} \
    --pretrained_ssdiff_path ${PRETRAINED_SSDIFF} \
    --train_dataset ${TRAIN_DATASET} \
    --val_dataset ${VAL_DATASET} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --fusion_channels ${FUSION_CHANNELS} \
    --num_fusion_blocks ${NUM_FUSION_BLOCKS} \
    --loss_l1_weight ${L1_WEIGHT} \
    --loss_l2_weight ${L2_WEIGHT} \
    --loss_consistency_weight ${CONSISTENCY_WEIGHT} \
    --device ${DEVICE} \
    --save_dir ${SAVE_DIR} \
    --save_interval 10 \
    --log_interval 100

echo ""
echo "========================================"
echo "训练完成！"
echo "模型保存在: ${SAVE_DIR}"
echo "========================================"

