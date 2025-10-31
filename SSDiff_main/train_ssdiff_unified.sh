#!/bin/bash

# SSDiff统一训练脚本
# 结合VAE/ControlNet/CLIP/RAM和知识蒸馏

# 基础配置
PRETRAINED_SSDIFF="/home/zelilin/data/pansharpening/SSDiff_main/results/10-15-17-45/model060000.pt"
LORA_CHECKPOINT_PATH=""  # 可选，留空则从头训练LoRA
DATA_DIR="./dataset"
OUTPUT_DIR="./experiments/ssdiff_unified_$(date +%Y%m%d_%H%M%S)"

# 训练模式选择
# l1: 仅直接监督
# vsd: 仅知识蒸馏
# mixed: 混合训练（推荐）
LOSS_MODE="mixed"
LAMBDA_L1=1.0
LAMBDA_VSD=0.5

# 训练参数
BATCH_SIZE=4
LEARNING_RATE=5e-5
MAX_STEPS=120000
CHECKPOINT_STEPS=500

# LoRA配置
LORA_RANK=4
TRAIN_LORA="true"

# VAE配置
USE_VAE="False"
VAE_LATENT_DIM=256
TRAIN_VAE="False"
VAE_LR=1e-5
USE_KL_LOSS="false"
LAMBDA_KL=0.001
USE_PERCEPTUAL_LOSS="false"
LAMBDA_PERCEPTUAL=0.1

# ControlNet配置
USE_CONTROLNET="False"
MS_CHANNELS=8
TRAIN_CONTROLNET="False"
CONTROLNET_LR=2.5e-4  # ControlNet使用更大的学习率

# CLIP配置
USE_CLIP="False"
CLIP_MODEL_PATH="./model/SD21Base"
PROMPT="A high resolution satellite image with clear details"
TRAIN_CLIP="false"  # CLIP通常保持冻结

# RAM配置
USE_RAM="False"
RAM_MODEL_PATH="./ram/pretrained/ram_swin_large_14m.pth"

# ARConv渐进式训练配置
USE_ARCONV="False"
ARCONV_HW_RANGE="[1,9]"
ARCONV_WARMUP_STEPS=1000   # 前1000步使用Conv2d
ARCONV_ACTIVATE_STEPS=3000 # 3000步后开始训练ARConv
ARCONV_FIXSTEP=4000        # 4000步后固定ARConv配置

# 扩散模型参数
PREDICT_XSTART="false"
USE_DDIM="false"
TIMESTEP_RESPACING="1000"

# 其他参数
MIXED_PRECISION="fp16"
SEED=123

# GPU设置
GPUS="0,1,2"  # 使用的GPU编号，用逗号分隔
NUM_PROCESSES=3  # GPU数量，需要与GPUS中的数量一致

echo "=========================================="
echo "SSDiff统一训练 - $LOSS_MODE 模式"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "训练模式: $LOSS_MODE"
if [ "$LOSS_MODE" = "mixed" ]; then
    echo "  L1权重: $LAMBDA_L1"
    echo "  VSD权重: $LAMBDA_VSD"
fi
echo ""
echo "模块配置:"
echo "  VAE: $USE_VAE"
echo "  ControlNet: $USE_CONTROLNET (学习率: $CONTROLNET_LR)"
echo "  CLIP: $USE_CLIP"
echo "  RAM: $USE_RAM"
echo "  ARConv: $USE_ARCONV"
echo ""
echo "训练参数:"
echo "  批大小: $BATCH_SIZE (每个GPU)"
echo "  总批大小: $((BATCH_SIZE * NUM_PROCESSES))"
echo "  学习率: $LEARNING_RATE"
echo "  最大步数: $MAX_STEPS"
echo ""
echo "GPU配置:"
echo "  使用GPU: $GPUS"
echo "  进程数: $NUM_PROCESSES"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 保存配置到文件
cat > "$OUTPUT_DIR/config.txt" << EOF
训练配置
========================================
模式: $LOSS_MODE
L1权重: $LAMBDA_L1
VSD权重: $LAMBDA_VSD
VAE: $USE_VAE
ControlNet: $USE_CONTROLNET
CLIP: $USE_CLIP
RAM: $USE_RAM
ARConv: $USE_ARCONV
批大小: $BATCH_SIZE
学习率: $LEARNING_RATE
最大步数: $MAX_STEPS
========================================
EOF

# 运行多GPU并行训练
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    --mixed_precision $MIXED_PRECISION \
    --main_process_port 29500 \
    train_ssdiff_unified.py \
    --pretrained_ssdiff_path "$PRETRAINED_SSDIFF" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --loss_mode "$LOSS_MODE" \
    --lambda_l1 $LAMBDA_L1 \
    --lambda_vsd $LAMBDA_VSD \
    --seed $SEED \
    --train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_STEPS \
    --checkpointing_steps $CHECKPOINT_STEPS \
    --lora_rank $LORA_RANK \
    --train_lora "$TRAIN_LORA" \
    --use_vae "$USE_VAE" \
    --vae_latent_dim $VAE_LATENT_DIM \
    --train_vae "$TRAIN_VAE" \
    --vae_lr $VAE_LR \
    --use_kl_loss "$USE_KL_LOSS" \
    --lambda_kl $LAMBDA_KL \
    --use_perceptual_loss "$USE_PERCEPTUAL_LOSS" \
    --lambda_perceptual $LAMBDA_PERCEPTUAL \
    --use_controlnet "$USE_CONTROLNET" \
    --ms_channels $MS_CHANNELS \
    --train_controlnet "$TRAIN_CONTROLNET" \
    --controlnet_lr $CONTROLNET_LR \
    --use_clip "$USE_CLIP" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --prompt "$PROMPT" \
    --train_clip "$TRAIN_CLIP" \
    --use_ram "$USE_RAM" \
    --ram_model_path "$RAM_MODEL_PATH" \
    --use_arconv "$USE_ARCONV" \
    --arconv_hw_range "$ARCONV_HW_RANGE" \
    --arconv_warmup_steps $ARCONV_WARMUP_STEPS \
    --arconv_activate_steps $ARCONV_ACTIVATE_STEPS \
    --arconv_fixstep $ARCONV_FIXSTEP \
    --predict_xstart "$PREDICT_XSTART" \
    --use_ddim "$USE_DDIM" \
    --timestep_respacing "$TIMESTEP_RESPACING" \
    --lr_scheduler "constant" \
    --lr_warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --set_grads_to_none

echo "=========================================="
echo "训练完成!"
echo "模型保存在: $OUTPUT_DIR/checkpoints"
echo "=========================================="

