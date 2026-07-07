#!/usr/bin/env bash
# run_sft.sh: 脚本示例，用于一键启动 SFT 微调任务

### 配置区域 ###
# 使用的 GPU 设备
export CUDA_VISIBLE_DEVICES="0, 1"
# 如需允许超长上下文（可选）
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 模型及数据集路径
MODEL_PATH="/inspire/hdd/global_user/USER_ID/user/models/Qwen2.5-7B-Instruct"
DATASET_PATH="/inspire/hdd/WORKSPACE_ID/public-project/USER_ID/ifdr-main-v2/ifdr-main/evol/data/s1k_train.pt"

# 保存、日志及 checkpoint 路径
SAVE_DIR="./checkpoints"
CKPT_DIR="${SAVE_DIR}/ckpt"

# 训练超参数
TRAIN_BATCH_SIZE=1
MICRO_BATCH_SIZE=1
LEARNING_RATE=5e-6
MAX_EPOCHS=3
LR_WARMUP_RATIO=0.03

# LoRA 与量化配置
LORA_RANK=0
LOAD_IN_4BIT="--load_in_4bit"
FLASH_ATTN="--flash_attn"

# WandB 配置（如不使用，可注释掉）
USE_WANDB="--use_wandb"
WANDB_PROJECT=./wandb

### 执行命令 ###
python run_sft.py \
  --pretrain ${MODEL_PATH} \
  --model_type qwen_math_it \
  --dataset ${DATASET_PATH} \
  --input_key prompt \
  --output_key response \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --micro_train_batch_size $MICRO_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --lr_warmup_ratio $LR_WARMUP_RATIO \
  --max_epochs $MAX_EPOCHS \
  --save_path ${SAVE_DIR} \
  --ckpt_path ${CKPT_DIR} \
  --lora_rank $LORA_RANK \
  --bf16 \
  --dataset_name s1k_evol \
  $FLASH_ATTN 
  # $USE_WANDB $WANDB_PROJECT

# End of script
