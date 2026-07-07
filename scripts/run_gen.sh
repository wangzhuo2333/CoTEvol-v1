#!/bin/bash
# shellcheck disable=SC2068

data_name=$1
model_type=$2
checkpoint_file=$3
GPU_ID=$4
seed=$5

# bash ./scripts/run_gen.sh math qwen-math xx xx xx
CUDA_VISIBLE_DEVICES=${GPU_ID} python run_gen.py \
--checkpoint_file ${checkpoint_file} \
--model_max_length 1024 \
--seed ${seed} \
--model_type ${model_type} \
--data_name ${data_name} \
--gpu_id 0 \
--sample_data 10