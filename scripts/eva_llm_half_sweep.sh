#!/bin/bash
# shellcheck disable=SC2068

data_name=$1
model_type=$2
GPU_ID=$3
checkpoint_file=$4

# llama3.1-8B
# /extrahome0/download/Meta-Llama-3.1-8B-Base
# /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# llama2-7B
# /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf

if [ "$model_type" = "llama3-base" ]; then
    model_name_or_path=/extrahome0/download/Meta-Llama-3.1-8B-Base
elif [ "$model_type" = "llama3-it" ]; then
    model_name_or_path=/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
elif [ "$model_type" = "qwen-math" ]; then
    model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-1.5B/
else
    echo "Unknown model_type"
    model_name_or_path=""
fi

CUDA_VISIBLE_DEVICES=${GPU_ID:0:1} python eva_llm_sweep.py \
--model_name_or_path ${model_name_or_path} \
--checkpoint_file ${checkpoint_file} \
--model_max_length 2048 \
--seed 42 \
--model_type ${model_type} \
--data_name ${data_name} \
--gpu_id 0 \
--pattern_name "*/checkpoint*" \
--sample_data 0 \
--half "right" \
--do_test \
--reuse &

CUDA_VISIBLE_DEVICES=${GPU_ID:1:1} python eva_llm_sweep.py \
--model_name_or_path ${model_name_or_path} \
--checkpoint_file ${checkpoint_file} \
--model_max_length 2048 \
--seed 42 \
--model_type ${model_type} \
--data_name ${data_name} \
--gpu_id 0 \
--pattern_name "*/checkpoint*" \
--sample_data 0 \
--half "left" \
--do_test \
--reuse &&

CUDA_VISIBLE_DEVICES=${GPU_ID:0:1} python eva_llm_sweep.py \
--model_name_or_path ${model_name_or_path} \
--checkpoint_file ${checkpoint_file} \
--model_max_length 2048 \
--seed 42 \
--model_type ${model_type} \
--data_name ${data_name} \
--gpu_id 0 \
--pattern_name "*/checkpoint*" \
--sample_data 0 \
--half "all" \
--do_test \
--reuse

#--do_zst
#--reuse
#--do_zst \
#--do_zst False \
#--reuse False
#--eval_data_path /extrahome0/user/data/gsm8k_test.jsonl \