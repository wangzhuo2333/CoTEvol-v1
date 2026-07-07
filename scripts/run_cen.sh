#!/bin/bash
# shellcheck disable=SC2068
# read parameters
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

device=${args[0]}

# llama3.1-8B
# /extrahome0/download/Meta-Llama-3.1-8B-Base
# /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# llama2-7B
# /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf
deepspeed --include localhost:${device} --master_port 10001 run_cen.py \
--data_path /userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt \
--output_dir /extrahome0/user/output/ \
--dataset_name prm_r1 \
--model_name_or_path /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/ \
--learning_rate 1e-6 \
--max_prompt_length 512 \
--max_length 2048 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--eval_strategy no \
--logging_strategy steps \
--logging_steps 50 \
--save_strategy steps \
--save_steps 50 \
--no_remove_unused_columns \
--seed 42 \
--use_peft \
--lora_r 8 \
--lora_alpha 16 \
--load_in_8bit \
--bf16 \
--dataset_sample 0 \
--method_name dpo \
--model_type llama3.1-8b-it
