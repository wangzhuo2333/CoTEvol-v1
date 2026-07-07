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
model_type=${args[1]}

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

deepspeed --include localhost:${device} --master_port 10001 --module run_dpo \
   --save_path /extrahome0/user/output/ \
   --save_steps 100 \
   --logging_steps 100 \
   --eval_steps -1 \
   --train_batch_size 16 \
   --micro_train_batch_size 1 \
   --pretrain ${model_name_or_path} \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --beta 0.5 \
   --dataset /userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt \
   --chosen_key chosen \
   --rejected_key rejected \
   --dataset_name prm_r1 \
   --method_name open_dpo \
   --model_type ${model_type} \
   --dataset_sample 0 \
   --gradient_checkpointing \
   --lora_rank 0 \
   --lora_alpha 32
