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
dataset_name=${args[2]}
model_name_or_path=${args[3]}

if [ "$dataset_name" = "prm_r1" ]; then
    dataset=/userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt
    max_len=2048
elif [ "$dataset_name" = "prm_sft" ]; then
    dataset=/extrahome0/user/data/prm_sft/prm800k_train_eval.pt
    max_len=2048
elif [ "$dataset_name" = "gsm_dpo_if1" ]; then
    dataset=/extrahome0/user/data/gsm_sft.pt
    max_len=2048
else
    echo "Unknown dataset_name"
    dataset=""
fi

deepspeed --include localhost:${device} --master_port 10001 --module run_dpo \
   --save_path /extrahome0/user/output/ \
   --save_steps -1 \
   --logging_steps -1 \
   --eval_steps -1 \
   --train_batch_size 32 \
   --micro_train_batch_size 8 \
   --pretrain ${model_name_or_path} \
   --bf16 \
   --max_epochs 1 \
   --max_len ${max_len} \
   --zero_stage 1 \
   --learning_rate 1e-5 \
   --beta 0.5 \
   --dataset ${dataset} \
   --chosen_key chosen \
   --rejected_key rejected \
   --dataset_name ${dataset_name} \
   --method_name ifdr \
   --model_type qwen-math \
   --dataset_sample 0 \
   --gradient_checkpointing \
   --lora_rank 0 \
   --lora_alpha 32
