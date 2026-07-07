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
dataset_name=${args[1]}
model_type=${args[2]}

if [ "$dataset_name" = "prm_r1" ]; then
    dataset=/userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt
    max_len=2048
elif [ "$dataset_name" = "prm_sft" ]; then
    dataset=/extrahome0/user/data/prm_sft/prm800k_train_eval.pt
    max_len=2048
elif [ "$dataset_name" = "gsm_sft" ]; then
    dataset=/extrahome0/user/data/gsm_sft.pt
    max_len=1024
elif [ "$dataset_name" = "math_sft" ]; then
    dataset=/userhome/Research_HUB/Reasoning/prm800k/prm800k/math_train_valid_test/MATH_train_valid_test500.pt
    max_len=1024
elif [ "$dataset_name" = "math_p20" ]; then
    dataset=/userhome/Research_HUB/Reasoning/prm800k/prm800k/math_train_valid_test/MATH_train_valid_test500_p=20%.pt
    max_len=1024
elif [ "$dataset_name" = "s1k_short" ]; then
    dataset=/extrahome0/user/data/s1k/s1k_short_cot.pt
    max_len=2048
elif [ "$dataset_name" = "s1k_evol" ]; then
    dataset=/media/user/data/ga/s1k/s1k_evol_cot.pt
    max_len=2048
elif [ "$dataset_name" = "math7500_short" ]; then
    dataset=/media/user/data/ga/math7500/math7500_short_cot.pt
    max_len=1024
    micro_train_batch_size=2
    max_epochs=3
else
    echo "Unknown dataset_name"
    dataset=""
fi

if [ "$model_type" = "llama3-base" ]; then
    model_name_or_path=/extrahome0/download/Meta-Llama-3.1-8B-Base
    micro_train_batch_size=1
    zero_stage=2
#    learning_rate=7e-7
    learning_rate=5e-6
    max_epochs=10
elif [ "$model_type" = "llama3-it" ]; then
    model_name_or_path=/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
    micro_train_batch_size=1
    zero_stage=2
    learning_rate=7e-7
    max_epochs=10
elif [ "$model_type" = "qwen-it" ]; then
#    model_name_or_path=/extrahome0/HF_models/Qwen2.5-7B-Instruct/
    model_name_or_path=/media/user/pretrained/Qwen2.5-7B-Instruct
    zero_stage=2
    learning_rate=2e-4
elif [ "$model_type" = "qwen-math" ]; then
    model_name_or_path=/extrahome0/user/model/Qwen2.5-Math-1.5B/
    micro_train_batch_size=8
    zero_stage=1
else
    echo "Unknown model_type"
    model_name_or_path=""
fi

deepspeed --include localhost:${device} --master_port 10001 --module run_sft \
   --save_path /media/user/output/ \
   --save_steps -1 \
   --logging_steps 2 \
   --eval_steps -1 \
   --train_batch_size 16 \
   --micro_train_batch_size ${micro_train_batch_size} \
   --pretrain ${model_name_or_path} \
   --bf16 \
   --max_epochs ${max_epochs} \
   --max_len ${max_len} \
   --zero_stage ${zero_stage} \
   --learning_rate ${learning_rate} \
   --dataset ${dataset} \
   --input_key prompt \
   --output_key response \
   --dataset_name ${dataset_name} \
   --method_name sft \
   --model_type ${model_type} \
   --dataset_sample 32 \
   --gradient_checkpointing \
   --lora_rank 0 \
   --lora_alpha 32 \
   --flash_attn \
#   --adam_offload
#   --use_wandb True \
#   --wandb_project evol_cot


##!/bin/bash
## shellcheck disable=SC2068
## read parameters
#idx=0
#for i in $@
#do
#  args[${idx}]=$i
#  let "idx=${idx}+1"
#done
#
#device=${args[0]}
#
## llama3.1-8B
## /extrahome0/download/Meta-Llama-3.1-8B-Base
## /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
## llama2-7B
## /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf
#deepspeed --include localhost:${device} --master_port 10001 run_sft.py \
#--data_path /userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt \
#--output_dir /extrahome0/user/output/ \
#--dataset_name prm_r1 \
#--model_name_or_path /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/ \
#--learning_rate 5e-6 \
#--max_seq_length 2048 \
#--num_train_epochs 1 \
#--per_device_train_batch_size 1 \
#--gradient_accumulation_steps 8 \
#--eval_strategy no \
#--logging_strategy steps \
#--logging_steps 50 \
#--save_strategy steps \
#--save_steps 50 \
#--seed 42 \
#--use_peft \
#--lora_r 8 \
#--lora_alpha 16 \
#--load_in_8bit \
#--bf16 \
#--dataset_sample 0 \
#--method_name sft \
#--model_type llama3.1-8b-it
