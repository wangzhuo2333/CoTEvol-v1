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
#model_type=${args[1]}

# llama3.1-8B
# /extrahome0/download/Meta-Llama-3.1-8B-Base
# /userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
# llama2-7B
# /userhome/Research_HUB/RLHF/trlx/examples/cl_rlhf/download/Llama-2-7b-hf
##set -x
#if [ "$model_type" = "llama3-base" ]; then
#    model_name_or_path=/extrahome0/download/Meta-Llama-3.1-8B-Base
#elif [ "$model_type" = "llama3-it" ]; then
#    model_name_or_path=/userhome/Research_HUB/RLHF/trlx/examples/ul_rlhf/download/Meta-Llama-3.1-8B-Instruct/hf/
#else
#    echo "Unknown model_type"
#    model_name_or_path=""
#fi

model_name_or_path=/extrahome0/HF_models/Qwen2.5-7B-Instruct/

#read -r -d '' training_commands <<EOF
#  run_dpo \
#   --save_path /extrahome0/user/output/ \
#   --save_steps 100 \
#   --logging_steps 100 \
#   --eval_steps -1 \
#   --train_batch_size 16 \
#   --micro_train_batch_size 1 \
#   --pretrain /extrahome0/download/Meta-Llama-3.1-8B-Base \
#   --bf16 \
#   --max_epochs 1 \
#   --max_len 2048 \
#   --zero_stage 1 \
#   --learning_rate 1e-6 \
#   --beta 0.5 \
#   --dataset /userhome/Research_HUB/Reasoning/data_dir/prm800k_cut_train_valid_test.pt \
#   --chosen_key chosen \
#   --rejected_key rejected \
#   --dataset_name prm_r1 \
#   --method_name open_dpo \
#   --model_type ${model_type} \
#   --dataset_sample 0 \
#   --gradient_checkpointing \
#   --lora_rank 16 \
#   --lora_alpha 32
#EOF
    # --load_in_4bit
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
    # --flash_attn
    # --apply_chat_template
    # --load_checkpoint

#if [[ ${1} != "slurm" ]]; then
#    deepspeed --module $training_commands
#fi

#deepspeed --include localhost:${device} --master_port 10001 --module $training_commands
#deepspeed --include localhost:${device} --master_port 10001 --module test \
#   --save_path /extrahome0/user/output/ \
#   --save_steps 2 \
#   --logging_steps 1 \
#   --eval_steps -1 \
#   --train_batch_size 16 \
#   --micro_train_batch_size 1 \
#   --pretrain ${model_name_or_path} \
#   --bf16 \
#   --max_epochs 1 \
#   --max_len 20000 \
#   --zero_stage 1 \
#   --learning_rate 1e-6 \
#   --dataset /extrahome0/user/data/s1k/s1k_short_cot.pt \
#   --dataset_name prm_r1 \
#   --method_name open_dpo \
#   --model_type ${model_type} \
#   --dataset_sample 64 \
#   --gradient_checkpointing \
#   --lora_rank 0 \
#   --lora_alpha 32 \
#   --output_key response \
#   --eval_steps 1 \

echo ${device:0:1}
echo ${device:1:1}
echo ${device:2:2} # 22: 23, 12: 12, 23: 23,
echo ${device:3:3}
